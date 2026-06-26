#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
KEYSTONE probe: reduce a 32x32 tile to a scalar via the MATRIX engine (2-stage ones-matmul) instead of
the pool reduce. Tests the load-bearing assumption: does matmul accumulation hold fp32-ish precision in a
bf16-DST kernel (no global fp32_dest_acc_en)? If yes, the reduce can RE-FUSE into the bf16 m17 arithmetic
kernel (one dispatch) — the pool reduce_tile FAILED this exact case at rel_err 1.0 (Stage 2 finding).

reduce(T) = 1.(ones @ T) -> colsums  2.(colsums @ ones) -> total  (each matmul = 32-term MACs, fp32 internal).
Tile = signed high-dynamic-range (a-grad shape: base * dx^2) so a naive 1024-term bf16 sum is catastrophic.
Gate: rel_err < 2e-2.  Run bf16-DST and fp32-DST; compare.
"""
import torch, ttnn
HOME = (1, 1)

READER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t T=get_arg_val<uint32_t>(2), ones=get_arg_val<uint32_t>(3), nb=get_arg_val<uint32_t>(4);
    cb_reserve_back(0,1); noc_async_read(get_noc_addr(sx,sy,T), get_write_ptr(0), nb);
    cb_reserve_back(1,1); noc_async_read(get_noc_addr(sx,sy,ones), get_write_ptr(1), nb);
    noc_async_read_barrier();
    cb_push_back(0,1); cb_push_back(1,1);
}
"""

# 2-stage matmul reduce: cb0=T, cb1=ones, cb2=colsums(mid), cb16=total
COMPUTE = r"""
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    cb_wait_front(0,1); cb_wait_front(1,1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(1, 0, 2);   // matmul: in0->SrcB, in1->SrcA
    // ---- stage 1: ones @ T -> colsums (every row = column sums) ----
    matmul_init(1, 0);
    cb_reserve_back(2,1);
    tile_regs_acquire();
    matmul_tiles(1, 0, 0, 0, 0);                              // dst0 = ones @ T
    tile_regs_commit(); tile_regs_wait();
    pack_tile(0, 2);
    tile_regs_release();
    cb_push_back(2,1);
    // ---- stage 2: colsums @ ones -> total (every cell = grand total) ----
    cb_wait_front(2,1);
    matmul_init(2, 1);
    cb_reserve_back(16,1);
    tile_regs_acquire();
    matmul_tiles(2, 1, 0, 0, 0);                              // dst0 = colsums @ ones
    tile_regs_commit(); tile_regs_wait();
    pack_tile(0, 16);
    tile_regs_release();
    cb_push_back(16,1);
    cb_pop_front(0,1); cb_pop_front(1,1); cb_pop_front(2,1);
}
"""

WRITER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1), co=get_arg_val<uint32_t>(2), nb=get_arg_val<uint32_t>(3);
    cb_wait_front(16,1);
    noc_async_write(get_read_ptr(16), get_noc_addr(sx,sy,co), nb);
    noc_async_write_barrier();
    cb_pop_front(16,1);
}
"""


def l1(dev, data=None):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(crs, [32, 32], ttnn.ShardOrientation.ROW_MAJOR))
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, 32, 32).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)


def run(dev, T, fp32):
    NB = 32 * 32 * 4
    Tt, onest, outt = l1(dev, T), l1(dev, torch.ones(32, 32)), l1(dev)
    hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(*HOME)); sx, sy = hp.x, hp.y
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])

    def rt(a):
        r = ttnn.RuntimeArgs(); r[HOME[0]][HOME[1]] = a; return r
    cbf = lambda i: ttnn.CBDescriptor(total_size=2 * NB, core_ranges=crs,
             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=NB)])
    ks = lambda s, a, cfg: ttnn.KernelDescriptor(kernel_source=s, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
             core_ranges=crs, runtime_args=rt(a), compile_time_args=[], config=cfg)
    cfg = ttnn.ComputeConfigDescriptor()
    if fp32:
        cfg.fp32_dest_acc_en = True
    prog = ttnn.ProgramDescriptor(kernels=[
        ks(READER, [sx, sy, Tt.buffer_address(), onest.buffer_address(), NB], ttnn.ReaderConfigDescriptor()),
        ks(COMPUTE, [], cfg),
        ks(WRITER, [sx, sy, outt.buffer_address(), NB], ttnn.WriterConfigDescriptor())],
        semaphores=[], cbs=[cbf(0), cbf(1), cbf(2), cbf(16)])
    ttnn.generic_op([Tt, outt], prog)
    got = float(ttnn.to_torch(outt).reshape(32, 32)[0, 0])
    gold = float(T.double().sum())
    return got, gold, abs(got - gold) / (abs(gold) + 1e-9)


def main():
    torch.manual_seed(0)
    ii, jj = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    dx = (jj - 16).float()
    scenes = {
        "a-grad (base*dx^2 signed)": (torch.randn(32, 32) * 0.01) * (dx ** 2),     # the case pool-reduce failed
        "signed large N(0,1e3)":      torch.randn(32, 32) * 1e3,
        "near-cancel":                torch.randn(32, 32) - torch.randn(32, 32).mean(),
    }
    dev = ttnn.open_device(device_id=0)
    try:
        print(f"  {'scene':28s} {'mode':9s} {'gold':>13s} {'matmul':>13s} {'rel_err':>10s}")
        worst = {"bf16": 0.0, "fp32": 0.0}
        for name, T in scenes.items():
            for fp32, tag in ((False, "bf16-dst"), (True, "fp32-dst")):
                got, gold, e = run(dev, T, fp32)
                worst["fp32" if fp32 else "bf16"] = max(worst["fp32" if fp32 else "bf16"], e)
                print(f"  {name:28s} {tag:9s} {gold:13.4f} {got:13.4f} {e:10.2e}", flush=True)
        print(f"\n  WORST bf16-dst={worst['bf16']:.2e}  fp32-dst={worst['fp32']:.2e}  (gate 2e-2)")
        print(f"  bf16-dst matmul reduce: {'HOLDS — re-fusion viable (pool-reduce failed this at 1.0)' if worst['bf16']<2e-2 else 'FAILS — need fp32-dst (no re-fusion)'}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
