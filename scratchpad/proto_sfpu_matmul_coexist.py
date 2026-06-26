#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
INCREMENTAL de-risk for the matmul re-fusion: do an SFPU op (input*2) THEN a matmul-ones reduce in ONE
compute kernel. Tests the load-bearing UNKNOWN: can init_sfpu (SFPU arithmetic) and
compute_kernel_hw_startup<SrcOrder::Reverse>+matmul_init (matrix engine) COEXIST / re-init within one
kernel body? If this works, fusing m17-arith + matmul-reduce is viable; if it hangs/errors, redirect to a
separate (bf16) reduce dispatch. Gate: device scalar == host sum(input*2), rel < 2e-2.
"""
import torch, ttnn
HOME = (1, 1)

READER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t in=get_arg_val<uint32_t>(2), ones=get_arg_val<uint32_t>(3), nb=get_arg_val<uint32_t>(4);
    cb_reserve_back(0,1); noc_async_read(get_noc_addr(sx,sy,in), get_write_ptr(0), nb);
    cb_reserve_back(1,1); noc_async_read(get_noc_addr(sx,sy,ones), get_write_ptr(1), nb);
    noc_async_read_barrier();
    cb_push_back(0,1); cb_push_back(1,1);
}
"""

COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#define TWO 0x40000000u
void kernel_main() {
    cb_wait_front(0,1); cb_wait_front(1,1);
    // ===== PHASE A: SFPU — dst = input * 2 -> CB 2 =====
    init_sfpu(0,16); binop_with_scalar_tile_init();
    cb_reserve_back(2,1);
    tile_regs_acquire();
    copy_tile_init(0); copy_tile(0,0,0); mul_unary_tile(0, TWO);
    tile_regs_commit(); tile_regs_wait();
    pack_tile(0,2);
    tile_regs_release();
    cb_push_back(2,1);
    // ===== PHASE B: MATRIX engine — 2-stage ones-matmul reduce of CB2 =====
    cb_wait_front(2,1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(1, 2, 16);   // reconfigure for matmul mid-kernel
    matmul_init(1, 2);
    cb_reserve_back(3,1);
    tile_regs_acquire();
    matmul_tiles(1, 2, 0, 0, 0);                              // ones @ (in*2) = colsums
    tile_regs_commit(); tile_regs_wait();
    pack_tile(0,3);
    tile_regs_release();
    cb_push_back(3,1);
    cb_wait_front(3,1);
    matmul_init(3, 1);
    cb_reserve_back(16,1);
    tile_regs_acquire();
    matmul_tiles(3, 1, 0, 0, 0);                              // colsums @ ones = total
    tile_regs_commit(); tile_regs_wait();
    pack_tile(0,16);
    tile_regs_release();
    cb_push_back(16,1);
    cb_pop_front(0,1); cb_pop_front(1,1); cb_pop_front(2,1); cb_pop_front(3,1);
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


def main():
    torch.manual_seed(0)
    T = torch.randn(32, 32) * 3.0
    dev = ttnn.open_device(device_id=0)
    try:
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
        prog = ttnn.ProgramDescriptor(kernels=[
            ks(READER, [sx, sy, Tt.buffer_address(), onest.buffer_address(), NB], ttnn.ReaderConfigDescriptor()),
            ks(COMPUTE, [], ttnn.ComputeConfigDescriptor()),
            ks(WRITER, [sx, sy, outt.buffer_address(), NB], ttnn.WriterConfigDescriptor())],
            semaphores=[], cbs=[cbf(0), cbf(1), cbf(2), cbf(3), cbf(16)])
        ttnn.generic_op([Tt, outt], prog)
        got = float(ttnn.to_torch(outt).reshape(32, 32)[0, 0])
        gold = float((T.double() * 2).sum())
        e = abs(got - gold) / (abs(gold) + 1e-9)
        print(f"  SFPU(*2)+matmul-reduce coexist: device={got:.4f} gold={gold:.4f} rel={e:.2e} -> "
              f"{'OK — SFPU+matmul COEXIST in one kernel' if e < 2e-2 else 'FAIL'}", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
