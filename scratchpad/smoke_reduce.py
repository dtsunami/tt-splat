#!/usr/bin/env python3
"""De-risk the ONE unproven primitive: in-kernel reduce_tile<SUM, REDUCE_SCALAR> via generic_op.
Sum a [32,32] tile -> scalar at [0,0]. If this works, the fused backward is feasible."""
import sys, struct, torch, ttnn
HOME = (1, 1)

READER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t da=get_arg_val<uint32_t>(2), nb=get_arg_val<uint32_t>(3);
    cb_reserve_back(0,1); noc_async_read(get_noc_addr(sx,sy,da), get_write_ptr(0), nb);
    noc_async_read_barrier();
    cb_push_back(0,1);
    // reduce SUM scaler: f32 1.0 in the first row (16 words) of each of 4 faces (256-u32 stride), rest 0
    cb_reserve_back(1,1);
    volatile tt_l1_ptr uint32_t* sp = (volatile tt_l1_ptr uint32_t*)get_write_ptr(1);
    for (uint32_t k=0;k<1024;k++) sp[k]=0;
    const uint32_t ONE=0x3f800000u;
    for (uint32_t face=0; face<4; face++)
        for (uint32_t k=0;k<16;k++) sp[face*256 + k]=ONE;
    cb_push_back(1,1);
}
"""
COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/reduce.h"
void kernel_main() {
    cb_wait_front(0,1); cb_wait_front(1,1);
    cb_reserve_back(16,1);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(0, 1, 16);
    tile_regs_acquire();
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(0, 1, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, 16);
    tile_regs_release();
    reduce_uninit();
    cb_push_back(16,1);
    cb_pop_front(0,1); cb_pop_front(1,1);
}
"""
WRITER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t co=get_arg_val<uint32_t>(2), nb=get_arg_val<uint32_t>(3);
    cb_wait_front(16,1);
    noc_async_write(get_read_ptr(16), get_noc_addr(sx,sy,co), nb);
    noc_async_write_barrier();
    cb_pop_front(16,1);
}
"""


def l1(dev, data=None):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    sh = ttnn.ShardSpec(crs, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sh)
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, 32, 32).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        data = torch.rand(32, 32)
        gold = float(data.sum())
        da_t, out_t = l1(dev, data), l1(dev)
        hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(*HOME)); sx, sy = hp.x, hp.y
        NB = 32 * 32 * 4
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])

        def rt(a):
            r = ttnn.RuntimeArgs(); r[HOME[0]][HOME[1]] = a; return r
        cbf = lambda idx: ttnn.CBDescriptor(total_size=2 * NB, core_ranges=crs,
                 format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=NB)])
        ks = lambda src, rta, cfg, cta=[]: ttnn.KernelDescriptor(
            kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=crs, runtime_args=rt(rta), compile_time_args=cta, config=cfg)
        prog = ttnn.ProgramDescriptor(kernels=[
            ks(READER, [sx, sy, da_t.buffer_address(), NB], ttnn.ReaderConfigDescriptor()),
            ks(COMPUTE, [], ttnn.ComputeConfigDescriptor()),
            ks(WRITER, [sx, sy, out_t.buffer_address(), NB], ttnn.WriterConfigDescriptor())],
            semaphores=[], cbs=[cbf(0), cbf(1), cbf(16)])
        ttnn.generic_op([da_t, out_t], prog)
        got = ttnn.to_torch(out_t).reshape(32, 32)
        val = float(got[0, 0])
        print(f"reduce: device[0,0]={val:.4f}  gold={gold:.4f}  out.sum={float(got.sum()):.4f} "
              f"out.max={float(got.max()):.4f}  rel_err={abs(val-gold)/abs(gold):.2e}")
        print("SMOKE_REDUCE_OK" if abs(val - gold) / abs(gold) < 0.02 else "SMOKE_REDUCE_FAIL")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
