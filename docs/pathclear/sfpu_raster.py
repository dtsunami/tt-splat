#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SFPU rasterizer pathclear (milestone 1): fuse one Gaussian's alpha eval into a single
COMPUTE kernel on the SFPU/TRISC engine (vs M3's many ttnn-op dispatches).

  alpha = op * exp(-0.5 * (a*dx^2 + 2b*dx*dy + c*dy^2)),  dx=PX-cx, dy=PY-cy

All on-chip in the compute kernel (dst registers): copy PX/PY -> dx,dy via scalar sub ->
dx^2,dy^2,dx*dy via dst-binary mul -> scale -> sum -> *-0.5 -> exp_tile -> *op. Validated
vs CPU golden. Proves the SFPU does the Gaussian eval (last engine in the kernel-diff table).
"""
import struct, math, torch, ttnn

HOME = (0, 0)
# one Gaussian
CX, CY, OP = 16.0, 14.0, 0.8
SX, SY, TH = 4.0, 2.0, 0.5

def f2u(x): return struct.unpack("<I", struct.pack("<f", float(x)))[0]

def conic():
    ct, st = math.cos(TH), math.sin(TH)
    R = torch.tensor([[ct, -st], [st, ct]])
    cov = R @ torch.diag(torch.tensor([SX*SX, SY*SY])) @ R.T
    M = torch.inverse(cov)
    return float(M[0,0]), float(M[0,1]), float(M[1,1])

READER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t px=get_arg_val<uint32_t>(2), py=get_arg_val<uint32_t>(3), nbytes=get_arg_val<uint32_t>(4);
    cb_reserve_back(0,1); noc_async_read(get_noc_addr(sx,sy,px), get_write_ptr(0), nbytes);
    cb_reserve_back(1,1); noc_async_read(get_noc_addr(sx,sy,py), get_write_ptr(1), nbytes);
    noc_async_read_barrier(); cb_push_back(0,1); cb_push_back(1,1);
}
"""

COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    uint32_t cxb=get_arg_val<uint32_t>(0), cyb=get_arg_val<uint32_t>(1), ab=get_arg_val<uint32_t>(2);
    uint32_t twob=get_arg_val<uint32_t>(3), cb=get_arg_val<uint32_t>(4), opb=get_arg_val<uint32_t>(5);
    uint32_t nhalf=get_arg_val<uint32_t>(6);
    init_sfpu(0, 16);
    binop_with_scalar_tile_init();
    cb_wait_front(0,1); cb_wait_front(1,1); cb_reserve_back(16,1);
    tile_regs_acquire();
    copy_tile_init(0); copy_tile(0,0,0);          // dst0 = PX
    copy_tile_init(1); copy_tile(1,0,1);          // dst1 = PY
    sub_unary_tile(0, cxb);                        // dst0 = dx
    sub_unary_tile(1, cyb);                        // dst1 = dy
    mul_binary_tile_init();
    mul_binary_tile(0,0,2);                        // dst2 = dx*dx
    mul_binary_tile(1,1,3);                        // dst3 = dy*dy
    mul_binary_tile(0,1,4);                        // dst4 = dx*dy
    mul_unary_tile(2, ab);                         // a*dx2
    mul_unary_tile(3, cb);                         // c*dy2
    mul_unary_tile(4, twob);                       // 2b*dxdy
    add_binary_tile_init();
    add_binary_tile(2,3,2);                        // a*dx2 + c*dy2
    add_binary_tile(2,4,2);                        // power
    mul_unary_tile(2, nhalf);                      // -0.5*power
    exp_tile_init(); exp_tile(2);                  // exp(...)
    mul_unary_tile(2, opb);                        // * op = alpha
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(2, 16);
    tile_regs_release();
    cb_push_back(16,1); cb_pop_front(0,1); cb_pop_front(1,1);
}
"""

WRITER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t out=get_arg_val<uint32_t>(2), nbytes=get_arg_val<uint32_t>(3);
    cb_wait_front(16,1);
    noc_async_write(get_read_ptr(16), get_noc_addr(sx,sy,out), nbytes);
    noc_async_write_barrier(); cb_pop_front(16,1);
}
"""

def l1(dev, data=None):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    sh = ttnn.ShardSpec(crs, [32,32], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sh)
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1,1,32,32]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1,1,32,32).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

def main():
    dev = ttnn.open_device(device_id=0)
    try:
        a, b, c = conic()
        ii, jj = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
        PX, PY = jj.float(), ii.float()
        px_t, py_t, out_t = l1(dev, PX), l1(dev, PY), l1(dev)
        hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(*HOME)); sx, sy = hp.x, hp.y
        NB = 32*32*4

        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
        def rt(args):
            r = ttnn.RuntimeArgs(); r[HOME[0]][HOME[1]] = args; return r
        cbf = lambda idx: ttnn.CBDescriptor(total_size=2*NB, core_ranges=crs,
                 format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=NB)])
        rdr = ttnn.KernelDescriptor(kernel_source=READER, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                 core_ranges=crs, runtime_args=rt([sx,sy,px_t.buffer_address(),py_t.buffer_address(),NB]),
                 compile_time_args=[], config=ttnn.ReaderConfigDescriptor())
        cmp = ttnn.KernelDescriptor(kernel_source=COMPUTE, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                 core_ranges=crs, runtime_args=rt([f2u(CX),f2u(CY),f2u(a),f2u(2*b),f2u(c),f2u(OP),f2u(-0.5)]),
                 compile_time_args=[], config=ttnn.ComputeConfigDescriptor())
        wtr = ttnn.KernelDescriptor(kernel_source=WRITER, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                 core_ranges=crs, runtime_args=rt([sx,sy,out_t.buffer_address(),NB]),
                 compile_time_args=[], config=ttnn.WriterConfigDescriptor())
        prog = ttnn.ProgramDescriptor(kernels=[rdr,cmp,wtr], semaphores=[],
                 cbs=[cbf(0),cbf(1),cbf(16)])
        ttnn.generic_op([px_t, out_t], prog)

        got = ttnn.to_torch(out_t).reshape(32,32)
        dx, dy = PX-CX, PY-CY
        gold = OP*torch.exp(-0.5*(a*dx*dx + 2*b*dx*dy + c*dy*dy))
        mse = float(((got-gold)**2).mean()); psnr = 10*math.log10(float(gold.max())**2/max(mse,1e-12))
        print(f"conic=({a:.4f},{b:.4f},{c:.4f})")
        print(f"alpha MSE={mse:.3e}  PSNR={psnr:.1f} dB  max_dev={float((got-gold).abs().max()):.3e}")
        print("SFPU_RASTER_OK" if mse < 1e-5 else "SFPU_RASTER_FAIL")
    finally:
        ttnn.close_device(dev)

if __name__ == "__main__":
    main()
