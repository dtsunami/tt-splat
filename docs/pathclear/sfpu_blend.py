#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SFPU rasterizer milestone 2: full front-to-back BLEND LOOP fused in ONE compute kernel.
Extends M5 (single-Gaussian eval) — now loops N Gaussians, keeping C and transmittance T
in dst registers across the loop:  C += T*alpha*col ; T *= (1-alpha).  One dispatch for the
whole tile render (vs M3's ~6 ttnn ops per Gaussian). Validated vs CPU golden.

dst layout: dst0=C, dst1=T (persist across loop); dst2-5 = scratch per Gaussian.
Float constants via hex bit-literals (0.0/1.0/-1.0/-0.5).
"""
import struct, math, torch, ttnn

HOME = (0, 0)
N = 8

def f2u(x): return struct.unpack("<I", struct.pack("<f", float(x)))[0]

READER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t px=get_arg_val<uint32_t>(2), py=get_arg_val<uint32_t>(3), nb=get_arg_val<uint32_t>(4);
    cb_reserve_back(0,1); noc_async_read(get_noc_addr(sx,sy,px), get_write_ptr(0), nb);
    cb_reserve_back(1,1); noc_async_read(get_noc_addr(sx,sy,py), get_write_ptr(1), nb);
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
#define ZERO 0x00000000u
#define ONE  0x3F800000u
#define NEG1 0xBF800000u
#define NHALF 0xBF000000u
void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    init_sfpu(0, 16);
    binop_with_scalar_tile_init();
    cb_wait_front(0,1); cb_wait_front(1,1); cb_reserve_back(16,1);
    tile_regs_acquire();
    // C=dst0=0, T=dst1=1   (derive from PX so no zeros CB needed)
    copy_tile_init(0); copy_tile(0,0,0); mul_unary_tile(0, ZERO);
    copy_tile(0,0,1); mul_unary_tile(1, ZERO); add_unary_tile(1, ONE);
    for (uint32_t g = 0; g < N; g++) {
        uint32_t b = g*7;
        uint32_t cxb=get_arg_val<uint32_t>(b+0), cyb=get_arg_val<uint32_t>(b+1);
        uint32_t ab=get_arg_val<uint32_t>(b+2), twob=get_arg_val<uint32_t>(b+3);
        uint32_t cc=get_arg_val<uint32_t>(b+4), opb=get_arg_val<uint32_t>(b+5), colb=get_arg_val<uint32_t>(b+6);
        copy_tile_init(0); copy_tile(0,0,2); sub_unary_tile(2, cxb);   // dx
        copy_tile_init(1); copy_tile(1,0,3); sub_unary_tile(3, cyb);   // dy
        mul_binary_tile_init();
        mul_binary_tile(2,2,4); mul_unary_tile(4, ab);                 // a*dx2 -> dst4
        mul_binary_tile(3,3,5); mul_unary_tile(5, cc);                 // c*dy2 -> dst5
        mul_binary_tile(2,3,2); mul_unary_tile(2, twob);              // 2b*dxdy -> dst2
        add_binary_tile_init();
        add_binary_tile(4,5,4); add_binary_tile(4,2,4);               // power -> dst4
        mul_unary_tile(4, NHALF);
        exp_tile_init(); exp_tile(4);                                 // exp -> dst4
        mul_unary_tile(4, opb);                                       // alpha -> dst4
        mul_binary_tile_init(); mul_binary_tile(1,4,5);              // T*alpha -> dst5
        mul_unary_tile(5, colb);                                      // *col = contrib
        add_binary_tile_init(); add_binary_tile(0,5,0);              // C += contrib
        mul_unary_tile(4, NEG1); add_unary_tile(4, ONE);            // 1-alpha -> dst4
        mul_binary_tile_init(); mul_binary_tile(1,4,1);             // T *= (1-alpha)
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, 16);
    tile_regs_release();
    cb_push_back(16,1); cb_pop_front(0,1); cb_pop_front(1,1);
}
"""

WRITER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t out=get_arg_val<uint32_t>(2), nb=get_arg_val<uint32_t>(3);
    cb_wait_front(16,1);
    noc_async_write(get_read_ptr(16), get_noc_addr(sx,sy,out), nb);
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
        g = torch.Generator().manual_seed(2)
        cx = torch.rand(N,generator=g)*32; cy = torch.rand(N,generator=g)*32
        sx_ = 3+torch.rand(N,generator=g)*5; sy_ = 3+torch.rand(N,generator=g)*5
        th = torch.rand(N,generator=g)*math.pi
        op = 0.4+torch.rand(N,generator=g)*0.4; col = 0.3+torch.rand(N,generator=g)*0.6
        depth = torch.rand(N,generator=g); order = torch.argsort(depth).tolist()
        abc = []
        for i in range(N):
            ct,st = math.cos(th[i]), math.sin(th[i])
            R = torch.tensor([[ct,-st],[st,ct]])
            cov = R@torch.diag(torch.tensor([sx_[i]**2, sy_[i]**2]))@R.T
            M = torch.inverse(cov); abc.append((float(M[0,0]),float(M[0,1]),float(M[1,1])))

        ii,jj = torch.meshgrid(torch.arange(32),torch.arange(32),indexing="ij")
        PX,PY = jj.float(), ii.float()
        # golden (front->back)
        C = torch.zeros(32,32); T = torch.ones(32,32)
        for i in order:
            a,b,c = abc[i]; dx,dy = PX-float(cx[i]), PY-float(cy[i])
            al = float(op[i])*torch.exp(-0.5*(a*dx*dx+2*b*dx*dy+c*dy*dy))
            C = C + T*al*float(col[i]); T = T*(1-al)
        gold = C

        px_t,py_t,out_t = l1(dev,PX), l1(dev,PY), l1(dev)
        hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(*HOME)); sx,sy = hp.x,hp.y
        NB = 32*32*4
        # params flat in blend order
        params = []
        for i in order:
            a,b,c = abc[i]
            params += [f2u(cx[i]),f2u(cy[i]),f2u(a),f2u(2*b),f2u(c),f2u(op[i]),f2u(col[i])]

        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME),ttnn.CoreCoord(*HOME))])
        def rt(a):
            r = ttnn.RuntimeArgs(); r[HOME[0]][HOME[1]] = a; return r
        cbf = lambda idx: ttnn.CBDescriptor(total_size=2*NB, core_ranges=crs,
                 format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=NB)])
        rdr = ttnn.KernelDescriptor(kernel_source=READER, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                 core_ranges=crs, runtime_args=rt([sx,sy,px_t.buffer_address(),py_t.buffer_address(),NB]),
                 compile_time_args=[], config=ttnn.ReaderConfigDescriptor())
        cmp = ttnn.KernelDescriptor(kernel_source=COMPUTE, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                 core_ranges=crs, runtime_args=rt(params), compile_time_args=[N], config=ttnn.ComputeConfigDescriptor())
        wtr = ttnn.KernelDescriptor(kernel_source=WRITER, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                 core_ranges=crs, runtime_args=rt([sx,sy,out_t.buffer_address(),NB]),
                 compile_time_args=[], config=ttnn.WriterConfigDescriptor())
        ttnn.generic_op([px_t,out_t], ttnn.ProgramDescriptor(kernels=[rdr,cmp,wtr], semaphores=[], cbs=[cbf(0),cbf(1),cbf(16)]))

        got = ttnn.to_torch(out_t).reshape(32,32)
        mse = float(((got-gold)**2).mean()); psnr = 10*math.log10(float(gold.max())**2/max(mse,1e-12))
        print(f"N={N} blend-loop fused kernel  MSE={mse:.3e}  PSNR={psnr:.1f} dB  max_dev={float((got-gold).abs().max()):.3e}")
        print("SFPU_BLEND_OK" if mse < 1e-4 else "SFPU_BLEND_FAIL")
    finally:
        ttnn.close_device(dev)

if __name__ == "__main__":
    main()
