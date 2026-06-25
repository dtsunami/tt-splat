#!/usr/bin/env python3
"""Fused backward SFPU kernel — single Gaussian (S=0), verify the 7 grad-product tiles vs torch.
Isolates the arithmetic+recip+copy_dest_values before adding the reverse-loop S/T_run state."""
import struct, math, torch, ttnn
HOME = (1, 1)


def f2u(x): return struct.unpack("<I", struct.pack("<f", float(x)))[0]
NHALF, NEG1, ONE, TWO, NEG2 = f2u(-0.5), f2u(-1.0), f2u(1.0), f2u(2.0), f2u(-2.0)

READER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t px=get_arg_val<uint32_t>(2), py=get_arg_val<uint32_t>(3);
    uint32_t dl=get_arg_val<uint32_t>(4), tf=get_arg_val<uint32_t>(5), nb=get_arg_val<uint32_t>(6);
    cb_reserve_back(0,1); noc_async_read(get_noc_addr(sx,sy,px), get_write_ptr(0), nb);
    cb_reserve_back(1,1); noc_async_read(get_noc_addr(sx,sy,py), get_write_ptr(1), nb);
    cb_reserve_back(2,1); noc_async_read(get_noc_addr(sx,sy,dl), get_write_ptr(2), nb);
    cb_reserve_back(3,1); noc_async_read(get_noc_addr(sx,sy,tf), get_write_ptr(3), nb);
    noc_async_read_barrier();
    cb_push_back(0,1); cb_push_back(1,1); cb_push_back(2,1); cb_push_back(3,1);
}
"""

COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"
#include "api/dataflow/circular_buffer.h"
#define NHALF 0xBF000000u
#define NEG2  0xC0000000u
#define TWO   0x40000000u
#define NEG1  0xBF800000u
#define ONE   0x3F800000u
void kernel_main() {
    uint32_t cx=get_arg_val<uint32_t>(0), cy=get_arg_val<uint32_t>(1), a=get_arg_val<uint32_t>(2);
    uint32_t twob=get_arg_val<uint32_t>(3), c=get_arg_val<uint32_t>(4), op=get_arg_val<uint32_t>(5), col=get_arg_val<uint32_t>(6), b=get_arg_val<uint32_t>(7);
    init_sfpu(0,16); binop_with_scalar_tile_init(); copy_dest_values_init();
    recip_tile_init(); exp_tile_init();
    cb_wait_front(0,1); cb_wait_front(1,1); cb_wait_front(2,1); cb_wait_front(3,1);
    cb_reserve_back(16,1); cb_reserve_back(17,1); cb_reserve_back(18,1);
    cb_reserve_back(19,1); cb_reserve_back(20,1); cb_reserve_back(21,1); cb_reserve_back(22,1);
    tile_regs_acquire();
    // dx -> d0, dy -> d1
    copy_tile_init(0); copy_tile(0,0,0); sub_unary_tile(0, cx);
    copy_tile_init(1); copy_tile(1,0,1); sub_unary_tile(1, cy);
    mul_binary_tile_init(); add_binary_tile_init();
    // power -> d12
    mul_binary_tile(0,0,12); mul_unary_tile(12, a);
    mul_binary_tile(1,1,13); mul_unary_tile(13, c); add_binary_tile(12,13,12);
    mul_binary_tile(0,1,13); mul_unary_tile(13, twob); add_binary_tile(12,13,12);
    // gexp -> d10
    mul_unary_tile(12, NHALF); exp_tile(12); copy_dest_values(12,10);
    // alpha -> d11
    copy_dest_values(10,11); mul_unary_tile(11, op);
    // rec=1/(1-alpha) -> d12
    copy_dest_values(11,12); mul_unary_tile(12, NEG1); add_unary_tile(12, ONE); recip_tile(12);
    // T_i = T_final*rec -> d13
    copy_tile_init(3); copy_tile(3,0,13); mul_binary_tile(13,12,13);
    // w = T_i*alpha -> d14
    copy_dest_values(13,14); mul_binary_tile(14,11,14);
    // dLda = (T_i*col) * dLdC -> d15    (S=0 so dCda = T_i*col)
    copy_dest_values(13,15); mul_unary_tile(15, col);
    copy_tile_init(2); copy_tile(2,0,12); mul_binary_tile(15,12,15);
    // base = dLda*alpha*(-0.5) -> d2
    copy_dest_values(15,2); mul_binary_tile(2,11,2); mul_unary_tile(2, NHALF);
    // ---- 7 products into d3..d9 ----
    copy_tile_init(2); copy_tile(2,0,11); mul_binary_tile(11,14,3);          // g_col = dLdC*w   (reload dLdC->d11)
    mul_binary_tile(15,10,4);                                                // g_op  = dLda*gexp
    mul_binary_tile(0,0,11); mul_binary_tile(11,2,5);                        // g_a   = base*dx^2
    mul_binary_tile(1,1,11); mul_binary_tile(11,2,7);                        // g_c   = base*dy^2
    mul_binary_tile(0,1,11); mul_unary_tile(11, TWO); mul_binary_tile(11,2,6); // g_b = base*2dxdy
    copy_dest_values(0,11); mul_unary_tile(11, a); copy_dest_values(1,13); mul_unary_tile(13, b);
    // note: twob = 2b but cx-grad needs b; pass b separately below via half? -> handled host-side (see harness)
    add_binary_tile(11,13,11); mul_unary_tile(11, NEG2); mul_binary_tile(11,2,8);  // g_cx = base*-2(a dx + b dy)
    copy_dest_values(0,11); mul_unary_tile(11, b); copy_dest_values(1,13); mul_unary_tile(13, c);
    add_binary_tile(11,13,11); mul_unary_tile(11, NEG2); mul_binary_tile(11,2,9);  // g_cy = base*-2(b dx + c dy)
    tile_regs_commit(); tile_regs_wait();
    pack_tile(3,16); pack_tile(4,17); pack_tile(5,18); pack_tile(7,19);
    pack_tile(6,20); pack_tile(8,21); pack_tile(9,22);
    tile_regs_release();
    cb_push_back(16,1); cb_push_back(17,1); cb_push_back(18,1); cb_push_back(19,1);
    cb_push_back(20,1); cb_push_back(21,1); cb_push_back(22,1);
    cb_pop_front(0,1); cb_pop_front(1,1); cb_pop_front(2,1); cb_pop_front(3,1);
}
"""

WRITER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1), nb=get_arg_val<uint32_t>(2);
    for (uint32_t j=0;j<7;j++){
        uint32_t cb=16+j; uint32_t dst=get_arg_val<uint32_t>(3+j);
        cb_wait_front(cb,1);
        noc_async_write(get_read_ptr(cb), get_noc_addr(sx,sy,dst), nb);
        noc_async_write_barrier();
        cb_pop_front(cb,1);
    }
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
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        ii, jj = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
        PX, PY = jj.double(), ii.double()
        cx, cy = 14.0, 17.0
        a, b, c = 0.06, 0.003, 0.05
        op, col = 0.6, 0.7
        Tf = (0.5 + 0.4 * torch.rand(32, 32)).double()       # arbitrary T_final
        dLdC = torch.rand(32, 32).double()
        # torch golden (S=0): dCda=T_i*col, T_i=Tf/(1-alpha)
        dx, dy = PX - cx, PY - cy
        gexp = torch.exp(-0.5 * (a*dx*dx + 2*b*dx*dy + c*dy*dy)); alpha = op * gexp
        Ti = Tf / (1 - alpha); w = Ti * alpha
        dLda = dLdC * (Ti * col); base = dLda * alpha * (-0.5)
        gold = {"col": (dLdC*w).sum(), "op": (dLda*gexp).sum(), "a": (base*dx*dx).sum(),
                "b": (base*2*dx*dy).sum(), "c": (base*dy*dy).sum(),
                "cx": (base*(-2)*(a*dx+b*dy)).sum(), "cy": (base*(-2)*(b*dx+c*dy)).sum()}

        px_t, py_t, dl_t, tf_t = l1(dev, PX), l1(dev, PY), l1(dev, dLdC), l1(dev, Tf)
        outs = [l1(dev) for _ in range(7)]
        hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(*HOME)); sx, sy = hp.x, hp.y
        NB = 32*32*4
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])

        def _ccfg():
            return ttnn.ComputeConfigDescriptor()    # bf16 dst (16 slots); fp32_dest halves to 8 -> aliases d8-15

        def rt(arr):
            r = ttnn.RuntimeArgs(); r[HOME[0]][HOME[1]] = arr; return r
        cbf = lambda i: ttnn.CBDescriptor(total_size=2*NB, core_ranges=crs,
                 format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=NB)])
        ks = lambda s, arr, cfg, cta=[]: ttnn.KernelDescriptor(
            kernel_source=s, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=crs, runtime_args=rt(arr), compile_time_args=cta, config=cfg)
        # NOTE: kernel uses 'twob' slot for BOTH 2b (power) and b (centers). Pass 2b; the center grads
        # then use 2b in place of b -> WRONG. Fix: pass b in the 'twob' arg-slot used by center code.
        # Simpler for this isolation test: set b such that 2b and b distinct matters -> we pass real (a,2b,c,op,col)
        # and ACCEPT the cx/cy use 2b (documented); compare only col/op/a/b/c here, fix centers in next iter.
        comp_args = [f2u(cx), f2u(cy), f2u(a), f2u(2*b), f2u(c), f2u(op), f2u(col), f2u(b)]
        out_addr = [o.buffer_address() for o in outs]
        prog = ttnn.ProgramDescriptor(kernels=[
            ks(READER, [sx, sy, px_t.buffer_address(), py_t.buffer_address(), dl_t.buffer_address(), tf_t.buffer_address(), NB], ttnn.ReaderConfigDescriptor()),
            ks(COMPUTE, comp_args, _ccfg()),
            ks(WRITER, [sx, sy, NB] + out_addr, ttnn.WriterConfigDescriptor())],
            semaphores=[], cbs=[cbf(i) for i in (0, 1, 2, 3, 16, 17, 18, 19, 20, 21, 22)])
        ttnn.generic_op([px_t, outs[0]], prog)

        names = ["col", "op", "a", "c", "b", "cx", "cy"]   # pack order: 16=col,17=op,18=a,19=c,20=b,21=cx,22=cy
        got = {n: float(ttnn.to_torch(outs[i]).reshape(32, 32).sum()) for i, n in enumerate(names)}
        scale = max(abs(float(gold[n])) for n in got)        # gradient magnitude scale (Adam normalizes per-group)
        print(f"grad   device      golden    abs_err/scale  (scale={scale:.1f})")
        worst = 0.0
        for n in ["col", "op", "a", "b", "c", "cx", "cy"]:
            g, go = got[n], float(gold[n]); e = abs(g-go)/scale; worst = max(worst, e)
            print(f"  {n:3} {g:11.4f} {go:11.4f}  {e:.2e}")
        print(f"BWD_KERNEL_ALL7 worst_err/scale={worst:.2e} -> {'OK' if worst<2e-2 else 'FAIL'}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
