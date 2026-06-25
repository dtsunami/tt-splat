#!/usr/bin/env python3
"""Fused backward, K-Gaussian reverse loop. S(d10)/T_run(d11) kept RESIDENT in dst across the loop
(loaded only on g==0); each Gaussian packs its 7 products. Tests if dst persists across the per-
Gaussian acquire/pack/release. Verify K=2 vs torch full backward (with suffix-S)."""
import struct, math, torch, ttnn
HOME = (1, 1)
KG = 2


def f2u(x): return struct.unpack("<I", struct.pack("<f", float(x)))[0]

READER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t px=get_arg_val<uint32_t>(2), py=get_arg_val<uint32_t>(3);
    uint32_t dl=get_arg_val<uint32_t>(4), tf=get_arg_val<uint32_t>(5), sz=get_arg_val<uint32_t>(6), nb=get_arg_val<uint32_t>(7);
    cb_reserve_back(0,1); noc_async_read(get_noc_addr(sx,sy,px), get_write_ptr(0), nb);
    cb_reserve_back(1,1); noc_async_read(get_noc_addr(sx,sy,py), get_write_ptr(1), nb);
    cb_reserve_back(2,1); noc_async_read(get_noc_addr(sx,sy,dl), get_write_ptr(2), nb);
    cb_reserve_back(3,1); noc_async_read(get_noc_addr(sx,sy,tf), get_write_ptr(3), nb);
    cb_reserve_back(4,1); noc_async_read(get_noc_addr(sx,sy,sz), get_write_ptr(4), nb);
    noc_async_read_barrier();
    cb_push_back(0,1); cb_push_back(1,1); cb_push_back(2,1); cb_push_back(3,1); cb_push_back(4,1);
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
    constexpr uint32_t K = get_compile_time_arg_val(0);
    init_sfpu(0,16); binop_with_scalar_tile_init(); copy_dest_values_init();
    recip_tile_init(); exp_tile_init(); mul_binary_tile_init(); add_binary_tile_init();
    cb_wait_front(0,1); cb_wait_front(1,1); cb_wait_front(2,1); cb_wait_front(3,1); cb_wait_front(4,1);
    for (uint32_t g=0; g<K; g++) {
        uint32_t b0=g*8;
        uint32_t cx=get_arg_val<uint32_t>(b0+0), cy=get_arg_val<uint32_t>(b0+1), a=get_arg_val<uint32_t>(b0+2);
        uint32_t twob=get_arg_val<uint32_t>(b0+3), c=get_arg_val<uint32_t>(b0+4), op=get_arg_val<uint32_t>(b0+5),
                 col=get_arg_val<uint32_t>(b0+6), bb=get_arg_val<uint32_t>(b0+7);
        cb_reserve_back(16,1); cb_reserve_back(17,1); cb_reserve_back(18,1); cb_reserve_back(19,1);
        cb_reserve_back(20,1); cb_reserve_back(21,1); cb_reserve_back(22,1);
        tile_regs_acquire();
        if (g==0) { copy_tile_init(4); copy_tile(4,0,10); copy_tile_init(3); copy_tile(3,0,11); }  // S=0, T_run=T_final
        copy_tile_init(0); copy_tile(0,0,0); sub_unary_tile(0, cx);    // d0=dx
        copy_tile_init(1); copy_tile(1,0,1); sub_unary_tile(1, cy);    // d1=dy
        // power -> d12 ; gexp -> d12
        mul_binary_tile(0,0,12); mul_unary_tile(12,a);
        mul_binary_tile(1,1,13); mul_unary_tile(13,c); add_binary_tile(12,13,12);
        mul_binary_tile(0,1,13); mul_unary_tile(13,twob); add_binary_tile(12,13,12);
        mul_unary_tile(12,NHALF); exp_tile(12);                        // d12=gexp
        copy_dest_values(12,13); mul_unary_tile(13,op);                // d13=alpha
        copy_dest_values(13,14); mul_unary_tile(14,NEG1); add_unary_tile(14,ONE); recip_tile(14);  // d14=rec
        mul_binary_tile(11,14,11);                                     // d11=T_i (=new T_run, in place)
        copy_dest_values(11,15); mul_binary_tile(15,13,15);            // d15=w
        // dCda -> d2 = T_i*col - S*rec
        copy_dest_values(11,2); mul_unary_tile(2,col);                 // d2=T_i*col
        copy_dest_values(10,3); mul_binary_tile(3,14,3);               // d3=S*rec
        sub_binary_tile(2,3,2);                                        // d2=dCda
        // newS = S + w*col -> d10 (in place)
        copy_dest_values(15,3); mul_unary_tile(3,col); add_binary_tile(10,3,10);  // d10=newS
        // dLda -> d2 (keep dLdC in d3)
        copy_tile_init(2); copy_tile(2,0,3); mul_binary_tile(2,3,2);   // d2=dLda, d3=dLdC
        mul_binary_tile(2,12,4);                                       // d4=g_op (dLda*gexp)
        mul_binary_tile(2,13,2); mul_unary_tile(2,NHALF);             // d2=base
        mul_binary_tile(3,15,3);                                       // d3=g_col (dLdC*w)
        mul_binary_tile(0,0,5); mul_binary_tile(5,2,5);               // d5=g_a
        mul_binary_tile(1,1,7); mul_binary_tile(7,2,7);               // d7=g_c
        mul_binary_tile(0,1,6); mul_unary_tile(6,TWO); mul_binary_tile(6,2,6);  // d6=g_b
        copy_dest_values(0,8); mul_unary_tile(8,a); copy_dest_values(1,12); mul_unary_tile(12,bb);
        add_binary_tile(8,12,8); mul_unary_tile(8,NEG2); mul_binary_tile(8,2,8);   // d8=g_cx
        copy_dest_values(0,9); mul_unary_tile(9,bb); copy_dest_values(1,12); mul_unary_tile(12,c);
        add_binary_tile(9,12,9); mul_unary_tile(9,NEG2); mul_binary_tile(9,2,9);   // d9=g_cy
        tile_regs_commit(); tile_regs_wait();
        pack_tile(3,16); pack_tile(4,17); pack_tile(5,18); pack_tile(7,19);
        pack_tile(6,20); pack_tile(8,21); pack_tile(9,22);
        tile_regs_release();
        cb_push_back(16,1); cb_push_back(17,1); cb_push_back(18,1); cb_push_back(19,1);
        cb_push_back(20,1); cb_push_back(21,1); cb_push_back(22,1);
    }
    cb_pop_front(0,1); cb_pop_front(1,1); cb_pop_front(2,1); cb_pop_front(3,1); cb_pop_front(4,1);
}
"""

WRITER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1), K=get_arg_val<uint32_t>(2), nb=get_arg_val<uint32_t>(3);
    // 7 output tensors (one per grad), each holds K tiles. args 4..10 = base addrs.
    for (uint32_t g=0; g<K; g++) {
        for (uint32_t j=0; j<7; j++) {
            uint32_t cb=16+j; uint32_t base=get_arg_val<uint32_t>(4+j);
            cb_wait_front(cb,1);
            noc_async_write(get_read_ptr(cb), get_noc_addr(sx,sy,base + g*nb), nb);
            noc_async_write_barrier();
            cb_pop_front(cb,1);
        }
    }
}
"""


def l1(dev, data=None, ntiles=1):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(crs, [32*ntiles, 32], ttnn.ShardOrientation.ROW_MAJOR))
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32*ntiles, 32]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, 32, 32).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        ii, jj = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
        PX, PY = jj.double(), ii.double()
        # K=2 gaussians, processed in this (reverse-depth) order
        G = [dict(cx=12.0, cy=16.0, a=0.06, b=0.004, c=0.05, op=0.6, col=0.7),
             dict(cx=18.0, cy=14.0, a=0.05, b=-0.003, c=0.06, op=0.5, col=0.4)]
        dLdC = torch.rand(32, 32).double()
        Tfinal = torch.ones(32, 32).double()                 # forward final-T; recon walks back from here
        # torch golden: reverse pass with suffix-S, reconstruct T
        gold = {k: [0.0]*KG for k in ["col", "op", "a", "b", "c", "cx", "cy"]}
        S = torch.zeros(32, 32).double(); Trun = Tfinal.clone()
        for g in range(KG):
            G_ = G[g]; dx, dy = PX-G_["cx"], PY-G_["cy"]
            gexp = torch.exp(-0.5*(G_["a"]*dx*dx + 2*G_["b"]*dx*dy + G_["c"]*dy*dy)); alpha = G_["op"]*gexp
            rec = 1/(1-alpha); Ti = Trun*rec; w = Ti*alpha
            dCda = Ti*G_["col"] - S*rec; dLda = dLdC*dCda; base = dLda*alpha*(-0.5)
            gold["col"][g] = float((dLdC*w).sum()); gold["op"][g] = float((dLda*gexp).sum())
            gold["a"][g] = float((base*dx*dx).sum()); gold["b"][g] = float((base*2*dx*dy).sum())
            gold["c"][g] = float((base*dy*dy).sum())
            gold["cx"][g] = float((base*(-2)*(G_["a"]*dx+G_["b"]*dy)).sum())
            gold["cy"][g] = float((base*(-2)*(G_["b"]*dx+G_["c"]*dy)).sum())
            S = S + w*G_["col"]; Trun = Ti

        px_t, py_t, dl_t = l1(dev, PX), l1(dev, PY), l1(dev, dLdC)
        tf_t, sz_t = l1(dev, Tfinal), l1(dev, torch.zeros(32, 32))
        outs = [l1(dev, ntiles=KG) for _ in range(7)]           # 7 grads, each [K tiles]
        hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(*HOME)); sx, sy = hp.x, hp.y
        NB = 32*32*4
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])

        def rt(arr):
            r = ttnn.RuntimeArgs(); r[HOME[0]][HOME[1]] = arr; return r
        cbf = lambda i, d=2: ttnn.CBDescriptor(total_size=d*NB, core_ranges=crs,
                 format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=NB)])
        ks = lambda s, arr, cfg, cta=[]: ttnn.KernelDescriptor(
            kernel_source=s, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=crs, runtime_args=rt(arr), compile_time_args=cta, config=cfg)
        cargs = []
        for g in G:
            cargs += [f2u(g["cx"]), f2u(g["cy"]), f2u(g["a"]), f2u(2*g["b"]), f2u(g["c"]), f2u(g["op"]), f2u(g["col"]), f2u(g["b"])]
        oaddr = [o.buffer_address() for o in outs]
        prog = ttnn.ProgramDescriptor(kernels=[
            ks(READER, [sx, sy, px_t.buffer_address(), py_t.buffer_address(), dl_t.buffer_address(), tf_t.buffer_address(), sz_t.buffer_address(), NB], ttnn.ReaderConfigDescriptor()),
            ks(COMPUTE, cargs, ttnn.ComputeConfigDescriptor(), [KG]),
            ks(WRITER, [sx, sy, KG, NB] + oaddr, ttnn.WriterConfigDescriptor())],
            semaphores=[], cbs=[cbf(i) for i in (0, 1, 2, 3, 4)] + [cbf(i, KG+1) for i in (16, 17, 18, 19, 20, 21, 22)])
        ttnn.generic_op([px_t, outs[0]], prog)

        names = ["col", "op", "a", "c", "b", "cx", "cy"]        # pack order 16..22
        scale = max(abs(gold[k][g]) for k in gold for g in range(KG))
        worst = 0.0
        print(f"K={KG} fused backward  (scale={scale:.1f})")
        for i, n in enumerate(names):
            t = ttnn.to_torch(outs[i]).reshape(KG, 32, 32)
            for g in range(KG):
                dv = float(t[g].sum()); go = gold[n][g]; e = abs(dv-go)/scale; worst = max(worst, e)
                print(f"  g{g} {n:3} dev={dv:9.3f} gold={go:9.3f} err/scale={e:.2e}")
        print(f"BWD_KLOOP worst={worst:.2e} -> {'OK' if worst<2e-2 else 'FAIL'}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
