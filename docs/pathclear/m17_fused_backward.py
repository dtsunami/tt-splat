#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# M17 — FUSED BACKWARD SFPU KERNEL (Phase 2b core). STATUS: COMPLETE single-tile, VERIFIED on silicon.
# K-Gaussian reverse-blend backward fused in ONE generic_op compute kernel: recompute alpha, reconstruct
# T via recip (T_i=T_run/(1-alpha)), suffix-S accumulation, all 7 grads {cx,cy,a,b,c,op,col}. K=2 and
# K=4 match torch full backward to err/scale 8.0e-3 (bf16 dst). S/T_run state flows through CB RECURRENCE:
# the READER (dataflow) copies newS/newT (cb26/27) back into the S/T_run input CBs (cb24/25) each iter
# via L1->L1 NoC read; COMPUTE is uniform (load S/T from CB, pack 7 products + newS/newT). dst-persistence
# across acquire/pack/release does NOT work -> CB recurrence is the answer.
# KEY: bf16 dst=16 slots (uses d0-15); fp32_dest_acc HALVES to 8 (do NOT enable); needs copy_dest_values.h
# + tile_move_copy.h + eltwise_unary/recip.h; scalars=f2u bits; reduce=host ttnn.sum (no in-kernel reduce).
# REMAINING for full 2b: (1) multi-tile (M14 sharded grid, per-tile culled lists); (2) M2 scatter-add for
# cross-tile grad accumulation; (3) wire behind DeviceRaster.backward (server/device_raster.py). At
# TT_SIZE=32 the single tile IS the whole image -> directly usable for device training now.

"""Fused backward, K-Gaussian reverse loop via CB RECURRENCE.
READER feeds S/T_run input CBs (24/25): S0=0,T0=Tfinal initially, then newS/newT (26/27) from compute
each iter via L1->L1 NoC copy. COMPUTE is uniform (load S/T from CB, pack 7 products + newS/newT).
Verify K=2 and K=4 vs torch full backward."""
import struct, math, torch, ttnn
HOME = (1, 1)


def f2u(x): return struct.unpack("<I", struct.pack("<f", float(x)))[0]

READER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t px=get_arg_val<uint32_t>(2), py=get_arg_val<uint32_t>(3), dl=get_arg_val<uint32_t>(4);
    uint32_t tf=get_arg_val<uint32_t>(5), sz=get_arg_val<uint32_t>(6), nb=get_arg_val<uint32_t>(7), K=get_arg_val<uint32_t>(8);
    uint32_t so=get_arg_val<uint32_t>(9), to=get_arg_val<uint32_t>(10);   // export final S/T (chunk threading)
    cb_reserve_back(0,1); noc_async_read(get_noc_addr(sx,sy,px), get_write_ptr(0), nb);
    cb_reserve_back(1,1); noc_async_read(get_noc_addr(sx,sy,py), get_write_ptr(1), nb);
    cb_reserve_back(2,1); noc_async_read(get_noc_addr(sx,sy,dl), get_write_ptr(2), nb);
    cb_reserve_back(24,1); noc_async_read(get_noc_addr(sx,sy,sz), get_write_ptr(24), nb);   // S0 = 0
    cb_reserve_back(25,1); noc_async_read(get_noc_addr(sx,sy,tf), get_write_ptr(25), nb);   // T0 = Tfinal
    noc_async_read_barrier();
    cb_push_back(0,1); cb_push_back(1,1); cb_push_back(2,1); cb_push_back(24,1); cb_push_back(25,1);
    for (uint32_t g=1; g<K; g++) {                  // recurrence: newS/newT -> S/T_run
        cb_wait_front(26,1); cb_wait_front(27,1);
        cb_reserve_back(24,1); noc_async_read(get_noc_addr(sx,sy,get_read_ptr(26)), get_write_ptr(24), nb);
        cb_reserve_back(25,1); noc_async_read(get_noc_addr(sx,sy,get_read_ptr(27)), get_write_ptr(25), nb);
        noc_async_read_barrier();
        cb_push_back(24,1); cb_push_back(25,1);
        cb_pop_front(26,1); cb_pop_front(27,1);
    }
    cb_wait_front(26,1); cb_wait_front(27,1);          // final newS/newT (last Gaussian) -> export
    noc_async_write(get_read_ptr(26), get_noc_addr(sx,sy,so), nb);
    noc_async_write(get_read_ptr(27), get_noc_addr(sx,sy,to), nb);
    noc_async_write_barrier();
    cb_pop_front(26,1); cb_pop_front(27,1);
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
    cb_wait_front(0,1); cb_wait_front(1,1); cb_wait_front(2,1);
    for (uint32_t g=0; g<K; g++) {
        uint32_t b0=g*8;
        uint32_t cx=get_arg_val<uint32_t>(b0+0), cy=get_arg_val<uint32_t>(b0+1), a=get_arg_val<uint32_t>(b0+2);
        uint32_t twob=get_arg_val<uint32_t>(b0+3), c=get_arg_val<uint32_t>(b0+4), op=get_arg_val<uint32_t>(b0+5),
                 col=get_arg_val<uint32_t>(b0+6), bb=get_arg_val<uint32_t>(b0+7);
        cb_wait_front(24,1); cb_wait_front(25,1);
        cb_reserve_back(16,1); cb_reserve_back(17,1); cb_reserve_back(18,1); cb_reserve_back(19,1);
        cb_reserve_back(20,1); cb_reserve_back(21,1); cb_reserve_back(22,1);
        cb_reserve_back(26,1); cb_reserve_back(27,1);
        tile_regs_acquire();
        copy_tile_init(24); copy_tile(24,0,10);                       // d10=S
        copy_tile_init(25); copy_tile(25,0,11);                       // d11=T_run
        copy_tile_init(0); copy_tile(0,0,0); sub_unary_tile(0, cx);   // d0=dx
        copy_tile_init(1); copy_tile(1,0,1); sub_unary_tile(1, cy);   // d1=dy
        mul_binary_tile(0,0,12); mul_unary_tile(12,a);
        mul_binary_tile(1,1,13); mul_unary_tile(13,c); add_binary_tile(12,13,12);
        mul_binary_tile(0,1,13); mul_unary_tile(13,twob); add_binary_tile(12,13,12);
        mul_unary_tile(12,NHALF); exp_tile(12);                       // d12=gexp
        copy_dest_values(12,13); mul_unary_tile(13,op);               // d13=alpha
        copy_dest_values(13,14); mul_unary_tile(14,NEG1); add_unary_tile(14,ONE); recip_tile(14);  // d14=rec
        mul_binary_tile(11,14,11);                                    // d11=T_i (newT, in place)
        copy_dest_values(11,15); mul_binary_tile(15,13,15);           // d15=w
        copy_dest_values(11,2); mul_unary_tile(2,col);                // d2=T_i*col
        copy_dest_values(10,3); mul_binary_tile(3,14,3);              // d3=S*rec
        sub_binary_tile(2,3,2);                                       // d2=dCda
        copy_dest_values(15,3); mul_unary_tile(3,col); add_binary_tile(10,3,10);  // d10=newS
        copy_tile_init(2); copy_tile(2,0,3); mul_binary_tile(2,3,2);  // d2=dLda, d3=dLdC
        mul_binary_tile(2,12,4);                                      // d4=g_op
        mul_binary_tile(2,13,2); mul_unary_tile(2,NHALF);            // d2=base
        mul_binary_tile(3,15,3);                                      // d3=g_col
        mul_binary_tile(0,0,5); mul_binary_tile(5,2,5);              // d5=g_a
        mul_binary_tile(1,1,7); mul_binary_tile(7,2,7);              // d7=g_c
        mul_binary_tile(0,1,6); mul_unary_tile(6,TWO); mul_binary_tile(6,2,6);  // d6=g_b
        copy_dest_values(0,8); mul_unary_tile(8,a); copy_dest_values(1,12); mul_unary_tile(12,bb);
        add_binary_tile(8,12,8); mul_unary_tile(8,NEG2); mul_binary_tile(8,2,8);   // d8=g_cx
        copy_dest_values(0,9); mul_unary_tile(9,bb); copy_dest_values(1,12); mul_unary_tile(12,c);
        add_binary_tile(9,12,9); mul_unary_tile(9,NEG2); mul_binary_tile(9,2,9);   // d9=g_cy
        tile_regs_commit(); tile_regs_wait();
        pack_tile(3,16); pack_tile(4,17); pack_tile(5,18); pack_tile(7,19);
        pack_tile(6,20); pack_tile(8,21); pack_tile(9,22);
        pack_tile(10,26); pack_tile(11,27);
        tile_regs_release();
        cb_push_back(16,1); cb_push_back(17,1); cb_push_back(18,1); cb_push_back(19,1);
        cb_push_back(20,1); cb_push_back(21,1); cb_push_back(22,1);
        cb_push_back(26,1); cb_push_back(27,1);
        cb_pop_front(24,1); cb_pop_front(25,1);
    }
    cb_pop_front(0,1); cb_pop_front(1,1); cb_pop_front(2,1);
}
"""

WRITER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1), K=get_arg_val<uint32_t>(2), nb=get_arg_val<uint32_t>(3);
    for (uint32_t g=0; g<K; g++)
        for (uint32_t j=0; j<7; j++) {
            uint32_t cb=16+j; uint32_t base=get_arg_val<uint32_t>(4+j);
            cb_wait_front(cb,1);
            noc_async_write(get_read_ptr(cb), get_noc_addr(sx,sy,base + g*nb), nb);
            noc_async_write_barrier();
            cb_pop_front(cb,1);
        }
}
"""


def l1(dev, data=None, nt=1):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(crs, [32*nt, 32], ttnn.ShardOrientation.ROW_MAJOR))
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32*nt, 32]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, 32, 32).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)


def run(dev, G):
    K = len(G)
    torch.manual_seed(0)
    ii, jj = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    PX, PY = jj.double(), ii.double()
    dLdC = torch.rand(32, 32).double(); Tfinal = torch.ones(32, 32).double()
    gold = {k: [0.0]*K for k in ["col", "op", "a", "b", "c", "cx", "cy"]}
    S = torch.zeros(32, 32).double(); Trun = Tfinal.clone()
    for g in range(K):
        q = G[g]; dx, dy = PX-q["cx"], PY-q["cy"]
        gexp = torch.exp(-0.5*(q["a"]*dx*dx + 2*q["b"]*dx*dy + q["c"]*dy*dy)); alpha = q["op"]*gexp
        rec = 1/(1-alpha); Ti = Trun*rec; w = Ti*alpha
        dCda = Ti*q["col"] - S*rec; dLda = dLdC*dCda; base = dLda*alpha*(-0.5)
        gold["col"][g] = float((dLdC*w).sum()); gold["op"][g] = float((dLda*gexp).sum())
        gold["a"][g] = float((base*dx*dx).sum()); gold["b"][g] = float((base*2*dx*dy).sum())
        gold["c"][g] = float((base*dy*dy).sum())
        gold["cx"][g] = float((base*(-2)*(q["a"]*dx+q["b"]*dy)).sum())
        gold["cy"][g] = float((base*(-2)*(q["b"]*dx+q["c"]*dy)).sum())
        S = S + w*q["col"]; Trun = Ti

    px, py, dl = l1(dev, PX), l1(dev, PY), l1(dev, dLdC)
    tf, sz = l1(dev, Tfinal), l1(dev, torch.zeros(32, 32))
    outs = [l1(dev, nt=K) for _ in range(7)]
    hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(*HOME)); sx, sy = hp.x, hp.y
    NB = 32*32*4
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])

    def rt(arr):
        r = ttnn.RuntimeArgs(); r[HOME[0]][HOME[1]] = arr; return r
    cbf = lambda i, d: ttnn.CBDescriptor(total_size=d*NB, core_ranges=crs,
             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=NB)])
    ks = lambda s, arr, cfg, cta=[]: ttnn.KernelDescriptor(
        kernel_source=s, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=crs, runtime_args=rt(arr), compile_time_args=cta, config=cfg)
    cargs = []
    for q in G:
        cargs += [f2u(q["cx"]), f2u(q["cy"]), f2u(q["a"]), f2u(2*q["b"]), f2u(q["c"]), f2u(q["op"]), f2u(q["col"]), f2u(q["b"])]
    oaddr = [o.buffer_address() for o in outs]
    cbs = [cbf(i, 2) for i in (0, 1, 2, 24, 25, 26, 27)] + [cbf(i, 3) for i in (16, 17, 18, 19, 20, 21, 22)]
    prog = ttnn.ProgramDescriptor(kernels=[
        ks(READER, [sx, sy, px.buffer_address(), py.buffer_address(), dl.buffer_address(), tf.buffer_address(), sz.buffer_address(), NB, K], ttnn.ReaderConfigDescriptor()),
        ks(COMPUTE, cargs, ttnn.ComputeConfigDescriptor(), [K]),
        ks(WRITER, [sx, sy, K, NB] + oaddr, ttnn.WriterConfigDescriptor())],
        semaphores=[], cbs=cbs)
    ttnn.generic_op([px, outs[0]], prog)

    names = ["col", "op", "a", "c", "b", "cx", "cy"]
    scale = max(abs(gold[k][g]) for k in gold for g in range(K))
    worst = 0.0
    for i, n in enumerate(names):
        t = ttnn.to_torch(outs[i]).reshape(K, 32, 32)
        for g in range(K):
            e = abs(float(t[g].sum())-gold[n][g])/scale; worst = max(worst, e)
    print(f"K={K} fused backward  worst_err/scale={worst:.2e} -> {'OK' if worst<2e-2 else 'FAIL'}")
    return worst


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        G2 = [dict(cx=12.0, cy=16.0, a=0.06, b=0.004, c=0.05, op=0.6, col=0.7),
              dict(cx=18.0, cy=14.0, a=0.05, b=-0.003, c=0.06, op=0.5, col=0.4)]
        run(dev, G2)
        G4 = G2 + [dict(cx=20.0, cy=20.0, a=0.07, b=0.002, c=0.05, op=0.45, col=0.55),
                   dict(cx=10.0, cy=22.0, a=0.05, b=-0.004, c=0.07, op=0.7, col=0.3)]
        run(dev, G4)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
