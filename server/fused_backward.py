#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Phase 2b: reusable single-tile (32x32) FUSED BACKWARD — one generic_op kernel computes all 7 grads
for K Gaussians (reverse-blend, recip-reconstruct-T, suffix-S via CB recurrence). Verified vs torch
(K=2/K=4, err/scale 8e-3). Drop-in replacement for the O(N) ttnn-op backward in device_raster, at H=W=32.

  grads = fused_backward(dev, params_rev, dLdC, Tfinal)
    params_rev : list of K dicts {cx,cy,a,b,c,op,col} in REVERSE depth order (back-to-front)
    dLdC       : torch [32,32] upstream dL/dC for this channel
    Tfinal     : torch [32,32] final transmittance from the forward
    -> dict {cx,cy,a,b,c,op,col: np.ndarray[K]}  (same reverse order as params_rev)

Full kernel + verification: docs/pathclear/m17_fused_backward.py.
"""
from __future__ import annotations
import struct
import numpy as np
import torch
import ttnn

HOME = (1, 1)
TS = 32
FUSED_K = 16                          # fixed compile-time K (pad chunks with no-op dummies) -> 1 JIT compile
_DUMMY_G = {"cx": 0.0, "cy": 0.0, "a": 1.0, "b": 0.0, "c": 1.0, "op": 0.0, "col": 0.0}   # op=0 -> no contribution


def f2u(x): return struct.unpack("<I", struct.pack("<f", float(x)))[0]

# kernels (verbatim from the verified m17 K-loop kernel)
from pathlib import Path as _P
_m17 = (_P(__file__).resolve().parent.parent / "docs" / "pathclear" / "m17_fused_backward.py").read_text()
_g = {}
exec(compile(_m17, "m17", "exec"), _g)          # pulls in READER, COMPUTE, WRITER strings + f2u
READER, COMPUTE, WRITER = _g["READER"], _g["COMPUTE"], _g["WRITER"]

_NAMES = ["col", "op", "a", "c", "b", "cx", "cy"]   # CB pack order 16..22

# ============================================================================================
# STAGE 2 — on-device per-tile reduce via a SEPARATE fp32 reduce dispatch (gate FB_S2=1). A fused
# single-kernel reduce was tried but can't reach the precision needed (a/c grads are signed dx^2/dy^2
# sums; bf16 dst -> rel 1.0), and enforce_fp32_accumulation
# requires global fp32_dest_acc_en, which halves dst 16->8 and breaks m17's 16-slot arithmetic. So:
#   dispatch-1 = m17 (writes 7 full-tile products to L1, unchanged)
#   dispatch-2 = these RED kernels (fp32_dest_acc_en=True): stream each L1 product tile -> fp32 reduce ->
#                scalar -> ONE packed output tile/core at [g,j]. Products never round-trip host.
# ============================================================================================
READER_RED = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1), nb=get_arg_val<uint32_t>(2), K=get_arg_val<uint32_t>(3);
    cb_reserve_back(23,1);
    volatile tt_l1_ptr uint32_t* sp=(volatile tt_l1_ptr uint32_t*)get_write_ptr(23);
    for (uint32_t k=0;k<1024;k++) sp[k]=0;
    for (uint32_t f=0;f<4;f++) for (uint32_t k=0;k<16;k++) sp[f*256+k]=0x3F800000u;
    cb_push_back(23,1);
    for (uint32_t g=0; g<K; g++)
        for (uint32_t j=0; j<7; j++) {
            uint32_t base=get_arg_val<uint32_t>(4+j);           // out_j per-core shard base
            cb_reserve_back(0,1);
            noc_async_read(get_noc_addr(sx,sy, base + g*nb), get_write_ptr(0), nb);
            noc_async_read_barrier();
            cb_push_back(0,1);
        }
}
"""

COMPUTE_RED = r"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    cb_wait_front(23,1);
    compute_kernel_hw_startup(0, 23, 16);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR, true>(0, 23, 16);
    for (uint32_t i=0; i<7*K; i++) {
        cb_wait_front(0,1); cb_reserve_back(16,1);
        tile_regs_acquire();
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR, true>(0, 23, 0, 0, 0);
        tile_regs_commit(); tile_regs_wait();
        pack_tile(0,16);
        tile_regs_release();
        cb_push_back(16,1); cb_pop_front(0,1);
    }
    reduce_uninit<true>();
}
"""

WRITER_RED = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1), nb=get_arg_val<uint32_t>(2), K=get_arg_val<uint32_t>(3);
    uint32_t base=get_arg_val<uint32_t>(4);   // output base; scalar -> logical [g, j*4] (16B-aligned)
    uint32_t toff=get_arg_val<uint32_t>(5);   // float offset of this chunk's tile in the accumulator (S3); 0 for S2
    for (uint32_t g=0; g<K; g++)              // 4-byte noc writes MUST be 16B-aligned (col stride 4 floats)
        for (uint32_t j=0; j<7; j++) {
            uint32_t col = j*4;                                   // logical column (0,4,8,...,24)
            uint32_t face = (col < 16) ? 0u : 1u;                 // tile face0=cols0-15, face1=cols16-31
            uint32_t off  = (toff + face*256u + g*16u + (col - face*16u)) * 4u;   // byte offset
            cb_wait_front(16,1);
            noc_async_write(get_read_ptr(16), get_noc_addr(sx,sy, base + off), 4);
            noc_async_write_barrier();
            cb_pop_front(16,1);
        }
}
"""


def _block(dev, grid, totH, totW, shard_h, data=None):
    sh = ttnn.ShardSpec(grid, [shard_h, TS], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, sh)
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, totH, totW]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, totH, totW).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)


def fused_backward_grid(dev, cxv, cyv, av, bv, cv, opv, colv, tile_lists, ntx, nty, Wp, Hp, gi, Tfin):
    """STAGE A — grid-sharded backward: ONE dispatch per (chunk, channel), all tiles parallel (vs the
    host-tile-loop). Block-shard PX/PY/dLdC/T/S per tile; each core runs the m17 kernel on its tile's
    chunk of <=FUSED_K Gaussians; S/T threaded across chunks as on-device block-sharded tensors. Products
    drain per chunk -> host accumulate. Returns geomg{}, colg[3] over the valid Gaussian indexing."""
    GX, GY = ntx, nty
    N = cxv.shape[0]
    geomg = {k: __import__("numpy").zeros(N) for k in ("cx", "cy", "a", "b", "c", "op")}
    import numpy as np
    colg = [np.zeros(N) for _ in range(3)]
    maxc = max((len(l) for l in tile_lists), default=0)
    if maxc == 0:
        return geomg, colg
    nbatch = (maxc + FUSED_K - 1) // FUSED_K
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
    ii, jj = torch.meshgrid(torch.arange(Hp), torch.arange(Wp), indexing="ij")
    PXt = _block(dev, grid, Hp, Wp, TS, jj.float())
    PYt = _block(dev, grid, Hp, Wp, TS, ii.float())
    coords = {(gx, gy): (lambda hp: (hp.x, hp.y))(dev.worker_core_from_logical_core(ttnn.CoreCoord(gx, gy)))
              for gx in range(GX) for gy in range(GY)}
    NB = TS * TS * 4
    SHF = FUSED_K * TS                                                  # output shard height (FUSED_K tiles)
    cbf = lambda i, d: ttnn.CBDescriptor(total_size=d * NB, core_ranges=grid,
             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=NB)])
    ks = lambda s, rt, cfg, cta=[]: ttnn.KernelDescriptor(
        kernel_source=s, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=grid, runtime_args=rt, compile_time_args=cta, config=cfg)

    def tile_chunk(t, c):    # tile t's Gaussians [c*K:(c+1)*K] padded to FUSED_K, in REVERSE order
        lst = tile_lists[t][::-1][c * FUSED_K:(c + 1) * FUSED_K]
        return lst, lst + [None] * (FUSED_K - len(lst))

    import os as _os, time as _time
    _PROF = _os.environ.get("FB_PROF", "0") == "1"
    _S3 = _os.environ.get("FB_S3", "0") == "1"     # Stage 3: on-device accumulate, drain ONCE per channel
    _S2 = _os.environ.get("FB_S2", "0") == "1" and not _S3   # Stage 2: in-kernel reduce, per-chunk readback
    _red = _S2 or _S3                              # both need the fp32 reduce dispatch
    _tac = {"alloc": 0.0, "args": 0.0, "prog": 0.0, "disp": 0.0, "readback": 0.0, "accum": 0.0}
    def _now(): return _time.perf_counter()
    # ---- STAGE 1: precompute channel-INVARIANT geometry args ONCE; patch only `col` per channel ----
    _t = _now()
    _DC = [f2u(_DUMMY_G["cx"]), f2u(_DUMMY_G["cy"]), f2u(_DUMMY_G["a"]), f2u(2 * _DUMMY_G["b"]),
           f2u(_DUMMY_G["c"]), f2u(_DUMMY_G["op"]), f2u(_DUMMY_G["col"]), f2u(_DUMMY_G["b"])]   # cached dummy block
    geo_args = {}                                       # (c,gx,gy) -> (cargs[8*FUSED_K], real [(slot,gid)])
    for c in range(nbatch):
        for gx in range(GX):
            for gy in range(GY):
                _, padded = tile_chunk(gy * ntx + gx, c)
                cargs = []; real = []
                for si, i in enumerate(padded):
                    if i is None:
                        cargs += _DC
                    else:
                        cargs += [f2u(cxv[i]), f2u(cyv[i]), f2u(av[i]), f2u(2 * bv[i]), f2u(cv[i]),
                                  f2u(opv[i]), 0, f2u(bv[i])]    # col (idx 6) patched per channel below
                        real.append((si, i))
                geo_args[(c, gx, gy)] = (cargs, real)
    outs = [_block(dev, grid, GY * SHF, Wp, SHF) for _ in range(7)]   # dispatch-1 (m17) products in L1
    out_addrs = [o.buffer_address() for o in outs]
    if _S3:                                          # drain-once accumulator: nbatch tiles/core (chunk c -> tile c)
        out_acc = _block(dev, grid, GY * nbatch * TS, GX * TS, nbatch * TS)
    elif _S2:                                         # per-chunk compact output: ONE tile/core, scalar at [g, j*4]
        out_s2 = _block(dev, grid, GY * TS, GX * TS, TS)
    _tac["args"] += _now() - _t

    for k in range(3):
        dLt = _block(dev, grid, Hp, Wp, TS, torch.from_numpy(gi[:, :, k].copy()))
        Tt = _block(dev, grid, Hp, Wp, TS, torch.from_numpy(Tfin.copy()))
        St = _block(dev, grid, Hp, Wp, TS, torch.zeros(Hp, Wp))
        for c in range(nbatch):
            _t = _now()
            Sout = _block(dev, grid, Hp, Wp, TS)
            Tout = _block(dev, grid, Hp, Wp, TS)
            _tac["alloc"] += _now() - _t; _t = _now()
            rt_r, rt_c, rt_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for gx in range(GX):
                for gy in range(GY):
                    sx, sy = coords[(gx, gy)]
                    rt_r[gx][gy] = [sx, sy, PXt.buffer_address(), PYt.buffer_address(), dLt.buffer_address(),
                                    Tt.buffer_address(), St.buffer_address(), NB, FUSED_K,
                                    Sout.buffer_address(), Tout.buffer_address()]
                    cargs, real = geo_args[(c, gx, gy)]
                    cc = cargs[:]                          # copy; patch only the per-channel col words
                    for (si, i) in real:
                        cc[si * 8 + 6] = f2u(colv[k][i])
                    rt_c[gx][gy] = cc
                    rt_w[gx][gy] = [sx, sy, FUSED_K, NB] + out_addrs
            _tac["args"] += _now() - _t; _t = _now()
            prog = ttnn.ProgramDescriptor(kernels=[       # dispatch-1: m17 -> 7 full-tile products in L1
                ks(READER, rt_r, ttnn.ReaderConfigDescriptor()),
                ks(COMPUTE, rt_c, ttnn.ComputeConfigDescriptor(), [FUSED_K]),
                ks(WRITER, rt_w, ttnn.WriterConfigDescriptor())],
                semaphores=[], cbs=[cbf(i, 2) for i in (0, 1, 2, 24, 25, 26, 27)] + [cbf(i, 3) for i in range(16, 23)])
            _tac["prog"] += _now() - _t; _t = _now()
            ttnn.generic_op([PXt, outs[0]], prog)
            if _red:                                      # dispatch-2: fp32 reduce each L1 product tile -> scalar
                out_t = out_acc if _S3 else out_s2
                toff = c * (TS * TS) if _S3 else 0         # chunk c -> its own tile in the accumulator
                rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for gx in range(GX):
                    for gy in range(GY):
                        sx, sy = coords[(gx, gy)]
                        rr[gx][gy] = [sx, sy, NB, FUSED_K] + out_addrs
                        rc[gx][gy] = []
                        rw[gx][gy] = [sx, sy, NB, FUSED_K, out_t.buffer_address(), toff]
                cfg32 = ttnn.ComputeConfigDescriptor(); cfg32.fp32_dest_acc_en = True
                prog2 = ttnn.ProgramDescriptor(kernels=[
                    ks(READER_RED, rr, ttnn.ReaderConfigDescriptor()),
                    ks(COMPUTE_RED, rc, cfg32, [FUSED_K]),
                    ks(WRITER_RED, rw, ttnn.WriterConfigDescriptor())],
                    semaphores=[], cbs=[cbf(0, 4), cbf(16, 4), cbf(23, 1)])
                ttnn.generic_op([outs[0], out_t], prog2)
                if _S2 and _os.environ.get("FB_DBG", "0") == "1" and k == 0 and c == 0:
                    dbg = ttnn.to_torch(out_s2).reshape(GY, TS, GX, TS)
                    for j, o in enumerate(outs):
                        ref_full = ttnn.to_torch(o).reshape(GY, FUSED_K, TS, GX, TS).sum(dim=(2, 4))
                        dev_red = dbg[:, :FUSED_K, :, j * 4]
                        e = (dev_red - ref_full).abs().max().item(); s = ref_full.abs().max().item() + 1e-9
                        print(f"   DBG out[{j}]={_NAMES[j]:3}: dev-reduce vs host-sum-of-same-products "
                              f"max_abs={e:.3e} rel={e/s:.3e}", flush=True)
            _tac["disp"] += _now() - _t; _t = _now()
            if not _S3:                                   # S2 / baseline: readback + accumulate PER CHUNK
                if _S2:    # ONE small readback; fp32-reduced scalar at logical [g, j*4] (16B-aligned writes)
                    tt = ttnn.to_torch(out_s2).reshape(GY, TS, GX, TS)
                    hs = [tt[:, :FUSED_K, :, j * 4].numpy() for j in range(7)]
                else:
                    # reduce each [GY*FK*TS, GX*TS] output to per-(tile-row, slot, tile-col) scalars in ONE torch op
                    hs = [ttnn.to_torch(o).reshape(GY, FUSED_K, TS, GX, TS).sum(dim=(2, 4)).numpy() for o in outs]
                _tac["readback"] += _now() - _t; _t = _now()
                for gx in range(GX):                      # vectorized over slots (no per-element float())
                    for gy in range(GY):
                        lst, _ = tile_chunk(gy * ntx + gx, c)
                        if not lst:
                            continue
                        idx = np.asarray(lst); L = idx.shape[0]
                        for gi_i, name in enumerate(_NAMES):
                            vals = hs[gi_i][gy, :L, gx]
                            (colg[k] if name == "col" else geomg[name])[idx] += vals
                _tac["accum"] += _now() - _t
            St, Tt = Sout, Tout
        if _S3:                                           # Stage 3: drain the accumulator ONCE per channel
            _t = _now()
            tt = ttnn.to_torch(out_acc).reshape(GY, nbatch, TS, GX, TS)   # [gy, chunk, row, gx, col]
            _tac["readback"] += _now() - _t; _t = _now()
            for gx in range(GX):
                for gy in range(GY):
                    rev = tile_lists[gy * ntx + gx][::-1]
                    L = len(rev)
                    if not L:
                        continue
                    idx = np.asarray(rev)
                    block = tt[gy, :, :FUSED_K, gx, :].reshape(nbatch * FUSED_K, TS)   # row = chunk*FK+slot = rev idx
                    for gi_i, name in enumerate(_NAMES):
                        vals = block[:L, gi_i * 4].numpy()
                        (colg[k] if name == "col" else geomg[name])[idx] += vals
            _tac["accum"] += _now() - _t
    if _PROF:
        print("      [fb_grid] " + "  ".join(f"{kk}={1e3*vv:.1f}" for kk, vv in _tac.items())
              + f"  nbatch={nbatch} tiles={GX*GY}", flush=True)
    return geomg, colg


def _l1(dev, data=None, nt=1):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(crs, [TS * nt, TS], ttnn.ShardOrientation.ROW_MAJOR))
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, TS * nt, TS]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, TS, TS).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)


def fused_backward(dev, params_rev, dLdC, Tfinal, ox=0, oy=0, S_init=None, return_state=False):
    """ox,oy = this tile's GLOBAL top-left pixel (so dx = (global x) - cx). dLdC/Tfinal/S_init are the
    tile's 32x32 sub-regions; Gaussian cx,cy are global. For chunking a dense tile, pass S_init (prev
    chunk's S) + Tfinal (prev chunk's T) and return_state=True to get the chunk's final S,T to thread on.
    Multi-tile = loop tiles + host-sum the per-Gaussian grads (cross-tile = host scalar sum).
    params_rev must be <= FUSED_K; padded with no-op dummies so K is a fixed compile-time constant."""
    nreal = len(params_rev)
    assert nreal <= FUSED_K, f"chunk {nreal} > FUSED_K {FUSED_K}"
    params_rev = list(params_rev) + [_DUMMY_G] * (FUSED_K - nreal)   # pad after real Gaussians (S/T unaffected)
    K = FUSED_K
    ii, jj = torch.meshgrid(torch.arange(TS), torch.arange(TS), indexing="ij")
    px, py = _l1(dev, (jj + ox).float()), _l1(dev, (ii + oy).float())
    dl = _l1(dev, dLdC.double())
    tf = _l1(dev, Tfinal.double())
    sz = _l1(dev, S_init.double() if S_init is not None else torch.zeros(TS, TS))
    s_out, t_out = _l1(dev), _l1(dev)                       # exported final S/T (chunk threading)
    outs = [_l1(dev, nt=K) for _ in range(7)]
    hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(*HOME)); sx, sy = hp.x, hp.y
    NB = TS * TS * 4
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])

    def rt(arr):
        r = ttnn.RuntimeArgs(); r[HOME[0]][HOME[1]] = arr; return r
    cbf = lambda i, d: ttnn.CBDescriptor(total_size=d * NB, core_ranges=crs,
             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=NB)])
    ks = lambda s, arr, cfg, cta=[]: ttnn.KernelDescriptor(
        kernel_source=s, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=crs, runtime_args=rt(arr), compile_time_args=cta, config=cfg)
    cargs = []
    for q in params_rev:
        cargs += [f2u(q["cx"]), f2u(q["cy"]), f2u(q["a"]), f2u(2 * q["b"]), f2u(q["c"]),
                  f2u(q["op"]), f2u(q["col"]), f2u(q["b"])]
    oaddr = [o.buffer_address() for o in outs]
    cbs = [cbf(i, 2) for i in (0, 1, 2, 24, 25, 26, 27)] + [cbf(i, 3) for i in (16, 17, 18, 19, 20, 21, 22)]
    prog = ttnn.ProgramDescriptor(kernels=[
        ks(READER, [sx, sy, px.buffer_address(), py.buffer_address(), dl.buffer_address(),
                    tf.buffer_address(), sz.buffer_address(), NB, K,
                    s_out.buffer_address(), t_out.buffer_address()], ttnn.ReaderConfigDescriptor()),
        ks(COMPUTE, cargs, ttnn.ComputeConfigDescriptor(), [K]),
        ks(WRITER, [sx, sy, K, NB] + oaddr, ttnn.WriterConfigDescriptor())],
        semaphores=[], cbs=cbs)
    ttnn.generic_op([px, outs[0]], prog)
    grads = {}
    for i, n in enumerate(_NAMES):
        t = ttnn.to_torch(outs[i]).reshape(K, TS, TS)
        grads[n] = t.reshape(K, -1).sum(dim=1).numpy()          # host reduce per-Gaussian (exact)
    if return_state:
        S = ttnn.to_torch(s_out).reshape(TS, TS).clone()
        T = ttnn.to_torch(t_out).reshape(TS, TS).clone()
        return grads, S, T
    return grads
