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
    _tac = {"alloc": 0.0, "args": 0.0, "prog": 0.0, "disp": 0.0, "readback": 0.0, "accum": 0.0}
    def _now(): return _time.perf_counter()
    for k in range(3):
        dLt = _block(dev, grid, Hp, Wp, TS, torch.from_numpy(gi[:, :, k].copy()))
        Tt = _block(dev, grid, Hp, Wp, TS, torch.from_numpy(Tfin.copy()))
        St = _block(dev, grid, Hp, Wp, TS, torch.zeros(Hp, Wp))
        for c in range(nbatch):
            _t = _now()
            outs = [_block(dev, grid, GY * SHF, Wp, SHF) for _ in range(7)]
            Sout = _block(dev, grid, Hp, Wp, TS)
            Tout = _block(dev, grid, Hp, Wp, TS)
            _tac["alloc"] += _now() - _t; _t = _now()
            rt_r, rt_c, rt_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for gx in range(GX):
                for gy in range(GY):
                    sx, sy = coords[(gx, gy)]; t = gy * ntx + gx
                    _, padded = tile_chunk(t, c)
                    rt_r[gx][gy] = [sx, sy, PXt.buffer_address(), PYt.buffer_address(), dLt.buffer_address(),
                                    Tt.buffer_address(), St.buffer_address(), NB, FUSED_K,
                                    Sout.buffer_address(), Tout.buffer_address()]
                    cargs = []
                    for i in padded:
                        q = (_DUMMY_G if i is None else
                             {"cx": cxv[i], "cy": cyv[i], "a": av[i], "b": bv[i], "c": cv[i], "op": opv[i], "col": colv[k][i]})
                        cargs += [f2u(q["cx"]), f2u(q["cy"]), f2u(q["a"]), f2u(2 * q["b"]), f2u(q["c"]),
                                  f2u(q["op"]), f2u(q["col"]), f2u(q["b"])]
                    rt_c[gx][gy] = cargs
                    rt_w[gx][gy] = [sx, sy, FUSED_K, NB] + [o.buffer_address() for o in outs]
            _tac["args"] += _now() - _t; _t = _now()
            prog = ttnn.ProgramDescriptor(kernels=[
                ks(READER, rt_r, ttnn.ReaderConfigDescriptor()),
                ks(COMPUTE, rt_c, ttnn.ComputeConfigDescriptor(), [FUSED_K]),
                ks(WRITER, rt_w, ttnn.WriterConfigDescriptor())],
                semaphores=[], cbs=[cbf(i, 2) for i in (0, 1, 2, 24, 25, 26, 27)] + [cbf(i, 3) for i in range(16, 23)])
            _tac["prog"] += _now() - _t; _t = _now()
            ttnn.generic_op([PXt, outs[0]], prog)
            _tac["disp"] += _now() - _t; _t = _now()
            # reduce each [GY*FK*TS, GX*TS] output to per-(tile-row, slot, tile-col) scalars in ONE torch op
            hs = [ttnn.to_torch(o).reshape(GY, FUSED_K, TS, GX, TS).sum(dim=(2, 4)).numpy() for o in outs]
            _tac["readback"] += _now() - _t; _t = _now()
            for gx in range(GX):                          # 9 tiles, vectorized over slots (no per-element float())
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
