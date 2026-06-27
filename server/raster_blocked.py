#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Tile-BLOCK raster (the >352px path) — process >grid tiles by LOOPING blocks of <=ncores tiles, each block's
tiles row-major TILE-LIST sharded onto the usable compute grid. The M14 forward + m17 backward kernels are
REUSED VERBATIM (1 tile/core); only the harness loops + gathers. The <=352px channel-parallel / s4 paths in
render_device/fused_backward stay the fast path; this kicks in only when the image exceeds the worker grid.

Proven on silicon: forward = scratchpad/proto_R1_tileblock_fwd.py (6.75e-4 vs host golden); backward =
scratchpad/proto_R2_tileblock_bwd.py (bit-exact 1.8e-7 vs fused_backward_grid base).
"""
from __future__ import annotations
import os
import numpy as np
import torch
import ttnn

import sfpu_raster_scaled as M14                                          # noqa: E402
from sfpu_raster_scaled import READER as FR, COMPUTE as FC, WRITER as FW, B, TS, f2u as ff2u, DUMMY   # noqa: E402
from fused_backward import (READER as BR, COMPUTE as BC, WRITER as BW, f2u, FUSED_K, _DUMMY_G, _NAMES,  # noqa: E402
                            READER_RF, COMPUTE_RF, WRITER_S4)  # s4 matmul-engine in-kernel reduce (drain-once)

_GEOM = ("cx", "cy", "a", "b", "c", "op")
NB = TS * TS * 4


def usable_grid(dev):
    gs = dev.compute_with_storage_grid_size()
    return gs.x, gs.y


def needs_blocking(dev, ntx, nty):
    """True when the tile grid exceeds the usable worker grid (the >352px path). Env-forceable for tests."""
    if os.environ.get("TT_FORCE_BLOCKED", "0") == "1":
        return True
    mx, my = usable_grid(dev)
    return ntx > mx or nty > my or ntx * nty > mx * my


def _grid(dev):
    mx, my = usable_grid(dev)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(mx - 1, my - 1))])
    coords = {(gx, gy): (lambda hp: (hp.x, hp.y))(dev.worker_core_from_logical_core(ttnn.CoreCoord(gx, gy)))
              for gx in range(mx) for gy in range(my)}
    return mx, my, grid, coords


def _tileshard(dev, grid, ncores, stiles, data=None):
    """HEIGHT_SHARDED [stiles*TS, TS]/core on [ncores*stiles*TS, TS], TILE_LAYOUT — row-major tile-list shard."""
    Hh = ncores * stiles * TS
    sh = ttnn.ShardSpec(grid, [stiles * TS, TS], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sh)
    shape = ttnn.Shape([1, 1, Hh, TS])
    if data is None:
        return ttnn.allocate_tensor_on_device(shape, ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, Hh, TS).float(), dtype=ttnn.float32,
                           layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


def _pixel_coords(bt, ntx, ncores):
    PX = np.zeros((ncores * TS, TS), np.float32); PY = np.zeros((ncores * TS, TS), np.float32)
    for local, t in enumerate(bt):
        tx, ty = t % ntx, t // ntx
        PX[local * TS:(local + 1) * TS, :] = (np.arange(TS) + tx * TS)[None, :]
        PY[local * TS:(local + 1) * TS, :] = (np.arange(TS) + ty * TS)[:, None]
    return PX, PY


def raster_rgb_blocked(dev, tile_lists, ntx, nty, cx, cy, a, b2, c, op, col3, Wp, Hp, want_T=False):
    """Forward render of all 3 channels over all tiles, block-looped. Returns ([R,G,B] np[Hp,Wp], Tfin|None)."""
    mx, my, grid, coords = _grid(dev)
    ncores = mx * my
    ntiles = ntx * nty
    nblocks = (ntiles + ncores - 1) // ncores
    chans = [np.zeros((Hp, Wp), np.float64) for _ in range(3)]
    Tfin = np.ones((Hp, Wp), np.float64) if want_T else None
    # vectorized arg-pack: f2u all params ONCE (float32 bit-view) -> per-dispatch is a numpy gather, not
    # a per-Gaussian struct.pack loop. Bit-identical to ff2u. (GDDR param streaming is the production follow-on.)
    v = lambda arr: np.ascontiguousarray(arr, np.float32).view(np.uint32)
    cxu, cyu, au, twobu, cu, opu = v(cx), v(cy), v(a), v(2 * np.asarray(b2)), v(c), v(op)
    col3u = [v(col3[ch]) for ch in range(3)]
    dB = np.array(DUMMY, np.uint32)
    cbf = lambda i: ttnn.CBDescriptor(total_size=2 * NB, core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=NB)])
    cfg = ttnn.ComputeConfigDescriptor(); cfg.fp32_dest_acc_en = True; cfg.math_approx_mode = False
    mk = lambda src, rt, cf, cta=[]: ttnn.KernelDescriptor(kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid, runtime_args=rt,
            compile_time_args=cta, config=cf)

    for k in range(nblocks):
        bt = list(range(k * ncores, min((k + 1) * ncores, ntiles)))
        PX, PY = _pixel_coords(bt, ntx, ncores)
        PXt = _tileshard(dev, grid, ncores, 1, torch.from_numpy(PX))
        PYt = _tileshard(dev, grid, ncores, 1, torch.from_numpy(PY))
        maxc = max((len(tile_lists[t]) for t in bt), default=0)
        nbatch = max(1, (maxc + B - 1) // B)
        for ch in range(3):
            accC = _tileshard(dev, grid, ncores, 1, torch.zeros(ncores * TS, TS))
            accT = _tileshard(dev, grid, ncores, 1, torch.ones(ncores * TS, TS))
            for d in range(nbatch):
                rt_r, rt_c, rt_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for gx in range(mx):
                    for gy in range(my):
                        local = gy * mx + gx
                        sx, sy = coords[(gx, gy)]
                        w = np.tile(dB, (B, 1))            # [B,7] DUMMY-filled
                        if local < len(bt):
                            lst = tile_lists[bt[local]][d * B:(d + 1) * B]
                            if lst:
                                g = np.asarray(lst); L = g.shape[0]
                                w[:L, 0] = cxu[g]; w[:L, 1] = cyu[g]; w[:L, 2] = au[g]
                                w[:L, 3] = twobu[g]; w[:L, 4] = cu[g]; w[:L, 5] = opu[g]; w[:L, 6] = col3u[ch][g]
                        rt_r[gx][gy] = [sx, sy, PXt.buffer_address(), PYt.buffer_address(),
                                        accC.buffer_address(), accT.buffer_address(), NB]
                        rt_c[gx][gy] = w.reshape(-1).tolist()
                        rt_w[gx][gy] = [sx, sy, accC.buffer_address(), accT.buffer_address(), NB]
                prog = ttnn.ProgramDescriptor(kernels=[
                    mk(FR, rt_r, ttnn.ReaderConfigDescriptor()), mk(FC, rt_c, cfg, [B]),
                    mk(FW, rt_w, ttnn.WriterConfigDescriptor())],
                    semaphores=[], cbs=[cbf(i) for i in (0, 1, 2, 3, 16, 17)])
                ttnn.generic_op([PXt, accC], prog)
            backC = ttnn.to_torch(accC).reshape(ncores, TS, TS).numpy()
            backT = ttnn.to_torch(accT).reshape(ncores, TS, TS).numpy() if (want_T and ch == 0) else None
            for local, t in enumerate(bt):
                tx, ty = t % ntx, t // ntx
                chans[ch][ty * TS:(ty + 1) * TS, tx * TS:(tx + 1) * TS] = backC[local]
                if backT is not None:
                    Tfin[ty * TS:(ty + 1) * TS, tx * TS:(tx + 1) * TS] = backT[local]
            accC.deallocate(); accT.deallocate()
        PXt.deallocate(); PYt.deallocate()
    return chans, Tfin


def fused_backward_blocked(dev, cx, cy, a, b2, c, op, colv, tile_lists, ntx, nty, Wp, Hp, gp, Tfin):
    """Backward (base m17 + host reduce) over all tiles, block-looped. Returns (geomg dict, colg[3])."""
    mx, my, grid, coords = _grid(dev)
    ncores = mx * my
    ntiles = ntx * nty
    nblocks = (ntiles + ncores - 1) // ncores
    N = len(cx)
    geomg = {k: np.zeros(N) for k in _GEOM}
    colg = [np.zeros(N) for _ in range(3)]
    v = lambda arr: np.ascontiguousarray(arr, np.float32).view(np.uint32)        # vectorized f2u (see fwd)
    cxu, cyu, au, twobu, cu, opu, bu = (v(cx), v(cy), v(a), v(2 * np.asarray(b2)), v(c), v(op), v(b2))
    colvu = [v(colv[ch]) for ch in range(3)]
    d8 = np.array([f2u(_DUMMY_G["cx"]), f2u(_DUMMY_G["cy"]), f2u(_DUMMY_G["a"]), f2u(2 * _DUMMY_G["b"]),
                   f2u(_DUMMY_G["c"]), f2u(_DUMMY_G["op"]), f2u(_DUMMY_G["col"]), f2u(_DUMMY_G["b"])], np.uint32)
    cbf = lambda i, depth: ttnn.CBDescriptor(total_size=depth * NB, core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=NB)])
    ks = lambda s, rt, cf, cta=[]: ttnn.KernelDescriptor(kernel_source=s,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid, runtime_args=rt,
            compile_time_args=cta, config=cf)

    def tchunk(t, d):
        lst = tile_lists[t][::-1][d * FUSED_K:(d + 1) * FUSED_K]
        return lst, lst + [None] * (FUSED_K - len(lst))

    for k in range(nblocks):
        bt = list(range(k * ncores, min((k + 1) * ncores, ntiles)))
        PX, PY = _pixel_coords(bt, ntx, ncores)
        Tt0 = np.ones((ncores * TS, TS), np.float32)
        for local, t in enumerate(bt):
            tx, ty = t % ntx, t // ntx
            Tt0[local * TS:(local + 1) * TS, :] = Tfin[ty * TS:(ty + 1) * TS, tx * TS:(tx + 1) * TS]
        PXt = _tileshard(dev, grid, ncores, 1, torch.from_numpy(PX))
        PYt = _tileshard(dev, grid, ncores, 1, torch.from_numpy(PY))
        maxc = max((len(tile_lists[t]) for t in bt), default=0)
        nbatch = max(1, (maxc + FUSED_K - 1) // FUSED_K)
        ones_t = _tileshard(dev, grid, ncores, 1, torch.ones(ncores * TS, TS))    # s4 matmul-reduce ones
        for ch in range(3):
            out_acc = _tileshard(dev, grid, ncores, nbatch)                       # drain-once accumulator (chunk c -> tile c)
            dl = np.zeros((ncores * TS, TS), np.float32)
            for local, t in enumerate(bt):
                tx, ty = t % ntx, t // ntx
                dl[local * TS:(local + 1) * TS, :] = gp[ty * TS:(ty + 1) * TS, tx * TS:(tx + 1) * TS, ch]
            dLt = _tileshard(dev, grid, ncores, 1, torch.from_numpy(dl))
            Tt = _tileshard(dev, grid, ncores, 1, torch.from_numpy(Tt0.copy()))
            St = _tileshard(dev, grid, ncores, 1, torch.zeros(ncores * TS, TS))
            for d in range(nbatch):
                Sout = _tileshard(dev, grid, ncores, 1); Tout = _tileshard(dev, grid, ncores, 1)
                toff = d * (TS * TS)
                rt_r, rt_c, rt_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for gx in range(mx):
                    for gy in range(my):
                        local = gy * mx + gx
                        sx, sy = coords[(gx, gy)]
                        w = np.tile(d8, (FUSED_K, 1))      # [FUSED_K,8] DUMMY-filled
                        if local < len(bt):
                            lst = tile_lists[bt[local]][::-1][d * FUSED_K:(d + 1) * FUSED_K]
                            if lst:
                                g = np.asarray(lst); L = len(g)
                                w[:L, 0] = cxu[g]; w[:L, 1] = cyu[g]; w[:L, 2] = au[g]; w[:L, 3] = twobu[g]
                                w[:L, 4] = cu[g]; w[:L, 5] = opu[g]; w[:L, 6] = colvu[ch][g]; w[:L, 7] = bu[g]
                        rt_r[gx][gy] = [sx, sy, PXt.buffer_address(), PYt.buffer_address(), dLt.buffer_address(),
                                        Tt.buffer_address(), St.buffer_address(), NB, FUSED_K,
                                        ones_t.buffer_address(), Sout.buffer_address(), Tout.buffer_address()]
                        rt_c[gx][gy] = w.reshape(-1).tolist()
                        rt_w[gx][gy] = [sx, sy, NB, FUSED_K, out_acc.buffer_address(), toff]
                prog = ttnn.ProgramDescriptor(kernels=[
                    ks(READER_RF, rt_r, ttnn.ReaderConfigDescriptor()),
                    ks(COMPUTE_RF, rt_c, ttnn.ComputeConfigDescriptor(), [FUSED_K]),
                    ks(WRITER_S4, rt_w, ttnn.WriterConfigDescriptor())],
                    semaphores=[], cbs=[cbf(i, 2) for i in (0, 1, 2, 5, 6, 24, 25, 26, 27)]
                                       + [cbf(i, 2) for i in range(8, 15)] + [cbf(i, 3) for i in range(16, 23)])
                ttnn.generic_op([PXt, out_acc], prog)
                St, Tt = Sout, Tout
            tt = ttnn.to_torch(out_acc).reshape(ncores, nbatch, TS, TS)           # drain ONCE per channel (scalars, no full-tile readback)
            for local, t in enumerate(bt):
                rev = tile_lists[t][::-1]; L = len(rev)
                if not L:
                    continue
                idx = np.asarray(rev)
                blk = tt[local, :, :FUSED_K, :].reshape(nbatch * FUSED_K, TS)     # row c*FUSED_K+g = reversed gid
                for gi_i, name in enumerate(_NAMES):
                    vals = blk[:L, gi_i * 4].numpy()
                    (colg[ch] if name == "col" else geomg[name])[idx] += vals
            out_acc.deallocate()
        ones_t.deallocate(); PXt.deallocate(); PYt.deallocate()
    return geomg, colg
