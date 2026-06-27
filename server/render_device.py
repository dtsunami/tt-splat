#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Phase-1 device render: rasterize the actual 3DGS scene ON the Blackhole for the dashboard display.

Projection + SH color + tile binning run on the HOST (cheap, reuses the verified train_real math);
the per-pixel conic alpha-blend runs ON THE DEVICE via the M14 scaled rasterizer (sfpu_raster_scaled:
per-tile culling, unbounded N via batched dispatch, persistent L1 C/T accumulators, fp32 accum).
The kernel blends a SCALAR color, so RGB = 3 passes (R,G,B) sharing one binning.

Training math stays on the host (gradients via train_real.render); only the dashboard's display frame
is rendered on-device. So this is PSU-safe: a culled ~1200-Gaussian render is dispatch-bound and light
(the full-grid raster measured ~76W). The heavy/virus regime is device *training* (Phase 2), which is
where the power_ramp RampController guard becomes load-bearing — see [[bh-psu-power-virus-reboot]].

Device is opened ONCE (persistent) and per-size resources are cached; comfy SDXL must be down (single
owner). Gate from train_tt via TT_DEVICE_RENDER=1.
"""
from __future__ import annotations
import os, sys, subprocess, re
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "pathclear"))
import ttnn                                              # noqa: E402
from train_real import project_general, sh_eval         # noqa: E402  host projection + SH color
from bin_sort import bin_and_sort                        # noqa: E402  M6 tile binning + depth sort
import sfpu_raster_scaled as M14                          # noqa: E402  READER/COMPUTE/WRITER, B, TS, f2u, DUMMY, block_l1

_POWER_CEILING = float(os.environ.get("TT_POWER_CEILING", "240"))   # preflight guard (W)
_DEV = None
_CACHE: dict = {}                                        # (Wp,Hp) -> cached device resources


def _preflight_power():
    """One-time best-effort guard: if the card is already near the PSU ceiling at open, refuse the
    device path (caller falls back to host). A render itself is light; this catches a hot/contended card."""
    try:
        out = subprocess.run(["tt-smi", "-s"], capture_output=True, text=True, timeout=25).stdout
        m = re.search(r'"power":\s*"?\s*([0-9.]+)', out)
        if m and float(m.group(1)) > _POWER_CEILING:
            raise RuntimeError(f"device power {m.group(1)}W > ceiling {_POWER_CEILING}W at open — "
                               "refusing device render (card hot/contended)")
    except FileNotFoundError:
        pass   # tt-smi not on PATH; proceed (render is light)


def _device():
    global _DEV
    if _DEV is None:
        _preflight_power()
        _DEV = ttnn.open_device(device_id=0)
    return _DEV


def _resources(dev, Wp, Hp):
    """Per-size device resources (grid + pixel-coord shards + worker NoC coords), built once."""
    key = (Wp, Hp)
    if key not in _CACHE:
        TS = M14.TS
        GX, GY = Wp // TS, Hp // TS
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
        ii, jj = torch.meshgrid(torch.arange(Hp), torch.arange(Wp), indexing="ij")
        PXt = M14.block_l1(dev, grid, Wp, Hp, jj.float())     # global x pixel coords, one tile/core
        PYt = M14.block_l1(dev, grid, Wp, Hp, ii.float())     # global y
        coords = {}
        for gx in range(GX):
            for gy in range(GY):
                hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(gx, gy))
                coords[(gx, gy)] = (hp.x, hp.y)
        _CACHE[key] = dict(GX=GX, GY=GY, grid=grid, PXt=PXt, PYt=PYt, coords=coords)
    return _CACHE[key]


def _raster_channel(dev, res, tile_lists, ntx, nbatch, cx, cy, ca, cb, cc, op, colch, Wp, Hp, want_T=False):
    """One scalar-color device raster pass (the M14 batched dispatch) -> (Hp,Wp) numpy.
    want_T=True also returns the final transmittance T (geometry-only, same across channels)."""
    GX, GY, grid = res["GX"], res["GY"], res["grid"]
    PXt, PYt, coords = res["PXt"], res["PYt"], res["coords"]
    TS, Bb = M14.TS, M14.B
    NB = TS * TS * 4
    accC = M14.block_l1(dev, grid, Wp, Hp, torch.zeros(Hp, Wp))
    accT = M14.block_l1(dev, grid, Wp, Hp, torch.ones(Hp, Wp))

    cbf = lambda idx: ttnn.CBDescriptor(total_size=2 * NB, core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=NB)])

    def compute_cfg():
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.fp32_dest_acc_en = True; cfg.math_approx_mode = False
        return cfg

    def params_for(t, d):
        lst = tile_lists[t][d * Bb:(d + 1) * Bb]
        out = []
        for k in range(Bb):
            if k < len(lst):
                i = lst[k]
                out += [M14.f2u(cx[i]), M14.f2u(cy[i]), M14.f2u(ca[i]), M14.f2u(2 * cb[i]),
                        M14.f2u(cc[i]), M14.f2u(op[i]), M14.f2u(colch[i])]
            else:
                out += M14.DUMMY
        return out

    def dispatch(d):
        rt_r, rt_c, rt_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for gx in range(GX):
            for gy in range(GY):
                sx_, sy_ = coords[(gx, gy)]; t = gy * ntx + gx
                rt_r[gx][gy] = [sx_, sy_, PXt.buffer_address(), PYt.buffer_address(),
                                accC.buffer_address(), accT.buffer_address(), NB]
                rt_c[gx][gy] = params_for(t, d)
                rt_w[gx][gy] = [sx_, sy_, accC.buffer_address(), accT.buffer_address(), NB]
        mk = lambda src, rt, cfg, cta=[]: ttnn.KernelDescriptor(
            kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=grid, runtime_args=rt, compile_time_args=cta, config=cfg)
        prog = ttnn.ProgramDescriptor(kernels=[
            mk(M14.READER, rt_r, ttnn.ReaderConfigDescriptor()),
            mk(M14.COMPUTE, rt_c, compute_cfg(), [Bb]),
            mk(M14.WRITER, rt_w, ttnn.WriterConfigDescriptor())],
            semaphores=[], cbs=[cbf(i) for i in (0, 1, 2, 3, 16, 17)])
        ttnn.generic_op([PXt, accC], prog)

    for d in range(nbatch):
        dispatch(d)
    out = ttnn.to_torch(accC).reshape(Hp, Wp).numpy().copy()
    T = ttnn.to_torch(accT).reshape(Hp, Wp).numpy().copy() if want_T else None
    accC.deallocate(); accT.deallocate()
    return (out, T) if want_T else out


def _resources_par(dev, res, Wp, Hp, CH):
    """Lazily build + cache the CH-band parallel grid: channels stacked on vertical core-bands
    (logical y = band*GY + gy). Returns None (cached) when the band-grid exceeds the worker grid."""
    if "par" in res:
        return res["par"]
    TS = M14.TS; GX, GY = res["GX"], res["GY"]
    try:
        gs = dev.compute_with_storage_grid_size(); maxX, maxY = gs.x, gs.y
    except Exception:
        maxX = maxY = 0
    if not (maxY and GX <= maxX and GY * CH <= maxY):
        res["par"] = None
        return None
    GYx = GY * CH
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GYx - 1))])
    ii, jj = torch.meshgrid(torch.arange(Hp), torch.arange(Wp), indexing="ij")
    PXt = M14.block_l1(dev, grid, Wp, GYx * TS, torch.cat([jj] * CH, dim=0).float())   # pixel coords per band
    PYt = M14.block_l1(dev, grid, Wp, GYx * TS, torch.cat([ii] * CH, dim=0).float())
    coords = {}
    for gx in range(GX):
        for ly in range(GYx):
            hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(gx, ly))
            coords[(gx, ly)] = (hp.x, hp.y)
    res["par"] = dict(GX=GX, GY=GY, GYx=GYx, grid=grid, PXt=PXt, PYt=PYt, coords=coords)
    return res["par"]


def _raster_rgb(dev, res, tile_lists, ntx, nbatch, cx, cy, ca, cb, cc, op, col3, Wp, Hp, want_T=False):
    """Render all 3 color channels in ONE banded dispatch set (R/G/B on stacked core-bands) instead of
    3 serial M14 passes. Geometry is channel-invariant; only `col` differs per band. col3 = [R,G,B]
    arrays. Returns (list[3] of (Hp,Wp) numpy, T or None). Falls back to serial when bands don't fit."""
    CH = 3
    par = _resources_par(dev, res, Wp, Hp, CH) if os.environ.get("RAST_CHANPAR", "1") == "1" else None
    if par is None:                                       # serial fallback (band-grid doesn't fit / disabled)
        chans, T = [], None
        for k in range(CH):
            r = _raster_channel(dev, res, tile_lists, ntx, nbatch, cx, cy, ca, cb, cc, op, col3[k],
                                Wp, Hp, want_T=(want_T and k == 0))
            ck, T = r if (want_T and k == 0) else (r, T)
            chans.append(ck)
        return chans, T
    GX, GY, GYx, grid = par["GX"], par["GY"], par["GYx"], par["grid"]
    PXt, PYt, coords = par["PXt"], par["PYt"], par["coords"]
    TS, Bb = M14.TS, M14.B
    NB = TS * TS * 4
    accC = M14.block_l1(dev, grid, Wp, GYx * TS, torch.zeros(GYx * TS, Wp))
    accT = M14.block_l1(dev, grid, Wp, GYx * TS, torch.ones(GYx * TS, Wp))

    cbf = lambda idx: ttnn.CBDescriptor(total_size=2 * NB, core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=NB)])

    def compute_cfg():
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.fp32_dest_acc_en = True; cfg.math_approx_mode = False
        return cfg

    # precompute channel-INVARIANT geometry words ONCE per (tile, batch); patch only `col` per band
    geo = {}
    for d in range(nbatch):
        for t in range(ntx * (Hp // TS)):
            lst = tile_lists[t][d * Bb:(d + 1) * Bb]
            words = []; real = []
            for k in range(Bb):
                if k < len(lst):
                    i = lst[k]
                    words += [M14.f2u(cx[i]), M14.f2u(cy[i]), M14.f2u(ca[i]), M14.f2u(2 * cb[i]),
                              M14.f2u(cc[i]), M14.f2u(op[i]), 0]      # col (slot k*7+6) patched per band
                    real.append((k * 7 + 6, i))
                else:
                    words += M14.DUMMY
            geo[(t, d)] = (words, real)

    def dispatch(d):
        rt_r, rt_c, rt_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for gx in range(GX):
            for gy in range(GY):
                t = gy * ntx + gx
                words, real = geo[(t, d)]
                for b in range(CH):
                    ly = b * GY + gy; sx_, sy_ = coords[(gx, ly)]
                    rt_r[gx][ly] = [sx_, sy_, PXt.buffer_address(), PYt.buffer_address(),
                                    accC.buffer_address(), accT.buffer_address(), NB]
                    cc_words = words[:]; cb_ = col3[b]                # copy; patch only the per-channel col words
                    for (slot, i) in real:
                        cc_words[slot] = M14.f2u(cb_[i])
                    rt_c[gx][ly] = cc_words
                    rt_w[gx][ly] = [sx_, sy_, accC.buffer_address(), accT.buffer_address(), NB]
        mk = lambda src, rt, cfg, cta=[]: ttnn.KernelDescriptor(
            kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=grid, runtime_args=rt, compile_time_args=cta, config=cfg)
        prog = ttnn.ProgramDescriptor(kernels=[
            mk(M14.READER, rt_r, ttnn.ReaderConfigDescriptor()),
            mk(M14.COMPUTE, rt_c, compute_cfg(), [Bb]),
            mk(M14.WRITER, rt_w, ttnn.WriterConfigDescriptor())],
            semaphores=[], cbs=[cbf(i) for i in (0, 1, 2, 3, 16, 17)])
        ttnn.generic_op([PXt, accC], prog)

    for d in range(nbatch):
        dispatch(d)
    full = ttnn.to_torch(accC).reshape(CH, Hp, Wp)        # band b (rows b*Hp:) = channel b's image
    chans = [full[b].numpy().copy() for b in range(CH)]
    T = ttnn.to_torch(accT).reshape(CH, Hp, Wp)[0].numpy().copy() if want_T else None   # geometry-only
    accC.deallocate(); accT.deallocate()
    return chans, T


def render_device(P, cam, H, W) -> np.ndarray:
    """Render P from camera `cam` ON the Blackhole. Returns (H,W,3) float32 in [0,1].
    Matches train_real.render's projection/SH/opacity; the per-pixel blend runs on-device (M14)."""
    dev = _device()
    Rv, tv, fx, fy, ppx, ppy = cam[:6]
    with torch.no_grad():
        u, v, zc, (ca, cb, cc) = project_general(P, Rv, tv, fx, fy, ppx, ppy)
        cam_center = -Rv.T @ tv
        dirs = P["mean"] - cam_center
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-9)
        col = sh_eval(P["sh"], dirs, P["deg"])           # [N,3] in [0,1]
        op = torch.sigmoid(P["op"])

    u, v, zc = u.numpy(), v.numpy(), zc.numpy()
    ca, cb, cc = ca.numpy(), cb.numpy(), cc.numpy()
    op, col = op.numpy(), col.numpy()

    valid = zc > 1e-4                                     # cull behind-camera (matches host render)
    u, v, ca, cb, cc, op, zc = (a[valid] for a in (u, v, ca, cb, cc, op, zc))
    col = col[valid]
    if u.shape[0] == 0:
        return np.zeros((H, W, 3), np.float32)

    # binning needs 2D variance (σ²), recovered by inverting the conic (a,b,c)=Σ2⁻¹
    detc = ca * cc - cb * cb
    detc = np.where(np.abs(detc) < 1e-12, 1e-12, detc)
    var_x = np.clip(cc / detc, 0.25, None)
    var_y = np.clip(ca / detc, 0.25, None)

    TS = M14.TS
    Wp = ((W + TS - 1) // TS) * TS                        # pad to 32-multiple; crop after
    Hp = ((H + TS - 1) // TS) * TS
    res = _resources(dev, Wp, Hp)

    s_gid, _s_tile, ranges, ntx, nty, _total = bin_and_sort(u, v, var_x, var_y, zc, Wp, Hp, ts=TS)
    tile_lists = [s_gid[ranges[t, 0]:ranges[t, 1]].tolist() for t in range(ntx * nty)]
    max_count = max((len(l) for l in tile_lists), default=0)
    if max_count == 0:
        return np.zeros((H, W, 3), np.float32)
    nbatch = (max_count + M14.B - 1) // M14.B

    chans, _ = _raster_rgb(dev, res, tile_lists, ntx, nbatch, u, v, ca, cb, cc, op,
                           [col[:, k] for k in range(3)], Wp, Hp)
    img = np.stack(chans, axis=-1)[:H, :W, :]            # (Hp,Wp,3) -> crop to (H,W,3)
    return np.clip(img, 0.0, 1.0).astype(np.float32)
