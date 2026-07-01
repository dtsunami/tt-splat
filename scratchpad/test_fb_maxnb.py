#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Validate the _MAXNB L1-cap windowing in fused_backward_blocked (the program-69 CB/L1 clash fix).

Scene has ONE hyper-dense tile so nbatch = ceil(maxc/FUSED_K) is large. We run the *production*
fused_backward_blocked three ways and compare:
  (1) _MAXNB huge  -> single window == pre-fix behavior (the reference for bit-exactness)
  (2) _MAXNB = 2   -> many windows  -> MUST be bit-identical to (1)
  (3) fused_backward_grid(base)     -> independent reference for absolute correctness (rel < 1e-2)
"""
import sys
from pathlib import Path
import numpy as np
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "pathclear"))
from sfpu_raster_scaled import scene, TS                                   # noqa: E402
from bin_sort import bin_and_sort                                         # noqa: E402
import raster_blocked as RB                                               # noqa: E402
from fused_backward import fused_backward_grid, FUSED_K                    # noqa: E402
from proto_R2_tileblock_bwd import host_forward_T                         # noqa: E402

_GEOM = ("cx", "cy", "a", "b", "c", "op")


def worst_rel(ga, ca, gb, cb):
    w = 0.0
    for name in _GEOM:
        e = np.abs(ga[name] - gb[name]).max(); s = np.abs(gb[name]).max() + 1e-9
        w = max(w, e / s)
    for ch in range(3):
        e = np.abs(ca[ch] - cb[ch]).max(); s = np.abs(cb[ch]).max() + 1e-9
        w = max(w, e / s)
    return w


def main():
    W = H = 256                                  # 8x8 tiles
    Ng = 120
    cx, cy, sx, sy, op, col, depth, abc = scene(3, W, H, Ng)
    cx, cy, op, col = cx.numpy(), cy.numpy(), op.numpy(), col.numpy()
    a = np.array([t[0] for t in abc]); b2 = np.array([t[1] for t in abc]); c = np.array([t[2] for t in abc])
    sx, sy, depth = sx.numpy(), sy.numpy(), depth.numpy()

    # inject a hyper-dense cluster into tile (3,3): NDENSE tiny Gaussians -> nbatch >> _MAXNB
    NDENSE = 240
    rng = np.random.default_rng(7)
    cxc = 3 * TS + 16 + rng.standard_normal(NDENSE) * 1.5
    cyc = 3 * TS + 16 + rng.standard_normal(NDENSE) * 1.5
    cx = np.concatenate([cx, cxc]); cy = np.concatenate([cy, cyc])
    a = np.concatenate([a, np.full(NDENSE, 0.5)]); b2 = np.concatenate([b2, np.zeros(NDENSE)])
    c = np.concatenate([c, np.full(NDENSE, 0.5)]); op = np.concatenate([op, np.full(NDENSE, 0.1)])
    col = np.concatenate([col, rng.random(NDENSE).astype(col.dtype)])
    sx = np.concatenate([sx, np.full(NDENSE, 1.0, sx.dtype)])
    sy = np.concatenate([sy, np.full(NDENSE, 1.0, sy.dtype)])
    depth = np.concatenate([depth, np.full(NDENSE, depth.mean(), depth.dtype)])

    s_gid, _st, ranges, ntx, nty, _tot = bin_and_sort(cx, cy, sx**2, sy**2, depth, W, H, ts=TS)
    tile_lists = [s_gid[ranges[t, 0]:ranges[t, 1]].tolist() for t in range(ntx * nty)]
    maxc = max(len(l) for l in tile_lists)
    nbatch = (maxc + FUSED_K - 1) // FUSED_K
    print(f"scene: N={len(cx)}  tiles={ntx}x{nty}  densest tile={maxc} gaussians  nbatch={nbatch}", flush=True)

    abc_list = list(zip(a.tolist(), b2.tolist(), c.tolist()))
    Tfin = host_forward_T(torch.from_numpy(cx), torch.from_numpy(cy), torch.from_numpy(op),
                          torch.from_numpy(col), abc_list, tile_lists, W, H, ntx)
    gp = np.random.default_rng(0).standard_normal((H, W, 3)).astype(np.float64) * 0.01
    colv = [col, col * 0.7, col * 1.3]

    dev = ttnn.open_device(device_id=0)
    try:
        RB._MAXNB = 100000                                  # single window == pre-fix
        gref, cref = RB.fused_backward_blocked(dev, cx, cy, a, b2, c, op, colv, tile_lists,
                                               ntx, nty, W, H, gp, Tfin)
        print("  (1) single-window done", flush=True)
        RB._MAXNB = 2                                       # force many windows
        gwin, cwin = RB.fused_backward_blocked(dev, cx, cy, a, b2, c, op, colv, tile_lists,
                                               ntx, nty, W, H, gp, Tfin)
        print("  (2) windowed (_MAXNB=2) done", flush=True)
        ggrid, cgrid = fused_backward_grid(dev, cx, cy, a, b2, c, op, colv, tile_lists,
                                           ntx, nty, W, H, gp, Tfin, stage="base")
        print("  (3) grid(base) done", flush=True)

        # grid s4 path (the production resident-loop stage) — also has the out_acc growth; test its windowing
        import fused_backward as FB
        FB._MAXNB = 100000
        gs4r, cs4r = fused_backward_grid(dev, cx, cy, a, b2, c, op, colv, tile_lists,
                                         ntx, nty, W, H, gp, Tfin, stage="s4")
        FB._MAXNB = 2
        gs4w, cs4w = fused_backward_grid(dev, cx, cy, a, b2, c, op, colv, tile_lists,
                                         ntx, nty, W, H, gp, Tfin, stage="s4")
        print("  (4) grid(s4) single vs windowed done", flush=True)

        w_bitexact = worst_rel(gwin, cwin, gref, cref)
        w_abs = worst_rel(gwin, cwin, ggrid, cgrid)
        w_s4 = worst_rel(gs4w, cs4w, gs4r, cs4r)
        print(f"\n  blocked windowed vs single-window (MUST be ~0): worst rel = {w_bitexact:.3e}")
        print(f"  blocked windowed vs grid(base)      (< 1e-2):    worst rel = {w_abs:.3e}")
        print(f"  grid(s4) windowed vs single-window  (MUST be ~0): worst rel = {w_s4:.3e}")
        ok = (w_bitexact < 1e-9) and (w_abs < 1e-2) and (w_s4 < 1e-9)
        print(f"  -> {'PASS' if ok else 'FAIL'}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
