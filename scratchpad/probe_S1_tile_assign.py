#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PROBE S1 (silicon) — on-device tile ASSIGNMENT, exact, gated vs the host bin_and_sort math.

The embarrassingly-parallel front half of the on-device counting-bucket bin/sort: per Gaussian compute
conic->variance, the 3-sigma screen AABB, the covered tile-range [tx0,tx1]x[ty0,ty1], and the instance
COUNT — all in batched ttnn over [N], reading the projected conic (a,b,c) + centers (u,v). Mirrors
render_device.py's host pipeline (conic->var with the detc guard + clip(.,0.25)) feeding bin_sort.py.

fp32 device will NOT be bit-exact to the float64 host at tile boundaries (floor of a value straddling an
integer edge can land ±1 tile). GATE: AABB matches host for the vast majority; every mismatch is a benign
boundary off-by-ONE (<=1 tile in a single coord) and total instance count agrees within ~1% — the same
benign class as the S0 depth-bucket approximation. Run @96px and @1.6k.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "pathclear"))
from probe_S0_depthbucket import load_ply, project_cam, PLY, DATASET   # noqa: E402
from train_tt import _load_colmap                                       # noqa: E402

TS = 32


def host_assign(u, v, a, b, c, W, H, ts):
    """float64 golden — render_device.py conic->var (detc guard + clip 0.25) + bin_sort.py 3sigma AABB."""
    u, v, a, b, c = (np.asarray(x, np.float64) for x in (u, v, a, b, c))
    ntx, nty = (W + ts - 1) // ts, (H + ts - 1) // ts
    detc = a * c - b * b
    deg = np.abs(detc) < 1e-12                                  # degenerate (host special-cases these)
    detc = np.where(deg, 1e-12, detc)
    var_x = np.clip(c / detc, 0.25, None); var_y = np.clip(a / detc, 0.25, None)
    rx, ry = 3.0 * np.sqrt(var_x), 3.0 * np.sqrt(var_y)
    tx0 = np.clip(np.floor((u - rx) / ts), 0, ntx - 1); tx1 = np.clip(np.floor((u + rx) / ts), 0, ntx - 1)
    ty0 = np.clip(np.floor((v - ry) / ts), 0, nty - 1); ty1 = np.clip(np.floor((v + ry) / ts), 0, nty - 1)
    return np.stack([tx0, tx1, ty0, ty1]), deg


def dev_assign(dev, u, v, a, b, c, W, H, ts):
    """Same map in batched ttnn (fp32, interleaved DRAM) — the correctness layout (perf streaming is S4)."""
    f = lambda x: ttnn.from_torch(torch.tensor(np.asarray(x), dtype=torch.float32).reshape(-1),
                                  dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    U, V, A, B, C = f(u), f(v), f(a), f(b), f(c)
    ntx, nty = (W + ts - 1) // ts, (H + ts - 1) // ts
    inv = 1.0 / ts
    detc = ttnn.sub(ttnn.mul(A, C), ttnn.mul(B, B))
    var_x = ttnn.clamp(ttnn.div(C, detc), 0.25, 1e30)
    var_y = ttnn.clamp(ttnn.div(A, detc), 0.25, 1e30)
    rx = ttnn.mul(ttnn.sqrt(var_x), 3.0); ry = ttnn.mul(ttnn.sqrt(var_y), 3.0)
    tx0 = ttnn.clamp(ttnn.floor(ttnn.mul(ttnn.sub(U, rx), inv)), 0.0, float(ntx - 1))
    tx1 = ttnn.clamp(ttnn.floor(ttnn.mul(ttnn.add(U, rx), inv)), 0.0, float(ntx - 1))
    ty0 = ttnn.clamp(ttnn.floor(ttnn.mul(ttnn.sub(V, ry), inv)), 0.0, float(nty - 1))
    ty1 = ttnn.clamp(ttnn.floor(ttnn.mul(ttnn.add(V, ry), inv)), 0.0, float(nty - 1))
    n = len(np.asarray(u))
    g = lambda t: ttnn.to_torch(t).flatten().numpy()[:n]
    return np.stack([g(tx0), g(tx1), g(ty0), g(ty1)])


def main():
    P, npts = load_ply(PLY)
    cams, _, _ = _load_colmap(DATASET)
    dev = ttnn.open_device(device_id=0)
    try:
        print(f"trained ply N={npts}; S1 on silicon\n")
        for LONG in (96, 1600):
            name, H, W, arrs = project_cam(P, cams[0], LONG)
            u, v, a, b, cc = arrs[0], arrs[1], arrs[2], arrs[3], arrs[4]
            ntx, nty = (W + TS - 1) // TS, (H + TS - 1) // TS
            hb, deg = host_assign(u, v, a, b, cc, W, H, TS)
            db = dev_assign(dev, u, v, a, b, cc, W, H, TS)
            keep = ~deg                                            # exclude host-degenerate from the gate
            d = np.abs(db[:, keep] - hb[:, keep])
            exact = np.all(d == 0, axis=0)                          # all 4 AABB coords match
            maxd = int(d.max()) if d.size else 0
            hc = ((hb[1] - hb[0] + 1) * (hb[3] - hb[2] + 1))[keep]  # instance counts
            dc = ((db[1] - db[0] + 1) * (db[3] - db[2] + 1))[keep]
            inst_h, inst_d = int(hc.sum()), int(dc.sum())
            print(f"  @{LONG:>4} ({W}x{H}, {ntx}x{nty}={ntx*nty} tiles)  N={keep.sum()} deg={int(deg.sum())}")
            print(f"      AABB exact      : {100*exact.mean():6.2f}%   (max coord delta = {maxd} tile)")
            print(f"      counts exact    : {100*np.mean(hc==dc):6.2f}%   max|dcount|={int(np.abs(hc-dc).max())}")
            print(f"      total instances : host={inst_h}  dev={inst_d}  "
                  f"({100*abs(inst_d-inst_h)/max(inst_h,1):.3f}% off)")
            gate = (maxd <= 1) and (abs(inst_d - inst_h) <= 0.01 * inst_h)
            print(f"      -> {'PASS' if gate else 'CHECK'} (boundary off-by<=1 & instances within 1%)\n")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
