#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PROTO S2+S3 (host, math-first) — the counting-bucket bin/sort ALGORITHM, before the device port.

Key simplification: S2 (assemble) + S3 (within-tile depth order) UNIFY into ONE counting sort on a
COMPOSITE integer key  key = tile_id * D + depth_bucket  (D=64 from S0). Counting sort = histogram(key)
-> exclusive-scan -> scatter — exactly ttnn.cumsum + the m2_scatter_gather owner-write. So the whole
device sort is one scan + one scatter on a bounded key in [0, ntiles*D).

Pipeline (each step maps to a device primitive):
  S1 AABB+counts            -> per-Gaussian (already on silicon, probe_S1)
  exclusive-scan(counts)    -> instance offsets            [ttnn.cumsum]
  instance expansion        -> (tile_id, gid, depth)       [reader gather by offset — the E5 substrate]
  key = tile_id*D + bucket  -> composite key               [ttnn elementwise]
  histogram(key)->scan->scatter -> sorted s_gid + ranges   [cumsum + m2 owner-scatter]

GATE: per-tile gid SET identical to host bin_and_sort (same instances; order differs by the S0-approved
depth-BUCKET vs exact-depth — that's the whole point). Run @1600px (the target).
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "pathclear"))
from probe_S0_depthbucket import load_ply, project_cam, PLY, DATASET   # noqa: E402
from probe_S1_tile_assign import host_assign                            # noqa: E402
from train_tt import _load_colmap                                       # noqa: E402
from bin_sort import bin_and_sort                                       # noqa: E402

TS = 32
D = 64


def conic_to_var(a, b, c):
    detc = a * c - b * b
    detc = np.where(np.abs(detc) < 1e-12, 1e-12, detc)
    return np.clip(c / detc, 0.25, None), np.clip(a / detc, 0.25, None)


def counting_assemble(u, v, a, b, cc, zc, W, H, ts, D):
    """The device-mirroring counting-bucket sort. Returns s_gid, ranges, total."""
    N = len(u)
    ntx, nty = (W + ts - 1) // ts, (H + ts - 1) // ts
    aabb, _deg = host_assign(u, v, a, b, cc, W, H, ts)
    tx0, tx1, ty0, ty1 = aabb.astype(np.int64)
    bw, bh = tx1 - tx0 + 1, ty1 - ty0 + 1
    counts = bw * bh

    # exclusive scan(counts) -> per-Gaussian instance offset      [ttnn.cumsum]
    offsets = np.cumsum(counts) - counts
    total = int(counts.sum())

    # instance expansion: enumerate each Gaussian's AABB box       [reader gather by offset]
    gid = np.repeat(np.arange(N, dtype=np.int64), counts)
    local = np.arange(total) - np.repeat(offsets, counts)
    bwr = np.repeat(bw, counts)
    tx = np.repeat(tx0, counts) + local % bwr
    ty = np.repeat(ty0, counts) + local // bwr
    tile_id = ty * ntx + tx
    dep = zc[gid]

    # composite key = tile_id*D + global depth bucket              [ttnn elementwise]
    zmin, zmax = dep.min(), dep.max() + 1e-9
    bucket = np.clip(((dep - zmin) / (zmax - zmin) * D).astype(np.int64), 0, D - 1)
    key = tile_id * D + bucket

    # counting sort on key: histogram -> exclusive-scan -> scatter [cumsum + m2 owner-scatter]
    nkey = ntx * nty * D
    hist = np.bincount(key, minlength=nkey)
    base = np.cumsum(hist) - hist                                  # per-key start offset
    # device: each instance writes to base[key] + (stable rank within key) via owner-increment.
    # host mirror: stable counting-sort scatter == argsort(key, stable).
    order = np.argsort(key, kind="stable")
    s_gid = gid[order]
    s_tile = tile_id[order]

    # per-tile [start,end)
    ranges = np.zeros((ntx * nty, 2), dtype=np.int64)
    uniq, starts = np.unique(s_tile, return_index=True)
    ends = np.append(starts[1:], len(s_tile))
    ranges[uniq, 0] = starts; ranges[uniq, 1] = ends
    return s_gid, ranges, total, base, hist


def main():
    P, npts = load_ply(PLY)
    cams, _, _ = _load_colmap(DATASET)
    print(f"trained ply N={npts}; PROTO S2+S3 counting-bucket sort (D={D})\n")
    for LONG in (96, 1600):
        name, H, W, arrs = project_cam(P, cams[0], LONG)
        u, v, a, b, cc, zc = arrs[0], arrs[1], arrs[2], arrs[3], arrs[4], arrs[6]
        ntx, nty = (W + TS - 1) // TS, (H + TS - 1) // TS

        var_x, var_y = conic_to_var(a, b, cc)
        gh, th, rh, ntx2, nty2, total_h = bin_and_sort(u, v, var_x, var_y, zc, W, H, ts=TS)
        gd, rd, total_d, base, hist = counting_assemble(u, v, a, b, cc, zc, W, H, TS, D)

        # GATE: per-tile gid SET identical (instances match; order differs by depth bucket)
        bad = 0; checked = 0
        for t in range(ntx * nty):
            sh = set(gh[rh[t, 0]:rh[t, 1]].tolist())
            sd = set(gd[rd[t, 0]:rd[t, 1]].tolist())
            if sh:
                checked += 1
                if sh != sd:
                    bad += 1
        # within-tile ORDER monotonic in depth-bucket (S3 correctness)
        mono = True
        for t in range(0, ntx * nty, max(1, ntx * nty // 200)):     # sample tiles
            g = gd[rd[t, 0]:rd[t, 1]]
            if len(g) > 1:
                zb = np.clip(((zc[g] - zc.min()) / (zc.max() - zc.min() + 1e-9) * D).astype(int), 0, D - 1)
                if np.any(np.diff(zb) < 0):
                    mono = False; break
        print(f"  @{LONG:>4} ({W}x{H}, {ntx}x{nty}={ntx*nty} tiles)  instances host={total_h} dev={total_d}")
        print(f"      per-tile gid SET match : {checked-bad}/{checked} tiles "
              f"({'PASS' if bad == 0 else f'{bad} MISMATCH'})")
        print(f"      within-tile depth order: {'monotonic-in-bucket PASS' if mono else 'NON-MONO FAIL'}")
        print(f"      key space = {ntx*nty*D} buckets  (hist nonempty={int((hist>0).sum())})\n")
    print("Maps to device: cumsum(counts)+gather expansion, then cumsum(hist)+m2-scatter on key=tile*D+bucket.")


if __name__ == "__main__":
    main()
