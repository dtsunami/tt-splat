#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Tile binning + depth sort — the 3DGS stage Tensix is worst at (data-dependent sort), so we
run it on a GENERAL-PURPOSE target. Host-first here (a legitimate heterogeneous target: sort
on host/GPU while Tensix does the dense math); the interface is target-agnostic so it retargets
to the NVIDIA GPU (radix sort) or x280 later.

Pipeline (matches gsplat/Inria):
  per Gaussian: 3-sigma screen AABB -> tiles covered
  expand to (tile_id, depth, gid) instances (Gaussians duplicated across covered tiles)
  SORT by (tile_id, depth)              <- the hard part; trivial on CPU/GPU
  extract per-tile [start,end) ranges

Verified against brute force; benchmarked at scale to confirm host/GPU is a fine target.
"""
import time
import numpy as np

TS = 16                      # tile size (px)


def bin_and_sort(cx, cy, sxx, syy, depth, W, H, ts=TS):
    """centers (cx,cy), covariance diagonals (sxx,syy) [variance], depth -> sorted instances."""
    ntx = (W + ts - 1) // ts
    nty = (H + ts - 1) // ts
    rx = 3.0 * np.sqrt(sxx)            # 3-sigma half-extents (AABB of the ellipse)
    ry = 3.0 * np.sqrt(syy)
    tx0 = np.clip(np.floor((cx - rx) / ts).astype(np.int64), 0, ntx - 1)
    tx1 = np.clip(np.floor((cx + rx) / ts).astype(np.int64), 0, ntx - 1)
    ty0 = np.clip(np.floor((cy - ry) / ts).astype(np.int64), 0, nty - 1)
    ty1 = np.clip(np.floor((cy + ry) / ts).astype(np.int64), 0, nty - 1)
    counts = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)        # tiles per Gaussian
    total = int(counts.sum())

    # expand to instances (vectorized): for each Gaussian, enumerate its tile box
    gid = np.repeat(np.arange(len(cx), dtype=np.int64), counts)
    # within-gaussian local index -> (dx,dy) offset in its box
    local = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    bw = np.repeat(tx1 - tx0 + 1, counts)
    dx = local % bw
    dy = local // bw
    tx = np.repeat(tx0, counts) + dx
    ty = np.repeat(ty0, counts) + dy
    tile_id = ty * ntx + tx
    dep = depth[gid]

    # SORT by (tile_id primary, depth secondary)  -- the heterogeneous-target hot spot
    order = np.lexsort((dep, tile_id))
    s_tile = tile_id[order]; s_gid = gid[order]

    # per-tile [start,end)
    ranges = np.zeros((ntx * nty, 2), dtype=np.int64)
    uniq, starts = np.unique(s_tile, return_index=True)
    ends = np.append(starts[1:], len(s_tile))
    ranges[uniq, 0] = starts; ranges[uniq, 1] = ends
    return s_gid, s_tile, ranges, ntx, nty, total


def brute_force_tile(t, cx, cy, sxx, syy, depth, W, H, ts=TS):
    ntx = (W + ts - 1) // ts
    tx, ty = t % ntx, t // ntx
    rx, ry = 3.0*np.sqrt(sxx), 3.0*np.sqrt(syy)
    x0 = np.floor((cx-rx)/ts); x1 = np.floor((cx+rx)/ts)
    y0 = np.floor((cy-ry)/ts); y1 = np.floor((cy+ry)/ts)
    hit = (np.clip(x0,0,ntx-1) <= tx) & (tx <= np.clip(x1,0,ntx-1)) & \
          (np.clip(y0,0,(H+ts-1)//ts-1) <= ty) & (ty <= np.clip(y1,0,(H+ts-1)//ts-1))
    ids = np.where(hit)[0]
    return ids[np.argsort(depth[ids])]


def main():
    rng = np.random.default_rng(0)

    # ---- correctness: small scene, verify every tile vs brute force ----
    W, H, N = 128, 128, 200
    cx = rng.uniform(0, W, N); cy = rng.uniform(0, H, N)
    sxx = rng.uniform(2, 30, N); syy = rng.uniform(2, 30, N)
    depth = rng.uniform(0, 1, N)
    s_gid, s_tile, ranges, ntx, nty, total = bin_and_sort(cx, cy, sxx, syy, depth, W, H)
    ok = True
    for t in range(ntx * nty):
        st, en = ranges[t]
        got = s_gid[st:en]
        exp = brute_force_tile(t, cx, cy, sxx, syy, depth, W, H)
        if not np.array_equal(got, exp):
            ok = False; break
    print(f"correctness  W={W} H={H} N={N}  tiles={ntx*nty} instances={total} "
          f"dup={total/N:.1f}x  ->  {'OK' if ok else 'MISMATCH @tile %d'%t}")

    # ---- benchmark at scale: is host/GPU a fine target? ----
    for N in (10_000, 100_000, 1_000_000):
        W = H = 1024
        cx = rng.uniform(0, W, N); cy = rng.uniform(0, H, N)
        sxx = rng.uniform(2, 25, N); syy = rng.uniform(2, 25, N)
        depth = rng.uniform(0, 1, N)
        t0 = time.perf_counter()
        s_gid, s_tile, ranges, ntx, nty, total = bin_and_sort(cx, cy, sxx, syy, depth, W, H)
        dt = (time.perf_counter() - t0) * 1e3
        print(f"  N={N:>9,}  instances={total:>10,} ({total/N:.1f}x)  "
              f"bin+sort={dt:7.1f} ms  ({total/dt/1e3:.1f} M-inst/s on host CPU)")

    print("interface (device<->target): SEND per-Gaussian {cx,cy,cov_xx,cov_yy,depth} (5 floats);"
          " RETURN sorted gid[] + per-tile [start,end). Sort is target-agnostic (CPU lexsort now;"
          " GPU radix / x280 next).")
    print("BIN_SORT_OK" if ok else "BIN_SORT_FAIL")


if __name__ == "__main__":
    main()
