#!/usr/bin/env python3
# E4: does the hash-home inbox (distinct per-(source,gid) slots) fit L1 under worst-case replication?
# Replication(g) = #tiles its 3-sigma AABB covers = fan_in for that gid (one source per tile). Per owner
# inbox bytes = sum over owned gids of replication(g) * 7 grads * 4B. Check vs ~512KB/core budget
# (1.46MB L1 total minus code+CBs+raster shards). Host-only (mirrors bin_sort.py AABB math).
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "tt-splat" / "docs" / "pathclear"))
from bin_sort import bin_and_sort, TS as BIN_TS

NCH = 7; BYTES = 4
L1_BUDGET = 512 * 1024            # conservative accumulator budget per core

def replication(cx, cy, sxx, syy, W, H, ts):
    ntx, nty = (W + ts - 1) // ts, (H + ts - 1) // ts
    rx, ry = 3 * np.sqrt(sxx), 3 * np.sqrt(syy)
    tx0 = np.clip(np.floor((cx - rx) / ts), 0, ntx - 1); tx1 = np.clip(np.floor((cx + rx) / ts), 0, ntx - 1)
    ty0 = np.clip(np.floor((cy - ry) / ts), 0, nty - 1); ty1 = np.clip(np.floor((cy + ry) / ts), 0, nty - 1)
    return ((tx1 - tx0 + 1) * (ty1 - ty0 + 1)).astype(np.int64)

def scene(kind, N, W, H, rng):
    if kind == "spread-small":          # corgi-like: spread, small footprints
        cx, cy = rng.uniform(0, W, N), rng.uniform(0, H, N); var = rng.uniform(1, 6, N)
    elif kind == "clustered":           # all near center, moderate footprint (the test_resident worst case)
        cx = np.clip(rng.normal(W/2, W*0.06, N), 0, W); cy = np.clip(rng.normal(H/2, H*0.06, N), 0, H); var = rng.uniform(2, 12, N)
    elif kind == "large-gaussians":     # big footprints -> 10-25x dup (bin_sort.py:100 regime)
        cx, cy = rng.uniform(0, W, N), rng.uniform(0, H, N); var = rng.uniform(20, 120, N)
    return cx, cy, var

def analyze(kind, N, W, H, ncores, rng):
    cx, cy, var = scene(kind, N, W, H, rng)
    rep = replication(cx, cy, var, var, W, H, BIN_TS)
    # hash-home: owner(g)=g%ncores; per-owner inbox = sum of replication of its gids
    owner = np.arange(N) % ncores
    per_owner_bytes = np.zeros(ncores, np.int64)
    np.add.at(per_owner_bytes, owner, rep * NCH * BYTES)
    worst = int(per_owner_bytes.max()); mean = float(per_owner_bytes.mean())
    cap_N = (L1_BUDGET / (per_owner_bytes.max() / max(1, (N // ncores)))) * ncores if per_owner_bytes.max() else 0
    print(f"  {kind:16s} N={N:6d} {W}x{H} ncores={ncores} | rep mean={rep.mean():5.2f} max={rep.max():3d} "
          f"p99={int(np.percentile(rep,99)):3d} | inbox/owner worst={worst/1024:7.1f}KB mean={mean/1024:6.1f}KB "
          f"-> {'FITS' if worst < L1_BUDGET else 'SPILL'} (budget {L1_BUDGET//1024}KB)", flush=True)
    return worst

rng = np.random.default_rng(0)
print(f"E4 inbox sizing under replication (7 grads x 4B/slot, ~{L1_BUDGET//1024}KB/core budget, ts={BIN_TS})")
ncores = 120
for kind in ("spread-small", "clustered", "large-gaussians"):
    for N in (65536, 262144, 1_000_000):
        analyze(kind, N, 256, 256, ncores, rng)
    print()
print("Interpretation: 'spread-small'/'clustered' are realistic; 'large-gaussians' is the pathological")
print("upper bound. Where worst>budget, that scene/N needs GDDR tiering (plan Stage 7b) or smaller owners-")
print("per-tile. The crossover N is the trigger for tiering, not a blocker for the measured regime.")
