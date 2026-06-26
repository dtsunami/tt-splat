#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage 2 end-to-end: compare fused_backward_grid FB_S2=0 (host reduce) vs FB_S2=1 (in-kernel fp32
reduce + compact readback) on the SAME scene. Reports FB_PROF buckets + correctness (grads must agree)."""
import sys, os
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path.home() / "tt-splat" / "server"))
sys.path.insert(0, str(Path.home() / "tt-splat" / "docs" / "pathclear"))
import ttnn
from bin_sort import bin_and_sort
import fused_backward as FB

SZ = int(os.environ.get("BN_SZ", "96"))
N = int(os.environ.get("BN_N", "1024"))
CLUSTER = os.environ.get("BN_CLUSTER", "0") == "1"   # pack into a corner -> inflate nbatch (worst case)
Wp = Hp = SZ
ntx = nty = SZ // 32
np.random.seed(0); torch.manual_seed(0)

if CLUSTER:
    cx = np.random.rand(N) * 40; cy = np.random.rand(N) * 40        # all in top-left ~1 tile
else:
    cx = np.random.rand(N) * Wp; cy = np.random.rand(N) * Hp
sx = 4 + np.random.rand(N) * 4; sy = 4 + np.random.rand(N) * 4
a = 1 / sx**2; c = 1 / sy**2; b = np.zeros(N)
op = 0.4 + np.random.rand(N) * 0.4; col = [0.3 + np.random.rand(N) * 0.5 for _ in range(3)]
depth = np.random.rand(N)
s_gid, _, ranges, nx, ny, _ = bin_and_sort(cx, cy, sx**2, sy**2, depth, Wp, Hp, ts=32)
tl = [s_gid[ranges[t, 0]:ranges[t, 1]].tolist() for t in range(nx * ny)]
maxc = max(len(l) for l in tl); nbatch = (maxc + FB.FUSED_K - 1) // FB.FUSED_K
gi = np.random.rand(Hp, Wp, 3); Tfin = np.ones((Hp, Wp))

dev = ttnn.open_device(device_id=0)
os.environ["FB_PROF"] = "1"
try:
    print(f"scene: SZ={SZ} N={N} tiles={nx}x{ny} maxc={maxc} nbatch={nbatch} cluster={CLUSTER}", flush=True)
    import time
    res = {}
    for s2, s3, name in (("0", "0", "baseline"), ("1", "0", "S2"), ("0", "1", "S3")):
        os.environ["FB_S2"] = s2; os.environ["FB_S3"] = s3
        FB.fused_backward_grid(dev, cx, cy, a, b, c, op, col, tl, ntx, nty, Wp, Hp, gi, Tfin)   # warm (JIT)
        t0 = time.perf_counter()
        g, cg = FB.fused_backward_grid(dev, cx, cy, a, b, c, op, col, tl, ntx, nty, Wp, Hp, gi, Tfin)
        wall = 1e3 * (time.perf_counter() - t0)
        res[name] = (g, cg, wall)
        print(f"  {name:9s} total_wall={wall:.1f} ms", flush=True)
    g0, c0, _ = res["baseline"]
    scale = max(abs(g0[k]).max() for k in g0) + 1e-9
    for name in ("S2", "S3"):
        g1, c1, _ = res[name]
        worst = max((abs(g1[k] - g0[k]).max() / scale) for k in g0)
        worst = max(worst, max((abs(c1[k] - c0[k]).max() / scale) for k in range(3)))
        print(f"  {name} vs baseline worst rel = {worst:.2e} -> {'OK' if worst < 2e-2 else 'FAIL'}  "
              f"SPEEDUP={res['baseline'][2] / res[name][2]:.2f}x", flush=True)
finally:
    ttnn.close_device(dev)
