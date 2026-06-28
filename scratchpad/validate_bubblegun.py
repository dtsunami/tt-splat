#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""On-silicon validation of the bubble gun: screen click -> un-project -> spawn -> resident realloc.
Round-trip check: a spawned splat re-projects back near the click pixel."""
import os
os.environ.setdefault("TT_METAL_HOME", "/home/starboy/tt-metal")
import sys, math
from pathlib import Path
import numpy as np, torch
ROOT = Path.home() / "tt-splat"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "server")); sys.path.insert(0, str(ROOT / "docs" / "pathclear"))
import ttnn
from device_resident import DeviceResidentTrainer
from render_device import _device
from train_tt import _project_uvz, _unproject_spawn

torch.manual_seed(0)
H = W = 64
N, deg, K = 150, 1, 4
mean = torch.empty(N, 3, dtype=torch.float64)
mean[:, 0] = torch.rand(N).double() * 2 - 1
mean[:, 1] = torch.rand(N).double() * 2 - 1
mean[:, 2] = 2 + torch.rand(N).double() * 2
P0 = {"mean": mean, "scale": torch.full((N, 3), math.log(0.3), dtype=torch.float64),
      "quat": torch.tensor([[1., 0, 0, 0]]).repeat(N, 1).double(),
      "op": torch.full((N,), 2.0, dtype=torch.float64),
      "sh": torch.randn(N, K, 3, dtype=torch.float64) * 0.3, "deg": deg}
cam = (torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 80., 80., 32., 32., "t")
gt = torch.rand(H, W, 3, dtype=torch.float64)

dev = _device()
tr = DeviceResidentTrainer(dev, P0, deg=deg)
u, v, zc = _project_uvz(tr.params_host(), cam)
pts = [[32, 32], [20, 40], [45, 25]]                 # three "trigger pulls"
mns, cls = _unproject_spawn(u, v, zc, cam, pts, gt, n_per=2, brush=1.5)
print("RESULT spawned:", mns.shape, "colors:", cls.shape)

n0 = tr.N
nn = tr.spawn(mns, cls)
print("RESULT N:", n0, "->", nn, "(expect +%d)" % mns.shape[0])

ph = tr.params_host()
u2, v2, zc2 = _project_uvz(ph, cam)
M = mns.shape[0]
us, vs, zs = u2[-M:], v2[-M:], zc2[-M:]              # spawned splats are appended last
in_front = bool((zs > 1e-4).all())
# first spawn (2 per point) should re-project near pts[0]=[32,32] (within ~3px of the brush jitter)
rt_err = float(np.hypot(us[0] - 32, vs[0] - 32))
print("RESULT roundtrip[0]:", round(float(us[0]), 1), round(float(vs[0]), 1), "~click [32,32], err", round(rt_err, 2), "px")
print("RESULT colors_match_gt[0]:",
      np.allclose(cls[0], gt[int(np.clip(vs[0], 0, H - 1)), int(np.clip(us[0], 0, W - 1))].numpy(), atol=0.25))
l, _ = tr.step(cam, gt)
print("RESULT step_after_spawn loss:", round(float(l), 5), "N:", tr.N)
print("RESULT", "BUBBLEGUN_OK" if nn == n0 + M and in_front and rt_err < 4.0 else "FAIL")

# --- eraser: cull a brushed region ---
from train_tt import _select_region
u3, v3, zc3 = _project_uvz(tr.params_host(), cam)
sel = _select_region(u3, v3, zc3, [[32, 32]], brush=12.0)
ncull = int(sel.sum()); nbefore = tr.N
nn2 = tr.cull(~torch.as_tensor(sel))
print("RESULT cull:", ncull, "selected; N", nbefore, "->", nn2)
l2, _ = tr.step(cam, gt)
print("RESULT step_after_cull loss:", round(float(l2), 5), "N:", tr.N)
print("RESULT", "CULL_OK" if (ncull > 0 and nn2 == nbefore - ncull) or (ncull == 0 and nn2 == nbefore) else "CULL_FAIL")
