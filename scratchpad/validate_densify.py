#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""On-silicon validation: device-resident adaptive densification (clone/split/prune + Adam realloc).
Accumulate the screen-space positional grad over a few steps, densify, and confirm N grows and the
resident loop keeps stepping at the new count."""
import os
os.environ.setdefault("TT_METAL_HOME", "/home/starboy/tt-metal")
import sys, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path.home() / "tt-splat" / "server"))
sys.path.insert(0, str(Path.home() / "tt-splat" / "docs" / "pathclear"))
import ttnn
from device_resident import DeviceResidentTrainer
from render_device import _device

torch.manual_seed(0)
H = W = 64
N, deg, K = 200, 1, 4
mean = torch.empty(N, 3, dtype=torch.float64)
mean[:, 0] = torch.rand(N).double() * 2 - 1
mean[:, 1] = torch.rand(N).double() * 2 - 1
mean[:, 2] = 2 + torch.rand(N).double() * 2
P0 = {"mean": mean, "scale": torch.full((N, 3), math.log(0.3), dtype=torch.float64),
      "quat": torch.tensor([[1., 0, 0, 0]]).repeat(N, 1).double(),
      "op": torch.full((N,), 2.0, dtype=torch.float64),       # sigmoid(2)=0.88 -> keep (test growth, not prune)
      "sh": torch.randn(N, K, 3, dtype=torch.float64) * 0.3, "deg": deg}
cam = (torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 80., 80., 32., 32., "t")
gt = torch.rand(H, W, 3, dtype=torch.float64)

dev = _device()
tr = DeviceResidentTrainer(dev, P0, deg=deg)
for _ in range(8):                                            # build up the densify signal
    tr.step(cam, gt)
print("RESULT gpos_accum_steps:", tr._gacc, "nonzero_grad_gaussians:", int((tr._gpos > 0).sum()))

n0 = tr.N
st = tr.densify(grad_threshold=0.0, n_max=5000)
print("RESULT densify:", st, "trainer.N:", tr.N)
ph = tr.params_host()
print("RESULT shapes_after:", {k: tuple(ph[k].shape) for k in ('mean', 'scale', 'quat', 'op', 'sh')})

ok = (st['n_after'] == tr.N and ph['mean'].shape[0] == tr.N and tuple(ph['sh'].shape[1:]) == (K, 3)
      and tr.N == n0 - st['prune'] - st['split'] + st['clone'] + 2 * st['split'])
for _ in range(3):                                            # must keep stepping at the new N
    l, _img = tr.step(cam, gt)
print("RESULT step_after_densify loss:", round(float(l), 5), "N:", tr.N)
print("RESULT", "DENSIFY_RESIDENT_OK" if ok and tr.N > 0 else "FAIL")
