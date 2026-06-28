#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""On-silicon validation of this session's resident-loop changes:
   - stage_timings now emits the 'loss' stage time
   - --profile (TT_PROFILE) adds live per-step device compute-util (util/dev_us/cores)
   - resident dashboard commands: prune / reset_opacities / clamp_scale / set_lr
Small N @ 64px (dispatch-bound, PSU-safe)."""
import os
os.environ["TT_PROFILE"] = "1"
os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"
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
N, deg, K = 64, 1, 4
mean = torch.empty(N, 3, dtype=torch.float64)
mean[:, 0] = torch.rand(N).double() * 2 - 1
mean[:, 1] = torch.rand(N).double() * 2 - 1
mean[:, 2] = 2 + torch.rand(N).double() * 2
P0 = {"mean": mean, "scale": torch.full((N, 3), math.log(0.2), dtype=torch.float64),
      "quat": torch.tensor([[1., 0, 0, 0]]).repeat(N, 1).double(),
      "op": torch.linspace(-1.0, 1.0, N, dtype=torch.float64),   # varied -> prune actually removes some
      "sh": torch.randn(N, K, 3, dtype=torch.float64) * 0.3, "deg": deg}
cam = (torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 80., 80., 32., 32., "t")
gt = torch.rand(H, W, 3, dtype=torch.float64)

dev = _device()
tr = DeviceResidentTrainer(dev, P0, deg=deg)
print("RESULT profiler_on:", tr.profiler is not None)
for _ in range(4):
    l, _ = tr.step(cam, gt)
log = tr.step_log[-1]
print("RESULT step_log_keys:", sorted(log.keys()))
print("RESULT has_loss_stage:", "loss" in log, "loss_ms:", round(log.get("loss", -1), 3))
print("RESULT util:", log.get("util"), "dev_us:", log.get("dev_us"), "cores:", log.get("cores"))

# --- resident commands ---
n0 = tr.N
nk = tr.prune(0.5)                          # keep sigmoid(op) > 0.5  -> ~half
print("RESULT prune:", n0, "->", nk, "trainer.N:", tr.N)
l, _ = tr.step(cam, gt)                      # must still step (shapes consistent after re-slice)
print("RESULT step_after_prune_ok loss:", round(l, 5))

tr.reset_opacities()
op = ttnn.to_torch(tr.adam.p["op"]).reshape(-1)
print("RESULT reset_op mean:", round(float(op.mean()), 3), "expect ~", round(math.log(0.01 / 0.99), 3))

tr.clamp_scale(math.log(0.15))
sc = ttnn.to_torch(tr.adam.p["scale"])
print("RESULT clamp_scale max:", round(float(sc.max()), 4), "expect <=", round(math.log(0.15), 4))

lr_before = tr.adam.lr["mean"]
tr.set_lr(0.5)
print("RESULT set_lr mean:", lr_before, "->", tr.adam.lr["mean"])

l, _ = tr.step(cam, gt)
print("RESULT final_step_ok loss:", round(l, 5))
print("RESULT ALL_DONE")
