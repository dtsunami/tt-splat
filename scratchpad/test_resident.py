#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage E gate: the device-resident loop.  (1) one-step 3D grads match the host-autograd render_train
path; (2) it CONVERGES (loss drops, PSNR climbs) with the inner loop fully on-device; (3) per-stage perf."""
import sys, math
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path.home() / "tt-splat" / "server"))
sys.path.insert(0, str(Path.home() / "tt-splat" / "docs" / "pathclear"))
import ttnn
from train_real import render
import device_raster as DR
from device_resident import DeviceResidentTrainer
from render_device import _device          # shared persistent device handle (render_train uses this too)

torch.manual_seed(0)
H = W = 64
N, deg, K = 24, 1, 4
mean = torch.empty(N, 3, dtype=torch.float64)
mean[:, 0] = torch.rand(N).double() * 2 - 1
mean[:, 1] = torch.rand(N).double() * 2 - 1
mean[:, 2] = 2 + torch.rand(N).double() * 2
P0 = {"mean": mean, "scale": torch.full((N, 3), math.log(0.2), dtype=torch.float64),
      "quat": torch.tensor([[1., 0, 0, 0]]).repeat(N, 1).double(),
      "op": torch.zeros(N, dtype=torch.float64),
      "sh": torch.randn(N, K, 3, dtype=torch.float64) * 0.3, "deg": deg}
OPT = ["mean", "scale", "quat", "op", "sh"]
cam = (torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 80., 80., 32., 32., "t")
gt = torch.rand(H, W, 3, dtype=torch.float64)

dev = _device()                          # single shared handle — avoid double-open deadlock
try:
    # ---- (1) grad-equivalence vs host-autograd render_train ----
    print("(1) running host-autograd render_train (JIT warm-up)...", flush=True)
    P = {k: (P0[k].clone().requires_grad_(True) if k in OPT else P0[k]) for k in P0}
    img = DR.render_train(P, cam, H, W)
    loss = ((img - gt.float()) ** 2).mean()
    loss.backward()
    print("    render_train + backward done", flush=True)
    ref = {k: P[k].grad.detach().clone().double() for k in OPT}

    tr = DeviceResidentTrainer(dev, P0, deg=deg)
    l0, _ = tr.step(cam, gt)
    print(f"=== Stage E: device-resident loop ===")
    print(f"(1) grad-equivalence vs host-autograd render_train  (loss host {float(loss):.6f} / resident {l0:.6f})")
    worst = 0.0
    for k in OPT:
        gd = tr.last_g3[k].double()
        rel = (gd - ref[k]).norm().item() / (ref[k].norm().item() + 1e-12)
        worst = max(worst, rel)
        print(f"    grad[{k:5}] rel={rel:.2e}")
    print(f"    worst_rel={worst:.2e}  -> {'OK' if worst < 5e-2 else 'FAIL'}")

    # ---- (2) convergence: fresh trainer, descend toward gt ----
    tr2 = DeviceResidentTrainer(dev, P0, deg=deg)
    psnr = lambda mse: 10 * math.log10(1.0 / max(mse, 1e-12))
    losses = []
    STEPS = int(__import__("os").environ.get("STEPS", "40"))
    for s in range(STEPS):
        l, im = tr2.step(cam, gt)
        losses.append(l)
        if s == 0 or (s + 1) % 10 == 0:
            print(f"    step {s+1:3d}  loss {l:.5f}", flush=True)
    print(f"(2) convergence over {STEPS} steps:  loss {losses[0]:.5f} -> {losses[-1]:.5f}   "
          f"PSNR {psnr(losses[0]):.2f} -> {psnr(losses[-1]):.2f} dB  "
          f"-> {'OK' if losses[-1] < losses[0] * 0.6 else 'FAIL'}")

    # ---- (3) perf ----
    print(f"(3) {tr2.report()}")
finally:
    ttnn.close_device(dev)
