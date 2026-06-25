#!/usr/bin/env python3
"""Gradient check: DeviceRaster bridge (on-device fwd+bwd) vs host train_real.render torch-autograd."""
import sys, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path.home() / "tt-splat" / "server"))
sys.path.insert(0, str(Path.home() / "tt-splat" / "docs" / "pathclear"))
from train_real import render, sh_dim
import device_raster as DR

torch.manual_seed(0)
H = W = 64
N, deg, K = 8, 1, 4
mean = torch.empty(N, 3, dtype=torch.float64)
mean[:, 0] = torch.rand(N).double() * 2 - 1
mean[:, 1] = torch.rand(N).double() * 2 - 1
mean[:, 2] = 2 + torch.rand(N).double() * 2
P = {"mean": mean, "scale": torch.full((N, 3), math.log(0.2), dtype=torch.float64),
     "quat": torch.tensor([[1., 0, 0, 0]]).repeat(N, 1).double(),
     "op": torch.zeros(N, dtype=torch.float64),
     "sh": torch.randn(N, K, 3, dtype=torch.float64) * 0.3, "deg": deg}
OPT = ["mean", "scale", "quat", "op", "sh"]
cam = (torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 80., 80., 32., 32., "t")
ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
PX, PY = jj.double(), ii.double()
gt = torch.rand(H, W, 3, dtype=torch.float64)


def grads_from(img, gt_):
    for k in OPT:
        if P[k].grad is not None: P[k].grad = None
    loss = ((img - gt_) ** 2).mean()
    loss.backward()
    return {k: P[k].grad.detach().clone().double() for k in OPT}, float(loss)


for k in OPT:
    P[k].requires_grad_(True)

gh, lh = grads_from(render(P, cam, H, W, PX, PY), gt)                 # host (float64)
gd, ld = grads_from(DR.render_train(P, cam, H, W), gt.float())        # device (float32)

print(f"host loss {lh:.6f}  device loss {ld:.6f}")
worst = 0.0
for k in OPT:
    num = (gd[k] - gh[k]).norm().item()
    den = gh[k].norm().item() + 1e-12
    rel = num / den
    worst = max(worst, rel)
    print(f"  grad[{k:5}]  ||host||={den:.3e}  rel_err={rel:.3e}")
print(f"DEVICE_GRAD64  worst_rel_err={worst:.3e}  -> {'OK' if worst < 5e-2 else 'FAIL'}")
