#!/usr/bin/env python3
"""Convergence: optimize P with DEVICE gradients (render_train) toward a target. Loss must drop."""
import sys, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path.home() / "tt-splat" / "server"))
sys.path.insert(0, str(Path.home() / "tt-splat" / "docs" / "pathclear"))
from train_real import render
import device_raster as DR

torch.manual_seed(1)
H = W = 64
N, deg, K = 8, 1, 4
def make_P():
    m = torch.empty(N, 3, dtype=torch.float64)
    m[:, 0] = torch.rand(N).double() * 1.6 - 0.8; m[:, 1] = torch.rand(N).double() * 1.6 - 0.8
    m[:, 2] = 2.5 + torch.rand(N).double()
    return {"mean": m, "scale": torch.full((N, 3), math.log(0.22), dtype=torch.float64),
            "quat": torch.tensor([[1., 0, 0, 0]]).repeat(N, 1).double(),
            "op": torch.zeros(N, dtype=torch.float64),
            "sh": torch.randn(N, K, 3, dtype=torch.float64) * 0.4, "deg": deg}
cam = (torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 80., 80., 32., 32., "t")
ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"); PX, PY = jj.double(), ii.double()

gtP = make_P()
target = render(gtP, cam, H, W, PX, PY).clamp(0, 1).float().detach()        # host golden target

P = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in gtP.items()}    # perturbed init
P["mean"] = P["mean"] + torch.randn(N, 3).double() * 0.15
P["sh"] = P["sh"] + torch.randn(N, K, 3).double() * 0.2
P["op"] = P["op"] + torch.randn(N).double() * 0.3
OPT = ["mean", "scale", "quat", "op", "sh"]
for k in OPT: P[k].requires_grad_(True)
lr = {"mean": .01, "scale": .01, "quat": .01, "op": .02, "sh": .01}
opt = torch.optim.Adam([{"params": [P[k]], "lr": lr[k]} for k in OPT])
psnr = lambda a, b: 10 * math.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12))

print(f"device-gradient convergence  H={H} N={N}")
for step in range(1, 31):
    opt.zero_grad()
    img = DR.render_train(P, cam, H, W)                 # device fwd+bwd
    loss = ((img - target) ** 2).mean()
    loss.backward(); opt.step()
    if step == 1 or step % 10 == 0:
        print(f"  step {step:3d}  loss={float(loss):.6f}  PSNR={psnr(img.detach(), target):.1f} dB")
final = psnr(DR.render_train(P, cam, H, W).detach(), target)
print(f"DEVICE_CONVERGE64  final PSNR={final:.1f} dB -> {'OK' if final > 28 else 'FAIL'}")
