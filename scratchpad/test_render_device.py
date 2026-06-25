#!/usr/bin/env python3
"""Unit test: render_device (Blackhole) vs train_real.render (host) on a synthetic 3D scene."""
import sys, math
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path.home() / "tt-splat" / "server"))
sys.path.insert(0, str(Path.home() / "tt-splat" / "docs" / "pathclear"))

from train_real import init_from_points, render
import render_device as RD

torch.manual_seed(0)
H = W = 96
N = 300
xyz = torch.empty(N, 3, dtype=torch.float64)
xyz[:, 0] = (torch.rand(N) * 2 - 1).double()          # x in [-1,1]
xyz[:, 1] = (torch.rand(N) * 2 - 1).double()          # y in [-1,1]
xyz[:, 2] = (2 + torch.rand(N) * 3).double()          # z in [2,5], in front
rgb = torch.rand(N, 3, dtype=torch.float64)
P = init_from_points(xyz, rgb, sh_degree=0)           # deg 0 -> view-independent color

Rv = torch.eye(3, dtype=torch.float64)
tv = torch.zeros(3, dtype=torch.float64)
fx = fy = 150.0
cam = (Rv, tv, fx, fy, W / 2, H / 2, "test")

ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
PX, PY = jj.double(), ii.double()

host = render(P, cam, H, W, PX, PY).clamp(0, 1).float().numpy()   # (H,W,3)
dev = RD.render_device(P, cam, H, W)                              # (H,W,3)

mse = float(((host - dev) ** 2).mean())
psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
print(f"host range [{host.min():.3f},{host.max():.3f}] mean {host.mean():.3f}")
print(f"dev  range [{dev.min():.3f},{dev.max():.3f}] mean {dev.mean():.3f}")
print(f"RENDER_DEVICE_TEST  MSE={mse:.3e}  PSNR={psnr:.1f} dB  -> {'OK' if psnr > 28 else 'FAIL'}")
