#!/usr/bin/env python3
"""Gate (b) for recipe gap #1 — inject a known pose error into one camera, confirm pose-opt drives the
effective pose back toward truth and the view's PSNR climbs above the fixed-pose baseline.

Isolated test: Gaussians frozen at ground truth, only the camera's 6-DoF correction is trainable
(the host autograd path, end to end). CPU only, no device."""
import sys, math
from pathlib import Path
import numpy as np, torch

R = Path.home() / "tt-splat"
sys.path.insert(0, str(R / "server")); sys.path.insert(0, str(R / "docs" / "pathclear"))
import pose_opt
from train_real import render, sh_dim, C0
from train3d import scene, cameras, F, PP, H as TH, W as TW

torch.manual_seed(0)
dt = torch.float64
ii, jj = torch.meshgrid(torch.arange(TH), torch.arange(TW), indexing="ij")
PX, PY = jj.double(), ii.double()

# ground-truth scene + colours (deg-1 SH)
GT = scene(1, 24)
g = torch.Generator().manual_seed(3)
rgb = (0.2 + torch.rand(24, 3, generator=g) * 0.6)
GT["sh"] = torch.zeros(24, sh_dim(1), 3, dtype=dt)
GT["sh"][:, 0] = (rgb - 0.5) / C0
GT["deg"] = 1
cams = [(Rm, tm, float(F), float(F), float(PP[0]), float(PP[1]), f"cam{i}")
        for i, (Rm, tm) in enumerate(cameras(6, seed=0))]

# render GT images
with torch.no_grad():
    targets = [render(GT, c, TH, TW, PX, PY).clamp(0, 1) for c in cams]

psnr = lambda a, b: 10 * math.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12))


def geodesic_deg(Ra, Rb):
    c = (torch.trace(Ra @ Rb.T) - 1) / 2
    return math.degrees(math.acos(float(c.clamp(-1, 1))))


# inject a known pose error into camera 0 (3.5 deg rotation about a tilted axis + a translation offset)
ci = 0
R_true, t_true = cams[ci][0].to(dt), cams[ci][1].to(dt)
err_axis = torch.tensor([0.4, -0.8, 0.45], dtype=dt); err_axis /= err_axis.norm()
err_ang = math.radians(3.5)
R_err = pose_opt.so3_exp(err_axis * err_ang) @ R_true
t_err = t_true + torch.tensor([0.04, -0.03, 0.05], dtype=dt)
cam_bad = (R_err, t_err) + tuple(cams[ci][2:])

# fixed-pose baseline (the ghosting view)
with torch.no_grad():
    base_psnr = psnr(render(GT, cam_bad, TH, TW, PX, PY).clamp(0, 1), targets[ci])
rot0 = geodesic_deg(R_err, R_true)
trn0 = float((t_err - t_true).norm())

# pose-opt: Gaussians FROZEN at GT, only camera 0's correction trains
P = {k: GT[k].detach().clone() for k in ("mean", "scale", "quat", "op", "sh")}; P["deg"] = 1
pose = pose_opt.PoseOptimizer(1, lr=5e-3, reg=1e-5)
for step in range(400):
    cam_use = pose.corrected_cam(0, cam_bad, differentiable=True)
    img = render(P, cam_use, TH, TW, PX, PY)
    loss = ((img - targets[ci]) ** 2).mean()
    pose.zero_grad(); loss.backward(); pose.step()

with torch.no_grad():
    cam_fit = pose.corrected_cam(0, cam_bad)
    R_eff, t_eff = cam_fit[0], cam_fit[1]
    fit_psnr = psnr(render(P, cam_fit, TH, TW, PX, PY).clamp(0, 1), targets[ci])
rot1 = geodesic_deg(R_eff, R_true)
trn1 = float((t_eff - t_true).norm())

print(f"rotation err (deg):  {rot0:.3f} -> {rot1:.3f}")
print(f"translation err   :  {trn0:.4f} -> {trn1:.4f}")
print(f"view PSNR (dB)    :  baseline(fixed-bad)={base_psnr:.2f} -> pose-opt={fit_psnr:.2f}")
ok = (rot1 < rot0 * 0.25) and (trn1 < trn0 * 0.25) and (fit_psnr > base_psnr + 3.0)
print("RESULT", "POSE_E2E_OK" if ok else "POSE_E2E_FAIL",
      f"(rot {rot1/rot0:.2%} trn {trn1/trn0:.2%} dPSNR +{fit_psnr-base_psnr:.1f})")
