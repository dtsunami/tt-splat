#!/usr/bin/env python3
"""Gate (a) for recipe gap #1 — finite-difference grad-check of the RESIDENT analytic pose Jacobian.

The resident path computes dL/dδ = Σ_g J_pose,gᵀ·(Jᵀ[dL/du,dL/dv]) from device 2D grads.  Here we
verify pose_opt.resident_grad's formula against a numerical perturbation of the camera, on a loss that
depends on the camera ONLY through (u,v) (conic/colour/opacity/order frozen) — which is exactly the term
the analytic models.  CPU only, no device."""
import sys, math
from pathlib import Path
import numpy as np, torch

R = Path.home() / "tt-splat"
sys.path.insert(0, str(R / "server")); sys.path.insert(0, str(R / "docs" / "pathclear"))
import pose_opt
from train_real import project_general, sh_eval

torch.manual_seed(0)
dt = torch.float64
N, H, W = 30, 48, 48
fx = fy = 60.0; cx = cy = 24.0
# a camera looking down +z; gaussians scattered in front
R0 = torch.eye(3, dtype=dt)
t0 = torch.tensor([0.1, -0.05, 0.0], dtype=dt)
mean = torch.empty(N, 3, dtype=dt)
mean[:, 0] = (torch.rand(N, dtype=dt) * 2 - 1) * 0.8
mean[:, 1] = (torch.rand(N, dtype=dt) * 2 - 1) * 0.8
mean[:, 2] = 2.0 + torch.rand(N, dtype=dt) * 1.5
P = {"mean": mean,
     "scale": torch.full((N, 3), math.log(0.15), dtype=dt),
     "quat": torch.tensor([[1., 0, 0, 0]], dtype=dt).repeat(N, 1),
     "op": torch.logit(torch.full((N,), 0.5, dtype=dt)),
     "sh": torch.randn(N, 4, 3, dtype=dt) * 0.3, "deg": 1}
gt = torch.rand(H, W, 3, dtype=dt)
ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
PX, PY = jj.double(), ii.double()


def project_uvz(R, t):
    u, v, zc, _ = project_general(P, R, t, fx, fy, cx, cy)
    return u, v, zc


# frozen conic/colour/opacity/order at delta=0 (the analytic holds u,v as the only camera dependence)
with torch.no_grad():
    u0, v0, zc0, (ca, cb, cc) = project_general(P, R0, t0, fx, fy, cx, cy)
    cam_center = -R0.T @ t0
    dirs = P["mean"] - cam_center; dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-9)
    col = sh_eval(P["sh"], dirs, P["deg"])
    op = torch.sigmoid(P["op"])
    order = torch.argsort(zc0).tolist()


def frozen_render(u, v):
    """Composite with frozen conic/colour/op/order; only (u,v) vary."""
    C = torch.zeros(H, W, 3, dtype=dt); T = torch.ones(H, W, dtype=dt)
    for i in order:
        if zc0[i] <= 0:
            continue
        dx, dy = PX - u[i], PY - v[i]
        al = (op[i] * torch.exp(-0.5 * (ca[i] * dx * dx + 2 * cb[i] * dx * dy + cc[i] * dy * dy))).clamp(max=0.99)
        w = T * al
        C = C + w[..., None] * col[i]
        T = T * (1 - al)
    return C


def loss_from_uv(u, v):
    return ((frozen_render(u, v) - gt) ** 2).mean()


# --- analytic dL/dδ via resident_grad (from device-style 2D grads) ---
u_leaf = u0.clone().requires_grad_(); v_leaf = v0.clone().requires_grad_()
L = loss_from_uv(u_leaf, v_leaf); L.backward()
du, dv = u_leaf.grad.numpy(), v_leaf.grad.numpy()
screen = dict(u=u0.numpy(), v=v0.numpy(), zc=zc0.numpy(), du=du, dv=dv,
              valid=np.ones(N, bool))
cam = (R0, t0, fx, fy, cx, cy, "t")
po = pose_opt.PoseOptimizer(1, lr=1e-3, reg=0.0)
po.resident_grad(0, screen, cam)
analytic = po.delta.grad[0].numpy().copy()      # [dω(3), dt(3)]

# --- finite-difference dL/dδ: perturb camera by δ, recompute u(δ),v(δ) (mc varies fully), frozen render ---
def loss_at_delta(delta):
    d = torch.tensor(delta, dtype=dt)
    Rd = pose_opt.so3_exp(d[:3])
    Rp = Rd @ R0; tp = Rd @ t0 + d[3:]
    u, v, _ = project_uvz(Rp, tp)
    return float(loss_from_uv(u, v))

h = 1e-6
fd = np.zeros(6)
for k in range(6):
    dp = np.zeros(6); dm = np.zeros(6); dp[k] = h; dm[k] = -h
    fd[k] = (loss_at_delta(dp) - loss_at_delta(dm)) / (2 * h)

rel = np.abs(analytic - fd) / (np.abs(fd) + 1e-12)
print("component:   [   dωx       dωy       dωz       dtx       dty       dtz ]")
print("analytic :", np.array2string(analytic, precision=6, floatmode="fixed"))
print("finite-d :", np.array2string(fd, precision=6, floatmode="fixed"))
print("rel err  :", np.array2string(rel, precision=2e0 and 3, floatmode="fixed"))
maxrel = float(rel.max())
print(f"RESULT max_rel_err={maxrel:.2e}")
print("RESULT", "POSE_GRADCHECK_OK" if maxrel < 1e-4 else "POSE_GRADCHECK_FAIL")
