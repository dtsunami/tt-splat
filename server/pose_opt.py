#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Camera pose optimization (gsplat `pose_opt`) — recipe gap #1.

COLMAP poses carry calibration/registration error that shows up as ghosting / double-images.
gsplat makes the extrinsics trainable to absorb it; tt-splat fixed them from COLMAP.  This module
adds a trainable per-camera 6-DoF correction `δ = (ω, t)` (so(3) tangent + translation, init 0).

The correction is applied IN CAMERA SPACE:  mc' = Exp(ω)·mc0 + t  where mc0 = R0·X + t0 is the
COLMAP camera-space point.  Realised by building the corrected extrinsics

    R' = Exp(ω)·R0 ,   t' = Exp(ω)·t0 + t

and handing them to the projector exactly like a normal camera (no device-kernel change — the cameras
are host params).  Two paths consume it:

  HOST path (train_real.render autograd): `corrected_cam(..., differentiable=True)` returns R',t' as
    torch leaves so `loss.backward()` flows into δ.  Exact (captures conic/colour/depth coupling too).

  DEVICE-RESIDENT path: the device already reads back, per Gaussian, dL/du, dL/dv (grads2d cx/cy) and
    u,v,zc.  Reconstruct mc from (u,v,zc)+intrinsics, then
        dL/dmc = Jᵀ·[dL/du, dL/dv]   (J = perspective Jacobian),
        dL/dt  = Σ_g dL/dmc_g ,       dL/dω = Σ_g mc_g × dL/dmc_g .
    This is the MVP mean-projection term (dominant); the conic-vs-pose coupling is a second-order
    refinement (left out, like gsplat's small-correction regime).  Both paths step the SAME host Adam.

The interactive `pose_nudge(camera, ω, t)` reuses this state — it's the game's "grab the camera" handle.
"""
from __future__ import annotations
import numpy as np
import torch


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues SO(3) exponential. omega:(3,) torch -> R:(3,3) torch. Autograd-safe at ω=0
    (theta enters only through theta² = ω·ω, an even function, so the sqrt kink at 0 is masked by the
    clamp and the first-order term reduces to the generator I + [ω]×)."""
    dtype, device = omega.dtype, omega.device
    theta2 = (omega * omega).sum()
    theta = torch.sqrt(theta2.clamp_min(1e-12))
    wx, wy, wz = omega[0], omega[1], omega[2]
    z = torch.zeros((), dtype=dtype, device=device)
    K = torch.stack([torch.stack([z, -wz, wy]),
                     torch.stack([wz, z, -wx]),
                     torch.stack([-wy, wx, z])])
    A = torch.sin(theta) / theta                      # -> 1 as theta->0
    Bc = (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12)   # -> 1/2 as theta->0
    eye = torch.eye(3, dtype=dtype, device=device)
    return eye + A * K + Bc * (K @ K)


class PoseOptimizer:
    """Per-camera 6-DoF correction δ=(ω,t) with a host Adam shared by both train paths."""

    def __init__(self, n_cams: int, lr: float = 1e-3, reg: float = 1e-4,
                 dtype: torch.dtype = torch.float64):
        self.n = int(n_cams)
        self.reg = float(reg)
        self.dtype = dtype
        self.delta = torch.zeros(self.n, 6, dtype=dtype, requires_grad=True)  # [:, :3]=ω, [:, 3:]=t
        self.opt = torch.optim.Adam([self.delta], lr=float(lr))

    # ---- camera construction ----
    def corrected_cam(self, i: int, cam, differentiable: bool = False):
        """Return cam with corrected extrinsics R'=Exp(ω)R0, t'=Exp(ω)t0+t. When differentiable, the
        returned R',t' carry autograd back to δ (host path); else they're detached (resident path)."""
        R0 = cam[0] if torch.is_tensor(cam[0]) else torch.as_tensor(np.asarray(cam[0]))
        t0 = cam[1] if torch.is_tensor(cam[1]) else torch.as_tensor(np.asarray(cam[1]))
        R0 = R0.to(self.dtype); t0 = t0.to(self.dtype).reshape(3)
        d = self.delta[i] if differentiable else self.delta[i].detach()
        Rd = so3_exp(d[:3])
        Rp = Rd @ R0
        tp = Rd @ t0 + d[3:]
        return (Rp, tp) + tuple(cam[2:])

    # ---- gradient sources ----
    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)

    def resident_grad(self, i: int, screen: dict, cam):
        """Set δ[i].grad from the device's per-Gaussian screen-space grads (analytic mean-projection
        term). screen: dict(u,v,zc,du,dv,valid) numpy; cam: the corrected cam (for intrinsics)."""
        fx, fy, cx, cy = float(cam[2]), float(cam[3]), float(cam[4]), float(cam[5])
        u, v, zc = screen["u"], screen["v"], screen["zc"]
        du, dv = screen["du"], screen["dv"]
        valid = np.asarray(screen["valid"], bool) & (zc > 1e-6)
        grad = torch.zeros_like(self.delta)
        if valid.any():
            u, v, zc, du, dv = (a[valid].astype(np.float64) for a in (u, v, zc, du, dv))
            xc = (u - cx) * zc / fx
            yc = (v - cy) * zc / fy
            dxc = du * (fx / zc)                                   # J^T·[du,dv] -> dL/dmc
            dyc = dv * (fy / zc)
            dzc = du * (-fx * xc / (zc * zc)) + dv * (-fy * yc / (zc * zc))
            dmc = np.stack([dxc, dyc, dzc], axis=-1)              # [M,3]
            mc = np.stack([xc, yc, zc], axis=-1)                  # [M,3]
            dt = dmc.sum(0)                                        # dL/dt
            domega = np.cross(mc, dmc).sum(0)                     # Σ mc × dL/dmc = dL/dω
            grad[i] = torch.as_tensor(np.concatenate([domega, dt]), dtype=self.dtype)
        self.delta.grad = grad

    # ---- step (applies the L2 reg prior on δ, then Adam) ----
    def step(self):
        if self.delta.grad is None:
            return
        if self.reg:
            with torch.no_grad():
                self.delta.grad = self.delta.grad + self.reg * self.delta
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    # ---- interactive "grab the camera" nudge ----
    def nudge(self, i: int, omega=(0.0, 0.0, 0.0), trans=(0.0, 0.0, 0.0)):
        with torch.no_grad():
            self.delta[i, :3] += torch.as_tensor(np.asarray(omega, np.float64), dtype=self.dtype)
            self.delta[i, 3:] += torch.as_tensor(np.asarray(trans, np.float64), dtype=self.dtype)

    def magnitude(self, i: int):
        """(|ω| in rad, |t| in world units) of camera i's current correction — for logging/gate checks."""
        d = self.delta[i].detach()
        return float(d[:3].norm()), float(d[3:].norm())
