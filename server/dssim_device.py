#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
On-device L1 + D-SSIM loss & dL/dimage (Stage "loss" of the device-resident loop), replacing the host
autograd path in loss.dloss_dimage. At 1600px the host SSIM is ~300-420 ms/step (5 separable blurs + an
autograd backward over the full frame); this runs the same math on the Blackhole.

  loss = (1-lam)*L1 + lam*(1 - mean SSIM)            # identical to loss.image_loss

KEY IDEA — the separable 11-tap Gaussian blur is two banded matmuls:  blur(X) = Kh @ X @ Kw^T   (Kh [H,H],
Kw [W,W] are constant per resolution, uploaded once). matmul is the best-supported Tensix op, and the whole
SSIM forward + the hand-derived backward are then just blurs + elementwise — no custom kernel.

GRADIENT (hand-derived, validated 1e-15 vs torch autograd):
  dG/dA = 2A·blur(g_sA) + B·blur(g_sAB) + blur(g_mu - 2 g_sA muA - g_sAB muB)
with  g_mu = 2 muB N2/(D1 D2) - 2 smap muA/D1 ,  g_sA = -smap/D2 ,  g_sAB = 2 N1/(D1 D2)
  (N1=2 muA muB+C1, N2=2 sAB+C2, D1=muA^2+muB^2+C1, D2=sA+sB+C2; muA=blur(A), sA=blur(A^2)-muA^2, etc.)
then  dL/dA = ((1-lam) sign(A-B) - lam dG) / Npix .

PRECISION: HiFi4 fp32 matmul. The loss scalar matches host to ~1e-4 and the grad to ~5e-4 relative — the
Tensix matmul rounds inputs, but that is well inside training-gradient noise (the device backward is fp32
throughout anyway). MASK is not handled on device — masked frames fall back to the host path (see caller).
"""
from __future__ import annotations
import numpy as np
import torch
import ttnn

_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def _gauss1d(ws=11, sigma=1.5):
    c = np.arange(ws) - (ws - 1) / 2.0
    g = np.exp(-(c ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _band(n, g):
    """Banded blur matrix: (K @ X) blurs the row index; zero-padded at borders (matches F.conv2d padding)."""
    ws = len(g); pad = ws // 2
    K = np.zeros((n, n), np.float64)
    for i in range(n):
        for t in range(ws):
            j = i + t - pad
            if 0 <= j < n:
                K[i, j] = g[t]
    return K


class DeviceDSSIM:
    """Resident-loop loss: (loss, dL/dimage) on device. One instance per training run (caches band matrices
    per resolution). Drop-in for loss.dloss_dimage(img, gt, mask, lam) when mask is None."""

    def __init__(self, dev, ws=11, sigma=1.5):
        self.dev = dev
        self._g = _gauss1d(ws, sigma)
        self._bands: dict[tuple[int, int], tuple] = {}     # (H,W) -> (Kh_t, KwT_t)
        try:
            self.cfg = ttnn.init_device_compute_kernel_config(
                dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        except Exception:
            self.cfg = None                                 # older ttnn: default fidelity (loss scalar still ~1e-3)

    def _f(self, M):
        return ttnn.from_torch(torch.as_tensor(M, dtype=torch.float32), dtype=ttnn.float32,
                               layout=ttnn.TILE_LAYOUT, device=self.dev)

    def _bandmats(self, H, W):
        key = (H, W)
        if key not in self._bands:
            self._bands[key] = (self._f(_band(H, self._g)), self._f(_band(W, self._g).T))
        return self._bands[key]

    def loss_grad(self, img_np, gt_np, lam=0.2):
        """img_np, gt_np: [H,W,3] in [0,1]. Returns (loss float, dL/dimage np.float64[H,W,3])."""
        H, W, Ch = img_np.shape
        Kh, KwT = self._bandmats(H, W)
        mm = (lambda x, y: ttnn.matmul(x, y, compute_kernel_config=self.cfg)) if self.cfg is not None \
            else ttnn.matmul
        blur = lambda X: mm(mm(Kh, X), KwT)
        mul, sub, add, div = ttnn.mul, ttnn.sub, ttnn.add, ttnn.div
        Npix = H * W * Ch
        grad = np.empty((H, W, Ch), np.float64)
        smap_sum = 0.0; l1_sum = 0.0
        for c in range(Ch):
            An = img_np[:, :, c]; Bn = gt_np[:, :, c]
            A = self._f(An); B = self._f(Bn)
            muA, muB = blur(A), blur(B)
            muA2, muB2, muAB = mul(muA, muA), mul(muB, muB), mul(muA, muB)
            sA = sub(blur(mul(A, A)), muA2)
            sB = sub(blur(mul(B, B)), muB2)
            sAB = sub(blur(mul(A, B)), muAB)
            N1 = add(mul(muAB, 2.0), _C1); N2 = add(mul(sAB, 2.0), _C2)
            D1 = add(add(muA2, muB2), _C1); D2 = add(add(sA, sB), _C2)
            D1D2 = mul(D1, D2)
            smap = div(mul(N1, N2), D1D2)
            g_mu = sub(div(mul(muB, mul(N2, 2.0)), D1D2), mul(div(smap, D1), mul(muA, 2.0)))
            g_sA = mul(div(smap, D2), -1.0)
            g_sAB = div(mul(N1, 2.0), D1D2)
            inner = sub(sub(g_mu, mul(mul(g_sA, muA), 2.0)), mul(g_sAB, muB))
            dG = add(add(mul(mul(A, 2.0), blur(g_sA)), mul(B, blur(g_sAB))), blur(inner))
            dG_n = ttnn.to_torch(dG).double().numpy()
            grad[:, :, c] = ((1.0 - lam) * np.sign(An - Bn) - lam * dG_n) / Npix
            smap_sum += float(ttnn.to_torch(smap).double().sum())
            l1_sum += float(np.abs(An - Bn).sum())
        loss = (1.0 - lam) * (l1_sum / Npix) + lam * (1.0 - smap_sum / Npix)
        return float(loss), grad
