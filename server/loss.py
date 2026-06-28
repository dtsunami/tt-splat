#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
The 3DGS training loss: L1 + D-SSIM (closing the gsplat gap — tt-splat trained on pure MSE before this).

  loss = (1 - lambda_dssim) * L1  +  lambda_dssim * (1 - SSIM)

SSIM is the standard Gaussian-window (11x11, sigma 1.5) structural-similarity. It's a host elementwise op
(cheap next to the device render/backward), and it's pure torch — so `dL/dimage` comes from autograd,
correct by construction (the device backward consumes that gradient unchanged). Both train paths call this:
the host path lets `loss.backward()` flow through it; the device-resident path wraps the rendered image in a
grad tensor, backprops once, and feeds `img.grad` to the on-device raster backward.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def _gauss_1d(ws, sigma, device, dtype):
    c = torch.arange(ws, dtype=dtype, device=device) - (ws - 1) / 2.0
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    return g                                               # [ws], sums to 1


def ssim_map(a, b, ws=11, sigma=1.5):
    """Per-pixel-per-channel SSIM of two [H,W,3] images in [0,1] (torch). Returns [H,W,3].

    The Gaussian window is SEPARABLE, so each blur is two 1D depthwise convs (11+11 taps) instead of one
    11x11 conv (121 taps) — same result (~1e-6), ~5x fewer MACs. Big deal on the CPU host path where this
    runs every step; combine with the float32 dloss_dimage below and the loss stage drops ~13x."""
    dt, dev = a.dtype, a.device
    g = _gauss_1d(ws, sigma, dev, dt)
    wcol = g.reshape(1, 1, ws, 1).repeat(3, 1, 1, 1)       # depthwise [3,1,ws,1]
    wrow = g.reshape(1, 1, 1, ws).repeat(3, 1, 1, 1)       # depthwise [3,1,1,ws]
    pad = ws // 2

    def blur(x):                                           # separable Gaussian blur, groups=3 (per channel)
        return F.conv2d(F.conv2d(x, wcol, padding=(pad, 0), groups=3), wrow, padding=(0, pad), groups=3)

    A = a.permute(2, 0, 1).unsqueeze(0)                    # [1,3,H,W]
    B = b.permute(2, 0, 1).unsqueeze(0)
    muA = blur(A); muB = blur(B)
    muA2, muB2, muAB = muA * muA, muB * muB, muA * muB
    sA = blur(A * A) - muA2
    sB = blur(B * B) - muB2
    sAB = blur(A * B) - muAB
    smap = ((2 * muAB + _C1) * (2 * sAB + _C2)) / ((muA2 + muB2 + _C1) * (sA + sB + _C2))
    return smap[0].permute(1, 2, 0)                        # [H,W,3]


def image_loss(img, gt, mask=None, lambda_dssim=0.2):
    """img, gt: [H,W,3] torch in [0,1]. mask: [H,W] weights or None. Returns a differentiable scalar.
    lambda_dssim=0 -> pure L1; the 3DGS default is 0.2."""
    l1pp = (img - gt).abs()
    if mask is not None:
        m = mask.unsqueeze(-1)
        l1 = (l1pp * m).sum() / (m.sum() * 3 + 1e-8)
    else:
        l1 = l1pp.mean()
    if lambda_dssim is None or lambda_dssim <= 0:
        return l1
    smap = ssim_map(img.clamp(0, 1) if not img.requires_grad else img, gt)
    if mask is not None:
        m = mask.unsqueeze(-1)
        dssim = 1.0 - (smap * m).sum() / (m.sum() * 3 + 1e-8)
    else:
        dssim = 1.0 - smap.mean()
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim


def dloss_dimage(img_np, gt_np, mask_np=None, lambda_dssim=0.2):
    """For the device-resident path: given numpy render+gt (+optional mask), return (loss float,
    dL/dimage np[H,W,3]) by one autograd pass — feeds the on-device raster backward.

    Runs in FLOAT32: CPU float64 conv2d falls off the oneDNN fast path (this was a ~13x per-step host
    regression vs the old MSE — ~10s/step at 1600px). The gradient feeds an f32 device backward anyway,
    and f32 vs f64 here differs by ~1e-6 (grad-checked). Returns f64 so the caller's f64 buffers are
    unchanged."""
    it = torch.tensor(img_np, dtype=torch.float32, requires_grad=True)
    gt = torch.tensor(gt_np, dtype=torch.float32)
    mk = torch.tensor(mask_np, dtype=torch.float32) if mask_np is not None else None
    L = image_loss(it, gt, mk, lambda_dssim)
    L.backward()
    return float(L.detach()), it.grad.detach().double().numpy()
