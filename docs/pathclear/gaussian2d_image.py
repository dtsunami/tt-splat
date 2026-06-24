#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
tt-splat pathclear milestone 1: 2D anisotropic Gaussian -> image fit, on Blackhole.

Fits a single rotated 2D Gaussian to a 32x32 target image, recovering amplitude,
center, and the 2D *conic* (inverse-covariance) — the exact quantity the real 3DGS
rasterizer evaluates per pixel:

    I(p) = A * exp( -0.5 * [ a*dx^2 + 2b*dx*dy + c*dy^2 ] ),   d = p - center

Params (6): A, cx, cy, a, b, c.  Conic kept positive-definite by a cheap host projection.

ON SILICON (fp32): per-pixel forward (sub/mul/square/exp) + Adam from primitives.
HOST glue: reduce 6 gradient scalars + assemble grad tile + PD projection.

Still NO sort / binning / scatter — this grows the proven "green spine" toward the rasterizer.
"""
from __future__ import annotations
import math
import torch
import ttnn

W = 32
N = W * W
TILE = (W, W)
STEPS = 800
LR, B1, B2, EPS = 0.08, 0.9, 0.999, 1e-8

# ---- ground truth: rotated anisotropic Gaussian ----
A_T, CX_T, CY_T = 1.0, 16.0, 13.0
SX_T, SY_T, TH_T = 4.0, 1.6, 0.6           # std in px (x,y) + rotation (rad)


def conic_from(sx, sy, th):
    ct, st = math.cos(th), math.sin(th)
    R = torch.tensor([[ct, -st], [st, ct]])
    S = torch.diag(torch.tensor([sx * sx, sy * sy]))
    cov = R @ S @ R.T
    M = torch.inverse(cov)                  # conic = Sigma^-1
    return float(M[0, 0]), float(M[0, 1]), float(M[1, 1])


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        ii, jj = torch.meshgrid(torch.arange(W), torch.arange(W), indexing="ij")
        PX = jj.float()      # column = x
        PY = ii.float()      # row    = y

        aT, bT, cT = conic_from(SX_T, SY_T, TH_T)
        dxT, dyT = PX - CX_T, PY - CY_T
        qT = aT * dxT**2 + 2 * bT * dxT * dyT + cT * dyT**2
        target = A_T * torch.exp(-0.5 * qT) + 0.01 * torch.randn(TILE)

        def dev_t(t):
            return ttnn.from_torch(t.contiguous(), dtype=ttnn.float32,
                                   layout=ttnn.TILE_LAYOUT, device=dev)

        pxt, pyt, yt = dev_t(PX), dev_t(PY), dev_t(target)

        # params packed into tile lanes [0,0..5] = A, cx, cy, a, b, c
        names = ["A", "cx", "cy", "a", "b", "c"]
        p0 = [1.5, 16.0, 16.0, 1 / 9, 0.0, 1 / 9]      # wrong-ish, isotropic init
        ptile = torch.zeros(TILE);
        for k, val in enumerate(p0): ptile[0, k] = val
        param = dev_t(ptile)
        m = dev_t(torch.zeros(TILE))
        v = dev_t(torch.zeros(TILE))

        def get():
            pt = ttnn.to_torch(param)
            return [float(pt[0, k]) for k in range(6)]

        def render_loss(pr):
            A, cx, cy, a, b, c = pr
            dx, dy = PX - cx, PY - cy
            q = a * dx**2 + 2 * b * dx * dy + c * dy**2
            return A * torch.exp(-0.5 * q)

        print("truth     A=%.2f cx=%.1f cy=%.1f  conic(a,b,c)=(%.4f,%.4f,%.4f)" %
              (A_T, CX_T, CY_T, aT, bT, cT))

        for step in range(1, STEPS + 1):
            A, cx, cy, a, b, c = get()

            # ---- forward ON DEVICE ----
            dx = ttnn.sub(pxt, cx)
            dy = ttnn.sub(pyt, cy)
            dx2, dy2, dxdy = ttnn.square(dx), ttnn.square(dy), ttnn.mul(dx, dy)
            q = ttnn.add(ttnn.add(ttnn.mul(dx2, a), ttnn.mul(dxdy, 2 * b)), ttnn.mul(dy2, c))
            g = ttnn.exp(ttnn.mul(q, -0.5))
            I = ttnn.mul(g, A)
            r = ttnn.sub(I, yt)

            # ---- analytic gradients (device products, host reduction) ----
            rh = ttnn.to_torch(r); gh = ttnn.to_torch(g)
            dxh = ttnn.to_torch(dx); dyh = ttnn.to_torch(dy)
            dLdI = (2.0 / N) * rh
            grads = {
                "A":  (dLdI * gh).sum(),
                "cx": (dLdI * A * gh * (a * dxh + b * dyh)).sum(),
                "cy": (dLdI * A * gh * (b * dxh + c * dyh)).sum(),
                "a":  (dLdI * (-0.5 * A * gh * dxh * dxh)).sum(),
                "b":  (dLdI * (-A * gh * dxh * dyh)).sum(),
                "c":  (dLdI * (-0.5 * A * gh * dyh * dyh)).sum(),
            }
            gtile = torch.zeros(TILE)
            for k, nm in enumerate(names): gtile[0, k] = float(grads[nm])
            gt = dev_t(gtile)

            # ---- Adam ON DEVICE (fp32 primitives) ----
            m = ttnn.add(ttnn.mul(m, B1), ttnn.mul(gt, 1 - B1))
            v = ttnn.add(ttnn.mul(v, B2), ttnn.mul(ttnn.square(gt), 1 - B2))
            mhat = ttnn.mul(m, 1.0 / (1 - B1 ** step))
            vhat = ttnn.mul(v, 1.0 / (1 - B2 ** step))
            param = ttnn.sub(param, ttnn.mul(ttnn.div(mhat, ttnn.add(ttnn.sqrt(vhat), EPS)), LR))

            # ---- host PD projection on the conic (a,c>0, ac-b^2>0) ----
            A, cx, cy, a, b, c = get()
            a = max(a, 1e-3); c = max(c, 1e-3)
            lim = 0.99 * math.sqrt(a * c)
            b = max(-lim, min(lim, b))
            pt = ttnn.to_torch(param)
            pt[0, 3], pt[0, 4], pt[0, 5] = a, b, c
            param = dev_t(pt)

            if step == 1 or step % 100 == 0:
                loss = float((rh ** 2).mean())
                print("step %4d  loss=%.6f  A=%.3f cx=%.2f cy=%.2f (a,b,c)=(%.4f,%.4f,%.4f)" %
                      (step, loss, A, cx, cy, a, b, c))

        pr = get()
        final = render_loss(pr)
        mse = float(((final - target) ** 2).mean())
        psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
        print("recovered A=%.3f cx=%.2f cy=%.2f (a,b,c)=(%.4f,%.4f,%.4f)" %
              (pr[0], pr[1], pr[2], pr[3], pr[4], pr[5]))
        print("image MSE=%.6f  PSNR=%.2f dB" % (mse, psnr))
        ok = (abs(pr[1] - CX_T) < 0.6 and abs(pr[2] - CY_T) < 0.6 and psnr > 30.0)
        print("M1_OK" if ok else "M1_FAIL")

        # optional: dump target|recovered side-by-side
        try:
            from PIL import Image
            import numpy as np
            both = torch.cat([target.clamp(0, 1), final.clamp(0, 1)], dim=1).numpy()
            Image.fromarray((both * 255).astype(np.uint8)).resize((W * 8, W * 4), Image.NEAREST)\
                 .save("pathclear/m1_target_vs_recovered.png")
            print("wrote pathclear/m1_target_vs_recovered.png (target | recovered)")
        except Exception as e:
            print("(png skipped:", e, ")")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
