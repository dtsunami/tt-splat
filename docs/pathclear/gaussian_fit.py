#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
tt-splat pathclear milestone 0: PROVEN TRAINING FLOW on Blackhole via ttnn.

Reduced problem that de-risks the real 3DGS training machinery WITHOUT touching
the deferred walls (sort / scatter-add / rasterizer):

    Fit a 1D Gaussian   y = A * exp(-0.5 * ((x - mu) / sigma)^2)
    to noisy samples, recovering (A, mu, sigma) by gradient descent.

What runs ON SILICON (fp32):
  - forward Gaussian eval:  sub / mul / square / EXP       (SFPU pipeline)
  - the optimizer:          Adam built from ttnn primitives (mul/add/square/sqrt/div/sub)
What is host glue (cheap; moves on-device in a later milestone):
  - reducing elementwise products to the 3 gradient scalars + assembling the grad tile

NOTE: ttnn.operations.moreh.adam is BFLOAT16-only AND returns wrong results in this
build (verified: grad [0.5,-0.5,2.0] -> [-1.52,3.52,0.37], should be ~[0.95,1.05,0.95]),
so we roll Adam from proven fp32 primitives instead.

Proves: device SFPU math + on-device fp32 optimizer + convergence.
"""
from __future__ import annotations
import torch
import ttnn

NP = 1024                       # samples, laid in a full 32x32 tile (no padding)
TILE = (32, 32)
A0, MU0, SIG0 = 2.0, 1.0, 1.5   # ground truth
STEPS = 300
LR, B1, B2, EPS = 0.05, 0.9, 0.999, 1e-8


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        x = torch.linspace(-5.0, 5.0, NP, dtype=torch.float32).reshape(TILE)
        y = A0 * torch.exp(-0.5 * ((x - MU0) / SIG0) ** 2) + 0.01 * torch.randn(TILE)

        def dev_t(t):
            return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

        xt, yt = dev_t(x), dev_t(y)

        # parameters in a tile; lanes [0,0],[0,1],[0,2] = A, mu, sigma. rest stays 0.
        p = torch.zeros(TILE, dtype=torch.float32)
        p[0, 0], p[0, 1], p[0, 2] = 1.0, 0.0, 1.0       # deliberately-wrong init
        param = dev_t(p)
        m = dev_t(torch.zeros(TILE))                    # Adam 1st moment
        v = dev_t(torch.zeros(TILE))                    # Adam 2nd moment

        def get_params():
            pt = ttnn.to_torch(param)
            return float(pt[0, 0]), float(pt[0, 1]), float(pt[0, 2])

        print(f"truth     A={A0} mu={MU0} sigma={SIG0}")
        print("init      A={:.3f} mu={:.3f} sigma={:.3f}".format(*get_params()))

        for step in range(1, STEPS + 1):
            a, mu, s = get_params()
            inv_s = 1.0 / s

            # ---- forward ON DEVICE: g = exp(-0.5 z^2), ypred = A*g, r = ypred - y ----
            d = ttnn.sub(xt, mu)                        # x - mu
            z = ttnn.mul(d, inv_s)                      # (x-mu)/sigma
            g = ttnn.exp(ttnn.mul(ttnn.square(z), -0.5))
            ypred = ttnn.mul(g, a)
            r = ttnn.sub(ypred, yt)                     # residual

            # ---- analytic gradients (device elementwise; host reduces the sums) ----
            dLdyp = (2.0 / NP) * ttnn.to_torch(r)
            g_h, d_h = ttnn.to_torch(g), ttnn.to_torch(d)
            gA = float((dLdyp * g_h).sum())
            gM = float((dLdyp * a * g_h * d_h * inv_s ** 2).sum())
            gS = float((dLdyp * a * g_h * d_h * d_h * inv_s ** 3).sum())
            grad = torch.zeros(TILE, dtype=torch.float32)
            grad[0, 0], grad[0, 1], grad[0, 2] = gA, gM, gS
            gt = dev_t(grad)

            # ---- Adam update ON DEVICE (fp32, from primitives) ----
            m = ttnn.add(ttnn.mul(m, B1), ttnn.mul(gt, 1.0 - B1))
            v = ttnn.add(ttnn.mul(v, B2), ttnn.mul(ttnn.square(gt), 1.0 - B2))
            mhat = ttnn.mul(m, 1.0 / (1.0 - B1 ** step))
            vhat = ttnn.mul(v, 1.0 / (1.0 - B2 ** step))
            update = ttnn.mul(ttnn.div(mhat, ttnn.add(ttnn.sqrt(vhat), EPS)), LR)
            param = ttnn.sub(param, update)

            if step == 1 or step % 30 == 0:
                loss = float((ttnn.to_torch(r) ** 2).mean())
                print("step {:4d}  loss={:.6f}  A={:.3f} mu={:.3f} sigma={:.3f}".format(
                    step, loss, *get_params()))

        a, mu, s = get_params()
        print("recovered A={:.3f} mu={:.3f} sigma={:.3f}".format(a, mu, abs(s)))
        ok = abs(a - A0) < 0.1 and abs(mu - MU0) < 0.1 and abs(abs(s) - SIG0) < 0.1
        print("PATHCLEAR_OK" if ok else "PATHCLEAR_FAIL (params off)")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
