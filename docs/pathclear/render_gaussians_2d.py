#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Forward rasterizer pathclear: render N 2D Gaussians into an image by front-to-back
alpha compositing, ON SILICON (SFPU via ttnn ops), validated vs a CPU golden.

Per Gaussian (sorted near->far), over all pixels:
    alpha = clamp(opacity * exp(-0.5 * (a*dx^2 + 2b*dx*dy + c*dy^2)), 0, 0.99)
    C    += T * alpha * color
    T    *= (1 - alpha)

This is the 3DGS forward blend (grayscale here; RGB = 3 identical channels). Reuses the M1
conic eval. Proves the full forward render path before we wire forward+backward training.
"""
import math, struct
import torch
import ttnn

H = W = 64
N = 16                      # number of Gaussians


def make_scene(seed=0):
    g = torch.Generator().manual_seed(seed)
    s = {
        "cx": torch.rand(N, generator=g) * W,
        "cy": torch.rand(N, generator=g) * H,
        "sx": 3.0 + torch.rand(N, generator=g) * 7.0,
        "sy": 3.0 + torch.rand(N, generator=g) * 7.0,
        "th": torch.rand(N, generator=g) * math.pi,
        "op": 0.4 + torch.rand(N, generator=g) * 0.5,
        "col": 0.25 + torch.rand(N, generator=g) * 0.75,
        "depth": torch.rand(N, generator=g),
    }
    # conic (a,b,c) = Sigma^-1 from (sx,sy,theta)
    a = torch.empty(N); b = torch.empty(N); c = torch.empty(N)
    for i in range(N):
        ct, st = math.cos(s["th"][i]), math.sin(s["th"][i])
        R = torch.tensor([[ct, -st], [st, ct]])
        cov = R @ torch.diag(torch.tensor([s["sx"][i]**2, s["sy"][i]**2])) @ R.T
        M = torch.inverse(cov)
        a[i], b[i], c[i] = M[0, 0], M[0, 1], M[1, 1]
    s["a"], s["b"], s["c"] = a, b, c
    order = torch.argsort(s["depth"])      # near -> far
    return s, order


def golden(s, order):
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    PX, PY = jj.float(), ii.float()
    C = torch.zeros(H, W); T = torch.ones(H, W)
    for i in order.tolist():
        dx, dy = PX - float(s["cx"][i]), PY - float(s["cy"][i])
        power = float(s["a"][i])*dx*dx + 2*float(s["b"][i])*dx*dy + float(s["c"][i])*dy*dy
        alpha = (float(s["op"][i]) * torch.exp(-0.5 * power)).clamp(0.0, 0.99)
        C = C + T * alpha * float(s["col"][i])
        T = T * (1 - alpha)
    return C


def render_device(dev, s, order):
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

    def dt(t):
        return ttnn.from_torch(t.reshape(1, 1, H, W).float(), dtype=ttnn.float32,
                               layout=ttnn.TILE_LAYOUT, device=dev)

    PX, PY = dt(jj), dt(ii)
    C = dt(torch.zeros(H, W)); T = dt(torch.ones(H, W))
    for i in order.tolist():
        cx, cy = float(s["cx"][i]), float(s["cy"][i])
        a, b, c = float(s["a"][i]), float(s["b"][i]), float(s["c"][i])
        op, col = float(s["op"][i]), float(s["col"][i])
        dx = ttnn.sub(PX, cx)
        dy = ttnn.sub(PY, cy)
        power = ttnn.add(ttnn.add(ttnn.mul(ttnn.square(dx), a),
                                  ttnn.mul(ttnn.mul(dx, dy), 2 * b)),
                         ttnn.mul(ttnn.square(dy), c))
        alpha = ttnn.mul(ttnn.exp(ttnn.mul(power, -0.5)), op)
        alpha = ttnn.clamp(alpha, 0.0, 0.99)
        contrib = ttnn.mul(ttnn.mul(T, alpha), col)
        C = ttnn.add(C, contrib)
        one_minus = ttnn.add(ttnn.mul(alpha, -1.0), 1.0)
        T = ttnn.mul(T, one_minus)
    return ttnn.to_torch(C).reshape(H, W)


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        s, order = make_scene()
        gold = golden(s, order)
        got = render_device(dev, s, order)
        mse = float(((got - gold) ** 2).mean())
        peak = float(gold.max())
        psnr = 10 * math.log10((peak * peak) / max(mse, 1e-12))
        print(f"image {H}x{W}  N={N}  blend order=near->far")
        print(f"MSE={mse:.3e}  PSNR={psnr:.1f} dB  (vs CPU golden)")
        ok = mse < 1e-5
        print("RASTER_OK" if ok else "RASTER_FAIL")
        try:
            from PIL import Image
            import numpy as np
            both = torch.cat([gold.clamp(0, 1), got.clamp(0, 1)], dim=1).numpy()
            Image.fromarray((both * 255).astype(np.uint8)).resize((W * 8, H * 4), Image.NEAREST)\
                 .save("pathclear/render_golden_vs_device.png")
            print("wrote pathclear/render_golden_vs_device.png (golden | device)")
        except Exception as e:
            print("(png skipped:", e, ")")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
