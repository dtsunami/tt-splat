#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
2D Gaussian-splatting TRAINING LOOP on Blackhole — the capstone pathclear.

Forward render + per-pixel backward run ON SILICON (SFPU via ttnn ops, using the blend
backward verified bit-exactly vs torch.autograd in train2d_verify.py). Gradient reduction
+ Adam are host-side scalar work (negligible; M2 moves reduction on-device for scale).

Confidence gates:
  (1) train2d_verify.py: analytic backward == torch.autograd (machine precision).  [done]
  (2) step-1 cross-check: DEVICE grads == torch analytic grads (catches ttnn slips).
  (3) convergence: fit a known target render; loss drops, PSNR rises.

Opacity kept in (0,1) and conic PD by host projection => alpha never saturates (no clamp mask).
"""
import math
import torch
import ttnn

H = W = 64
N = 8
STEPS = 200
# per-parameter-group LR (params span scales: positions ~64 px, conic ~0.1, color ~0.5) — 3DGS-style
LR = {"cx": 0.5, "cy": 0.5, "a": 0.003, "b": 0.002, "c": 0.003, "op": 0.015, "col": 0.015}
B1, B2, EPS = 0.9, 0.999, 1e-8
PARAMS = ["cx", "cy", "a", "b", "c", "op", "col"]


def rand_scene(seed):
    g = torch.Generator().manual_seed(seed)
    s = {"cx": torch.rand(N, generator=g)*W, "cy": torch.rand(N, generator=g)*H,
         "a": 0.04+torch.rand(N, generator=g)*0.12, "b": (torch.rand(N, generator=g)-0.5)*0.03,
         "c": 0.04+torch.rand(N, generator=g)*0.12, "op": 0.3+torch.rand(N, generator=g)*0.45,
         "col": 0.3+torch.rand(N, generator=g)*0.6, "depth": torch.rand(N, generator=g)}
    return s, torch.argsort(s["depth"]).tolist()


def project(p):                       # keep conic PD + opacity in range (no clamp needed)
    p["a"] = p["a"].clamp_min(1e-3); p["c"] = p["c"].clamp_min(1e-3)
    lim = 0.99*(p["a"]*p["c"]).sqrt(); p["b"] = p["b"].clamp(-lim, lim)
    p["op"] = p["op"].clamp(0.02, 0.95)


# ---------- device render + backward (ttnn) ----------
def main():
    dev = ttnn.open_device(device_id=0)
    try:
        ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

        def dt(t):
            return ttnn.from_torch(t.reshape(1, 1, H, W).float(), dtype=ttnn.float32,
                                   layout=ttnn.TILE_LAYOUT, device=dev)

        def hsum(t):
            return float(ttnn.to_torch(t).sum())

        PX, PY = dt(jj.float()), dt(ii.float())

        def render(p, order, store=False):
            C = dt(torch.zeros(H, W)); T = dt(torch.ones(H, W)); rec = []
            for i in order:
                dx = ttnn.sub(PX, float(p["cx"][i])); dy = ttnn.sub(PY, float(p["cy"][i]))
                power = ttnn.add(ttnn.add(ttnn.mul(ttnn.square(dx), float(p["a"][i])),
                                          ttnn.mul(ttnn.mul(dx, dy), 2*float(p["b"][i]))),
                                 ttnn.mul(ttnn.square(dy), float(p["c"][i])))
                gexp = ttnn.exp(ttnn.mul(power, -0.5))
                alpha = ttnn.mul(gexp, float(p["op"][i]))     # = araw; op<1 => alpha<1
                w = ttnn.mul(T, alpha)
                if store:
                    rec.append((i, dx, dy, gexp, alpha, T, w))
                C = ttnn.add(C, ttnn.mul(w, float(p["col"][i])))
                T = ttnn.mul(T, ttnn.add(ttnn.mul(alpha, -1.0), 1.0))
            return (C, rec) if store else C

        def backward(p, order, rec, dC):
            grads = {k: torch.zeros(N) for k in PARAMS}
            S = dt(torch.zeros(H, W))
            for (i, dx, dy, gexp, alpha, Ti, w) in reversed(rec):
                col, ai, bi, ci = (float(p[k][i]) for k in ("col", "a", "b", "c"))
                grads["col"][i] = hsum(ttnn.mul(dC, w))
                one_m = ttnn.add(ttnn.mul(alpha, -1.0), 1.0)
                dCda = ttnn.sub(ttnn.mul(Ti, col), ttnn.div(S, one_m))
                dLda = ttnn.mul(dC, dCda)
                grads["op"][i] = hsum(ttnn.mul(dLda, gexp))
                base = ttnn.mul(dLda, ttnn.mul(alpha, -0.5))   # dL/dpower
                grads["a"][i] = hsum(ttnn.mul(base, ttnn.square(dx)))
                grads["b"][i] = hsum(ttnn.mul(base, ttnn.mul(ttnn.mul(dx, dy), 2.0)))
                grads["c"][i] = hsum(ttnn.mul(base, ttnn.square(dy)))
                grads["cx"][i] = hsum(ttnn.mul(base, ttnn.mul(ttnn.add(ttnn.mul(dx, ai), ttnn.mul(dy, bi)), -2.0)))
                grads["cy"][i] = hsum(ttnn.mul(base, ttnn.mul(ttnn.add(ttnn.mul(dx, bi), ttnn.mul(dy, ci)), -2.0)))
                S = ttnn.add(S, ttnn.mul(w, col))
            return grads

        # ---------- ground-truth target + init ----------
        gt, order = rand_scene(1)
        target_t = render(gt, order)
        target_h = ttnn.to_torch(target_t).reshape(H, W)
        # init = GT + noise: convergence test isolated from densification (a far random init
        # needs clone/split to converge — deferred). Recovery here proves the loop optimizes.
        gp = torch.Generator().manual_seed(3)
        p = {k: gt[k].clone() for k in gt}
        p["cx"] = p["cx"] + torch.randn(N, generator=gp)*3.0
        p["cy"] = p["cy"] + torch.randn(N, generator=gp)*3.0
        p["a"] = p["a"] * (1 + torch.randn(N, generator=gp)*0.12)
        p["c"] = p["c"] * (1 + torch.randn(N, generator=gp)*0.12)
        p["b"] = p["b"] + torch.randn(N, generator=gp)*0.005
        p["op"] = p["op"] + torch.randn(N, generator=gp)*0.05
        p["col"] = p["col"] + torch.randn(N, generator=gp)*0.05
        project(p)
        m = {k: torch.zeros(N) for k in PARAMS}; v = {k: torch.zeros(N) for k in PARAMS}

        def torch_analytic(p, target):            # fp32 host reference for the cross-check
            PXh, PYh = jj.float(), ii.float()
            C = torch.zeros(H, W); T = torch.ones(H, W); rec = []
            for i in order:
                dx, dy = PXh-float(p["cx"][i]), PYh-float(p["cy"][i])
                pw = float(p["a"][i])*dx*dx+2*float(p["b"][i])*dx*dy+float(p["c"][i])*dy*dy
                ge = torch.exp(-0.5*pw); al = float(p["op"][i])*ge; ww = T*al
                rec.append((i, dx, dy, ge, al, T.clone(), ww)); C = C+ww*float(p["col"][i]); T = T*(1-al)
            dC = (2.0/(H*W))*(C-target); g = {k: torch.zeros(N) for k in PARAMS}; S = torch.zeros(H, W)
            for (i, dx, dy, ge, al, Ti, ww) in reversed(rec):
                col, ai, bi, ci = (float(p[k][i]) for k in ("col", "a", "b", "c"))
                g["col"][i] = (dC*ww).sum(); dCda = Ti*col - S/(1-al); dLda = dC*dCda
                g["op"][i] = (dLda*ge).sum(); base = dLda*(-0.5*al)
                g["a"][i] = (base*dx*dx).sum(); g["b"][i] = (base*2*dx*dy).sum(); g["c"][i] = (base*dy*dy).sum()
                g["cx"][i] = (base*(-2*(ai*dx+bi*dy))).sum(); g["cy"][i] = (base*(-2*(bi*dx+ci*dy))).sum()
                S = S+ww*col
            return g

        print(f"2D GS training  H={H} W={W} N={N} steps={STEPS}")
        for step in range(1, STEPS+1):
            C, rec = render(p, order, store=True)
            dC = ttnn.mul(ttnn.sub(C, target_t), 2.0/(H*W))
            grads = backward(p, order, rec, dC)

            if step == 1:                          # gate (2): device grads vs torch analytic
                ref = torch_analytic(p, target_h)
                worst = max((grads[k]-ref[k]).abs().max().item()/(ref[k].abs().max().item()+1e-9) for k in PARAMS)
                print(f"  [gate2] device-vs-torch grad worst rel err = {worst:.2e} -> {'OK' if worst<1e-3 else 'FAIL'}")

            for k in PARAMS:                       # host Adam
                gk = grads[k]
                m[k] = B1*m[k]+(1-B1)*gk; v[k] = B2*v[k]+(1-B2)*gk*gk
                mh = m[k]/(1-B1**step); vh = v[k]/(1-B2**step)
                p[k] = p[k] - LR[k]*mh/(vh.sqrt()+EPS)
            project(p)

            if step == 1 or step % 25 == 0:
                loss = float(((ttnn.to_torch(C).reshape(H, W)-target_h)**2).mean())
                print(f"  step {step:3d}  loss={loss:.6f}")

        final = ttnn.to_torch(render(p, order)).reshape(H, W)
        mse = float(((final-target_h)**2).mean()); peak = float(target_h.max())
        psnr = 10*math.log10(peak*peak/max(mse, 1e-12))
        print(f"final MSE={mse:.3e}  PSNR={psnr:.1f} dB")
        print("TRAIN2D_OK" if psnr > 30 else "TRAIN2D_FAIL")
        try:
            from PIL import Image; import numpy as np
            row = torch.cat([target_h.clamp(0, 1), final.clamp(0, 1)], 1).numpy()
            Image.fromarray((row*255).astype(np.uint8)).resize((W*8, H*4), Image.NEAREST)\
                 .save("pathclear/train2d_target_vs_fit.png")
            print("wrote pathclear/train2d_target_vs_fit.png (target | fit)")
        except Exception as e:
            print("(png skipped:", e, ")")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
