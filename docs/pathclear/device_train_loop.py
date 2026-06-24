#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLOSE THE LOOP: an integrated DEVICE training loop for the 2D Gaussian rasterizer.

Composes the validated pieces into one loop where the heavy per-pixel work runs ON DEVICE:
  forward blend (ttnn) -> loss -> DEVICE BACKWARD (M15, reverse pass + ttnn.sum reduce) -> grads
  -> Adam update.  Fits a perturbed-init scene to a target; loss drops, params recover.

Per-pixel render + backward + reduction are on-device (ttnn). Adam runs on the device-reduced
gradient scalars (negligible scalar math; M0 proved a fully-device packed-tile Adam separately).
This is the device-resident training step; fusing fwd+bwd into custom kernels + fp32 reduce
(reduce_tile) + M2 scatter-add for multi-tile are the perf/scale follow-ups.
"""
import math, torch, ttnn

H = W = 32
N = 4
STEPS = 120
LR = {"cx": .4, "cy": .4, "a": .004, "b": .003, "c": .004, "op": .02, "col": .02}
B1, B2, EPS = 0.9, 0.999, 1e-8
KEYS = ["cx", "cy", "a", "b", "c", "op", "col"]


def gt_scene(seed=1):
    g = torch.Generator().manual_seed(seed)
    return {"cx": torch.rand(N, generator=g)*W, "cy": torch.rand(N, generator=g)*H,
            "a": 0.06+torch.rand(N, generator=g)*0.08, "b": (torch.rand(N, generator=g)-0.5)*0.02,
            "c": 0.06+torch.rand(N, generator=g)*0.08, "op": 0.4+torch.rand(N, generator=g)*0.3,
            "col": 0.35+torch.rand(N, generator=g)*0.4}


def project(p):
    p["a"] = p["a"].clamp_min(1e-3); p["c"] = p["c"].clamp_min(1e-3)
    lim = 0.99*(p["a"]*p["c"]).sqrt(); p["b"] = p["b"].clamp(-lim, lim)
    p["op"] = p["op"].clamp(0.02, 0.95)


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        dt = lambda t: ttnn.from_torch(t.reshape(1, 1, H, W).float(), dtype=ttnn.float32,
                                       layout=ttnn.TILE_LAYOUT, device=dev)
        PX, PY = dt(jj.float()), dt(ii.float())
        dsum = lambda t: float(ttnn.to_torch(ttnn.sum(t)).flatten()[0])

        order = list(range(N))

        def forward(p, store):
            C = dt(torch.zeros(H, W)); T = dt(torch.ones(H, W)); rec = []
            for i in order:
                cx, cy, a, b, c, op, col = (float(p[k][i]) for k in KEYS)
                dx = ttnn.sub(PX, cx); dy = ttnn.sub(PY, cy)
                power = ttnn.add(ttnn.add(ttnn.mul(ttnn.mul(dx, dx), a), ttnn.mul(ttnn.mul(dx, dy), 2*b)),
                                 ttnn.mul(ttnn.mul(dy, dy), c))
                gexp = ttnn.exp(ttnn.mul(power, -0.5)); alpha = ttnn.mul(gexp, op)
                if store: rec.append((i, dx, dy, gexp, alpha, T))
                C = ttnn.add(C, ttnn.mul(ttnn.mul(T, alpha), col))
                T = ttnn.mul(T, ttnn.add(ttnn.mul(alpha, -1.0), 1.0))
            return (C, rec) if store else C

        def backward(p, rec, dLdC):
            grads = {k: torch.zeros(N) for k in KEYS}; S = dt(torch.zeros(H, W))
            for (i, dx, dy, gexp, alpha, Ti) in reversed(rec):
                col, a, b, c = (float(p[k][i]) for k in ("col", "a", "b", "c"))
                w = ttnn.mul(Ti, alpha)
                grads["col"][i] = dsum(ttnn.mul(dLdC, w))
                one_m = ttnn.add(ttnn.mul(alpha, -1.0), 1.0)
                dLda = ttnn.mul(dLdC, ttnn.sub(ttnn.mul(Ti, col), ttnn.div(S, one_m)))
                grads["op"][i] = dsum(ttnn.mul(dLda, gexp))
                base = ttnn.mul(dLda, ttnn.mul(alpha, -0.5))
                grads["a"][i] = dsum(ttnn.mul(base, ttnn.mul(dx, dx)))
                grads["b"][i] = dsum(ttnn.mul(base, ttnn.mul(ttnn.mul(dx, dy), 2.0)))
                grads["c"][i] = dsum(ttnn.mul(base, ttnn.mul(dy, dy)))
                grads["cx"][i] = dsum(ttnn.mul(base, ttnn.mul(ttnn.add(ttnn.mul(dx, a), ttnn.mul(dy, b)), -2.0)))
                grads["cy"][i] = dsum(ttnn.mul(base, ttnn.mul(ttnn.add(ttnn.mul(dx, b), ttnn.mul(dy, c)), -2.0)))
                S = ttnn.add(S, ttnn.mul(w, col))
            return grads

        gt = gt_scene(1)
        target = ttnn.to_torch(forward(gt, store=False)).reshape(H, W)
        TGT = dt(target)

        gp = torch.Generator().manual_seed(5)            # perturbed init
        p = {k: gt[k].clone() for k in gt}
        p["cx"] += torch.randn(N, generator=gp)*2; p["cy"] += torch.randn(N, generator=gp)*2
        p["a"] *= 1+torch.randn(N, generator=gp)*0.1; p["c"] *= 1+torch.randn(N, generator=gp)*0.1
        p["op"] += torch.randn(N, generator=gp)*0.05; p["col"] += torch.randn(N, generator=gp)*0.05
        project(p)
        m = {k: torch.zeros(N) for k in KEYS}; v = {k: torch.zeros(N) for k in KEYS}

        def psnr(c):
            mse = float(((c-target)**2).mean()); return 10*math.log10(float(target.max())**2/max(mse, 1e-12))

        print(f"device training loop  H={H} N={N} steps={STEPS}  (fwd+bwd on device)")
        for step in range(1, STEPS+1):
            C, rec = forward(p, store=True)
            dLdC = ttnn.mul(ttnn.sub(C, TGT), 2.0/(H*W))
            grads = backward(p, rec, dLdC)
            for k in KEYS:                                # Adam on device-reduced grads
                g = grads[k]; m[k] = B1*m[k]+(1-B1)*g; v[k] = B2*v[k]+(1-B2)*g*g
                mh = m[k]/(1-B1**step); vh = v[k]/(1-B2**step)
                p[k] = p[k] - LR[k]*mh/(vh.sqrt()+EPS)
            project(p)
            if step == 1 or step % 20 == 0:
                cc = ttnn.to_torch(forward(p, store=False)).reshape(H, W)
                print(f"  step {step:3d}  loss={float(((cc-target)**2).mean()):.6f}  PSNR={psnr(cc):.1f} dB")
        final = ttnn.to_torch(forward(p, store=False)).reshape(H, W)
        print(f"final PSNR={psnr(final):.1f} dB")
        print("DEVICE_LOOP_OK" if psnr(final) > 35 else "DEVICE_LOOP_FAIL")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
