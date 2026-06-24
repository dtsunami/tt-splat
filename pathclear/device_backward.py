#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Item 3 (core): the DEVICE BACKWARD of the alpha-blend rasterizer, on silicon.

Forward render (N Gaussians, one 32x32 tile) runs on device via ttnn, storing per-Gaussian
alpha_i and T_i. The backward then runs ENTIRELY ON DEVICE: reverse pass maintaining the
suffix-color S, per-pixel gradient products (ttnn elementwise), reduced per-Gaussian to scalars
(ttnn.sum) -> grads for {cx,cy,a,b,c,opacity,color}. Verified against host autograd.

This is the reverse of the M5/M14 blend kernel expressed in ttnn ops (device-executed). The fused
custom reduce_tile kernel (fp32-accum) is the perf/precision swap; M2 scatter-add composes it
across tiles and M0 device-Adam consumes the grads — all validated separately.

Note: ttnn.sum uses bf16 accumulation (~0.1-1% reduction error) -> verify to a relative tol;
fp32 reduction (reduce_tile<SUM,REDUCE_SCALAR, enforce_fp32_accumulation>) tightens it.
"""
import math, torch, ttnn

H = W = 32
N = 4


def scene(seed):
    g = torch.Generator().manual_seed(seed)
    p = {"cx": torch.rand(N, generator=g)*W, "cy": torch.rand(N, generator=g)*H,
         "a": 0.05+torch.rand(N, generator=g)*0.1, "b": (torch.rand(N, generator=g)-0.5)*0.03,
         "c": 0.05+torch.rand(N, generator=g)*0.1, "op": 0.3+torch.rand(N, generator=g)*0.4,
         "col": 0.3+torch.rand(N, generator=g)*0.5}
    order = torch.argsort(torch.rand(N, generator=g)).tolist()
    target = torch.rand(H, W, generator=g)
    return p, order, target


def host_grads(p, order, target):
    """Host autograd reference."""
    P = {k: v.clone().requires_grad_() for k, v in p.items()}
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    PX, PY = jj.float(), ii.float()
    C = torch.zeros(H, W); T = torch.ones(H, W)
    for i in order:
        dx, dy = PX-P["cx"][i], PY-P["cy"][i]
        al = (P["op"][i]*torch.exp(-0.5*(P["a"][i]*dx*dx+2*P["b"][i]*dx*dy+P["c"][i]*dy*dy))).clamp(max=0.99)
        C = C + T*al*P["col"][i]; T = T*(1-al)
    loss = ((C-target)**2).mean(); loss.backward()
    return {k: P[k].grad.clone() for k in P}


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        p, order, target = scene(0)
        ref = host_grads(p, order, target)

        ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        dt = lambda t: ttnn.from_torch(t.reshape(1, 1, H, W).float(), dtype=ttnn.float32,
                                       layout=ttnn.TILE_LAYOUT, device=dev)
        PX, PY, TGT = dt(jj.float()), dt(ii.float()), dt(target)
        dsum = lambda t: float(ttnn.to_torch(ttnn.sum(t)).flatten()[0])      # device reduce -> scalar

        # ---- forward ON DEVICE, store per-Gaussian alpha_i, T_i ----
        C = dt(torch.zeros(H, W)); T = dt(torch.ones(H, W)); rec = []
        for i in order:
            cx, cy, a, b, c, op, col = (float(p[k][i]) for k in ("cx", "cy", "a", "b", "c", "op", "col"))
            dx = ttnn.sub(PX, cx); dy = ttnn.sub(PY, cy)
            power = ttnn.add(ttnn.add(ttnn.mul(ttnn.mul(dx, dx), a), ttnn.mul(ttnn.mul(dx, dy), 2*b)),
                             ttnn.mul(ttnn.mul(dy, dy), c))
            gexp = ttnn.exp(ttnn.mul(power, -0.5))
            alpha = ttnn.mul(gexp, op)            # op<1 -> alpha<1, clamp inactive
            rec.append((i, dx, dy, gexp, alpha, T))
            C = ttnn.add(C, ttnn.mul(ttnn.mul(T, alpha), col))
            T = ttnn.mul(T, ttnn.add(ttnn.mul(alpha, -1.0), 1.0))
        dLdC = ttnn.mul(ttnn.sub(C, TGT), 2.0/(H*W))

        # ---- backward ON DEVICE: reverse pass, per-Gaussian grads via ttnn.sum ----
        grads = {k: torch.zeros(N) for k in p}
        S = dt(torch.zeros(H, W))
        for (i, dx, dy, gexp, alpha, Ti) in reversed(rec):
            col, a, b, c = (float(p[k][i]) for k in ("col", "a", "b", "c"))
            w = ttnn.mul(Ti, alpha)
            grads["col"][i] = dsum(ttnn.mul(dLdC, w))
            one_m = ttnn.add(ttnn.mul(alpha, -1.0), 1.0)
            dCda = ttnn.sub(ttnn.mul(Ti, col), ttnn.div(S, one_m))
            dLda = ttnn.mul(dLdC, dCda)
            grads["op"][i] = dsum(ttnn.mul(dLda, gexp))
            base = ttnn.mul(dLda, ttnn.mul(alpha, -0.5))           # dL/dpower
            grads["a"][i] = dsum(ttnn.mul(base, ttnn.mul(dx, dx)))
            grads["b"][i] = dsum(ttnn.mul(base, ttnn.mul(ttnn.mul(dx, dy), 2.0)))
            grads["c"][i] = dsum(ttnn.mul(base, ttnn.mul(dy, dy)))
            grads["cx"][i] = dsum(ttnn.mul(base, ttnn.mul(ttnn.add(ttnn.mul(dx, a), ttnn.mul(dy, b)), -2.0)))
            grads["cy"][i] = dsum(ttnn.mul(base, ttnn.mul(ttnn.add(ttnn.mul(dx, b), ttnn.mul(dy, c)), -2.0)))
            S = ttnn.add(S, ttnn.mul(w, col))

        print(f"device backward vs host autograd  (N={N}, ttnn.sum bf16-accum reduction)")
        worst = 0.0
        for k in ["cx", "cy", "a", "b", "c", "op", "col"]:
            num = float((grads[k]-ref[k]).abs().max()); den = float(ref[k].abs().max())+1e-12
            rel = num/den; worst = max(worst, rel)
            print(f"  {k:3s}  max|Δ|={num:.3e}  rel={rel:.2e}")
        print(f"worst relative error = {worst:.2e}")
        print("DEVICE_BWD_OK" if worst < 2e-2 else "DEVICE_BWD_FAIL")   # bf16-reduce tol
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
