#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
VERIFIED TEST CASE: analytic backward-through-the-alpha-blend vs torch.autograd.

Confidence gate before any on-device training. We implement the 3DGS 2D forward
(front->back compositing) and a hand-derived analytic backward, then check the
analytic gradients of EVERY parameter {cx,cy,a,b,c,opacity,color} against autograd.

Forward (near->far):  alpha_i = clamp(op_i * exp(-0.5*(a dx^2 + 2b dx dy + c dy^2)), 0, .99)
                      C += T*alpha_i*col_i ;  T *= (1-alpha_i)
Backward (far->near):  dC/dcol_i = w_i = T_i*alpha_i
                       dC/dalpha_i = T_i*col_i - S_i/(1-alpha_i),  S_i = sum_{j behind} w_j col_j
                       then chain alpha_i -> params (clamp-masked).
"""
import math
import torch

H = W = 32
N = 8
torch.set_default_dtype(torch.float64)   # tight check in fp64; device runs fp32 separately


def scene(seed=0):
    g = torch.Generator().manual_seed(seed)
    P = {
        "cx": (torch.rand(N, generator=g) * W).requires_grad_(),
        "cy": (torch.rand(N, generator=g) * H).requires_grad_(),
        "a":  (0.05 + torch.rand(N, generator=g) * 0.15).requires_grad_(),
        "b":  ((torch.rand(N, generator=g) - 0.5) * 0.05).requires_grad_(),
        "c":  (0.05 + torch.rand(N, generator=g) * 0.15).requires_grad_(),
        "op": (0.3 + torch.rand(N, generator=g) * 0.5).requires_grad_(),
        "col":(0.3 + torch.rand(N, generator=g) * 0.6).requires_grad_(),
    }
    depth = torch.rand(N, generator=g)
    order = torch.argsort(depth).tolist()          # near -> far
    target = torch.rand(H, W, generator=g)
    return P, order, target


def grid():
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    return jj.double(), ii.double()


def forward(P, order, PX, PY):
    C = torch.zeros(H, W); T = torch.ones(H, W)
    for i in order:
        dx, dy = PX - P["cx"][i], PY - P["cy"][i]
        power = P["a"][i]*dx*dx + 2*P["b"][i]*dx*dy + P["c"][i]*dy*dy
        alpha = (P["op"][i] * torch.exp(-0.5*power)).clamp(0.0, 0.99)
        C = C + T*alpha*P["col"][i]
        T = T * (1 - alpha)
    return C


def analytic(P, order, PX, PY, target):
    p = {k: v.detach() for k, v in P.items()}
    C = torch.zeros(H, W); T = torch.ones(H, W); rec = []
    for i in order:
        dx, dy = PX - p["cx"][i], PY - p["cy"][i]
        power = p["a"][i]*dx*dx + 2*p["b"][i]*dx*dy + p["c"][i]*dy*dy
        gexp = torch.exp(-0.5*power)
        araw = p["op"][i]*gexp
        alpha = araw.clamp(0.0, 0.99)
        w = T*alpha
        rec.append((i, dx, dy, gexp, araw, alpha, T.clone(), w))
        C = C + w*p["col"][i]
        T = T * (1 - alpha)
    dC = (2.0/(H*W)) * (C - target)
    grads = {k: torch.zeros(N) for k in P}
    S = torch.zeros(H, W)                    # sum_{behind} w_j col_j
    for (i, dx, dy, gexp, araw, alpha, Ti, w) in reversed(rec):
        col = p["col"][i]
        grads["col"][i] = (dC * w).sum()
        dCda = Ti*col - S/(1 - alpha)        # dC/dalpha_i
        dLda = dC * dCda
        mask = (araw <= 0.99).double()       # clamp: zero grad where saturated
        grads["op"][i] = (dLda * gexp * mask).sum()
        base = dLda * (-0.5*araw) * mask     # dL/dpower
        grads["a"][i]  = (base * dx*dx).sum()
        grads["b"][i]  = (base * 2*dx*dy).sum()
        grads["c"][i]  = (base * dy*dy).sum()
        grads["cx"][i] = (base * (-2*(p["a"][i]*dx + p["b"][i]*dy))).sum()
        grads["cy"][i] = (base * (-2*(p["b"][i]*dx + p["c"][i]*dy))).sum()
        S = S + w*col
    return grads


def main():
    P, order, target = scene()
    PX, PY = grid()

    # reference: autograd
    C = forward(P, order, PX, PY)
    L = ((C - target)**2).mean()
    L.backward()
    ref = {k: P[k].grad.detach().clone() for k in P}

    ana = analytic(P, order, PX, PY, target)

    print(f"gradient check  H={H} W={W} N={N}  (analytic vs torch.autograd)")
    worst = 0.0
    for k in ["cx", "cy", "a", "b", "c", "op", "col"]:
        num = (ana[k] - ref[k]).abs().max().item()
        den = ref[k].abs().max().item() + 1e-12
        rel = num / den
        worst = max(worst, rel)
        print(f"  {k:3s}  max|Δ|={num:.3e}  rel={rel:.2e}")
    print(f"worst relative error = {worst:.2e}")
    print("VERIFY_OK" if worst < 1e-6 else "VERIFY_FAIL")


if __name__ == "__main__":
    main()
