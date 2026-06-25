#!/usr/bin/env python3
"""De-risk the fused-backward ALGORITHM (pure torch, no device):
the kernel won't store per-Gaussian T — it reconstructs T_i = T_run/(1-alpha_i) walking REVERSE from
the forward's final-T, recomputing alpha from params. Verify (1) reconstructed T == stored T, and
(2) the 7 grads == torch autograd. If this passes, the kernel just ports this arithmetic."""
import torch, math

torch.manual_seed(0)
TS = 32
K = 6                                   # gaussians in the tile
ii, jj = torch.meshgrid(torch.arange(TS), torch.arange(TS), indexing="ij")
PX, PY = jj.double(), ii.double()

# random 2D gaussians over the tile (screen-space conic params), depth-sorted order = 0..K-1
cx = torch.rand(K, dtype=torch.float64) * TS
cy = torch.rand(K, dtype=torch.float64) * TS
a = 0.05 + torch.rand(K, dtype=torch.float64) * 0.05
b = (torch.rand(K, dtype=torch.float64) - 0.5) * 0.01
c = 0.05 + torch.rand(K, dtype=torch.float64) * 0.05
op = 0.4 + torch.rand(K, dtype=torch.float64) * 0.4
col = 0.3 + torch.rand(K, dtype=torch.float64) * 0.6
order = list(range(K))

# ---- reference forward with autograd (front-to-back), stores per-gaussian T ----
P = {k: v.clone().requires_grad_(True) for k, v in
     dict(cx=cx, cy=cy, a=a, b=b, c=c, op=op, col=col).items()}
C = torch.zeros(TS, TS, dtype=torch.float64); T = torch.ones(TS, TS, dtype=torch.float64)
T_stored = {}
for i in order:
    dx, dy = PX - P["cx"][i], PY - P["cy"][i]
    power = P["a"][i]*dx*dx + 2*P["b"][i]*dx*dy + P["c"][i]*dy*dy
    gexp = torch.exp(-0.5*power); alpha = P["op"][i]*gexp
    T_stored[i] = T.detach().clone()
    C = C + T*alpha*P["col"][i]; T = T*(1-alpha)
T_final = T.detach().clone()

dLdC = torch.rand(TS, TS, dtype=torch.float64)           # arbitrary upstream grad
loss = (C * dLdC).sum()
loss.backward()
auto = {k: P[k].grad.detach().clone() for k in P}

# ---- the KERNEL'S algorithm: reverse pass, reconstruct T from T_final, recompute alpha ----
g = {k: torch.zeros(K, dtype=torch.float64) for k in P}
S = torch.zeros(TS, TS, dtype=torch.float64)
T_run = T_final.clone()
recon_err = 0.0
for i in reversed(order):
    dx, dy = PX - cx[i], PY - cy[i]
    power = a[i]*dx*dx + 2*b[i]*dx*dy + c[i]*dy*dy
    gexp = torch.exp(-0.5*power); alpha = op[i]*gexp
    one_m = 1 - alpha
    T_i = T_run / one_m                                  # <-- reconstruct (reciprocal)
    recon_err = max(recon_err, float((T_i - T_stored[i]).abs().max()))
    w = T_i * alpha
    dCda = T_i*col[i] - S/one_m
    dLda = dLdC * dCda
    g["col"][i] = (dLdC * w).sum()
    g["op"][i] = (dLda * gexp).sum()
    base = dLda * alpha * (-0.5)
    g["a"][i] = (base * dx*dx).sum()
    g["b"][i] = (base * 2*dx*dy).sum()
    g["c"][i] = (base * dy*dy).sum()
    g["cx"][i] = (base * (-2)*(a[i]*dx + b[i]*dy)).sum()
    g["cy"][i] = (base * (-2)*(b[i]*dx + c[i]*dy)).sum()
    S = S + w*col[i]
    T_run = T_i

worst = max(float((g[k]-auto[k]).abs().max() / (auto[k].abs().max()+1e-12)) for k in P)
print(f"T reconstruction max abs err = {recon_err:.2e}")
print(f"grad worst rel err vs autograd = {worst:.2e}")
print("PROTO_RECON_OK" if recon_err < 1e-9 and worst < 1e-9 else "PROTO_RECON_FAIL")
