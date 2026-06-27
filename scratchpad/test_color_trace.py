#!/usr/bin/env python3
# LAYER 1 (host, no hardware): trace fidelity. The recorded DAG (eval_dag) must equal the direct fp64
# oracle (_color_op_backward run with NumpyBackend), and both must match a torch reference of the SH/op
# backward. Proves the trace faithfully captures the extracted color/opacity backward before any codegen.
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home()/"tt-splat"/"server")); sys.path.insert(0, str(Path.home()/"tt-splat"/"docs"/"pathclear"))
import numpy as np, torch
from backend import NumpyBackend, eval_dag
from device_project_backward import _color_op_backward, _C0
from proj_fused import trace_color, color_inputs
from train_real import sh_eval

np.random.seed(0); torch.manual_seed(0)
N, deg = 64, 3; K = (deg + 1) ** 2

# ---- random inputs (host fp64) ----
sh = np.random.randn(N, K, 3) * 0.2
mean = np.random.randn(N, 3) * 0.4 + np.array([0., 0., 4.])
logit = np.random.randn(N) * 0.5
up = {k: np.random.randn(N) for k in ("op", "colR", "colG", "colB")}
# forward color aux (mirror device_project.project_color math, host)
cc_world = np.zeros(3)
dvec = mean - cc_world; inv = 1.0 / (np.linalg.norm(dvec, axis=1) + 1e-9)
x, y, z = dvec[:, 0] * inv, dvec[:, 1] * inv, dvec[:, 2] * inv
op_sig = 1.0 / (1.0 + np.exp(-logit))
# wb (C-weighted SH spatial basis) + pre (pre-clamp color) from torch sh_eval pieces
dirs = torch.tensor(np.stack([x, y, z], 1))
col_pre = sh_eval(torch.tensor(sh), dirs, deg).numpy()        # [N,3] pre-clamp? sh_eval clamps; recompute pre:
# pre_c = C0*sh[:,0,c]+0.5 + sum_k wb_k*sh[:,k,c]; build wb the same way device_project.project_color does
_C1 = 0.4886025119029199
_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
       -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
xx, yy, zz = x*x, y*y, z*z; xy, yz, xz = x*y, y*z, x*z
wb = [None, -_C1*y, _C1*z, -_C1*x,
      _C2[0]*xy, _C2[1]*yz, _C2[2]*(2*zz-xx-yy), _C2[3]*xz, _C2[4]*(xx-yy),
      _C3[0]*y*(3*xx-yy), _C3[1]*xy*z, _C3[2]*y*(4*zz-xx-yy),
      _C3[3]*z*(2*zz-3*xx-3*yy), _C3[4]*x*(4*zz-xx-yy), _C3[5]*z*(xx-yy), _C3[6]*x*(xx-3*yy)]
pre = []
for c in range(3):
    r = _C0*sh[:, 0, c] + 0.5
    for k in range(1, K): r = r + wb[k]*sh[:, k, c]
    pre.append(r)

def src(kind, *idx):
    if kind == "op_sig": return op_sig
    if kind == "inv": return inv
    if kind == "dir": return {"x": x, "y": y, "z": z}[idx[0]]
    if kind == "up": return up[idx[0]]
    if kind == "pre": return pre[idx[0]]
    if kind == "wb": return wb[idx[0]]
    if kind == "sh": return sh[:, idx[0], idx[1]]
    raise KeyError(kind)

# ---- (a) direct fp64 oracle via NumpyBackend ----
B = NumpyBackend()
ACn = {"pre": pre, "wb": wb, "x": x, "y": y, "z": z, "inv": inv}
gop_o, gsh_o, gmc_o = _color_op_backward(B, lambda k, c: sh[:, k, c], up, ACn, op_sig, deg)

# ---- (b) DAG via TraceBackend + eval_dag ----
nodes, outs, innames = trace_color(deg)
vals = eval_dag(nodes, color_inputs(deg, src))
def gv(name): return vals[outs[name]]

# ---- compare DAG vs direct oracle ----
def rel(a, b): a, b = np.asarray(a), np.asarray(b); return np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-30)
worst = rel(gv("gop"), gop_o)
for c in range(3):
    for k in range(K): worst = max(worst, rel(gv(f"gsh.{k}.{c}"), gsh_o[(k, c)]))
for d in range(3): worst = max(worst, rel(gv(f"gmean_color.{d}"), gmc_o[d]))
print(f"trace nodes={len(nodes)} inputs={len(innames)} outputs={len(outs)}")
print(f"DAG(eval_dag) vs direct fp64 oracle  worst rel = {worst:.2e}")

# ---- independent torch reference for gop + gsh (catch a wrong extraction, not just self-consistency) ----
shT = torch.tensor(sh, requires_grad=True); lT = torch.tensor(logit, requires_grad=True)
colT = sh_eval(shT, dirs, deg); opT = torch.sigmoid(lT)
L = (torch.tensor(up["colR"])*colT[:, 0] + torch.tensor(up["colG"])*colT[:, 1]
     + torch.tensor(up["colB"])*colT[:, 2] + torch.tensor(up["op"])*opT).sum()
L.backward()
rgop = rel(gv("gop"), lT.grad.numpy())
rgsh = max(rel(gv(f"gsh.{k}.{c}"), shT.grad.numpy()[:, k, c]) for k in range(K) for c in range(3))
print(f"vs torch autograd:  gop rel={rgop:.2e}   gsh rel={rgsh:.2e}")
ok = worst < 1e-12 and rgop < 1e-6 and rgsh < 1e-6
print("COLOR_TRACE_OK" if ok else "COLOR_TRACE_FAIL")
