#!/usr/bin/env python3
# LAYER 3 (silicon): the codegen'd fused color/op backward kernel vs the fp64 oracle, on REAL inputs
# derived from the ttnn forward. Oracle == ttnn (byte-identical, proven) == torch (test_proj_bwd), so
# fused == oracle within fp32 tolerance proves the whole tracer->lower->emit->silicon pipeline. Gate 1e-2.
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home()/"tt-splat"/"server")); sys.path.insert(0, str(Path.home()/"tt-splat"/"docs"/"pathclear"))
import numpy as np, torch, ttnn
import device_project as DP
from device_project_backward import _color_op_backward
from backend import NumpyBackend
from proj_fused import color_inputs, run_color_fused, build_color

np.random.seed(0); torch.manual_seed(0)
N, deg = 200, 3; K = (deg + 1) ** 2
mean = torch.randn(N, 3) * 0.4 + torch.tensor([0., 0., 4.])
sh = torch.randn(N, K, 3) * 0.2
op_logit = torch.randn(N) * 0.5
Rv, tv, fx, fy, cx, cy = torch.eye(3), torch.zeros(3), 100., 100., 48., 48.
cam = (Rv, tv, fx, fy, cx, cy, "t")
up = {k: np.random.randn(N) for k in ("op", "colR", "colG", "colB")}

dev = ttnn.open_device(device_id=0)
try:
    g = lambda t: ttnn.to_torch(t).flatten().numpy()[:N].astype(np.float64)
    _, _, _, AC = DP.project_color(dev, mean, sh, deg, cam, aux=True)
    op_sig = g(DP.project_op(dev, op_logit))
    shn = sh.numpy().astype(np.float64)
    pre = [g(AC["pre"][c]) for c in range(3)]
    wb = [None] + [g(AC["wb"][k]) for k in range(1, K)]
    x, y, z, inv = g(AC["x"]), g(AC["y"]), g(AC["z"]), g(AC["inv"])

    def src(kind, *idx):
        if kind == "op_sig": return op_sig
        if kind == "inv": return inv
        if kind == "dir": return {"x": x, "y": y, "z": z}[idx[0]]
        if kind == "up": return up[idx[0]]
        if kind == "pre": return pre[idx[0]]
        if kind == "wb": return wb[idx[0]]
        if kind == "sh": return shn[:, idx[0], idx[1]]
        raise KeyError(kind)
    inputs = color_inputs(deg, src)

    # fp64 oracle (== ttnn == torch) on the SAME real inputs
    ACn = {"pre": pre, "wb": wb, "x": x, "y": y, "z": z, "inv": inv}
    gop_o, gsh_o, gmc_o = _color_op_backward(NumpyBackend(), lambda k, c: shn[:, k, c], up, ACn, op_sig, deg)

    b = build_color(deg)
    print(f"kernel: instrs={len(b['prog'])} maxreg={b['maxreg']} n_in={b['n_in']} n_out={b['n_out']}")
    fused = run_color_fused(dev, inputs, deg)

    def rel(a, b_): a, b_ = np.asarray(a, np.float64), np.asarray(b_, np.float64); return np.linalg.norm(a-b_)/(np.linalg.norm(b_)+1e-30)
    wg = rel(fused["gop"], gop_o)
    ws = max(rel(fused[f"gsh.{k}.{c}"], gsh_o[(k, c)]) for k in range(K) for c in range(3))
    wm = max(rel(fused[f"gmean_color.{d}"], gmc_o[d]) for d in range(3))
    print(f"  gop  rel={wg:.2e}")
    print(f"  gsh  rel={ws:.2e}  (worst over {K}x3)")
    print(f"  gmean_color rel={wm:.2e}")
    worst = max(wg, ws, wm)
    print(f"FUSED_COLOR worst={worst:.2e} -> {'OK' if worst < 1e-2 else 'FAIL'}")
finally:
    ttnn.close_device(dev)
