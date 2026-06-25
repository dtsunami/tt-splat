#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage D device grad-check: server.device_project_backward.project_backward vs torch autograd of
project_general/sh_eval/sigmoid.  Target (fp32): rel err < ~1e-2 per parameter."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path.home() / "tt-splat" / "server"))
sys.path.insert(0, str(Path.home() / "tt-splat" / "docs" / "pathclear"))
import ttnn
from train_real import project_general, sh_eval
from device_project_backward import project_backward

torch.set_default_dtype(torch.float64)

dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    N, deg = 40, 3
    K = (deg + 1) ** 2
    P = {
        "mean": (torch.randn(N, 3) * 0.4 + torch.tensor([0., 0., 4.])).requires_grad_(True),
        "scale": (torch.randn(N, 3) * 0.3 - 1.5).requires_grad_(True),
        "quat": torch.randn(N, 4).requires_grad_(True),
        "op": (torch.randn(N) * 0.5).requires_grad_(True),
        "sh": (torch.randn(N, K, 3) * 0.2).requires_grad_(True),
        "deg": deg,
    }
    Rv, tv, fx, fy, cx, cy = torch.eye(3), torch.zeros(3), 100., 100., 48., 48.
    cam = (Rv, tv, fx, fy, cx, cy)

    u, v, zc, (ca, cb, cc) = project_general(P, Rv, tv, fx, fy, cx, cy)
    cam_center = -Rv.T @ tv
    dirs = P["mean"] - cam_center
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-9)
    col = sh_eval(P["sh"], dirs, deg)
    op = torch.sigmoid(P["op"])
    torch.manual_seed(1)
    gg = {k: torch.randn(N) for k in ("u", "v", "ca", "cb", "cc", "op")}
    gcol = torch.randn(N, 3)
    L = (gg["u"] * u + gg["v"] * v + gg["ca"] * ca + gg["cb"] * cb + gg["cc"] * cc + gg["op"] * op
         + (gcol * col).sum(-1)).sum()
    L.backward()
    ref = {k: P[k].grad.clone() for k in ("mean", "scale", "quat", "op", "sh")}

    up = dict(u=gg["u"], v=gg["v"], ca=gg["ca"], cb=gg["cb"], cc=gg["cc"], op=gg["op"],
              colR=gcol[:, 0], colG=gcol[:, 1], colB=gcol[:, 2])
    Pd = {k: (P[k].detach() if torch.is_tensor(P[k]) else P[k]) for k in P}
    got = project_backward(dev, Pd, cam, up)

    print(f"=== Stage D DEVICE backward grad-check (N={N}, deg={deg}) ===")
    allok = True
    for k in ("op", "sh", "mean", "scale", "quat"):
        r, gt = ref[k], got[k].double()
        rel = (gt - r).norm() / (r.norm() + 1e-12)
        ok = rel < 1e-2
        allok &= ok
        print(f"  {k:6s} rel={rel:.2e}  {'OK' if ok else 'FAIL'}")
    print("DEVICE_PROJ_BWD ALL OK" if allok else "DEVICE_PROJ_BWD SOME FAILED")
finally:
    ttnn.close_device(dev)
