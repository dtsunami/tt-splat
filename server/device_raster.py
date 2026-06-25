#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Phase-2a device-gradient training bridge: make the on-device 3DGS render DIFFERENTIABLE so training
gradients run on the Blackhole.

Reuses M16's proven full-image ttnn-op forward + reverse-pass backward (device_train_loop.py:50-79),
generalized to any (H,W,N) and wrapped in a torch.autograd.Function. The 2D Gaussian params (centers,
conic, opacity, colour) come from the HOST projection (train_real.project_general + sh_eval + sigmoid)
as differentiable torch tensors — so torch autograd carries the device grads on through projection/SH
back to the 3D params P, and the existing host Adam steps them.

Correctness-first: this is O(N) ttnn dispatches (slow → small scale only). The fused fwd+bwd SFPU
kernel (Phase 2b) swaps in behind this same DeviceRaster interface for real-scale max perf.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "pathclear"))
import ttnn                                            # noqa: E402
from train_real import project_general, sh_eval        # noqa: E402
from render_device import _device                      # noqa: E402  persistent device handle + preflight

KEYS = ("cx", "cy", "a", "b", "c", "op", "col")
_CTX: dict = {}                                        # (H,W) -> cached pixel-coord tensors + helpers


def _ctx(dev, H, W):
    key = (H, W)
    if key not in _CTX:
        ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        dt = lambda t: ttnn.from_torch(t.reshape(1, 1, H, W).float(), dtype=ttnn.float32,
                                       layout=ttnn.TILE_LAYOUT, device=dev)
        _CTX[key] = dict(H=H, W=W, dt=dt,
                         PX=dt(jj.float()), PY=dt(ii.float()),
                         dsum=lambda t: float(ttnn.to_torch(ttnn.sum(t)).flatten()[0]))
    return _CTX[key]


def _fwd(ct, p, order, store):
    """Front-to-back ttnn blend over `order`. Returns (C [1,1,H,W] device, rec). Mirrors M16 forward."""
    PX, PY, dt = ct["PX"], ct["PY"], ct["dt"]
    C = dt(torch.zeros(ct["H"], ct["W"])); T = dt(torch.ones(ct["H"], ct["W"])); rec = []
    for i in order:
        cx, cy, a, b, cc, op, col = (float(p[k][i]) for k in KEYS)
        dx = ttnn.sub(PX, cx); dy = ttnn.sub(PY, cy)
        power = ttnn.add(ttnn.add(ttnn.mul(ttnn.mul(dx, dx), a), ttnn.mul(ttnn.mul(dx, dy), 2 * b)),
                         ttnn.mul(ttnn.mul(dy, dy), cc))
        gexp = ttnn.exp(ttnn.mul(power, -0.5)); alpha = ttnn.mul(gexp, op)
        if store:
            rec.append((i, dx, dy, gexp, alpha, T))
        C = ttnn.add(C, ttnn.mul(ttnn.mul(T, alpha), col))
        T = ttnn.mul(T, ttnn.add(ttnn.mul(alpha, -1.0), 1.0))
    return C, rec


def _bwd(ct, p, rec, dLdC, N):
    """Reverse pass: per-Gaussian grads {cx,cy,a,b,c,op,col} from upstream dL/dC. Mirrors M16 backward."""
    dt, dsum = ct["dt"], ct["dsum"]
    grads = {k: np.zeros(N, np.float64) for k in KEYS}
    S = dt(torch.zeros(ct["H"], ct["W"]))
    for (i, dx, dy, gexp, alpha, Ti) in reversed(rec):
        col, a, b, cc = (float(p[k][i]) for k in ("col", "a", "b", "c"))
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
        grads["cy"][i] = dsum(ttnn.mul(base, ttnn.mul(ttnn.add(ttnn.mul(dx, b), ttnn.mul(dy, cc)), -2.0)))
        S = ttnn.add(S, ttnn.mul(w, col))
    return grads


class DeviceRaster(torch.autograd.Function):
    """Differentiable on-device RGB rasterizer. forward = 3 ttnn blend passes (one per channel, shared
    geometry); backward = M16 reverse pass per channel, geometry grads summed across channels."""

    @staticmethod
    def forward(ctx, cx, cy, a, b, c, op, colR, colG, colB, order, H, W):
        dev = _device()
        ct = _ctx(dev, H, W)
        geom = {k: t.detach().cpu().numpy().astype(np.float64)
                for k, t in zip(("cx", "cy", "a", "b", "c", "op"), (cx, cy, a, b, c, op))}
        cols = [t.detach().cpu().numpy().astype(np.float64) for t in (colR, colG, colB)]
        order = [int(i) for i in order]
        imgs, recs = [], []
        for colk in cols:
            p = dict(geom); p["col"] = colk
            C, rec = _fwd(ct, p, order, store=True)
            imgs.append(ttnn.to_torch(C).reshape(H, W))
            recs.append(rec)
        ctx.ct, ctx.geom, ctx.cols, ctx.order, ctx.recs = ct, geom, cols, order, recs
        ctx.N, ctx.in_dtype = len(geom["cx"]), cx.dtype
        return torch.stack(imgs, dim=-1).to(cx.dtype)         # [H,W,3]

    @staticmethod
    def backward(ctx, grad_img):                              # grad_img [H,W,3]
        ct, N = ctx.ct, ctx.N
        gi = grad_img.detach().cpu().numpy().astype(np.float64)
        geomg = {k: np.zeros(N, np.float64) for k in ("cx", "cy", "a", "b", "c", "op")}
        colg = [np.zeros(N, np.float64) for _ in range(3)]
        for k in range(3):
            p = dict(ctx.geom); p["col"] = ctx.cols[k]
            dLdC = ct["dt"](torch.from_numpy(gi[:, :, k].copy()))
            g = _bwd(ct, p, ctx.recs[k], dLdC, N)
            for key in ("cx", "cy", "a", "b", "c", "op"):
                geomg[key] += g[key]
            colg[k] = g["col"]
        tt = lambda arr: torch.from_numpy(arr).to(ctx.in_dtype)
        return (tt(geomg["cx"]), tt(geomg["cy"]), tt(geomg["a"]), tt(geomg["b"]), tt(geomg["c"]),
                tt(geomg["op"]), tt(colg[0]), tt(colg[1]), tt(colg[2]), None, None, None)


def render_train(P, cam, H, W) -> torch.Tensor:
    """Differentiable on-device render of P from camera `cam` -> [H,W,3] torch (grad flows to P).
    Drop-in for train_real.render in the training loss path when TT_DEVICE_TRAIN is on."""
    Rv, tv, fx, fy, ppx, ppy = cam[:6]
    u, v, zc, (ca, cb, cc) = project_general(P, Rv, tv, fx, fy, ppx, ppy)   # differentiable (float64)
    cam_center = -Rv.T @ tv
    dirs = P["mean"] - cam_center
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-9)
    col = sh_eval(P["sh"], dirs, P["deg"])                 # [N,3]
    op = torch.sigmoid(P["op"])
    zc_d = zc.detach()
    order = [int(i) for i in torch.argsort(zc_d).tolist() if float(zc_d[i]) > 1e-4]   # cull behind-cam
    f = lambda t: t.float()
    return DeviceRaster.apply(f(u), f(v), f(ca), f(cb), f(cc), f(op),
                              f(col[:, 0]), f(col[:, 1]), f(col[:, 2]), order, H, W)
