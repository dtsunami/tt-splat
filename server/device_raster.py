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
import sys, os
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "pathclear"))
import ttnn                                            # noqa: E402
from train_real import project_general, sh_eval        # noqa: E402
from render_device import _device, _resources, _raster_channel, _raster_rgb   # noqa: E402  M14 forward (image + final-T)
from bin_sort import bin_and_sort                       # noqa: E402  M6 per-tile cull + depth sort
import sfpu_raster_scaled as M14                         # noqa: E402  TS, B

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
    return C, rec, T          # T = final transmittance (fused backward needs it)


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
    """On-device RGB rasterizer. FORWARD = M14 fused multi-tile CULLED rasterizer (image + final-T,
    via M6 per-tile depth-sorted lists). BACKWARD = the fused backward kernel run per tile over the
    SAME culled lists, host-accumulated. Grads are w.r.t. the 2D Gaussian params; the caller backprops
    them through projection/SH to P. Replaces the old O(N) ttnn-op fwd/bwd (Phase 2b items 1+2)."""

    @staticmethod
    def forward(ctx, cx, cy, a, b, c, op, colR, colG, colB, zc, H, W):
        dev = _device()
        np64 = lambda t: t.detach().cpu().numpy().astype(np.float64)
        cxn, cyn, an, bn, cn, opn, zcn = map(np64, (cx, cy, a, b, c, op, zc))
        cols = [np64(colR), np64(colG), np64(colB)]
        N = cxn.shape[0]
        TS = M14.TS
        Wp, Hp = ((W + TS - 1)//TS)*TS, ((H + TS - 1)//TS)*TS
        valid = zcn > 1e-4                                    # cull behind-camera
        vidx = np.nonzero(valid)[0]
        ctx.N, ctx.in_dtype = N, cx.dtype
        if vidx.size == 0:
            ctx.empty = True
            return torch.zeros(H, W, 3, dtype=cx.dtype)
        cxv, cyv, av, bv, cv, opv, zcv = (arr[valid] for arr in (cxn, cyn, an, bn, cn, opn, zcn))
        colv = [cl[valid] for cl in cols]
        detc = av*cv - bv*bv; detc = np.where(np.abs(detc) < 1e-12, 1e-12, detc)
        var_x = np.clip(cv/detc, 0.25, None); var_y = np.clip(av/detc, 0.25, None)   # 2D variance from conic
        res = _resources(dev, Wp, Hp)
        s_gid, _st, ranges, ntx, nty, _tot = bin_and_sort(cxv, cyv, var_x, var_y, zcv, Wp, Hp, ts=TS)  # M6
        tile_lists = [s_gid[ranges[t, 0]:ranges[t, 1]].tolist() for t in range(ntx*nty)]
        maxc = max((len(l) for l in tile_lists), default=0)
        nbatch = (maxc + M14.B - 1)//M14.B
        chans, Tfin = _raster_rgb(dev, res, tile_lists, ntx, nbatch, cxv, cyv, av, bv, cv, opv, colv,
                                  Wp, Hp, want_T=True)        # M14 fused forward (banded R/G/B); final-T geometry-only
        img = np.stack(chans, axis=-1)[:H, :W, :]
        ctx.empty = False
        ctx.tile_lists, ctx.ntx, ctx.Tfin, ctx.vidx = tile_lists, ntx, Tfin, vidx
        ctx.cxv, ctx.cyv, ctx.av, ctx.bv, ctx.cv, ctx.opv, ctx.colv = cxv, cyv, av, bv, cv, opv, colv
        return torch.from_numpy(np.clip(img, 0.0, 1.0)).to(cx.dtype)

    @staticmethod
    def backward(ctx, grad_img):                              # grad_img [H,W,3]
        N = ctx.N
        geomg = {k: np.zeros(N, np.float64) for k in ("cx", "cy", "a", "b", "c", "op")}
        colg = [np.zeros(N, np.float64) for _ in range(3)]
        if not ctx.empty:
            from fused_backward import fused_backward_grid
            dev = _device()
            gi = grad_img.detach().cpu().numpy().astype(np.float64)
            tl, ntx, Tfin, vidx = ctx.tile_lists, ctx.ntx, ctx.Tfin, ctx.vidx
            Hp, Wp = Tfin.shape                               # padded tile grid
            if gi.shape[0] != Hp or gi.shape[1] != Wp:        # image may be non-32-multiple -> zero-pad
                g2 = np.zeros((Hp, Wp, 3), np.float64); g2[:gi.shape[0], :gi.shape[1], :] = gi; gi = g2
            cxv, cyv, av, bv, cv, opv, colv = ctx.cxv, ctx.cyv, ctx.av, ctx.bv, ctx.cv, ctx.opv, ctx.colv
            # STAGE A: grid-sharded backward — ONE dispatch per (chunk,channel), all tiles parallel
            gv, cgv = fused_backward_grid(dev, cxv, cyv, av, bv, cv, opv, colv, tl, ntx, Hp // 32, Wp, Hp, gi, Tfin,
                                          stage=os.environ.get("TT_FB_STAGE", "s3"))   # Stage 3 default
            for key in ("cx", "cy", "a", "b", "c", "op"):
                geomg[key][vidx] = gv[key]                    # valid-subset -> original Gaussian index
            for k in range(3):
                colg[k][vidx] = cgv[k]
        tt = lambda arr: torch.from_numpy(arr).to(ctx.in_dtype)
        return (tt(geomg["cx"]), tt(geomg["cy"]), tt(geomg["a"]), tt(geomg["b"]), tt(geomg["c"]),
                tt(geomg["op"]), tt(colg[0]), tt(colg[1]), tt(colg[2]), None, None, None)


def render_train(P, cam, H, W) -> torch.Tensor:
    """Differentiable on-device render of P from camera `cam` -> [H,W,3] torch (grad flows to P).
    M14 fused culled forward + fused culled backward. Drop-in for train_real.render under TT_DEVICE_TRAIN."""
    Rv, tv, fx, fy, ppx, ppy = cam[:6]
    u, v, zc, (ca, cb, cc) = project_general(P, Rv, tv, fx, fy, ppx, ppy)   # differentiable (float64)
    cam_center = -Rv.T @ tv
    dirs = P["mean"] - cam_center
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-9)
    col = sh_eval(P["sh"], dirs, P["deg"])                 # [N,3]
    op = torch.sigmoid(P["op"])
    f = lambda t: t.float()
    return DeviceRaster.apply(f(u), f(v), f(ca), f(cb), f(cc), f(op),
                              f(col[:, 0]), f(col[:, 1]), f(col[:, 2]), f(zc), H, W)
