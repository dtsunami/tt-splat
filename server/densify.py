#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Adaptive density control (clone / split / prune) for tt-splat's 3D Gaussian params.

The verified M7 operators (docs/pathclear/train2d_densify.py, +18 dB) generalized from the 2D toy
params to the real {mean[N,3], scale[N,3](log), quat[N,4], op[N], sh[N,K,3]} set.  It's a dynamic,
data-dependent reshape of the Gaussian set — a host/general-purpose op (same caveat as bin/sort),
sitting on top of the device-proven render+backward.  Both train loops call this between steps.

  prune : sigmoid(op) <= prune_op                      -> drop (floaters/transparent)
  clone : high positional-grad + SMALL gaussian        -> duplicate, mean nudged (fill detail)
  split : high positional-grad + LARGE gaussian        -> 2 children, scale/1.6, mean jittered by extent

The densify signal `gpos` is the per-Gaussian accumulated POSITIONAL-gradient magnitude (screen-space
‖∂L/∂(u,v)‖ on the device-resident path; ‖∂L/∂mean‖ as a 3D proxy on the host path), averaged over the
interval since the last densify.  Threshold tau defaults to mean+std (scale-robust, as M7 proved); an
absolute grad_threshold overrides it.  Like M7 we rebuild the optimizer fresh after a reshape (proven),
so no optimizer-state threading here.
"""
from __future__ import annotations
import math
import torch

_KEYS = ("mean", "scale", "quat", "op", "sh")
_LOG16 = math.log(1.6)
_C0 = 0.28209479177387814        # SH band-0 constant (f_dc = (rgb-0.5)/C0)


def spawn_gaussians(P, means, colors, *, scale_log=None, op_prob=0.12):
    """The 'bubble gun': APPEND new Gaussians at world `means` [M,3] with RGB `colors` [M,3] in [0,1].
    New splats get an isotropic small scale, identity rotation, a base SH colour, and a modest opacity,
    then join the optimizable set and refine from there. Returns the grown P."""
    dt = P["mean"].dtype
    M = int(means.shape[0])
    K = P["sh"].shape[1]
    if scale_log is None:                                       # ~half the mean existing extent -> detail-sized
        scale_log = float(torch.log(torch.exp(P["scale"].float()).mean())) - _LOG16
    sh = torch.zeros(M, K, 3, dtype=dt)
    sh[:, 0, :] = (colors.to(dt).clamp(0, 1) - 0.5) / _C0       # band-0 = the sampled base colour
    new = {"mean": means.to(dt), "scale": torch.full((M, 3), float(scale_log), dtype=dt),
           "quat": torch.tensor([[1., 0, 0, 0]], dtype=dt).repeat(M, 1),
           "op": torch.full((M,), math.log(op_prob / (1 - op_prob)), dtype=dt), "sh": sh}
    out = {k: torch.cat([P[k], new[k].detach()], dim=0).detach().clone() for k in _KEYS}
    if "deg" in P:
        out["deg"] = P["deg"]
    return out


def densify_3d(P, gpos, *, grad_threshold=0.0, prune_op=0.005, prune_big_mult=20.0,
               split_world=0.0, clone_jitter=0.02, n_max=200_000,
               max_growth=0.5, min_keep_frac=0.05, min_log_scale=-9.0,
               guard=False, prune_floor_eps=0.5):
    """P: dict of torch tensors (leading dim N).  gpos: torch [N] (interval-averaged).
    Returns (newP, stats) where stats = {clone, split, prune, n_before, n_after[, diverged]}.

    SAFETY RAILS (a degenerate densify can prune to N=0 and SIGFPE the device, or shrink split children
    into NaN-producing conics — both observed on real runs):
      • DIVERGENCE: if any param is non-finite, return P UNCHANGED + {diverged:True}. Densify on NaN
        computes keep=all-False (sigmoid(NaN)>thr is False) and nukes the whole model → empty-tensor crash.
      • KEEP-FLOOR: never prune below min_keep_frac·N in one cycle (keep the top-opacity instead).
      • GROWTH-CAP: at most max_growth·N new Gaussians per cycle (keep the highest-grad candidates).
      • SCALE-FLOOR: split children clamp log-scale ≥ min_log_scale so repeated splits can't make
        degenerate (conic-overflow → NaN) Gaussians.
    Plus prune_big_mult>0 prunes fog-blobs whose world size exceeds that multiple of the median."""
    keys = [k for k in _KEYS if k in P]
    with torch.no_grad():
        N = P["mean"].shape[0]
        op_raw = P["op"].reshape(N).float()
        if not (torch.isfinite(op_raw).all() and torch.isfinite(P["scale"]).all()
                and torch.isfinite(P["mean"]).all()):          # DIVERGENCE guard — bail UNCHANGED, don't nuke
            newP = {k: P[k].detach().clone() for k in keys}
            if "deg" in P:
                newP["deg"] = P["deg"]
            return newP, {"clone": 0, "split": 0, "prune": 0, "n_before": N, "n_after": N, "diverged": True}

        op = torch.sigmoid(op_raw)
        scale = P["scale"]
        world = torch.exp(scale.float())
        world = world.max(dim=1).values if world.dim() == 2 else world      # largest axis (world units)
        med = float(world.median())
        keep = op > prune_op                                  # PRUNE low-opacity floaters
        if prune_big_mult and prune_big_mult > 0 and med > 0:  # + PRUNE over-large fog blobs (scale safeguard)
            keep = keep & (world < float(prune_big_mult) * med)
        if guard:                                             # #2: PRUNE fully-collapsed splats — even the largest axis is
            maxlog = scale.max(dim=1).values if scale.dim() == 2 else scale.reshape(N)   # pinned at the scale floor ⇒
            keep = keep & (maxlog > (float(min_log_scale) + float(prune_floor_eps)))     # degenerate sub-pixel (conic → NaN)
        if int(keep.sum()) < max(1, int(min_keep_frac * N)):  # KEEP-FLOOR — never collapse the model
            topk = torch.topk(op, max(1, min(N, int(min_keep_frac * N)))).indices
            keep = torch.zeros(N, dtype=torch.bool); keep[topk] = True

        tau = float(grad_threshold) if grad_threshold and grad_threshold > 0 else \
            float(gpos.mean() + gpos.std())
        big = world > (float(split_world) if split_world and split_world > 0 else med * 1.5)
        grad_hi = gpos.reshape(N) > tau

        do_clone = keep & grad_hi & (~big)                    # CLONE small high-grad
        do_split = keep & grad_hi & big                       # SPLIT large high-grad
        if max_growth and max_growth > 0:                     # GROWTH-CAP — bound new Gaussians per cycle
            cand = do_clone | do_split
            budget = int(max_growth * N)
            if int(cand.sum()) > budget:
                ci = cand.nonzero(as_tuple=True)[0]
                drop = ci[torch.argsort(gpos.reshape(N)[ci], descending=True)[budget:]]
                m = torch.ones(N, dtype=torch.bool); m[drop] = False
                do_clone = do_clone & m; do_split = do_split & m
        survive = keep & (~do_split)                          # clone keeps original; split consumes it

        sidx = survive.nonzero(as_tuple=True)[0]
        cidx = do_clone.nonzero(as_tuple=True)[0]
        spidx = do_split.nonzero(as_tuple=True)[0]

        newP = {}
        for k in keys:
            parts = [P[k][sidx]]
            if cidx.numel():
                v = P[k][cidx].clone()
                if k == "mean":                               # nudge clones off the parent
                    v = v + torch.randn_like(v) * (clone_jitter * world[cidx].unsqueeze(-1))
                parts.append(v)
            if spidx.numel():
                ext = torch.exp(scale[spidx].float())         # sample children within the parent's extent
                for _ in range(2):
                    v = P[k][spidx].clone()
                    if k == "scale":
                        v = (v - _LOG16).clamp(min=min_log_scale)   # /1.6 in log-space; SCALE-FLOOR vs degenerate splits
                    elif k == "mean":
                        v = v + torch.randn_like(v.float()).to(v.dtype) * ext
                    parts.append(v)
            newP[k] = torch.cat(parts, dim=0)

        n_after = newP["mean"].shape[0]
        if n_after > n_max:                                   # cap (keep the strongest by opacity)
            keep_op = torch.sigmoid(newP["op"].reshape(n_after).float())
            idx = torch.topk(keep_op, n_max).indices
            newP = {k: newP[k][idx] for k in keys}
            n_after = n_max

        for k in keys:                                        # detach + contiguous for re-leafing / device upload
            newP[k] = newP[k].detach().clone()
        if guard:                                             # #2: keep quaternions UNIT — a zero-norm child quat makes
            q = newP["quat"].float()                          # recip(|q|)=inf in the Stage D backward (the qnorm divide)
            qn = q.norm(dim=1, keepdim=True)
            ident = torch.zeros_like(q); ident[:, 0] = 1.0    # degenerate → identity rotation
            newP["quat"] = torch.where(qn > 1e-8, q / qn.clamp_min(1e-8), ident).to(P["quat"].dtype)
        if "deg" in P:
            newP["deg"] = P["deg"]
        stats = {"clone": int(do_clone.sum()), "split": int(do_split.sum()),
                 "prune": int((~keep).sum()), "n_before": N, "n_after": n_after}
        return newP, stats
