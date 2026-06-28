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
               split_world=0.0, clone_jitter=0.02, n_max=200_000):
    """P: dict of torch tensors (leading dim N).  gpos: torch [N] (interval-averaged).
    Returns (newP, stats) where stats = {clone, split, prune, n_before, n_after}.
    Safeguard: prune_big_mult>0 also prunes fog-blobs whose world size exceeds that multiple of the
    median (the floater/over-large cull that keeps densification from diverging)."""
    keys = [k for k in _KEYS if k in P]
    with torch.no_grad():
        N = P["mean"].shape[0]
        op = torch.sigmoid(P["op"].reshape(N).float())

        scale = P["scale"]
        world = torch.exp(scale.float())
        world = world.max(dim=1).values if world.dim() == 2 else world      # largest axis (world units)
        keep = op > prune_op                                  # PRUNE low-opacity floaters
        if prune_big_mult and prune_big_mult > 0:             # + PRUNE over-large fog blobs (scale safeguard)
            keep = keep & (world < float(prune_big_mult) * float(world.median()))
        tau = float(grad_threshold) if grad_threshold and grad_threshold > 0 else \
            float(gpos.mean() + gpos.std())
        big = world > (float(split_world) if split_world and split_world > 0
                       else float(world.median()) * 1.5)
        grad_hi = gpos.reshape(N) > tau

        do_clone = keep & grad_hi & (~big)                    # CLONE small high-grad
        do_split = keep & grad_hi & big                       # SPLIT large high-grad
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
                        v = v - _LOG16                         # /1.6 in log-space
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
        if "deg" in P:
            newP["deg"] = P["deg"]
        stats = {"clone": int(do_clone.sum()), "split": int(do_split.sum()),
                 "prune": int((~keep).sum()), "n_before": N, "n_after": n_after}
        return newP, stats
