#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Device-resident Adam (Stage C of the device-resident training loop). Params + m/v live on the Blackhole
as ttnn fp32 tensors; the update is M0's proven primitive sequence (gaussian_fit.py) — `moreh.adam` is
bf16-only AND numerically wrong in this build, so we roll it from mul/add/square/sqrt/div/sub. Each param
group updates in ONE batched elementwise pass over its whole [N,...] tensor (no per-Gaussian loop).

  opt = DeviceAdam(dev, {"mean": t, ...}, lr={"mean": .01, ...})
  opt.step({"mean": grad_t, ...})        # grads as host torch (or ttnn) tensors
  P = opt.params()                       # -> dict of host torch tensors (only when you need them)
"""
from __future__ import annotations
import torch
import ttnn


def _dt(dev, t):
    return ttnn.from_torch(t.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)


class DeviceAdam:
    def __init__(self, dev, init: dict, lr: dict, b1=0.9, b2=0.999, eps=1e-8):
        self.dev, self.lr, self.b1, self.b2, self.eps = dev, dict(lr), b1, b2, eps
        self.keys = list(init.keys())
        self.p = {k: _dt(dev, v) for k, v in init.items()}
        self.m = {k: _dt(dev, torch.zeros_like(init[k])) for k in self.keys}
        self.v = {k: _dt(dev, torch.zeros_like(init[k])) for k in self.keys}
        self.t = 0

    def step(self, grads: dict):
        self.t += 1
        bc1 = 1.0 / (1.0 - self.b1 ** self.t)
        bc2 = 1.0 / (1.0 - self.b2 ** self.t)
        for k in self.keys:
            g = grads[k]
            if isinstance(g, torch.Tensor):
                g = _dt(self.dev, g)
            self.m[k] = ttnn.add(ttnn.mul(self.m[k], self.b1), ttnn.mul(g, 1.0 - self.b1))      # m=b1 m+(1-b1)g
            self.v[k] = ttnn.add(ttnn.mul(self.v[k], self.b2), ttnn.mul(ttnn.square(g), 1.0 - self.b2))  # v
            mhat = ttnn.mul(self.m[k], bc1)
            vhat = ttnn.mul(self.v[k], bc2)
            upd = ttnn.mul(ttnn.div(mhat, ttnn.add(ttnn.sqrt(vhat), self.eps)), self.lr[k])
            self.p[k] = ttnn.sub(self.p[k], upd)                                                 # p -= lr*mhat/(√vhat+eps)

    def params(self) -> dict:
        return {k: ttnn.to_torch(self.p[k]) for k in self.keys}

    def set_param(self, k, t):    # for densify/prune realloc (Stage E)
        self.p[k] = _dt(self.dev, t)
        self.m[k] = _dt(self.dev, torch.zeros_like(t))
        self.v[k] = _dt(self.dev, torch.zeros_like(t))
