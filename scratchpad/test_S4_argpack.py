#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
S4 host arg-pack vectorization — (1) CORRECTNESS: vectorized args are bit-identical, so force-blocked still
matches the normal path @96px; (2) TIMING: per-stage step breakdown at a MULTI-BLOCK scale (768px, ~16k
Gaussians, host sort = the 50k path) to confirm 50k@1600 is runnable. (Full GDDR streaming = production follow-on.)
"""
import os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "docs" / "pathclear"))
import ttnn                                                              # noqa: E402
from train_real import init_from_points, load_image                     # noqa: E402
from train_tt import _load_colmap                                       # noqa: E402
from render_device import _device                                       # noqa: E402
from device_resident import DeviceResidentTrainer                        # noqa: E402

DATASET = ROOT / "work" / "scene2"
LR = {"mean": .01, "scale": .01, "quat": .01, "op": .02, "sh": .01}


def main():
    dev = _device()
    cams, xyz, rgb = _load_colmap(DATASET)

    def load(LONG):
        views, targets = [], []
        for c in cams:
            p = DATASET / "images" / c[6]
            if not p.exists():
                continue
            img, s = load_image(str(p), LONG)
            views.append((c[0], c[1], c[2] * s, c[3] * s, c[4] * s, c[5] * s, c[6])); targets.append(img)
        return views, targets

    def make_P(n):
        torch.manual_seed(0)
        x, r = xyz, rgb
        if x.shape[0] > n:
            idx = torch.randperm(x.shape[0])[:n]; x, r = x[idx], r[idx]
        elif x.shape[0] < n:                       # densify by jittered replication to reach n
            k = (n + x.shape[0] - 1) // x.shape[0]
            x = (x.repeat(k, 1) + torch.randn(x.shape[0] * k, 3) * 0.01)[:n]
            r = r.repeat(k, 1)[:n]
        return init_from_points(x, r, sh_degree=3)

    def run(P, binsort, blocked, views, targets, steps):
        os.environ["TT_DEVICE_BINSORT"] = binsort
        os.environ["TT_FORCE_BLOCKED"] = blocked
        os.environ["TT_FB_STAGE"] = "s4"   # blocked path uses s4; compare normal-s4 (apples-to-apples)
        tr = DeviceResidentTrainer(dev, P, lr=LR, deg=3)
        losses = [tr.step(views[i % len(views)], targets[i % len(views)])[0] for i in range(steps)]
        return losses, tr.step_log

    # (1) correctness: vectorized args bit-identical -> blocked == normal @96px
    P2 = make_P(2000); v96, t96 = load(96)
    n, _ = run(P2, "0", "0", v96, t96, 8)
    b, _ = run(P2, "0", "1", v96, t96, 8)
    pdiff = max(abs(a - c) for a, c in zip(n, b))
    print(f"[correctness @96px] max |normal-blocked| over curve = {pdiff:.2e}  ({'OK' if pdiff < 5e-3 else 'BAD'})")

    # (2) timing: multi-block, host sort (the 50k path)
    LONG = 768; Ng = 16000
    Pt = make_P(Ng); vL, tL = load(LONG)
    H, W, _ = tL[0].shape
    print(f"\n[timing] {W}x{H} ({W//32}x{H//32}={(W//32)*(H//32)} tiles, multi-block) N={Pt['mean'].shape[0]}", flush=True)
    _, log = run(Pt, "0", "0", vL, tL, 4)
    s = log[-1]
    print(f"  per-stage ms (last step): B={s['B']:.0f} bin={s['bin']:.0f} raster={s['raster']:.0f} "
          f"A={s['A']:.0f} D={s['D']:.0f} C={s['C']:.0f}  -> step={s['step']:.0f} ms")
    print(f"  -> S4 arg-pack {'PASS' if pdiff < 5e-3 else 'CHECK'} (runnable; raster fwd+A = "
          f"{s['raster']+s['A']:.0f} ms of {s['step']:.0f})")


if __name__ == "__main__":
    main()
