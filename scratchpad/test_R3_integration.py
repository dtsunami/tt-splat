#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
R3 (silicon) — FULL FLOW integration of the tile-block raster into device_resident.

(1) PARITY @96px: force the blocked path (TT_FORCE_BLOCKED=1) vs the normal path, both base stage, same
    host sort -> loss curves must match tightly (proves the blocked fwd+bwd are correct inside the live loop).
(2) FULL FLOW @384px: device sort + auto-blocked raster (the path that used to FATAL on a dispatch core) ->
    training runs and loss DESCENDS. The 384px->1600px unlock, end-to-end.
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
    gs = dev.compute_with_storage_grid_size()
    print(f"worker grid {gs.x}x{gs.y}", flush=True)
    cams, xyz, rgb = _load_colmap(DATASET)
    torch.manual_seed(0)
    if xyz.shape[0] > 2000:
        idx = torch.randperm(xyz.shape[0])[:2000]; xyz, rgb = xyz[idx], rgb[idx]
    P = init_from_points(xyz, rgb, sh_degree=3)

    def load(LONG):
        views, targets = [], []
        for c in cams:
            p = DATASET / "images" / c[6]
            if not p.exists():
                continue
            img, s = load_image(str(p), LONG)
            views.append((c[0], c[1], c[2] * s, c[3] * s, c[4] * s, c[5] * s, c[6])); targets.append(img)
        return views, targets

    def run(binsort, blocked, views, targets, steps=12):
        os.environ["TT_DEVICE_BINSORT"] = binsort
        os.environ["TT_FORCE_BLOCKED"] = blocked
        os.environ["TT_FB_STAGE"] = "base"
        tr = DeviceResidentTrainer(dev, P, lr=LR, deg=3)
        return [tr.step(views[i % len(views)], targets[i % len(views)])[0] for i in range(steps)]

    # (1) parity @96px: normal vs forced-blocked
    v96, t96 = load(96)
    H, W, _ = t96[0].shape
    print(f"\n[parity @96px] {W}x{H} ({W//32}x{H//32} tiles)", flush=True)
    normal = run("0", "0", v96, t96)
    print("  normal-path done", flush=True)
    forced = run("0", "1", v96, t96)
    print("  forced-blocked done", flush=True)
    pdiff = max(abs(a - b) for a, b in zip(normal, forced))
    print(f"  normal {normal[0]:.5f}->{normal[-1]:.5f} | blocked {forced[0]:.5f}->{forced[-1]:.5f}")
    print(f"  max |normal-blocked| over curve = {pdiff:.2e}")

    # (2) full flow @384px: device sort + auto-blocked raster (used to FATAL)
    v384, t384 = load(384)
    H4, W4, _ = t384[0].shape
    print(f"\n[full flow @384px] {W4}x{H4} ({W4//32}x{H4//32} tiles) — device sort + auto tile-block", flush=True)
    full = run("1", "0", v384, t384)
    print(f"  loss {full[0]:.5f} -> {full[-1]:.5f}  ({(1-full[-1]/full[0])*100:+.1f}%)", flush=True)

    parity_ok = pdiff < 5e-3
    descends = full[-1] < full[0] * 0.97 and np.isfinite(full[-1])
    print(f"\n  parity(blocked==normal @96): {parity_ok} ({pdiff:.2e}) | 384px descends: {descends}")
    print(f"  -> {'R3 PASS' if parity_ok and descends else 'R3 CHECK'}")


if __name__ == "__main__":
    main()
