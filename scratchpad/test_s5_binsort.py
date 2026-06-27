#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
S5 (silicon) — device_binsort wired into the device-resident TRAINING loop, validated @384px.

Runs DeviceResidentTrainer on the real corgi at a grid-fitting resolution (~384px = 12x9 tiles, fits the
~110-core worker grid), twice: host bin_and_sort (TT_DEVICE_BINSORT=0) vs the on-device counting sort
(=1). Gates: (1) device-sort training LOSS DESCENDS; (2) the two loss curves track (parity — the only
difference is the S0-approved depth-BUCKET ordering vs exact lexsort). This is the nearest live training
run with the entire device bin/sort in the path, before the raster tile-block loop unlocks 1600px.
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
N_CAP = 4000
STEPS = 20
LR = {"mean": .01, "scale": .01, "quat": .01, "op": .02, "sh": .01}


def main():
    dev = _device()
    gs = dev.compute_with_storage_grid_size(); maxX, maxY = gs.x, gs.y
    LONG = min(384, maxX * 32, int(maxY * 32 / 0.75))                    # 4:3 -> H≈0.75*W; fit ntx<=maxX, nty<=maxY
    print(f"worker grid {maxX}x{maxY}; training @LONG={LONG}px")

    cams, xyz, rgb = _load_colmap(DATASET)
    torch.manual_seed(0)
    if xyz.shape[0] > N_CAP:
        idx = torch.randperm(xyz.shape[0])[:N_CAP]; xyz, rgb = xyz[idx], rgb[idx]
    P = init_from_points(xyz, rgb, sh_degree=3)

    views, targets = [], []
    img_dir = DATASET / "images"
    for c in cams:
        p = img_dir / c[6]
        if not p.exists():
            continue
        img, s = load_image(str(p), LONG)
        views.append((c[0], c[1], c[2] * s, c[3] * s, c[4] * s, c[5] * s, c[6])); targets.append(img)
    H, W, _ = targets[0].shape
    ntx, nty = (W + 31) // 32, (H + 31) // 32
    print(f"images {W}x{H} -> {ntx}x{nty}={ntx*nty} tiles (grid fits: {ntx <= maxX and nty <= maxY}); "
          f"{len(views)} views, N={P['mean'].shape[0]} Gaussians")

    def run(mode):
        os.environ["TT_DEVICE_BINSORT"] = mode
        tr = DeviceResidentTrainer(dev, P, lr=LR, deg=3)                 # clones P -> resident
        losses = []
        for i in range(STEPS):
            loss, _ = tr.step(views[i % len(views)], targets[i % len(views)])
            losses.append(loss)
        return losses

    print(f"\n[host  sort] running {STEPS} steps...")
    host = run("0")
    print(f"[device sort] running {STEPS} steps...")
    dvc = run("1")

    drop_h = (1 - host[-1] / host[0]) * 100
    drop_d = (1 - dvc[-1] / dvc[0]) * 100
    curve_diff = max(abs(h - d) for h, d in zip(host, dvc))
    print(f"\n  host   loss: {host[0]:.5f} -> {host[-1]:.5f}  ({drop_h:+.1f}%)")
    print(f"  device loss: {dvc[0]:.5f} -> {dvc[-1]:.5f}  ({drop_d:+.1f}%)")
    print(f"  max |host-device| loss over curve: {curve_diff:.2e}")
    descends = dvc[-1] < dvc[0] * 0.97
    parity = curve_diff < max(0.02 * host[0], 5e-3)                     # bucket-order tolerance
    print(f"  -> device-sort training DESCENDS: {descends} | host-parity: {parity}")
    print(f"  -> {'S5 PASS' if descends and parity else 'S5 CHECK'}")


if __name__ == "__main__":
    main()
