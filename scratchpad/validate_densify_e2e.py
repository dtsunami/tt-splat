#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: the REAL train_tt device-resident loop on the corgi scene with an early densify
schedule — proves auto-densify grows n_gaussians mid-run through the full dashboard contract."""
import os
os.environ["TT_DEVICE_RESIDENT"] = "1"          # must be set BEFORE importing train_tt (read at import)
os.environ["TT_SIZE"] = "64"
os.environ["TT_MAX_POINTS"] = "400"
os.environ["TT_DENSIFY"] = "1"
os.environ.setdefault("TT_METAL_HOME", "/home/starboy/tt-metal")
import sys
from pathlib import Path
ROOT = Path.home() / "tt-splat"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "server"))
SP = Path("/tmp/claude-1000/-home-starboy/e585199c-0ca1-4e6e-96ae-274e4ac91dee/scratchpad/densify_e2e")
from ttgs.config import TrainConfig
from ttgs.viewer.dashboard import TrainingController
import train_tt

cfg = TrainConfig()
cfg.iterations = 16
cfg.densify_from = 2; cfg.densify_until = 100; cfg.densify_every = 3   # densify early + often
cfg.dashboard_every = 2; cfg.sh_degree = 1
ctrl = TrainingController(output_dir=SP)
out = train_tt.run(ROOT / "work" / "scene", SP, cfg, None, dashboard=ctrl)

hist = ctrl.get_history()
ns = [h.get("n_gaussians") for h in hist if h.get("n_gaussians") is not None]
print("RESULT n_gaussians timeline:", ns)
print("RESULT grew:", bool(ns) and ns[-1] > ns[0], f"({ns[0] if ns else '?'} -> {ns[-1] if ns else '?'})")
print("RESULT ply_written:", Path(out).exists())
print("RESULT", "DENSIFY_E2E_OK" if ns and ns[-1] > ns[0] and Path(out).exists() else "FAIL")
