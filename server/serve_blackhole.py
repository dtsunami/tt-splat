#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Stand up the ttgs frontend/server on Blackhole, driving the Tenstorrent training backend.

Uses tt-splat's vendored ttgs FastAPI dashboard + controllers (forked from arcgs, lives in
../ttgs) and routes the training stage to our tt-splat pipeline (server/train_tt.py).

  python server/serve_blackhole.py --dataset work/scene --output work/tt_out --port 7860
  # then open http://localhost:7860/training

Env knobs for the host-reference render budget: TT_MAX_POINTS, TT_SIZE.
"""
import argparse, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # vendored ttgs package
sys.path.insert(0, str(Path(__file__).resolve().parent))         # train_tt

from ttgs.config import load as load_config
from ttgs.backend.detect import Backend, BackendInfo
from ttgs.viewer.dashboard import DashboardServer
from ttgs.viewer.pipeline_controller import PipelineController
import train_tt


def tt_backend() -> BackendInfo:
    """Probe Tenstorrent Blackhole WITHOUT importing ttnn — importing it initializes the cluster
    and would contend with any other device user (e.g. the comfy SDXL server on board p150).
    The host-reference train loop never opens the device; the device kernels do so explicitly."""
    import importlib.util
    dev = os.path.exists("/dev/tenstorrent/0")
    have_ttnn = importlib.util.find_spec("ttnn") is not None    # installed? — does NOT import/init
    note = ("device present + ttnn installed (host-reference loop; device kernels validated separately)"
            if dev and have_ttnn else "no /dev/tenstorrent or ttnn — host fallback")
    return BackendInfo(Backend.CPU, "Tenstorrent Blackhole", None, True, note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="COLMAP dataset dir (sparse/0 + images/)")
    ap.add_argument("--output", default="work/tt_out")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    dataset, output = Path(a.dataset), Path(a.output)

    cfg = load_config(None)
    cfg.train.iterations = a.steps
    cfg.train.dashboard_every = 5
    backend = tt_backend()
    print(f"backend: {backend}")

    frames = dataset / "images" if (dataset / "images").exists() else dataset
    pc = PipelineController(output_dir=output, cfg=cfg, backend=backend, frames_dir=frames)
    db = DashboardServer(pc, port=a.port)
    print(f"dashboard: http://localhost:{a.port}/training   (Ctrl-C to stop)")
    db.run_training(
        train_tt.run,
        dataset, output, cfg.train, backend,
        dashboard=pc.training, masks_dir=pc.masks_dir,
    )


if __name__ == "__main__":
    main()
