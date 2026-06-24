"""Stage 3: 3D Gaussian Splatting training via gsplat + PyTorch.

gsplat is a pure-Python 3DGS library that dispatches through PyTorch, so it
runs natively on Intel Arc XPU, NVIDIA CUDA, and CPU — no external binary
required.

Backend priority (matches ttgs backend detection):
  xpu      → torch.device("xpu")   Intel Arc via oneAPI / PyTorch XPU
  cuda     → torch.device("cuda")  NVIDIA
  directml → torch.device("cpu")   torch-directml not yet supported by gsplat
  cpu      → torch.device("cpu")   Always available; ~10-20× slower than GPU
"""

from __future__ import annotations

from pathlib import Path

from ttgs.backend.detect import BackendInfo
from ttgs.config import TrainConfig


def run(
    dataset_dir: Path,
    output_dir: Path,
    cfg: TrainConfig,
    backend: BackendInfo,
    resume: bool = False,
    viewer_port: int | None = None,
    dashboard=None,
    masks_dir: Path | None = None,
    excluded: set[str] | None = None,
) -> Path:
    """Run gsplat 3DGS training.

    Args:
        dataset_dir:  COLMAP undistorted dataset directory (contains sparse/ + images/).
        output_dir:   Where to write splat.ply and checkpoints.
        cfg:          TrainConfig parameters.
        backend:      Detected/selected compute backend.
        resume:       If True, resume from the last checkpoint in output_dir.
        viewer_port:  If set, start a live viser viewer on this port.
        dashboard:    TrainingController for interactive controls.
        masks_dir:    Directory of per-image mask PNGs (stem must match image stem).
        excluded:     Set of image filenames to skip during training.

    Returns:
        Path to the output splat.ply file.
    """
    from ttgs.stages.train_gsplat import run as gsplat_run

    return gsplat_run(
        dataset_dir, output_dir, cfg, backend,
        resume=resume, viewer_port=viewer_port, dashboard=dashboard,
        masks_dir=masks_dir, excluded=excluded,
    )
