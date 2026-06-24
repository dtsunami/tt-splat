"""Stage 2: Structure from Motion via COLMAP.

Runs the standard COLMAP pipeline:
  feature_extractor → matcher → mapper → image_undistorter

Output layout (COLMAP format, ready for OpenSplat):
  <output_dir>/
    images/           ← symlink or copy of input images
    sparse/0/
      cameras.bin
      images.bin
      points3D.bin
    (after undistort):
    images_undistorted/
    sparse_undistorted/
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ttgs.config import SfmConfig
from ttgs.tools import require_tool

console = Console()


def _run_colmap(args: list[str], colmap: str, label: str, env: dict | None = None) -> None:
    """Run a COLMAP sub-command, streaming stderr to the console."""
    cmd = [colmap] + args
    console.print(f"[dim]→ {' '.join(cmd[:6])} ...[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]sfm[/] {label}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("", total=None)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        output_lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            output_lines.append(line)

    proc.wait()
    if proc.returncode != 0:
        # Print last 30 lines of output for diagnostics
        tail = "".join(output_lines[-30:])
        raise RuntimeError(f"COLMAP {label} failed (exit {proc.returncode}):\n{tail}")


def run(
    images_dir: Path,
    output_dir: Path,
    cfg: SfmConfig,
    colmap_bin: str | None = None,
    masks_dir: Path | None = None,
) -> Path:
    """Run the full COLMAP SfM pipeline.

    Args:
        images_dir:  Directory containing input images.
        output_dir:  Root output directory for the COLMAP workspace.
        cfg:         SfmConfig parameters.
        colmap_bin:  Explicit path to the COLMAP executable (auto-detected if None).
        masks_dir:   Optional directory of grayscale mask PNGs (same stems as
                     images).  White=include, Black=exclude from feature detection.
                     Passed as --ImageReader.mask_path to COLMAP.

    Returns:
        Path to the final dataset directory (compatible with OpenSplat).
    """
    colmap = colmap_bin or require_tool(
        "colmap",
        "Download from https://github.com/colmap/colmap/releases\n"
        "Then set COLMAP_PATH in your .env or add the bin directory to PATH.",
    )

    db_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_0 = sparse_dir / "0"
    images_link = output_dir / "images"
    undistorted = output_dir / "undistorted"

    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # COLMAP's Qt6 platform plugins live in plugins/platforms/ one level above bin/.
    # When launched as a subprocess Qt can't find them without an explicit path.
    colmap_env = os.environ.copy()
    plugins_dir = Path(colmap).parent.parent / "plugins" / "platforms"
    if plugins_dir.is_dir():
        colmap_env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins_dir)

    # Determine the path to pass to COLMAP as --image_path.
    # On Linux/Mac we symlink images into the workspace for a self-contained layout.
    # On Windows, junctions don't work on network drives (mapped or UNC), so we
    # skip the link and point COLMAP directly at the source directory.
    if sys.platform != "win32" and not images_link.exists():
        images_link.symlink_to(images_dir.resolve())

    colmap_images_path = images_link if images_link.exists() else images_dir

    # 1 — Feature extraction
    fe_args = [
        "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(colmap_images_path),
        "--ImageReader.single_camera", "1" if cfg.single_camera else "0",
        "--ImageReader.camera_model", cfg.camera_model,
    ]
    if masks_dir is not None and masks_dir.exists():
        fe_args += ["--ImageReader.mask_path", str(masks_dir)]
        console.print(f"[dim]sfm masks: {masks_dir}[/]")
    _run_colmap(fe_args, colmap, "feature extraction", env=colmap_env)

    # 2 — Feature matching
    matcher_cmd = {
        "exhaustive": "exhaustive_matcher",
        "sequential": "sequential_matcher",
        "vocab_tree": "vocab_tree_matcher",
    }.get(cfg.matcher, "exhaustive_matcher")

    _run_colmap(
        [matcher_cmd, "--database_path", str(db_path)],
        colmap,
        f"matching ({cfg.matcher})",
        env=colmap_env,
    )

    # 3 — Sparse reconstruction (mapper)
    _run_colmap(
        [
            "mapper",
            "--database_path", str(db_path),
            "--image_path", str(colmap_images_path),
            "--output_path", str(sparse_dir),
        ],
        colmap,
        "mapping",
        env=colmap_env,
    )

    if not sparse_0.exists():
        raise RuntimeError(
            "COLMAP mapper produced no reconstruction. "
            "Try '--matcher sequential' for ordered video frames, "
            "or check that images have sufficient overlap."
        )

    if cfg.undistort:
        # 4 — Image undistortion (needed for 3DGS / OpenSplat)
        undistorted.mkdir(parents=True, exist_ok=True)
        _run_colmap(
            [
                "image_undistorter",
                "--image_path", str(colmap_images_path),
                "--input_path", str(sparse_0),
                "--output_path", str(undistorted),
                "--output_type", "COLMAP",
            ],
            colmap,
            "undistortion",
            env=colmap_env,
        )
        dataset_dir = undistorted
    else:
        dataset_dir = output_dir

    # Count registered images
    images_bin = sparse_0 / "images.bin"
    n_images = _count_registered_images(images_bin)
    console.print(
        f"[bold green]sfm[/] done — {n_images} images registered. "
        f"Dataset at [cyan]{dataset_dir}[/]"
    )
    return dataset_dir


def _count_registered_images(images_bin: Path) -> int:
    """Parse the binary images.bin to count registered cameras."""
    if not images_bin.exists():
        return 0
    try:
        import struct

        with open(images_bin, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
        return n
    except Exception:
        return 0
