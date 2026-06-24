"""Full ttgs pipeline: extract → sfm → train → export.

The Pipeline class orchestrates all four stages with shared configuration,
consistent logging, and graceful error handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.rule import Rule

from ttgs import __version__
from ttgs.backend.detect import BackendInfo, best as best_backend
from ttgs.config import Config, load as load_config
from ttgs.stages import extract, sfm, train, export

console = Console()


@dataclass
class PipelineResult:
    frames_dir: Optional[Path] = None
    dataset_dir: Optional[Path] = None
    splat_ply: Optional[Path] = None
    output_file: Optional[Path] = None
    success: bool = False
    error: Optional[str] = None


def run(
    source: Path,
    output_dir: Path,
    config_path: Path | None = None,
    device: str | None = None,
    colmap_bin: str | None = None,
    resume: bool = False,
    skip_extract: bool = False,
    skip_sfm: bool = False,
    viewer_port: int | None = None,
    dashboard_port: int | None = None,
) -> PipelineResult:
    """Run the full pipeline end-to-end.

    Args:
        source:        Video file or directory of images.
        output_dir:    Root output directory; sub-dirs are created per stage.
        config_path:   Optional TOML config to merge with defaults.
        device:        Override compute backend ("xpu", "cuda", "directml", "cpu").
        colmap_bin:    Explicit path to colmap executable.
        resume:        Resume interrupted training.
        skip_extract:  Skip frame extraction (source must be an image dir).
        skip_sfm:      Skip SfM (output_dir/<stage>/sfm must already exist).

    Returns:
        PipelineResult with paths to outputs and success/error status.
    """
    cfg = load_config(config_path)
    result = PipelineResult()

    console.print(Rule(f"[bold]ttgs v{__version__}[/]"))

    # --- Backend selection (happens early so we can warn before spending time on SfM) ---
    try:
        backend: BackendInfo = best_backend(device)
        _print_backend(backend)
    except ValueError as exc:
        result.error = str(exc)
        console.print(f"[bold red]error:[/] {exc}")
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    sfm_dir = output_dir / "sfm"
    train_dir = output_dir / "train"
    export_dir = output_dir / "export"

    # ── Stage 1: Extract ──────────────────────────────────────────────────────
    console.print(Rule("[cyan]Stage 1 / 4 — Extract[/]"))
    if skip_extract:
        result.frames_dir = source
        console.print("[dim]extract skipped[/]")
    else:
        try:
            result.frames_dir = extract.run(source, frames_dir, cfg.extract)
        except Exception as exc:
            result.error = str(exc)
            console.print(f"[bold red]extract failed:[/] {exc}")
            return result

    # ── Stage 2: SfM ─────────────────────────────────────────────────────────
    console.print(Rule("[cyan]Stage 2 / 4 — Structure from Motion[/]"))
    if skip_sfm:
        result.dataset_dir = sfm_dir / "undistorted"
        if not result.dataset_dir.exists():
            result.dataset_dir = sfm_dir
        console.print(f"[dim]sfm skipped — using {result.dataset_dir}[/]")
    else:
        try:
            result.dataset_dir = sfm.run(result.frames_dir, sfm_dir, cfg.sfm, colmap_bin)
        except Exception as exc:
            result.error = str(exc)
            console.print(f"[bold red]sfm failed:[/] {exc}")
            return result

    # ── Stage 3: Train ───────────────────────────────────────────────────────
    console.print(Rule("[cyan]Stage 3 / 4 — Training[/]"))
    try:
        if dashboard_port is not None:
            from ttgs.viewer.dashboard import DashboardServer
            from ttgs.viewer.pipeline_controller import PipelineController

            pc = PipelineController(
                output_dir=output_dir,
                source=source,
                frames_dir=result.frames_dir,
                cfg=cfg,
                backend=backend,
                colmap_bin=colmap_bin,
            )
            _db = DashboardServer(pc, port=dashboard_port)
            result.splat_ply = _db.run_training(
                train.run,
                result.dataset_dir,
                train_dir,
                cfg.train,
                backend,
                resume=resume,
                viewer_port=viewer_port,
                dashboard=pc.training,
                masks_dir=pc.masks_dir,
                excluded=pc.get_exclusions(),
            )
        else:
            result.splat_ply = train.run(
                result.dataset_dir,
                train_dir,
                cfg.train,
                backend,
                resume=resume,
                viewer_port=viewer_port,
                dashboard=None,
            )
    except Exception as exc:
        result.error = str(exc)
        console.print(f"[bold red]train failed:[/] {exc}")
        return result

    # ── Stage 4: Export ──────────────────────────────────────────────────────
    console.print(Rule("[cyan]Stage 4 / 4 — Export[/]"))
    if result.splat_ply is None:
        console.print("[yellow]export skipped[/] — training did not produce a splat.ply")
        return result
    try:
        result.output_file = export.run(result.splat_ply, export_dir, cfg.export)
    except Exception as exc:
        result.error = str(exc)
        console.print(f"[bold red]export failed:[/] {exc}")
        return result

    result.success = True
    console.print(Rule("[bold green]Pipeline complete[/]"))
    console.print(f"Output: [cyan]{result.output_file}[/]")
    console.print(f"View:   [bold]ttgs view {result.splat_ply}[/]")
    return result


def _print_backend(backend: BackendInfo) -> None:
    vram = f"{backend.vram_gb:.1f} GB VRAM" if backend.vram_gb else ""
    console.print(
        f"[bold]backend:[/] {backend.backend.value} — {backend.device_name}"
        + (f" ({vram})" if vram else "")
    )
