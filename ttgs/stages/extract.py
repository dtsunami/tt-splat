"""Stage 1: Extract frames from a video file using ffmpeg.

If the input is already a directory of images, this stage is skipped and
the directory is returned as-is.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Console

from ttgs.config import ExtractConfig
from ttgs.tools import require_tool

console = Console()


def _ffprobe_duration(video: Path) -> float | None:
    """Return video duration in seconds using ffprobe, or None on failure."""
    ffprobe = require_tool("ffprobe", "Install ffmpeg (includes ffprobe): https://ffmpeg.org/download.html")
    result = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return None


def _ffprobe_fps(video: Path) -> float | None:
    """Return the native frame-rate of a video, or None on failure."""
    ffprobe = require_tool("ffprobe", "Install ffmpeg (includes ffprobe): https://ffmpeg.org/download.html")
    result = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    try:
        num, den = result.stdout.strip().split("/")
        return float(num) / float(den)
    except Exception:
        return None


def run(
    source: Path,
    output_dir: Path,
    cfg: ExtractConfig,
) -> Path:
    """Extract frames from *source* into *output_dir*.

    Args:
        source:     Path to a video file OR an existing image directory.
        output_dir: Destination directory for extracted frames.
        cfg:        ExtractConfig parameters.

    Returns:
        Path to the directory containing extracted frames.
    """
    if source.is_dir():
        images = list(source.glob("*.jpg")) + list(source.glob("*.png"))
        if images:
            console.print(
                f"[bold cyan]extract[/] input is a directory with {len(images)} images — skipping ffmpeg"
            )
            return source
        raise ValueError(f"Input directory '{source}' contains no .jpg or .png images.")

    ffmpeg = require_tool(
        "ffmpeg",
        "Install ffmpeg: https://ffmpeg.org/download.html  or  winget install ffmpeg",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    native_fps = _ffprobe_fps(source) or 30.0
    duration = _ffprobe_duration(source)

    # Calculate how many frames we'd get at the requested sample rate
    sample_fps = cfg.fps
    if duration and cfg.max_frames > 0:
        estimated = int(duration * sample_fps)
        if estimated > cfg.max_frames:
            sample_fps = cfg.max_frames / duration
            console.print(
                f"[yellow]extract[/] capping at {cfg.max_frames} frames "
                f"(adjusted sample rate: {sample_fps:.2f} fps)"
            )

    console.print(
        f"[bold cyan]extract[/] sampling at [green]{sample_fps:.2f}[/] fps "
        f"(native: {native_fps:.1f} fps)"
    )

    cmd = [
        ffmpeg,
        "-i", str(source),
        "-vf", f"fps={sample_fps}",
    ]

    if cfg.max_width > 0:
        # Scale to max_width, preserve aspect ratio, ensure even dimensions
        cmd += ["-vf", f"fps={sample_fps},scale={cfg.max_width}:-2:flags=lanczos"]

    if cfg.format == "jpg":
        cmd += ["-q:v", str(max(1, min(31, 32 - cfg.quality // 3)))]
        out_pattern = str(output_dir / "frame_%06d.jpg")
    else:
        out_pattern = str(output_dir / "frame_%06d.png")

    cmd += ["-hide_banner", "-loglevel", "warning", "-stats", out_pattern]

    console.print(f"[dim]→ {' '.join(cmd)}[/]")

    result = subprocess.run(cmd, text=True, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    frames = sorted(output_dir.glob(f"*.{cfg.format}"))
    console.print(
        f"[bold green]extract[/] done — {len(frames)} frames written to [cyan]{output_dir}[/]"
    )
    return output_dir
