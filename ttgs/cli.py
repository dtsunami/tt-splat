"""ttgs command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Load .env before anything else so tool-path env vars are in effect.
# find_dotenv(usecwd=True) walks up from CWD, so it works from any subdirectory.
try:
    from dotenv import load_dotenv, find_dotenv as _find_dotenv
    _env_file: str | None = _find_dotenv(usecwd=True) or None
    if _env_file:
        load_dotenv(_env_file, override=True)
except ImportError:
    _env_file = None

from ttgs import __version__

app = typer.Typer(
    name="ttgs",
    help="Gaussian Splatting pipeline for Tenstorrent Blackhole (tt-splat).",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# ─── Common options ──────────────────────────────────────────────────────────

_device_opt = typer.Option(
    None,
    "--device", "-d",
    help="Compute backend: cuda | cpu  (auto-detected if omitted). The Blackhole "
         "path runs via [bold]ttgs blackhole[/].",
    metavar="BACKEND",
)
_config_opt = typer.Option(
    None,
    "--config", "-c",
    help="Path to a TOML config file to merge with defaults",
    metavar="FILE",
    exists=True,
    file_okay=True,
    dir_okay=False,
)


# ─── run ─────────────────────────────────────────────────────────────────────

@app.command()
def run(
    source: Path = typer.Argument(..., help="Video file or directory of images"),
    output: Path = typer.Option(
        Path("output"),
        "--output", "-o",
        help="Root output directory",
        metavar="DIR",
    ),
    device: Optional[str] = _device_opt,
    config: Optional[Path] = _config_opt,
    colmap: Optional[str] = typer.Option(None, "--colmap", help="Explicit path to colmap binary"),
    resume: bool = typer.Option(False, "--resume", help="Resume interrupted training"),
    skip_extract: bool = typer.Option(False, "--skip-extract", help="Skip frame extraction"),
    skip_sfm: bool = typer.Option(False, "--skip-sfm", help="Skip Structure from Motion"),
    live: bool = typer.Option(False, "--live", help="Launch live viser viewer during training"),
    viewer_port: int = typer.Option(8080, "--viewer-port", help="Port for the live viser viewer"),
    headless: bool = typer.Option(False, "-y", "--headless", help="Skip dashboard, run pipeline non-interactively"),
    dashboard_port: int = typer.Option(7860, "--dashboard-port", help="Port for the training dashboard"),
) -> None:
    """[bold]Run the full pipeline[/]: extract → sfm → train → export.

    By default the interactive dashboard opens at localhost:7860.
    Use [bold]-y[/] to run headlessly (legacy behaviour).

    SOURCE can be a video file (mp4, mov, avi, …) or a directory of JPEG/PNG images.

    Examples:

      ttgs run footage.mp4
      ttgs run footage.mp4 --output ./my_scene --device xpu
      ttgs run footage.mp4 -y                        # headless
      ttgs run ./frames --skip-extract --device cuda -y
    """
    from ttgs.pipeline import run as pipeline_run

    if not source.exists():
        console.print(f"[bold red]error:[/] source not found: {source}")
        raise typer.Exit(1)

    result = pipeline_run(
        source=source,
        output_dir=output,
        config_path=config,
        device=device,
        colmap_bin=colmap,
        resume=resume,
        skip_extract=skip_extract,
        skip_sfm=skip_sfm,
        viewer_port=viewer_port if live else None,
        dashboard_port=dashboard_port if not headless else None,
    )
    if not result.success:
        raise typer.Exit(1)


# ─── extract ─────────────────────────────────────────────────────────────────

@app.command()
def extract(
    source: Path = typer.Argument(..., help="Video file"),
    output: Path = typer.Option(Path("output/frames"), "--output", "-o", help="Output frames directory"),
    config: Optional[Path] = _config_opt,
) -> None:
    """Extract frames from a video file."""
    from ttgs.config import load as load_config
    from ttgs.stages.extract import run as extract_run

    cfg = load_config(config)
    try:
        extract_run(source, output, cfg.extract)
    except Exception as exc:
        console.print(f"[bold red]error:[/] {exc}")
        raise typer.Exit(1)


# ─── sfm ─────────────────────────────────────────────────────────────────────

@app.command()
def sfm(
    images: Path = typer.Argument(..., help="Directory of input images"),
    output: Path = typer.Option(Path("output/sfm"), "--output", "-o", help="Output SfM directory"),
    colmap: Optional[str] = typer.Option(None, "--colmap", help="Explicit path to colmap binary"),
    config: Optional[Path] = _config_opt,
) -> None:
    """Run Structure from Motion (COLMAP) on an image directory."""
    from ttgs.config import load as load_config
    from ttgs.stages.sfm import run as sfm_run

    cfg = load_config(config)
    try:
        sfm_run(images, output, cfg.sfm, colmap)
    except Exception as exc:
        console.print(f"[bold red]error:[/] {exc}")
        raise typer.Exit(1)


# ─── train ───────────────────────────────────────────────────────────────────

@app.command()
def train(
    dataset: Path = typer.Argument(..., help="COLMAP-format dataset directory"),
    output: Path = typer.Option(Path("output/train"), "--output", "-o", help="Output training directory"),
    device: Optional[str] = _device_opt,
    config: Optional[Path] = _config_opt,
    resume: bool = typer.Option(False, "--resume", help="Resume from last checkpoint"),
    live: bool = typer.Option(False, "--live", help="Launch live viser viewer during training"),
    viewer_port: int = typer.Option(8080, "--viewer-port", help="Port for the live viser viewer"),
    dashboard: bool = typer.Option(False, "--dashboard", help="Launch interactive training dashboard"),
    dashboard_port: int = typer.Option(7860, "--dashboard-port", help="Port for the training dashboard"),
) -> None:
    """Train 3D Gaussian Splatting on a COLMAP dataset (gsplat backend)."""
    from ttgs.backend.detect import best as best_backend
    from ttgs.config import load as load_config
    from ttgs.stages.train import run as train_run

    cfg = load_config(config)

    try:
        backend = best_backend(device)
        if dashboard:
            from ttgs.viewer.dashboard import DashboardServer
            from ttgs.viewer.pipeline_controller import PipelineController

            pc = PipelineController(
                output_dir=output, cfg=cfg, backend=backend,
                frames_dir=dataset.parent / "images" if (dataset.parent / "images").exists() else dataset,
            )
            _db = DashboardServer(pc, port=dashboard_port)
            _db.run_training(
                train_run,
                dataset, output, cfg.train, backend,
                resume=resume,
                viewer_port=viewer_port if live else None,
                dashboard=pc.training,
                masks_dir=pc.masks_dir,
            )
        else:
            train_run(
                dataset, output, cfg.train, backend,
                resume=resume,
                viewer_port=viewer_port if live else None,
            )
    except Exception as exc:
        console.print(f"[bold red]error:[/] {exc}")
        raise typer.Exit(1)


# ─── export ──────────────────────────────────────────────────────────────────

@app.command()
def export_cmd(
    ply: Path = typer.Argument(..., help="Input .ply file from training"),
    output: Path = typer.Option(Path("output/export"), "--output", "-o", help="Output directory"),
    fmt: str = typer.Option("splat", "--format", "-f", help="Output format: splat | ply"),
    config: Optional[Path] = _config_opt,
) -> None:
    """Convert a trained .ply to .splat or pass through .ply."""
    from ttgs.config import load as load_config
    from ttgs.stages.export import run as export_run

    cfg = load_config(config)
    cfg.export.format = fmt
    try:
        export_run(ply, output, cfg.export)
    except Exception as exc:
        console.print(f"[bold red]error:[/] {exc}")
        raise typer.Exit(1)


# Give the command a name that doesn't conflict with the stdlib keyword
app.command(name="export")(export_cmd)


# ─── serve ───────────────────────────────────────────────────────────────────

@app.command()
def serve(
    source: Path = typer.Argument(..., help="Video file or directory of images"),
    output: Path = typer.Option(
        Path("output"),
        "--output", "-o",
        help="Root output directory",
        metavar="DIR",
    ),
    device: Optional[str] = _device_opt,
    config: Optional[Path] = _config_opt,
    colmap: Optional[str] = typer.Option(None, "--colmap", help="Explicit path to colmap binary"),
    port: int = typer.Option(7860, "--port", "-p", help="Dashboard server port"),
) -> None:
    """[bold]Start the interactive dashboard[/] — full pipeline control from the browser.

    Opens a web UI where you can:
      - review / exclude / mask individual images
      - run or re-run any pipeline stage (extract, SfM, train, export)
      - interactively guide training (pause, prune, mask, focus camera)
      - connect an AI agent via MCP at /mcp

    Examples:

      ttgs serve footage.mp4
      ttgs serve footage.mp4 --output ./my_scene --port 8080
      ttgs serve ./frames --output ./my_scene
    """
    from ttgs.config import load as load_config
    from ttgs.viewer.dashboard import DashboardServer
    from ttgs.viewer.pipeline_controller import PipelineController

    if not source.exists():
        console.print(f"[bold red]error:[/] source not found: {source}")
        raise typer.Exit(1)

    cfg = load_config(config)
    output.mkdir(parents=True, exist_ok=True)

    # Detect backend early so it's cached
    bk = None
    try:
        from ttgs.backend.detect import best as best_backend
        bk = best_backend(device)
    except Exception:
        pass

    # If source is a directory, treat it as frames_dir directly
    frames_dir = source if source.is_dir() else None

    pc = PipelineController(
        output_dir=output,
        source=source if not source.is_dir() else None,
        frames_dir=frames_dir,
        cfg=cfg,
        backend=bk,
        colmap_bin=colmap,
    )

    # Mark extract done if we already have frames
    if frames_dir or (output / "frames").exists():
        pc._stages["extract"].status = "done"

    server = DashboardServer(pc, port=port)
    server.run()   # blocks — Ctrl-C to stop


# ─── view ────────────────────────────────────────────────────────────────────

@app.command()
def view(
    splat: Path = typer.Argument(..., help="Path to a .ply file from training"),
    port: int = typer.Option(8080, "--port", "-p", help="viser viewer port"),
    max_gaussians: int = typer.Option(
        0,
        "--max-gaussians",
        help="Limit Gaussians sent to viewer (0 = all). Use ~500000 for large scenes.",
    ),
) -> None:
    """Open a trained .ply in the viser viewer.

    Use the splat.ply produced by training, not the exported .splat file.
    The .ply format carries the full Gaussian parameters needed for rendering.

    Examples:

      ttgs view output/train/splat.ply
      ttgs view output/train/splat.ply --port 8081
      ttgs view output/train/splat.ply --max-gaussians 500000
    """
    from ttgs.config import ViewerConfig
    from ttgs.viewer.serve import run as serve_run

    if not splat.exists():
        console.print(f"[bold red]error:[/] file not found: {splat}")
        raise typer.Exit(1)

    vcfg = ViewerConfig(port=port)
    try:
        serve_run(splat, vcfg, max_gaussians=max_gaussians)
    except Exception as exc:
        console.print(f"[bold red]error:[/] {type(exc).__name__}: {exc}")
        raise typer.Exit(1)


# ─── blackhole ───────────────────────────────────────────────────────────────

@app.command()
def blackhole(
    dataset: Path = typer.Argument(..., help="COLMAP dataset dir (sparse/0 + images/)"),
    output: Path = typer.Option(Path("work/tt_out"), "--output", "-o",
                                help="Output directory", metavar="DIR"),
    port: int = typer.Option(7860, "--port", "-p", help="Dashboard server port"),
    steps: int = typer.Option(2000, "--steps", help="Training iterations"),
) -> None:
    """[bold]Train on Tenstorrent Blackhole[/] — the ttgs dashboard driving the TT pipeline.

    Stands up the FastAPI training dashboard (Render|GT|Diff, prune/densify/clamp,
    pause/stop, live metrics) and routes the training stage to the tt-splat Blackhole
    backend. Honors the TT_MAX_POINTS / TT_SIZE env knobs (set them in .env).

    Examples:

      ttgs blackhole work/scene
      ttgs blackhole work/scene --output work/tt_out --port 7860 --steps 2000

    Then open [bold]http://localhost:7860/training[/]
    """
    import subprocess
    repo = Path(__file__).resolve().parent.parent
    script = repo / "server" / "serve_blackhole.py"
    if not script.exists():
        console.print(f"[bold red]error:[/] {script} not found — "
                      "run from a tt-splat checkout (pip install -e .).")
        raise typer.Exit(1)
    if not dataset.exists():
        console.print(f"[bold red]error:[/] dataset not found: {dataset}")
        raise typer.Exit(1)
    cmd = [sys.executable, str(script), "--dataset", str(dataset),
           "--output", str(output), "--port", str(port), "--steps", str(steps)]
    console.print(f"[dim]→ {' '.join(cmd)}[/]")
    raise typer.Exit(subprocess.call(cmd))


# ─── info ────────────────────────────────────────────────────────────────────

@app.command()
def info() -> None:
    """Show system info: Blackhole device status, backends, tools, gsplat."""
    import os, shutil, importlib.util
    from ttgs.backend.detect import probe_all, Backend

    console.print(f"\n[bold]ttgs v{__version__}[/]  Python {sys.version.split()[0]}\n")

    # Show which .env file was loaded (or warn if none found)
    if _env_file:
        console.print(f"[dim].env:[/] {_env_file}")
    else:
        console.print("[yellow].env not found[/] — copy [bold].env.example[/] → [bold].env[/] to configure paths")
    console.print()

    # Tenstorrent Blackhole status — what a new user checks first
    tt = Table(title="Tenstorrent Blackhole", show_header=True, header_style="bold cyan")
    tt.add_column("Check", style="bold")
    tt.add_column("Status")
    ttsmi = shutil.which("tt-smi")
    tt.add_row("tt-smi (PATH)", f"[green]{ttsmi}[/]" if ttsmi else "[red]not found[/] — install tenstorrent-tools")
    tt.add_row("/dev/tenstorrent/0",
               "[green]present[/]" if os.path.exists("/dev/tenstorrent/0") else "[red]absent[/] — is the card seated / driver loaded?")
    tmh = os.environ.get("TT_METAL_HOME", "")
    tt.add_row("TT_METAL_HOME",
               f"[green]{tmh}[/]" if tmh and os.path.isdir(tmh)
               else (f"[yellow]{tmh} (dir missing)[/]" if tmh else "[red]unset[/] — set in .env"))
    have_ttnn = importlib.util.find_spec("ttnn") is not None
    tt.add_row("ttnn (import)",
               "[green]installed[/]" if have_ttnn else "[yellow]not installed[/] — host-reference loop only")
    console.print(tt)
    console.print()

    # gsplat status
    try:
        import gsplat
        gsplat_ver = getattr(gsplat, "__version__", "unknown")
        try:
            from gsplat import rasterization  # noqa: F401
            console.print(f"[dim]gsplat:[/] v{gsplat_ver} [green]ready[/] (optional CPU/CUDA reference path)")
        except Exception as exc:
            console.print(f"[yellow]gsplat:[/] v{gsplat_ver} — rasterization import failed: {exc}")
    except ImportError:
        console.print(
            r"[dim]gsplat:[/] not installed — fine for [bold]ttgs blackhole[/]. "
            r"Only the optional host reference path (ttgs train) needs it: [bold]uv pip install -e '.\[reference]'[/]"
        )
    console.print()

    # Backends
    backends = probe_all()
    table = Table(title="Compute Backends", show_header=True, header_style="bold cyan")
    table.add_column("Backend", style="bold")
    table.add_column("Device")
    table.add_column("VRAM")
    table.add_column("Status")
    table.add_column("Note")

    for backend, binfo in backends.items():
        status = "[green]available[/]" if binfo.available else "[red]unavailable[/]"
        vram = f"{binfo.vram_gb:.1f} GB" if binfo.vram_gb else "—"
        table.add_row(backend.value, binfo.device_name, vram, status, binfo.note)

    console.print(table)

    # External tools
    from ttgs.tools import TOOLS, find_tool

    t2 = Table(title="External Tools", show_header=True, header_style="bold cyan")
    t2.add_column("Tool", style="bold")
    t2.add_column("Env var", style="dim")
    t2.add_column("Env var value", style="dim")
    t2.add_column("Resolved path")

    for spec in TOOLS.values():
        env_val = os.environ.get(spec.env_var, "")
        path = find_tool(spec)
        if path:
            resolved = f"[green]{path}[/]"
        elif env_val:
            resolved = f"[red]not found[/] (path does not exist)"
        else:
            resolved = f"[red]not found[/]"
        t2.add_row(spec.name, spec.env_var, env_val or "[dim]—[/]", resolved)

    console.print(t2)


# ─── setup ───────────────────────────────────────────────────────────────────

@app.command()
def setup() -> None:
    """Print setup instructions for all backends and external tools."""
    console.print("""
[bold cyan]ttgs setup guide (Tenstorrent Blackhole)[/]

[bold]1. Tenstorrent stack[/] (device + runtime)
   - Driver + tt-smi: install the tenstorrent-tools package; check with [bold]tt-smi[/].
   - tt-metal: build the ~/tt-metal tree and its python_env (./create_venv.sh — brings
     torch + ttnn). Point [bold]TT_METAL_HOME[/] / [bold]TT_METAL_RUNTIME_ROOT[/] at it in .env.
   - Recover a wedged card with [bold]tt-smi -r 0[/].

[bold]2. ffmpeg[/] (frame extraction from video — optional)
   Linux: sudo apt install ffmpeg     (or set FFMPEG_PATH / FFPROBE_PATH in .env)

[bold]3. COLMAP[/] (structure from motion — camera poses + sparse points)
   Linux: sudo apt install colmap     (or set COLMAP_PATH in .env)

[bold]4. Install ttgs[/]
   pip install -e .                    (registers the `ttgs` command)

Then verify everything with [bold]ttgs info[/] and run [bold]ttgs blackhole work/scene[/].
""")


# ─── version ─────────────────────────────────────────────────────────────────

@app.command()
def version() -> None:
    """Print version and exit."""
    console.print(f"ttgs v{__version__}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    app()


if __name__ == "__main__":
    main()
