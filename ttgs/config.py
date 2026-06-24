"""Configuration loading and dataclasses for ttgs."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

_DEFAULTS_PATH = Path(__file__).parent.parent / "config" / "defaults.toml"


@dataclass
class ExtractConfig:
    fps: float = 2.0
    max_frames: int = 300
    max_width: int = -1
    format: str = "jpg"
    quality: int = 95


@dataclass
class SfmConfig:
    matcher: str = "exhaustive"
    single_camera: bool = True
    camera_model: str = "OPENCV"
    undistort: bool = True


@dataclass
class TrainConfig:
    iterations: int = 30_000
    sh_degree: int = 3
    resolution: int = -1
    lambda_dssim: float = 0.2
    log_every: int = 500
    save_every: int = 5_000
    densify_every: int = 100
    densify_from: int = 500
    densify_until: int = 15_000
    densify_grad_threshold: float = 0.0002
    opacity_reset_every: int = 3_000
    viewer_every: int = 100   # push live viewer update every N steps (when --live)
    dashboard_every: int = 25  # push dashboard snapshot every N steps; first frame always at step 1
    snapshot_every: int = 0    # save per-camera render snapshots every N steps (0 = disabled)


@dataclass
class ExportConfig:
    format: str = "splat"
    sort_by_opacity: bool = True
    max_gaussians: int = 0


@dataclass
class ViewerConfig:
    port: int = 8080
    auto_open: bool = True


@dataclass
class Config:
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    sfm: SfmConfig = field(default_factory=SfmConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive on base)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load(user_path: Path | None = None) -> Config:
    """Load config, merging defaults with an optional user TOML file."""
    with open(_DEFAULTS_PATH, "rb") as fh:
        data = tomllib.load(fh)

    if user_path is not None:
        with open(user_path, "rb") as fh:
            data = _merge(data, tomllib.load(fh))

    def _section(cls, key):
        return cls(**{k: v for k, v in data.get(key, {}).items() if k in cls.__dataclass_fields__})

    return Config(
        extract=_section(ExtractConfig, "extract"),
        sfm=_section(SfmConfig, "sfm"),
        train=_section(TrainConfig, "train"),
        export=_section(ExportConfig, "export"),
        viewer=_section(ViewerConfig, "viewer"),
    )
