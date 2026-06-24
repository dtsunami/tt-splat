"""viser-based viewer for .ply Gaussian splat files.

viser (https://github.com/nerfstudio-project/viser) serves a WebGL viewer at
localhost:PORT.  Open the printed URL in any browser — Chrome/Edge work best.

Only the standard 3DGS binary .ply format is accepted.  The .splat export
format is for web-only viewers and does not carry the data needed to
reconstruct covariance matrices.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from rich.console import Console

from ttgs.config import ViewerConfig
from ttgs.stages.export import _read_ply_gaussians

console = Console()

_SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))

# viser converts covariances to float16 internally.
# float16 max ≈ 65504  →  max safe scale = sqrt(65504) ≈ 256  →  max log-scale ≈ 5.5
# Clamp here so exp(scale)² never overflows float16 and produces NaN in the shader.
_MAX_LOG_SCALE = 5.0
_F16_MAX = 60000.0


def _props_to_viser(
    props: dict[str, np.ndarray],
    max_gaussians: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a PLY property dict to the four arrays viser expects.

    Returns:
        centers:     (N, 3) float32  world positions
        covariances: (N, 6) float32  upper-triangular of Σ = R S² Rᵀ
        rgbs:        (N, 3) float32  [0, 1] from DC SH coefficients
        opacities:   (N,)  float32  [0, 1] from logit-space opacity
    """
    n_total = len(props["x"])

    if max_gaussians > 0 and n_total > max_gaussians:
        # Sort by descending sigmoid(opacity) before truncating so we keep the
        # most visible Gaussians rather than an arbitrary prefix.
        opas = 1.0 / (1.0 + np.exp(-props["opacity"]))
        order = np.argsort(-opas)[:max_gaussians]
        props = {k: v[order] for k, v in props.items()}
        console.print(
            f"[dim]truncating {n_total:,} → {max_gaussians:,} Gaussians for viewer[/]"
        )

    N = len(props["x"])
    centers = np.stack([props["x"], props["y"], props["z"]], axis=1).astype(np.float32)

    # Scales: clamp log-scales then exponentiate
    log_scales = np.stack(
        [props["scale_0"], props["scale_1"], props["scale_2"]], axis=1
    ).clip(-_MAX_LOG_SCALE, _MAX_LOG_SCALE)
    scales = np.exp(log_scales).astype(np.float32)

    # Quaternions (w, x, y, z) — normalise
    quats = np.stack(
        [props["rot_0"], props["rot_1"], props["rot_2"], props["rot_3"]], axis=1
    ).astype(np.float32)
    norms = np.linalg.norm(quats, axis=1, keepdims=True).clip(min=1e-8)
    quats /= norms
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    # 3×3 rotation matrices from quaternions
    R = np.empty((N, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    # Covariance Σ = R @ diag(s²) @ Rᵀ — viser wants (N, 3, 3), not packed (N, 6).
    # viser converts covariances to float16 internally; nan_to_num handles both
    # overflow (→ float16 Inf) and residual NaN (np.clip passes NaN through unchanged).
    S2 = scales * scales  # (N, 3)
    covariances = np.nan_to_num(
        np.einsum("nik,nk,njk->nij", R, S2, R),
        nan=0.0, posinf=_F16_MAX, neginf=-_F16_MAX,
    ).astype(np.float32)

    # Colour from DC spherical-harmonic coefficients
    rgbs = np.stack(
        [
            np.clip(0.5 + _SH_C0 * props["f_dc_0"], 0.0, 1.0),
            np.clip(0.5 + _SH_C0 * props["f_dc_1"], 0.0, 1.0),
            np.clip(0.5 + _SH_C0 * props["f_dc_2"], 0.0, 1.0),
        ],
        axis=1,
    ).astype(np.float32)

    # Opacity: logit → sigmoid — viser wants (N, 1) not (N,)
    opacities = (1.0 / (1.0 + np.exp(-props["opacity"]))).astype(np.float32)[:, None]

    return centers, covariances, rgbs, opacities


def run(splat_path: Path, cfg: ViewerConfig, max_gaussians: int = 0) -> None:
    """Load *splat_path* into a viser viewer and block until Ctrl-C.

    Args:
        splat_path:     Path to a standard 3DGS .ply file.
        cfg:            ViewerConfig (port, etc.).
        max_gaussians:  If > 0, truncate to this many Gaussians (most opaque first).
                        Useful for very large scenes where the browser struggles.
    """
    try:
        import viser
    except ImportError:
        raise RuntimeError(
            "viser is not installed.\n"
            "Install it with:  pip install viser"
        )

    if not splat_path.exists():
        raise FileNotFoundError(f"File not found: {splat_path}")

    if splat_path.suffix.lower() != ".ply":
        raise ValueError(
            f"The viser viewer requires a .ply file (got {splat_path.suffix!r}).\n"
            "Use the training output (splat.ply) rather than the exported .splat.\n"
            "Example: ttgs view output/train/splat.ply"
        )

    console.print(f"[dim]loading {splat_path.name}…[/]")
    props = _read_ply_gaussians(splat_path)
    n = len(props["x"])
    console.print(f"[dim]{n:,} Gaussians[/]")

    centers, covariances, rgbs, opacities = _props_to_viser(props, max_gaussians)
    n_display = len(centers)

    server = viser.ViserServer(port=cfg.port)
    try:
        server.scene.add_gaussian_splats(
            name="/splat",
            centers=centers,
            covariances=covariances,
            rgbs=rgbs,
            opacities=opacities,
        )
    except Exception as exc:
        server.stop()
        raise RuntimeError(
            f"viser add_gaussian_splats failed ({type(exc).__name__}: {exc})\n"
            f"  Gaussians: {n_display:,}  "
            f"  covariances shape: {covariances.shape}  dtype: {covariances.dtype}\n"
            "Try --max-gaussians 500000 to reduce scene size."
        ) from exc

    url = f"http://localhost:{cfg.port}"
    console.print(
        f"[bold cyan]viewer[/] [cyan]{splat_path.name}[/] ({n_display:,} Gaussians)  "
        f"[link={url}]{url}[/link]  — Ctrl-C to stop"
    )

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        console.print("\n[yellow]viewer[/] stopped.")
    finally:
        server.stop()
