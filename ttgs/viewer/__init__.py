"""Viewer utilities for ttgs.

``serve.run``   — static viewer: load a .ply and open viser
``LiveViewer``  — live viewer: push Gaussian updates during training
"""

from __future__ import annotations

import numpy as np
from rich.console import Console

console = Console()

_SH_C0 = 0.28209479177387814

# viser converts covariances to float16 internally.
# float16 max ≈ 65504  →  max safe log-scale = ln(sqrt(65504)) ≈ 5.5
# np.clip does NOT fix NaN (NaN passes through unchanged), so we use
# nan_to_num after the einsum as the authoritative sanitisation step.
_MAX_LOG_SCALE = 5.0
_F16_MAX = 60000.0


class LiveViewer:
    """viser-backed live viewer that can be updated while training runs.

    Caps each update at *max_gaussians* (sorted by opacity descending) so the
    WebSocket message stays small enough that the browser connection never
    drops.  Camera position is preserved by viser between updates as long as
    the connection stays up.

    Usage::

        viewer = LiveViewer(port=8080)
        # inside training loop:
        viewer.update(means, scales, quats, opacities, sh_dc)
        viewer.stop()
    """

    def __init__(self, port: int = 8080, max_gaussians: int = 200_000) -> None:
        try:
            import viser
        except ImportError:
            raise RuntimeError(
                "viser is not installed.\n"
                "Install it with:  pip install viser"
            )
        self._server = viser.ViserServer(port=port, verbose=False)
        self._max_gaussians = max_gaussians
        url = f"http://localhost:{port}"
        console.print(
            f"[bold cyan]viewer[/] live viewer at [link={url}]{url}[/link]  "
            f"(updates capped at {max_gaussians:,} Gaussians)"
        )

    def update(
        self,
        means: np.ndarray,      # (N, 3) float32 — world positions
        scales: np.ndarray,     # (N, 3) float32 — log-space scales
        quats: np.ndarray,      # (N, 4) float32 — (w, x, y, z) quaternions
        opacities: np.ndarray,  # (N,)  float32 — logit-space opacities
        sh_dc: np.ndarray,      # (N, 3) float32 — DC SH colour coefficients
    ) -> None:
        """Push an updated set of Gaussians to the browser."""
        means     = np.asarray(means,     dtype=np.float32)
        scales    = np.asarray(scales,    dtype=np.float32)
        quats     = np.asarray(quats,     dtype=np.float32)
        opacities = np.asarray(opacities, dtype=np.float32)
        sh_dc     = np.asarray(sh_dc,     dtype=np.float32)

        # Drop any Gaussian where means, scales, OR quats contain NaN/Inf.
        # np.clip silently passes NaN through, so we must filter first.
        # NaN in scales or quats would produce NaN covariances → NaN billboard
        # vertex positions → WASM sorter crash ("memory access out of bounds").
        finite_mask = (
            np.isfinite(means).all(axis=1)
            & np.isfinite(scales).all(axis=1)
            & np.isfinite(quats).all(axis=1)
        )
        if not finite_mask.all():
            means     = means[finite_mask]
            scales    = scales[finite_mask]
            quats     = quats[finite_mask]
            opacities = opacities[finite_mask]
            sh_dc     = sh_dc[finite_mask]

        if len(means) == 0:
            return  # nothing valid to display yet

        # Truncate to the most opaque Gaussians so the WebSocket message stays
        # small enough that the browser connection doesn't drop.
        N = len(means)
        if self._max_gaussians > 0 and N > self._max_gaussians:
            opas_sigmoid = 1.0 / (1.0 + np.exp(-opacities.clip(-20, 20)))
            order = np.argpartition(-opas_sigmoid, self._max_gaussians)[: self._max_gaussians]
            means     = means[order]
            scales    = scales[order]
            quats     = quats[order]
            opacities = opacities[order]
            sh_dc     = sh_dc[order]
            N = self._max_gaussians

        # Scales: clamp then exponentiate so s² never overflows float16
        s = np.exp(scales.clip(-_MAX_LOG_SCALE, _MAX_LOG_SCALE))

        # Quaternions — normalise
        norms = np.linalg.norm(quats, axis=1, keepdims=True).clip(min=1e-8)
        q = quats / norms
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Rotation matrices from quaternions
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

        # Covariance Σ = R diag(s²) Rᵀ — viser wants (N, 3, 3).
        # nan_to_num replaces any residual NaN/Inf (clip won't catch NaN).
        S2 = s * s
        covariances = np.nan_to_num(
            np.einsum("nik,nk,njk->nij", R, S2, R),
            nan=0.0, posinf=_F16_MAX, neginf=-_F16_MAX,
        ).astype(np.float32)

        # Colour: DC SH → RGB [0, 1]
        rgbs = np.stack(
            [
                np.clip(0.5 + _SH_C0 * sh_dc[:, 0], 0.0, 1.0),
                np.clip(0.5 + _SH_C0 * sh_dc[:, 1], 0.0, 1.0),
                np.clip(0.5 + _SH_C0 * sh_dc[:, 2], 0.0, 1.0),
            ],
            axis=1,
        ).astype(np.float32)

        # Opacity: logit → sigmoid — viser requires (N, 1) not (N,)
        opas = (1.0 / (1.0 + np.exp(-opacities.clip(-20, 20))))[:, None].astype(np.float32)

        self._server.scene.add_gaussian_splats(
            name="/splat",
            centers=means,
            covariances=covariances,
            rgbs=rgbs,
            opacities=opas,
        )

    def stop(self) -> None:
        try:
            self._server.stop()
        except Exception:
            pass
