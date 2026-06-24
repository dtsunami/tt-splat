"""Stage 4: Export Gaussians from .ply to viewer-ready formats.

Supported output formats:
  splat  — antimatter15 binary format (32 bytes/Gaussian, used by SuperSplat,
            ksplat, and most WebGL viewers)
  ply    — pass-through (copy or symlink source .ply)

The .splat format packs each Gaussian as 32 bytes:
  [0:12]  position   3 × float32
  [12:24] scale      3 × float32 (linear, not log)
  [24:28] color+α    4 × uint8 (RGB from DC SH coefficients + sigmoid opacity)
  [28:32] rotation   4 × uint8 (normalised quaternion, mapped to [0, 255])
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from rich.console import Console

from ttgs.config import ExportConfig

console = Console()

# DC spherical harmonic coefficient → linear RGB multiplier
_SH_C0 = 0.28209479177387814


def _read_ply_gaussians(ply_path: Path) -> dict[str, np.ndarray]:
    """Parse a 3DGS .ply file and return its vertex properties as numpy arrays.

    Returns a dict mapping property name → 1-D float32 array.
    """
    with open(ply_path, "rb") as fh:
        # --- Parse ASCII header ---
        header_lines: list[str] = []
        while True:
            line = fh.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        properties: list[str] = []
        n_vertices = 0
        is_binary_little = False

        for line in header_lines:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("property float"):
                properties.append(line.split()[-1])
            elif line == "format binary_little_endian 1.0":
                is_binary_little = True

        if not is_binary_little:
            raise ValueError(
                f"Unsupported PLY format in {ply_path}. "
                "Expected binary_little_endian (standard 3DGS output)."
            )

        # --- Read binary data ---
        n_props = len(properties)
        raw = np.frombuffer(fh.read(n_vertices * n_props * 4), dtype="<f4")
        data = raw.reshape(n_vertices, n_props)

    return {name: data[:, i] for i, name in enumerate(properties)}


def _ply_to_splat(props: dict[str, np.ndarray], cfg: ExportConfig) -> bytes:
    """Convert parsed PLY properties into the .splat binary blob."""
    n = len(next(iter(props.values())))

    if cfg.max_gaussians > 0 and n > cfg.max_gaussians:
        console.print(
            f"[yellow]export[/] truncating from {n:,} to {cfg.max_gaussians:,} Gaussians"
        )
        n = cfg.max_gaussians

    # --- Position ---
    xyz = np.stack([props["x"], props["y"], props["z"]], axis=1)[:n]

    # --- Scale: stored as log in 3DGS output ---
    scale = np.exp(
        np.stack([props["scale_0"], props["scale_1"], props["scale_2"]], axis=1)[:n]
    ).astype(np.float32)

    # --- Opacity: stored as logit, apply sigmoid ---
    opacity_logit = props["opacity"][:n]
    opacity = (1.0 / (1.0 + np.exp(-opacity_logit))).astype(np.float32)

    # --- Colour from DC spherical harmonics ---
    r = np.clip(0.5 + _SH_C0 * props["f_dc_0"][:n], 0.0, 1.0)
    g = np.clip(0.5 + _SH_C0 * props["f_dc_1"][:n], 0.0, 1.0)
    b = np.clip(0.5 + _SH_C0 * props["f_dc_2"][:n], 0.0, 1.0)
    rgba = np.stack([r, g, b, opacity], axis=1)
    rgba_u8 = (rgba * 255).clip(0, 255).astype(np.uint8)

    # --- Rotation quaternion: normalise and map to [0, 255] ---
    quat = np.stack(
        [props["rot_0"][:n], props["rot_1"][:n], props["rot_2"][:n], props["rot_3"][:n]],
        axis=1,
    ).astype(np.float32)
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    quat = quat / norms
    quat_u8 = ((quat + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    # --- Optional sort by opacity (descending) ---
    if cfg.sort_by_opacity:
        order = np.argsort(-opacity)
        xyz = xyz[order]
        scale = scale[order]
        rgba_u8 = rgba_u8[order]
        quat_u8 = quat_u8[order]

    # --- Pack into 32-byte .splat records ---
    out = bytearray(n * 32)
    mv = memoryview(out).cast("B")

    pos_bytes = xyz.astype("<f4").tobytes()
    scale_bytes = scale.astype("<f4").tobytes()

    # Write each field into its slice
    for i in range(n):
        base = i * 32
        struct.pack_into("<3f", out, base, *xyz[i])
        struct.pack_into("<3f", out, base + 12, *scale[i])
        out[base + 24 : base + 28] = rgba_u8[i].tobytes()
        out[base + 28 : base + 32] = quat_u8[i].tobytes()

    return bytes(out)


def run(
    input_ply: Path,
    output_dir: Path,
    cfg: ExportConfig,
) -> Path:
    """Convert the 3DGS .ply output to the requested format.

    Args:
        input_ply:  Path to the OpenSplat output .ply file.
        output_dir: Directory to write the converted file.
        cfg:        ExportConfig parameters.

    Returns:
        Path to the written output file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = cfg.format.lower()

    if fmt == "ply":
        out_path = output_dir / "splat.ply"
        if out_path.resolve() != input_ply.resolve():
            import shutil
            shutil.copy2(input_ply, out_path)
        console.print(f"[bold green]export[/] .ply ready at [cyan]{out_path}[/]")
        return out_path

    if fmt == "splat":
        console.print(f"[bold cyan]export[/] reading {input_ply.name} ...")
        props = _read_ply_gaussians(input_ply)
        n_total = len(next(iter(props.values())))
        console.print(f"[bold cyan]export[/] {n_total:,} Gaussians — converting to .splat ...")

        blob = _ply_to_splat(props, cfg)
        n_out = len(blob) // 32

        out_path = output_dir / "splat.splat"
        out_path.write_bytes(blob)

        size_mb = len(blob) / 1024**2
        console.print(
            f"[bold green]export[/] {n_out:,} Gaussians → [cyan]{out_path}[/] ({size_mb:.1f} MB)"
        )
        return out_path

    raise ValueError(f"Unknown export format '{fmt}'. Choose 'splat' or 'ply'.")
