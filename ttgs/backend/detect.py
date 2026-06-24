"""Detect the best available compute backend for training.

Priority order (can be overridden with --device):
  1. xpu     — Intel Arc via native PyTorch XPU (oneAPI, PyTorch 2.6+)
  2. cuda    — NVIDIA via CUDA
  3. directml — Any DirectX 12 GPU on Windows (Arc, AMD, Intel) via torch-directml
  4. cpu     — Always available; usable but ~10–20x slower than GPU

For Intel Arc B70 on Windows the recommended path is:
  xpu  (requires oneAPI Base Toolkit + pip install torch>=2.6)
  directml (requires pip install torch-directml; no extra SDK needed)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Backend(str, Enum):
    XPU = "xpu"          # Intel Arc via native PyTorch XPU
    CUDA = "cuda"        # NVIDIA
    DIRECTML = "directml"  # Any DX12 GPU (Windows)
    CPU = "cpu"          # Fallback


@dataclass
class BackendInfo:
    backend: Backend
    device_name: str
    vram_gb: Optional[float]
    available: bool
    note: str = ""

    def __str__(self) -> str:
        vram = f"{self.vram_gb:.1f} GB" if self.vram_gb else "unknown VRAM"
        status = "available" if self.available else "unavailable"
        note = f" ({self.note})" if self.note else ""
        return f"{self.backend.value}: {self.device_name} [{vram}] — {status}{note}"


def _probe_xpu() -> BackendInfo:
    try:
        import torch
    except ImportError:
        return BackendInfo(Backend.XPU, "none", None, False, "torch not installed")

    ver = getattr(torch, "__version__", "?")

    if not hasattr(torch, "xpu"):
        return BackendInfo(Backend.XPU, "none", None, False,
                           f"torch {ver} has no xpu module — need PyTorch 2.6+ from pytorch.org/whl/xpu")

    try:
        available = torch.xpu.is_available()
    except Exception as exc:
        return BackendInfo(Backend.XPU, "none", None, False,
                           f"torch {ver} — torch.xpu.is_available() raised: {exc}")

    if available:
        idx = torch.xpu.current_device()
        name = torch.xpu.get_device_name(idx)
        try:
            props = torch.xpu.get_device_properties(idx)
            vram = props.total_memory / 1024**3
        except Exception:
            vram = None
        return BackendInfo(Backend.XPU, name, vram, True)

    # Dig into why it's unavailable
    device_count = getattr(torch.xpu, "device_count", lambda: 0)()
    if "+cpu" in ver:
        note = (f"torch {ver} is a CPU-only build — reinstall from the XPU index:\n"
                f"  pip install torch --index-url https://download.pytorch.org/whl/xpu")
    elif device_count == 0:
        note = (f"torch {ver} — no XPU devices enumerated; "
                f"check Level Zero runtime (sycl-ls or ttgs info --verbose)")
    else:
        note = f"torch {ver} — {device_count} XPU device(s) found but is_available() is False"
    return BackendInfo(Backend.XPU, "none", None, False, note)


def _probe_cuda() -> BackendInfo:
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            vram = torch.cuda.get_device_properties(idx).total_memory / 1024**3
            return BackendInfo(Backend.CUDA, name, vram, True)
        return BackendInfo(Backend.CUDA, "none", None, False, "CUDA not available")
    except ImportError:
        return BackendInfo(Backend.CUDA, "none", None, False, "torch not installed")


def _probe_directml() -> BackendInfo:
    try:
        import torch_directml  # type: ignore[import]

        count = torch_directml.device_count()
        if count > 0:
            name = torch_directml.device_name(0)
            # torch-directml doesn't expose VRAM queries directly
            return BackendInfo(Backend.DIRECTML, name, None, True)
        return BackendInfo(Backend.DIRECTML, "none", None, False, "no DX12 devices found")
    except ImportError:
        return BackendInfo(
            Backend.DIRECTML,
            "none",
            None,
            False,
            "torch-directml not installed",
        )


def probe_all() -> dict[Backend, BackendInfo]:
    """Return info for all backends, whether available or not."""
    return {
        Backend.XPU: _probe_xpu(),
        Backend.CUDA: _probe_cuda(),
        Backend.DIRECTML: _probe_directml(),
        Backend.CPU: BackendInfo(Backend.CPU, "CPU", None, True, "always available"),
    }


def best(preferred: str | None = None) -> BackendInfo:
    """Return the best available backend, respecting an optional preference.

    Args:
        preferred: One of "xpu", "cuda", "directml", "cpu", or None for auto.

    Raises:
        ValueError: If *preferred* names an unavailable backend.
    """
    results = probe_all()

    if preferred is not None:
        key = Backend(preferred.lower())
        info = results[key]
        if not info.available:
            raise ValueError(
                f"Backend '{preferred}' is not available: {info.note}\n"
                f"Run 'ttgs info' to see all backends."
            )
        return info

    # Auto-select: XPU → CUDA → DirectML → CPU
    for backend in (Backend.XPU, Backend.CUDA, Backend.DIRECTML, Backend.CPU):
        info = results[backend]
        if info.available:
            return info

    # CPU is always available so we should never reach here
    return results[Backend.CPU]


def opensplat_device_flag(backend: Backend) -> str:
    """Map our Backend enum to the --device flag OpenSplat understands."""
    mapping = {
        Backend.XPU: "xpu",
        Backend.CUDA: "cuda",
        Backend.DIRECTML: "cpu",  # OpenSplat has no DML; fall back to CPU for DML users
        Backend.CPU: "cpu",
    }
    return mapping[backend]
