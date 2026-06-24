"""External tool discovery: env-var overrides → PATH → hardcoded candidates.

Supported env vars (set in .env or your shell):
  FFMPEG_PATH      — path to ffmpeg binary or its parent directory
  FFPROBE_PATH     — path to ffprobe binary or its parent directory
  COLMAP_PATH      — path to colmap binary or its parent directory
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolSpec:
    name: str             # canonical executable name
    env_var: str          # e.g. "FFMPEG_PATH"
    purpose: str          # shown in `ttgs info`
    candidates: list[str] = field(default_factory=list)  # hardcoded fallback paths
    # Windows sometimes ships binaries with different capitalisation
    alt_names: list[str] = field(default_factory=list)


# Single source of truth for every external tool ttgs touches.
TOOLS: dict[str, ToolSpec] = {
    "ffmpeg": ToolSpec(
        name="ffmpeg",
        env_var="FFMPEG_PATH",
        purpose="frame extraction",
    ),
    "ffprobe": ToolSpec(
        name="ffprobe",
        env_var="FFPROBE_PATH",
        purpose="video probing",
    ),
    "colmap": ToolSpec(
        name="colmap",
        env_var="COLMAP_PATH",
        purpose="structure from motion",
        candidates=[
            r"C:\Program Files\COLMAP\COLMAP.bat",
            r"C:\Program Files\COLMAP\colmap.exe",
        ],
    ),
}


def find_tool(spec: ToolSpec) -> str | None:
    """Locate a tool executable.  Resolution order:

    1. ``<ENV_VAR>`` — directory (searches for binary inside) or full path
    2. ``shutil.which`` — standard PATH lookup, including alt_names
    3. Hardcoded ``candidates`` list

    Returns the resolved path string, or ``None`` if not found.
    """
    env_path = os.environ.get(spec.env_var)
    if env_path:
        p = Path(env_path)
        if p.is_dir():
            suffixes = ["", ".exe", ".bat"] if sys.platform == "win32" else [""]
            names = [spec.name] + spec.alt_names
            for name in names:
                for suffix in suffixes:
                    candidate = p / (name + suffix)
                    if candidate.exists():
                        return str(candidate)
        elif p.exists():
            return str(p)

    for name in [spec.name] + spec.alt_names:
        exe = shutil.which(name)
        if exe:
            return exe

    for c in spec.candidates:
        if Path(c).exists():
            return c

    return None


def require_tool(key: str, install_hint: str = "") -> str:
    """Like :func:`find_tool` but raises ``RuntimeError`` when absent.

    Args:
        key:          Key in :data:`TOOLS` (e.g. ``"colmap"``).
        install_hint: Extra text appended to the error message.

    Returns:
        Absolute path to the executable.
    """
    spec = TOOLS[key]
    path = find_tool(spec)
    if path:
        return path
    lines = [
        f"{spec.name} not found.",
        f"Set {spec.env_var} in your .env file or shell environment,",
        f"or add the binary to your PATH.",
    ]
    if install_hint:
        lines.append("")
        lines.append(install_hint)
    raise RuntimeError("\n".join(lines))
