"""Tenstorrent Blackhole hardware monitor — telemetry via `tt-smi -s`.

Drop-in replacement for the (Intel-Arc / Windows) GpuMonitor on the TT path. Shells the official
`tt-smi -s --snapshot_no_tty` JSON snapshot and exposes the same interface the dashboard expects
(``.available`` / ``.start()`` / ``.snapshot()``).  This is the SAME read-only telemetry source
``render_device._preflight_power`` already uses, so it's safe to run alongside a training process
(it reads ASIC power/temp/clock over the ARC mailbox; it does NOT open or reset the device).

Polling is lazy + TTL-cached: a snapshot only runs when something actually requests ``/gpu``, and at
most once per ``_TTL`` seconds, so an idle dashboard spawns no subprocesses.  Any failure degrades to
an empty dict (the panel just stays hidden), never an exception.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time


_TTL = 3.0        # seconds between actual tt-smi invocations
_TIMEOUT = 20     # tt-smi -s wall-clock budget


def _f(x, default=None):
    """Parse a possibly-whitespace-padded numeric string from tt-smi; tolerant of 'N/A'."""
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return default


class TtSmiMonitor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict = {}
        self._ts = 0.0
        self._bin = shutil.which("tt-smi")
        # Available iff tt-smi is on PATH, a device node exists, and the user hasn't opted out.
        self.available = bool(
            self._bin
            and os.path.exists("/dev/tenstorrent/0")
            and os.environ.get("TT_SMI_TELEM", "1") == "1"
        )

    def start(self) -> None:
        """No background thread — polling is lazy/on-demand. Present for interface parity."""
        return None

    def _run(self) -> dict:
        try:
            out = subprocess.run(
                [self._bin, "-s", "--snapshot_no_tty"],
                capture_output=True, text=True, timeout=_TIMEOUT,
            ).stdout
            d = json.loads(out)
            dev = (d.get("device_info") or [{}])[0]
            tel = dev.get("telemetry", {}) or {}
            lim = dev.get("limits", {}) or {}
            board = dev.get("board_info", {}) or {}
            snap = {
                "device":       board.get("board_type", "Blackhole"),
                "power_w":      _f(tel.get("power")),
                "tdp_w":        _f(lim.get("tdp_limit"), 150.0),
                "temp_c":       _f(tel.get("asic_temperature")),
                "temp_limit_c": _f(lim.get("thm_limit"), 110.0),
                "aiclk_mhz":    _f(tel.get("aiclk")),
                "aiclk_max_mhz": _f(lim.get("asic_fmax"), 1350.0),
                "voltage_v":    _f(tel.get("voltage")),
                "current_a":    _f(tel.get("current")),
                "tdc_a":        _f(lim.get("tdc_limit"), 200.0),
                "fan":          _f(tel.get("fan_speed")),
                "dram_speed":   board.get("dram_speed"),
            }
            return {k: v for k, v in snap.items() if v is not None}
        except Exception:
            return {}

    def snapshot(self) -> dict:
        if not self.available:
            return {}
        now = time.monotonic()
        with self._lock:
            if now - self._ts < _TTL and self._cache:
                return dict(self._cache)
        snap = self._run()       # outside the lock — tt-smi can take a second
        with self._lock:
            if snap:
                self._cache = snap
                self._ts = time.monotonic()
            return dict(self._cache)


# Back-compat alias so `from ttgs.backend.tt_monitor import GpuMonitor` works too.
GpuMonitor = TtSmiMonitor
