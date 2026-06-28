#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Live per-step DEVICE compute-utilization for the resident training loop (gated by --profile / TT_PROFILE=1).

The prebuilt ~/tt-metal tree is profiler-enabled (ENABLE_TRACY=ON), so the firmware auto-brackets EVERY
kernel launch with a `*-KERNEL` cycle zone — including our `ttnn.generic_op` raster/projection/Adam
dispatches, with no kernel-source edits.  Each step we call `ttnn.ReadDeviceProfiler(dev)` (flushes the
device-side zones to `generated/profiler/.logs/profile_log_device.csv`), parse only the NEW rows, and turn
the per-core kernel cycle spans into:

  * dev_us   — busiest COMPUTE core's summed kernel time this step (µs)  ≈ the device-compute critical path
  * util     — 100 * dev_us / step_wall_us  → "what % of the step the hottest Tensix core was computing"
  * cores    — number of distinct cores that ran a kernel this step

util is the honest compute-utilization signal: the rest of the step is host glue (arg-pack, readback,
bin/sort, dispatch latency) — exactly the overhead the perf work targets.  Everything degrades to None on
any parse miss, so a profiled run never breaks; an unprofiled run never touches this module.

LIMITATION: the on-chip marker DRAM holds ~12k markers/RISC and we read once per step, so a heavy step
(many ops/dispatches) overflows it and drops markers — util then UNDER-reads (it's a lower bound).  Reading
per-stage avoids the drops but the per-read cost (device round-trip + CSV flush) dominates the step, so we
keep the cheap once-per-step read for a live indicator.  For rigorous per-op utilization use the offline
`python -m tracy` flow (see docs/controls.html) — this is the live, at-a-glance signal.
"""
from __future__ import annotations
import os
from pathlib import Path

_KERNEL_ZONES = ("TRISC-KERNEL", "BRISC-KERNEL", "NCRISC-KERNEL")
_CSV_CAP_BYTES = 256 * 1024 * 1024     # MID_RUN_DUMP appends ~0.3 MB/step; truncate-to-header past this


def _csv_path() -> Path:
    base = os.environ.get("TT_METAL_PROFILER_DIR") or (
        os.path.join(os.environ.get("TT_METAL_HOME", ""), "generated", "profiler"))
    return Path(base) / ".logs" / "profile_log_device.csv"


class DeviceProfiler:
    """Incremental reader for the tt-metal device-profiler CSV. One instance per run."""

    def __init__(self):
        import ttnn
        self._ttnn = ttnn
        self.path = _csv_path()
        self.aiclk_mhz = 1350.0          # refined from the CSV header on first read
        self._off = 0                    # byte offset already consumed
        self._seen_ts = 0                # max kernel-START cycle seen (rewrite-safe filter)
        self._accum: dict = {}           # (cx,cy,risc) -> kernel-busy cycles, this step (drained per-stage)
        self.available = True
        # Seek past whatever is already in the file so we only attribute OUR steps.
        try:
            if self.path.exists():
                self._off = self.path.stat().st_size
        except Exception:
            self.available = False

    def _read_new_rows(self):
        """Return the CSV rows appended since the last read (handles append OR truncate-rewrite)."""
        try:
            size = self.path.stat().st_size
            if size < self._off:          # file was rewritten/rotated — restart after the header
                self._off = 0
            with open(self.path, "r") as f:
                if self._off == 0:
                    header = f.readline()                      # "ARCH: blackhole, CHIP_FREQ[MHz]: 1350, ..."
                    if "CHIP_FREQ" in header:
                        try:
                            self.aiclk_mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
                        except Exception:
                            pass
                    f.readline()                               # column header
                    self._off = f.tell()
                else:
                    f.seek(self._off)
                data = f.read()
                self._off = f.tell()
            return data.splitlines()
        except Exception:
            self.available = False
            return []

    def drain(self, dev) -> None:
        """Flush the device profiler buffer and fold this slice's kernel-busy cycles into the
        per-step accumulator. Call AFTER each device-stage sync — the on-chip marker DRAM holds
        only ~12k markers/RISC, so one read per step overflows on a real step (markers dropped)."""
        if not self.available:
            return
        try:
            self._ttnn.ReadDeviceProfiler(dev)
        except Exception:
            self.available = False
            return
        events: dict = {}
        new_max = self._seen_ts
        for ln in self._read_new_rows():
            c = ln.split(",")
            if len(c) < 12 or c[10].strip() not in _KERNEL_ZONES:
                continue
            typ = c[11].strip()
            try:
                cx, cy, risc, t = int(c[1]), int(c[2]), c[3].strip(), int(c[5])
            except (ValueError, IndexError):
                continue
            if typ == "ZONE_START" and t <= self._seen_ts:
                continue                                       # already attributed to an earlier slice
            new_max = max(new_max, t)
            events.setdefault((cx, cy, risc), []).append((t, typ))
        self._seen_ts = new_max
        for key, evs in events.items():                        # pair START->END per (core,risc), sum durations
            evs.sort()
            start = None
            for t, typ in evs:
                if typ == "ZONE_START":
                    start = t
                elif typ == "ZONE_END" and start is not None:
                    self._accum[key] = self._accum.get(key, 0) + (t - start); start = None
        self._cap_csv()                                        # bound the on-disk CSV mid-run

    def _read_header(self) -> str:
        try:
            with open(self.path, "r") as f:
                return f.readline() + f.readline()             # ARCH line + column header
        except Exception:
            return ""

    def _cap_csv(self) -> None:
        """Truncate the device CSV back to its 2-line header once it passes the cap. Safe: tt-metal
        appends (O_APPEND) so the next dump lands after the new EOF, and our offset/seen-ts filters
        already tolerate a shrunk file. Keeps a long --profile run from leaving a multi-GB file."""
        try:
            if self.path.stat().st_size <= _CSV_CAP_BYTES:
                return
            h = self._read_header()
            with open(self.path, "w") as f:
                f.write(h)
            self._off = len(h.encode())                        # _seen_ts (cycles) still filters stale zones
        except Exception:
            pass

    def close(self) -> None:
        """Run-end teardown: shrink the CSV to its header so a --profile run doesn't leave the
        multi-GB dump behind. Best-effort; never raises."""
        if not self.available:
            return
        try:
            h = self._read_header()
            with open(self.path, "w") as f:
                f.write(h)
        except Exception:
            pass

    def finalize(self, step_wall_ms: float) -> dict:
        """Turn the step's accumulated kernel cycles into {util, dev_us, cores}; reset for next step.
        util = busiest COMPUTE (TRISC) core's kernel time as a % of the step wall (rest = host glue)."""
        acc, self._accum = self._accum, {}
        if not acc:
            return {}
        compute_busy = max((cyc for (cx, cy, risc), cyc in acc.items() if risc.startswith("TRISC")), default=0)
        dev_us = compute_busy / self.aiclk_mhz                 # cycles / (cycles per µs) = µs
        wall_us = max(step_wall_ms * 1e3, 1e-6)
        phys_cores = len({(cx, cy) for (cx, cy, _risc) in acc})
        return {"dev_us": round(dev_us, 1),
                "util": round(min(100.0, 100.0 * dev_us / wall_us), 1),
                "cores": phys_cores}

    def step(self, dev, step_wall_ms: float) -> dict:
        """Convenience: one drain + finalize (sufficient for tiny single-dispatch workloads)."""
        self.drain(dev)
        return self.finalize(step_wall_ms)
