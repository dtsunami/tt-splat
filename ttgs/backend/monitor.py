"""GPU hardware monitor for Intel Arc via Level Zero Sysman.

Provides real-time metrics (temperature, VRAM, frequency, voltage,
power draw, utilization, fan speed) by calling the Level Zero System
Management API through ctypes.

No external Python dependencies — ze_loader.dll ships with the Intel
GPU driver on Windows.  Metrics that are unavailable are silently
skipped; callers get whatever the driver can report.
"""

from __future__ import annotations

import ctypes as ct
import os
import threading
import time
from typing import Any

# ── Level Zero result codes ──────────────────────────────────────────────────

_OK = 0  # ZE_RESULT_SUCCESS

# zes_structure_type_t values (from zes_api.h)
_STYPE_FREQ_STATE = 0x1B
_STYPE_MEM_STATE = 0x1C


# ── ctypes struct definitions ────────────────────────────────────────────────
# Only the *state* structs we actually read.  ctypes handles alignment on x64
# automatically (fields padded to their natural alignment).

class _FreqState(ct.Structure):
    """zes_freq_state_t — GPU or VRAM frequency + voltage."""
    _fields_ = [
        ("stype", ct.c_uint32),
        ("pNext", ct.c_void_p),
        ("currentVoltage", ct.c_double),   # Volts  (-1 if unknown)
        ("request", ct.c_double),          # MHz
        ("tdp", ct.c_double),              # MHz
        ("efficient", ct.c_double),        # MHz
        ("actual", ct.c_double),           # MHz
        ("throttleReasons", ct.c_uint32),  # bitmask
    ]


class _MemState(ct.Structure):
    """zes_mem_state_t — VRAM usage."""
    _fields_ = [
        ("stype", ct.c_uint32),
        ("pNext", ct.c_void_p),
        ("health", ct.c_uint32),
        ("free", ct.c_uint64),    # bytes
        ("size", ct.c_uint64),    # bytes
    ]


class _PowerEnergy(ct.Structure):
    """zes_power_energy_counter_t — cumulative energy counter."""
    _fields_ = [
        ("stype", ct.c_uint32),
        ("pNext", ct.c_void_p),
        ("energy", ct.c_uint64),     # microjoules
        ("timestamp", ct.c_uint64),  # microseconds
    ]


class _EngineStats(ct.Structure):
    """zes_engine_stats_t — cumulative engine busy counter."""
    _fields_ = [
        ("stype", ct.c_uint32),
        ("pNext", ct.c_void_p),
        ("activeTime", ct.c_uint64),  # microseconds
        ("timestamp", ct.c_uint64),   # microseconds
    ]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _vp(raw_int: int | None) -> ct.c_void_p:
    """Wrap a raw pointer int from a ctypes array into c_void_p so it
    is passed as a full 64-bit pointer on x64 (plain ints get truncated
    to c_long = 32-bit by the ctypes default argument conversion)."""
    return ct.c_void_p(raw_int)


# ── GpuMonitor ───────────────────────────────────────────────────────────────

class GpuMonitor:
    """Collect Intel Arc GPU metrics via Level Zero Sysman.

    >>> mon = GpuMonitor()
    >>> mon.start()          # background thread polls every 2 s
    >>> mon.snapshot()       # thread-safe latest readings dict
    >>> mon.stop()
    """

    def __init__(self) -> None:
        self._lib: Any = None
        self._device: ct.c_void_p | None = None
        self._temp_h: list[ct.c_void_p] = []
        self._mem_h: list[ct.c_void_p] = []
        self._freq_h: list[ct.c_void_p] = []
        self._power_h: list[ct.c_void_p] = []
        self._engine_h: list[ct.c_void_p] = []
        self._fan_h: list[ct.c_void_p] = []
        self._prev_energy: tuple[int, int] | None = None
        self._prev_engine: dict[int, tuple[int, int]] = {}
        self._lock = threading.Lock()
        self._latest: dict[str, Any] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._available = False
        self._init()

    @property
    def available(self) -> bool:
        return self._available

    # ── Initialisation ───────────────────────────────────────────────────

    def _init(self) -> None:
        try:
            self._lib = ct.CDLL("ze_loader.dll")
        except OSError:
            return

        device = self._try_zes_init() or self._try_ze_init()
        if device is None:
            return
        self._device = device

        self._temp_h = self._enum("zesDeviceEnumTemperatureSensors")
        self._mem_h = self._enum("zesDeviceEnumMemoryModules")
        self._freq_h = self._enum("zesDeviceEnumFrequencyDomains")
        self._power_h = self._enum("zesDeviceEnumPowerDomains")
        self._engine_h = self._enum("zesDeviceEnumEngineGroups")
        self._fan_h = self._enum("zesDeviceEnumFans")
        self._available = True

    def _try_zes_init(self) -> ct.c_void_p | None:
        """Separate Sysman init (Level Zero >= 1.5)."""
        try:
            if self._lib.zesInit(ct.c_uint32(0)) != _OK:
                return None
            n = ct.c_uint32(0)
            if self._lib.zesDriverGet(ct.byref(n), None) != _OK or n.value == 0:
                return None
            drv = (ct.c_void_p * n.value)()
            self._lib.zesDriverGet(ct.byref(n), drv)

            dc = ct.c_uint32(0)
            if self._lib.zesDeviceGet(_vp(drv[0]), ct.byref(dc), None) != _OK or dc.value == 0:
                return None
            dev = (ct.c_void_p * dc.value)()
            self._lib.zesDeviceGet(_vp(drv[0]), ct.byref(dc), dev)
            return _vp(dev[0])
        except Exception:
            return None

    def _try_ze_init(self) -> ct.c_void_p | None:
        """Combined init (older Level Zero — needs ZES_ENABLE_SYSMAN)."""
        try:
            os.environ.setdefault("ZES_ENABLE_SYSMAN", "1")
            if self._lib.zeInit(ct.c_uint32(0)) != _OK:
                return None
            n = ct.c_uint32(0)
            if self._lib.zeDriverGet(ct.byref(n), None) != _OK or n.value == 0:
                return None
            drv = (ct.c_void_p * n.value)()
            self._lib.zeDriverGet(ct.byref(n), drv)

            dc = ct.c_uint32(0)
            if self._lib.zeDeviceGet(_vp(drv[0]), ct.byref(dc), None) != _OK or dc.value == 0:
                return None
            dev = (ct.c_void_p * dc.value)()
            self._lib.zeDeviceGet(_vp(drv[0]), ct.byref(dc), dev)
            return _vp(dev[0])
        except Exception:
            return None

    def _enum(self, fn_name: str) -> list[ct.c_void_p]:
        """Enumerate Sysman domain handles (temp sensors, mem modules, …)."""
        try:
            fn = getattr(self._lib, fn_name)
            n = ct.c_uint32(0)
            if fn(self._device, ct.byref(n), None) != _OK or n.value == 0:
                return []
            arr = (ct.c_void_p * n.value)()
            fn(self._device, ct.byref(n), arr)
            return [_vp(arr[i]) for i in range(n.value)]
        except Exception:
            return []

    # ── Metric reading ───────────────────────────────────────────────────

    def _poll(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        lib = self._lib

        # Temperature (first sensor → GPU die)
        for h in self._temp_h[:1]:
            try:
                t = ct.c_double(0)
                if lib.zesTemperatureGetState(h, ct.byref(t)) == _OK and t.value > 0:
                    data["gpu_temp_c"] = round(t.value, 1)
            except Exception:
                pass

        # VRAM (first memory module → device memory)
        for h in self._mem_h[:1]:
            try:
                s = _MemState(stype=_STYPE_MEM_STATE)
                if lib.zesMemoryGetState(h, ct.byref(s)) == _OK and s.size > 0:
                    used = s.size - s.free
                    data["vram_used_gb"] = round(used / (1 << 30), 2)
                    data["vram_total_gb"] = round(s.size / (1 << 30), 1)
                    data["vram_used_pct"] = round(100 * used / s.size, 1)
            except Exception:
                pass

        # Frequency — domain 0 = GPU core, domain 1 = VRAM
        for i, h in enumerate(self._freq_h[:2]):
            try:
                s = _FreqState(stype=_STYPE_FREQ_STATE)
                if lib.zesFrequencyGetState(h, ct.byref(s)) != _OK:
                    continue
                if i == 0:
                    if s.actual > 0:
                        data["gpu_freq_mhz"] = round(s.actual)
                    if s.currentVoltage > 0:
                        data["gpu_voltage_v"] = round(s.currentVoltage, 3)
                    data["gpu_throttle"] = s.throttleReasons
                elif i == 1 and s.actual > 0:
                    data["vram_freq_mhz"] = round(s.actual)
            except Exception:
                pass

        # Power (W) — delta of cumulative energy counter
        for h in self._power_h[:1]:
            try:
                c = _PowerEnergy()
                if lib.zesPowerGetEnergyCounter(h, ct.byref(c)) == _OK and c.timestamp > 0:
                    if self._prev_energy is not None:
                        de = c.energy - self._prev_energy[0]
                        dt = c.timestamp - self._prev_energy[1]
                        if dt > 0:
                            data["gpu_power_w"] = round(de / dt, 1)  # µJ / µs = W
                    self._prev_energy = (c.energy, c.timestamp)
            except Exception:
                pass

        # GPU utilisation (%) — delta of engine active time
        for i, h in enumerate(self._engine_h[:1]):
            try:
                s = _EngineStats()
                if lib.zesEngineGetActivity(h, ct.byref(s)) == _OK and s.timestamp > 0:
                    if i in self._prev_engine:
                        da = s.activeTime - self._prev_engine[i][0]
                        dt = s.timestamp - self._prev_engine[i][1]
                        if dt > 0:
                            data["gpu_util_pct"] = round(min(100.0, 100.0 * da / dt), 1)
                    self._prev_engine[i] = (s.activeTime, s.timestamp)
            except Exception:
                pass

        # Fan speed (RPM)  — units arg: 0 = RPM
        for h in self._fan_h[:1]:
            try:
                speed = ct.c_int32(0)
                if lib.zesFanGetState(h, ct.c_uint32(0), ct.byref(speed)) == _OK:
                    if speed.value > 0:
                        data["fan_rpm"] = speed.value
            except Exception:
                pass

        data["timestamp"] = time.time()
        return data

    # ── Public API ───────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return the latest metrics dict (thread-safe)."""
        with self._lock:
            return dict(self._latest)

    def start(self, interval: float = 2.0) -> None:
        """Start background polling thread."""
        if self._running or not self._available:
            return
        self._running = True

        def _loop() -> None:
            while self._running:
                data = self._poll()
                with self._lock:
                    self._latest = data
                time.sleep(interval)

        self._thread = threading.Thread(target=_loop, daemon=True, name="gpu-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
