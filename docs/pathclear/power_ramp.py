#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Soft-start dI/dt mitigation harness — PROVEN on ttstar's BH p150a + 850W PSU (2026-06-24).

WHY: an open-loop idle->full-grid device load browns out the 850W PSU and reboots the whole host
(dI/dt transient exceeds PSU OCP / VR current slew — the efficiency-vs-ICCmax wall). A *gradual*
ramp to the same steady-state is fine: a continuous soft-start reached 268W with NO reboot.

WHAT: RampController runs `dispatch_fn(intensity)` continuously while ramping intensity 0->1 over
`ramp_s`, samples telemetry in a BACKGROUND thread (inline sampling starves the load -> duty 0 ->
power never leaves idle), and aborts on a power ceiling or a too-fast per-step jump.

3DGS FRAMING: this is for running the device 3DGS kernels (M13/M14 rasterizer, M16 training loop)
at full grid without tripping the PSU. The SAME harness wraps the real splatting dispatch — see
`raster_load()`, which ramps the M13 rasterizer's ACTIVE CORE GRID 1x1->11x10 linearly (cliff-free,
unlike ttnn.matmul whose auto-sharding hides a ~70W->268W core-count cliff). For heavy scaled runs
swap in the M14 batched dispatch. `matmul_load()` is the synthetic characterization load.

  python power_ramp.py --load matmul --pmax 230 --ramp 80     # characterize the PSU knee
  python power_ramp.py --load raster --pmax 230 --ramp 60     # ramp the real 3DGS rasterizer
"""
import sys, os, re, time, subprocess, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def tt_smi_telem():
    """One ARC telemetry sample: {power(W), aiclk(MHz), temp(C), fan(%)}. None on failure."""
    try:
        out = subprocess.run(["tt-smi", "-s"], capture_output=True, text=True, timeout=25).stdout
    except Exception:
        return {"power": None, "aiclk": None, "temp": None, "fan": None}

    def g(key):
        m = re.search(rf'"{key}":\s*"?\s*([0-9.]+)', out)
        return float(m.group(1)) if m else None

    a = g("aiclk")
    return {"power": g("power"), "aiclk": int(a) if a else None,
            "temp": g("asic_temperature"), "fan": g("fan_speed")}


class RampController:
    """Continuous soft-start ramp with background telemetry + closed-loop abort. Reusable across
    any device load (synthetic or real 3DGS dispatch). Returns the telemetry history (power curve)."""

    def __init__(self, pmax=230.0, dpmax=50.0, ramp_s=80.0, time_s=130.0,
                 sample_s=2.0, telem_fn=tt_smi_telem, log=print):
        self.pmax, self.dpmax = pmax, dpmax          # abort ceiling (W); max dP between samples (W)
        self.ramp_s, self.time_s = ramp_s, time_s    # ramp window; overall time cap
        self.sample_s = sample_s
        self.telem_fn, self.log = telem_fn, log

    def run(self, dispatch_fn):
        st = {"power": None, "aiclk": None, "temp": None, "fan": None,
              "abort": False, "reason": "", "peak": 0.0, "hist": []}

        def monitor():
            last_p = None
            while not st["abort"]:
                t = self.telem_fn()
                st.update(t)
                p = t["power"]
                if p is not None:
                    st["peak"] = max(st["peak"], p)
                    st["hist"].append((round(time.perf_counter() - t0, 1), p, t["aiclk"], t["temp"], t["fan"]))
                    if p > self.pmax:
                        st["abort"], st["reason"] = True, f"power {p}W > ceiling {self.pmax}W"
                    elif last_p is not None and p - last_p > self.dpmax:
                        st["abort"], st["reason"] = True, f"dP +{p-last_p:.0f}W/sample > {self.dpmax}W"
                    last_p = p
                time.sleep(self.sample_s)

        t0 = time.perf_counter()
        mon = threading.Thread(target=monitor, daemon=True)
        mon.start()
        time.sleep(self.sample_s + 1.0)              # let the first sample land
        self.log(f"ramp start: idle {st['power']}W {st['aiclk']}MHz {st['temp']}C "
                 f"-> ramp {self.ramp_s:.0f}s, ceiling {self.pmax}W")

        last_log = 0.0
        while not st["abort"] and (time.perf_counter() - t0) < self.time_s:
            frac = min(1.0, (time.perf_counter() - t0) / self.ramp_s)   # 0..1 soft-start
            dispatch_fn(frac)                                           # one continuous unit of work
            now = time.perf_counter()
            if now - last_log >= 4.0:
                self.log(f"  t={now-t0:5.1f}s  load={frac:4.2f}  {st['power']}W  "
                         f"{st['aiclk']}MHz  {st['temp']}C  fan{st['fan']}%")
                last_log = now

        st["abort"] = True
        mon.join(timeout=self.sample_s + 1)
        tag = f"ABORTED ({st['reason']})" if st["reason"] else "completed"
        self.log(f"ramp {tag} — peak ~{st['peak']}W, final {st['power']}W {st['aiclk']}MHz "
                 f"{st['temp']}C fan{st['fan']}%")
        return st


# ── Loads (dispatch_fn(intensity in [0,1]) -> one continuous unit of device work) ─────────────

def matmul_load(dev, szmin=512, szmax=4096, depth=8):
    """Synthetic characterization load: bf16 matmul whose SIZE scales with intensity. NOTE the
    matmul core-count cliff (flat ~70W then jumps to full grid) — fine for finding the PSU knee,
    but use raster_load / a synthetic CoreRangeSet kernel for cliff-free linear ramping."""
    import torch, ttnn
    state = {"sz": None, "ab": None}

    def dispatch(intensity):
        sz = max(szmin, ((int(szmin + intensity * (szmax - szmin))) // 256) * 256)  # 256-quantized
        if sz != state["sz"]:
            if state["ab"]:
                state["ab"][0].deallocate(); state["ab"][1].deallocate()
            t = lambda: ttnn.from_torch(torch.randn(sz, sz), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=dev)
            state["ab"], state["sz"] = (t(), t()), sz
        a, b = state["ab"]
        cs = [ttnn.matmul(a, b) for _ in range(depth)]
        _ = ttnn.to_torch(cs[-1])[0, 0].item()       # bounded sync
        for c in cs:
            c.deallocate()
    return dispatch


def raster_load(dev, gx_max=11, gy_max=10, reps=8):
    """The REAL 3DGS device rasterizer (M13) as a ramp load: intensity -> ACTIVE CORE GRID
    1x1 .. gx_max x gy_max. Linear core ramp (cliff-free) of the actual splatting kernel — this is
    the harness wrapping the production 3DGS dispatch. Light (dispatch-bound); for heavy scaled
    power swap in the M14 batched dispatch here."""
    import sfpu_raster_multitile as M
    cache = {}

    def dispatch(intensity):
        gx = max(1, int(round(intensity * gx_max)))
        gy = max(1, int(round(intensity * gy_max)))
        if (gx, gy) not in cache:
            cache[(gx, gy)] = M.scene(2, gx * M.TS, gy * M.TS)
        cx, cy, op, col, order, abc = cache[(gx, gy)]
        M.render_multitile(dev, gx, gy, cx, cy, op, col, order, abc, validate=False, reps=reps)
    return dispatch


def main():
    import argparse, ttnn
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", choices=["matmul", "raster"], default="matmul")
    ap.add_argument("--pmax", type=float, default=200.0, help="abort ceiling (W)")
    ap.add_argument("--ramp", type=float, default=80.0, help="seconds to ramp idle->full")
    ap.add_argument("--time", type=float, default=130.0, help="overall time cap (s)")
    a = ap.parse_args()

    dev = ttnn.open_device(device_id=0)
    try:
        load = matmul_load(dev) if a.load == "matmul" else raster_load(dev)
        st = RampController(pmax=a.pmax, ramp_s=a.ramp, time_s=a.time).run(load)
        print(f"RESULT peak={st['peak']}W final={st['power']}W aborted={bool(st['reason'])} {st['reason']}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
