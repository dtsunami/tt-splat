#!/usr/bin/env python3
"""Continuous soft-start ramp — device stays busy, intensity ramps over wall-clock time,
telemetry sampled in a BACKGROUND thread (so it doesn't starve the load), aborts on ceiling.
Size grows smoothly 512->4096 over RAMP seconds => controlled dI/dt, real sustained power."""
import sys, time, os, re, subprocess, threading, torch, ttnn

PMAX = float(os.environ.get("PMAX", "200"))   # abort ceiling (W)
TLIM = float(os.environ.get("TLIM", "120"))   # overall time cap (s)
RAMP = float(os.environ.get("RAMP", "80"))    # seconds to ramp min->max intensity

st = {"power": None, "aiclk": None, "temp": None, "fan": None, "abort": False, "peak": 0.0}

def monitor():
    while not st["abort"]:
        try:
            out = subprocess.run(["tt-smi", "-s"], capture_output=True, text=True, timeout=25).stdout
            f = lambda k, cast=float: (cast(re.search(rf'"{k}":\s*"?\s*([0-9.]+)', out).group(1))
                                       if re.search(rf'"{k}":\s*"?\s*([0-9.]+)', out) else None)
            st["power"], st["aiclk"] = f("power"), f("aiclk", lambda x: int(float(x)))
            st["temp"],  st["fan"]   = f("asic_temperature"), f("fan_speed")
            if st["power"]:
                st["peak"] = max(st["peak"], st["power"])
                if st["power"] > PMAX:
                    st["abort"] = True
                    print(f"  [monitor] ABORT: {st['power']}W > ceiling {PMAX}W", flush=True)
        except Exception:
            pass
        time.sleep(2.5)

dev = ttnn.open_device(device_id=0)
threading.Thread(target=monitor, daemon=True).start()
time.sleep(4)
print(f"RAMP2 start: idle {st['power']}W {st['aiclk']}MHz {st['temp']}C — continuous, ramp {RAMP:.0f}s, ceiling {PMAX}W", flush=True)

def mk(sz):
    t = lambda: ttnn.from_torch(torch.randn(sz, sz), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    return t(), t()

SZMIN, SZMAX = 512, 4096
t0 = time.perf_counter(); last = 0.0; cur = None; a = b = None
while not st["abort"] and (time.perf_counter() - t0) < TLIM:
    frac = min(1.0, (time.perf_counter() - t0) / RAMP)          # 0..1 smooth ramp
    sz = max(SZMIN, ((int(SZMIN + frac * (SZMAX - SZMIN))) // 32) * 32)
    if sz != cur:
        if a: a.deallocate(); b.deallocate()
        a, b = mk(sz); cur = sz
    cs = [ttnn.matmul(a, b) for _ in range(8)]                  # continuous — no idle gap
    _ = ttnn.to_torch(cs[-1])[0, 0].item()                     # bounded sync
    for c in cs: c.deallocate()
    now = time.perf_counter()
    if now - last >= 4:
        print(f"  t={now-t0:5.1f}s  sz={sz:<4} -> {st['power']}W  {st['aiclk']}MHz  {st['temp']}C  fan{st['fan']}%", flush=True)
        last = now
st["abort"] = True
print(f"RAMP2 done — peak ~{st['peak']}W, final {st['power']}W {st['aiclk']}MHz {st['temp']}C fan{st['fan']}%", flush=True)
ttnn.close_device(dev)
