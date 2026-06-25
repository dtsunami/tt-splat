#!/usr/bin/env python3
"""Soft-start Tensix power ramp — avoid the dI/dt that OCP-trips the PSU.

Ramps matmul SIZE (steady-state magnitude) and DUTY-CYCLE (PWM soft-start: small bursts with
shrinking idle gaps) so current rises gradually. Reads ARC telemetry between size steps and
ABORTS on a power ceiling or a too-fast per-step jump. Goal: reach elevated *sustained* power
without a transient. Conservative ceiling on the first run — well below the ~300W+ that rebooted.
"""
import sys, time, os, re, subprocess, torch, ttnn

PMAX  = float(os.environ.get("PMAX", "230"))    # abort ceiling (W) — under the ~300W+ trip
DPMAX = float(os.environ.get("DPMAX", "45"))    # max power jump per size-step (W)
TLIM  = float(os.environ.get("TLIM", "150"))    # overall time cap (s)

def telem():
    try:
        out = subprocess.run(["tt-smi", "-s"], capture_output=True, text=True, timeout=25).stdout
        gp = lambda k: re.search(rf'"{k}":\s*"?\s*([0-9.]+)', out)
        p, a = gp("power"), re.search(r'"aiclk":\s*"?\s*([0-9]+)', out)
        t, f = re.search(r'asic_temperature":\s*"?\s*([0-9.]+)', out), gp("fan_speed")
        return (float(p.group(1)) if p else None, int(a.group(1)) if a else None,
                float(t.group(1)) if t else None, float(f.group(1)) if f else None)
    except Exception:
        return (None, None, None, None)

dev = ttnn.open_device(device_id=0)
g = dev.compute_with_storage_grid_size()
print(f"RAMP start: grid {g.x}x{g.y}={g.x*g.y}, ceiling={PMAX}W — soft-start size+duty", flush=True)
p0, a0, t0, f0 = telem(); print(f"  idle: {p0}W {a0}MHz {t0}C fan{f0}%", flush=True)

t_start = time.perf_counter()
last_p = p0 or 22.0
SIZES = [256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096]
DUTY  = [1, 2, 4, 8, 16]                          # matmuls/burst — ramps within each size
GAP   = [0.20, 0.12, 0.07, 0.03, 0.0]            # idle gap between bursts, shrinks (PWM)
aborted = False
for sz in SIZES:
    if time.perf_counter() - t_start > TLIM:
        print("  time cap reached", flush=True); break
    a = ttnn.from_torch(torch.randn(sz, sz), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(torch.randn(sz, sz), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    for duty, gap in zip(DUTY, GAP):              # PWM soft-start within the size step
        cs = [ttnn.matmul(a, b) for _ in range(duty)]
        _ = ttnn.to_torch(cs[-1])[0, 0].item()    # bounded sync
        for c in cs:
            c.deallocate()
        if gap:
            time.sleep(gap)
    p, aclk, tC, fan = telem()
    dp = (p - last_p) if (p and last_p) else 0.0
    print(f"  sz={sz:<4} -> {p}W (+{dp:.0f})  {aclk}MHz  {tC}C  fan{fan}%", flush=True)
    a.deallocate(); b.deallocate()
    if p and p > PMAX:
        print(f"  ABORT: {p}W > ceiling {PMAX}W — backing off", flush=True); aborted = True; break
    if dp > DPMAX:
        print(f"  ABORT: dP +{dp:.0f}W in one step > {DPMAX}W — too steep", flush=True); aborted = True; break
    last_p = p or last_p
print(f"RAMP {'ABORTED' if aborted else 'completed full ramp'} — peak ~{last_p}W", flush=True)
ttnn.close_device(dev)
