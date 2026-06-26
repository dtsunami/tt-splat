#!/usr/bin/env python3
# E2 (revised): in-kernel reduce_tile<SUM,REDUCE_SCALAR> is ABANDONED/broken in this build (base
# smoke_reduce.py also returns 0). The proven on-device reduce here is ttnn.sum (M15/M16). Test whether
# ttnn-based on-device reduction holds the 2e-2 grad gate on 1024-term SIGNED sums (fp32), and time it
# vs the readback it would replace (E1: ~0.116ms/full-tile-block readback).
import sys, time
from pathlib import Path
import torch, ttnn

dev = ttnn.open_device(device_id=0)
def sync():
    try: ttnn.synchronize_device(dev)
    except Exception: pass

def dt(t):
    return ttnn.from_torch(t.reshape(1, 1, *t.shape).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

def reduce_sum(t):
    """On-device full-tile sum -> scalar. Try the forms that exist in this ttnn build."""
    d = dt(t)
    for kwargs in ({"dim": [2, 3]}, {"dim": [-2, -1]}, {"dim": 3}):
        try:
            r = ttnn.sum(d, **kwargs)
            return float(ttnn.to_torch(r).flatten()[0]) if kwargs.get("dim") in ([2, 3], [-2, -1]) \
                else float(ttnn.to_torch(ttnn.sum(r, dim=2)).flatten()[0])
        except Exception:
            continue
    raise RuntimeError("no ttnn.sum form worked")

try:
    torch.manual_seed(0)
    scenarios = {
        "positive ~U(0,1)":        torch.rand(32, 32),
        "signed ~N(0,1)":          torch.randn(32, 32),
        "signed large ~N(0,1e3)":  torch.randn(32, 32) * 1e3,
        "grad-like dLdC*a*dxdy":   (torch.randn(32, 32) * 0.01) * (torch.rand(32, 32) * 60 - 30),
        "near-cancel (sum~0)":     torch.randn(32, 32) - torch.randn(32, 32).mean(),
    }
    print("E2 on-device reduce precision via ttnn.sum (1024-term, vs host fp64). gate rel_err < 2e-2")
    print(f"  {'scenario':26s} {'gold':>14s} {'ttnn_sum':>14s} {'rel_err':>10s}")
    worst = 0.0
    for name, d in scenarios.items():
        gold = float(d.double().sum())
        got = reduce_sum(d)
        e = abs(got - gold) / (abs(gold) + 1e-9)
        worst = max(worst, e)
        print(f"  {name:26s} {gold:14.4f} {got:14.4f} {e:10.2e}", flush=True)
    print(f"\n  worst ttnn.sum rel_err = {worst:.2e}  -> {'OK (on-device reduce holds gate)' if worst < 2e-2 else 'FAIL'}")

    # speed: ttnn.sum over a [1,1,32,32] tile, per call, vs E1 readback (~0.116ms full block)
    d = dt(torch.randn(32, 32))
    for _ in range(5): ttnn.sum(d, dim=[2, 3])
    sync(); t0 = time.perf_counter()
    for _ in range(100): ttnn.sum(d, dim=[2, 3])
    sync(); ms = 1e3 * (time.perf_counter() - t0) / 100
    print(f"  ttnn.sum per-call = {ms:.3f}ms (a SEPARATE dispatch; vs in-kernel reduce which is free but broken)")
    print("  FINDING: in-kernel reduce_tile broken/abandoned -> Stage 2 reduces via ttnn.sum OR a RISC L1")
    print("           loop (m2-style, fp32 bit-exact). reduce_tile is NOT a dependency.")
finally:
    ttnn.close_device(dev)
