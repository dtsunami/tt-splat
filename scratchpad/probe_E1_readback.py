#!/usr/bin/env python3
# E1: is the Stage A readback latency-bound (call count) or bandwidth-bound (volume)?
# Time to_torch per call on (a) the real [GY*FK*32, GX*32] Stage-A output, (b) a single 32x32 tile,
# (c) the post-reduce [GY*FK, GX] scalar block. If per-call ms is ~flat across sizes -> latency-bound
# (call count is the lever, Stage 3); if it scales with bytes -> bandwidth-bound (reduce bytes, Stage 2).
import sys, time
from pathlib import Path
import torch, ttnn
sys.path.insert(0, str(Path.home() / "tt-splat" / "server"))
sys.path.insert(0, str(Path.home() / "tt-splat" / "docs" / "pathclear"))
from fused_backward import _block, TS, FUSED_K

dev = ttnn.open_device(device_id=0)
def sync():
    try: ttnn.synchronize_device(dev)
    except Exception: pass

def bench(label, totH, totW, shard_h, reps=200):
    GX = totW // TS
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, totH // shard_h - 1))])
    t = _block(dev, grid, totH, totW, shard_h, torch.randn(totH, totW))
    bytes_ = totH * totW * 4
    for _ in range(5): ttnn.to_torch(t)          # warm
    sync(); t0 = time.perf_counter()
    for _ in range(reps): ttnn.to_torch(t)
    sync(); ms = 1e3 * (time.perf_counter() - t0) / reps
    bw = bytes_ / (ms / 1e3) / 1e9
    print(f"  {label:32s} shape=[{totH},{totW}] {bytes_/1024:7.1f}KB  per_call={ms:6.3f}ms  BW={bw:6.2f}GB/s", flush=True)
    return ms, bytes_

try:
    # 96px scene: GX=GY=3, SHF=FUSED_K*TS
    GX = GY = 3; SHF = FUSED_K * TS; Wp = GX * TS
    print(f"E1 readback latency-vs-bandwidth (FUSED_K={FUSED_K}, TS={TS})")
    m_full, b_full = bench("(a) real Stage-A output", GY * SHF, Wp, SHF)
    m_tile, b_tile = bench("(b) single 32x32 tile", TS, TS, TS)
    m_red,  b_red  = bench("(c) post-reduce [GY*FK,GX*1]", GY * FUSED_K * TS, Wp, FUSED_K * TS)  # placeholder same alloc granule
    # tiny: a [32,32] holding GY*FK scalars (what a packed reduce drains)
    m_sc, b_sc = bench("(d) one 32x32 scalar block", TS, TS, TS)
    print(f"\n  full/tile per-call ratio = {m_full/m_tile:.2f}x  (bytes ratio = {b_full/b_tile:.0f}x)")
    verdict = "LATENCY-BOUND (call count) -> Stage 3 (drain-once) is the big lever; Stage 2 cuts the per-call floor too" \
              if (m_full/m_tile) < 3.0 else \
              "BANDWIDTH-BOUND (volume) -> Stage 2 (on-device reduce, ~1024x bytes) is the big lever"
    print(f"  VERDICT: {verdict}")
    # extrapolate current readback: ~21*nbatch calls; at nbatch=63 -> 1323 calls
    calls = 21 * 63
    print(f"  extrapolated current readback @ nbatch=63: {calls} calls x {m_full:.3f}ms = {calls*m_full:.0f}ms (measured ~260ms)")
finally:
    ttnn.close_device(dev)
