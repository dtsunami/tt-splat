#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Stage 2 de-risk: find the ttnn op sequence that does the PER-TILE spatial reduce ON DEVICE,
replacing fused_backward.py:139
    hs = ttnn.to_torch(o).reshape(GY, FUSED_K, TS, GX, TS).sum(dim=(2,4)).numpy()   # full readback + HOST sum

The output `o` is a BLOCK-SHARDED TILE_LAYOUT tensor [1,1, GY*FUSED_K*32, GX*32]: a grid of
(GY*FUSED_K) x GX independent 32x32 tiles (shard [FUSED_K*32, 32] per core = FUSED_K stacked tiles).
Goal: reduce each 32x32 tile -> one scalar => [GY, FUSED_K, GX], then read back only that tiny result.

E2 verdict: in-kernel reduce_tile is broken in this build -> reduce via ttnn.sum. The open question is
the reshape/permute to peel the GX-interleaved tiles onto a batch axis so ttnn.sum hits each tile alone.
Try candidates, gate rel_err < 2e-2 (bf16 accum) vs host fp64; time each vs the full-readback baseline.
"""
import sys, time
from pathlib import Path
import numpy as np, torch, ttnn

sys.path.insert(0, str(Path.home() / "tt-splat" / "server"))

TS = 32
FUSED_K = 16
GX, GY = 3, 3                  # exercise multi-tile both dims (corgi 96px = 3x3)
GYK = GY * FUSED_K
SHF = FUSED_K * TS
H, W = GY * SHF, GX * TS

dev = ttnn.open_device(device_id=0)


def sync():
    try: ttnn.synchronize_device(dev)
    except Exception: pass


def make_block(data):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
    sh = ttnn.ShardSpec(grid, [SHF, TS], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, sh)
    return ttnn.from_torch(data.reshape(1, 1, H, W).float(), dtype=ttnn.float32,
                           layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


def to_interleaved(o):
    for args in ((o, ttnn.DRAM_MEMORY_CONFIG), (o, ttnn.L1_MEMORY_CONFIG), (o,)):
        try: return ttnn.sharded_to_interleaved(*args)
        except Exception: continue
    raise RuntimeError("sharded_to_interleaved: no form worked")


# -------- candidate reduce sequences: o(block-sharded) -> np[GY, FUSED_K, GX] --------
def cand_reshape_permute_tile(o):
    oi = to_interleaved(o)
    r = ttnn.reshape(oi, (GYK, TS, GX, TS))      # h->(gyk,ih), w->(gx,iw)
    r = ttnn.permute(r, (0, 2, 1, 3))            # (GYK, GX, TS, TS)
    r = ttnn.sum(r, dim=[2, 3])                  # (GYK, GX)
    return ttnn.to_torch(r).reshape(GY, FUSED_K, GX).numpy()


def cand_rowmajor_permute(o):
    oi = to_interleaved(o)
    rm = ttnn.to_layout(oi, ttnn.ROW_MAJOR_LAYOUT)
    r = ttnn.reshape(rm, (GYK, TS, GX, TS))
    r = ttnn.permute(r, (0, 2, 1, 3))            # (GYK, GX, TS, TS)
    r = ttnn.reshape(r, (GYK * GX, TS * TS))
    r = ttnn.to_layout(r, ttnn.TILE_LAYOUT)
    r = ttnn.sum(r, dim=1)                        # (GYK*GX,)
    return ttnn.to_torch(r).reshape(GY, FUSED_K, GX).numpy()


def cand_two_axis_sum(o):
    oi = to_interleaved(o)
    r = ttnn.reshape(oi, (1, H, GX, TS))         # w->(gx,iw)
    r = ttnn.sum(r, dim=3)                        # (1,H,GX) sum over iw
    r = ttnn.reshape(r, (GYK, TS, GX))           # h->(gyk,ih)
    r = ttnn.sum(r, dim=1)                        # (GYK,GX) sum over ih
    return ttnn.to_torch(r).reshape(GY, FUSED_K, GX).numpy()


def cand_permute_then_2axis(o):
    oi = to_interleaved(o)
    r = ttnn.reshape(oi, (GYK, TS, GX, TS))
    r = ttnn.permute(r, (0, 2, 1, 3))            # (GYK, GX, TS, TS)
    r = ttnn.reshape(r, (GYK * GX, TS, TS))
    r = ttnn.sum(r, dim=[1, 2])
    return ttnn.to_torch(r).reshape(GY, FUSED_K, GX).numpy()


CANDS = [
    ("reshape+permute(tile)", cand_reshape_permute_tile),
    ("rowmajor+permute", cand_rowmajor_permute),
    ("two-axis-sum", cand_two_axis_sum),
    ("permute+2axis", cand_permute_then_2axis),
]

def cand_sharded_direct(o):
    # try reducing the BLOCK-SHARDED tensor directly (skip sharded_to_interleaved copy)
    r = ttnn.reshape(o, (GYK, TS, GX, TS))
    r = ttnn.permute(r, (0, 2, 1, 3))
    r = ttnn.sum(r, dim=[2, 3])
    return ttnn.to_torch(r).reshape(GY, FUSED_K, GX).numpy()


CANDS.append(("sharded-direct", cand_sharded_direct))

try:
    torch.manual_seed(0)
    data = torch.randn(H, W).double()
    ref = data.reshape(GY, FUSED_K, TS, GX, TS).sum(dim=(2, 4)).numpy()     # host gold [GY,FUSED_K,GX]
    o = make_block(data)

    # baseline: today's full readback + host sum
    sync(); t0 = time.perf_counter()
    for _ in range(20):
        base = ttnn.to_torch(o).reshape(GY, FUSED_K, TS, GX, TS).sum(dim=(2, 4)).numpy()
    sync(); base_ms = 1e3 * (time.perf_counter() - t0) / 20
    print(f"shape [1,1,{H},{W}]  GX={GX} GY={GY} FUSED_K={FUSED_K}  (one of 7 outputs)")
    print(f"  BASELINE full-readback+host-sum = {base_ms:.3f} ms/call (per output)\n")

    print(f"  {'candidate':24s} {'status':8s} {'rel_err':>10s} {'ms/call':>9s}")
    for name, fn in CANDS:
        try:
            got = fn(o)
            e = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9)
            sync(); t0 = time.perf_counter()
            for _ in range(20): fn(o)
            sync(); ms = 1e3 * (time.perf_counter() - t0) / 20
            ok = "OK" if e < 2e-2 else "GATEFAIL"
            print(f"  {name:24s} {ok:8s} {e:10.2e} {ms:9.3f}")
        except Exception as ex:
            print(f"  {name:24s} {'ERROR':8s} {'-':>10s} {'-':>9s}   {type(ex).__name__}: {str(ex)[:80]}")

    # ---- THE REAL TEST: 7 outputs per (chunk,channel). baseline = 7 readbacks; can batching amortize? ----
    print(f"\n  --- 7-output round (real line 139 does 7 per chunk*channel) ---")
    outs7 = [make_block(torch.randn(H, W).double()) for _ in range(7)]
    sync(); t0 = time.perf_counter()
    for _ in range(20):
        hs = [ttnn.to_torch(x).reshape(GY, FUSED_K, TS, GX, TS).sum(dim=(2, 4)).numpy() for x in outs7]
    sync(); b7 = 1e3 * (time.perf_counter() - t0) / 20
    print(f"  BASELINE 7x(readback+host-sum)            = {b7:.3f} ms")

    def batched7(outs7):
        oi = ttnn.concat([to_interleaved(x) for x in outs7], dim=2)   # [1,1,7*H,W]
        r = ttnn.reshape(oi, (7 * GYK, TS, GX, TS))
        r = ttnn.permute(r, (0, 2, 1, 3))
        r = ttnn.sum(r, dim=[2, 3])
        return ttnn.to_torch(r).reshape(7, GY, FUSED_K, GX).numpy()
    try:
        g = batched7(outs7)
        ref7 = np.stack([ttnn.to_torch(x).reshape(GY, FUSED_K, TS, GX, TS).sum(dim=(2, 4)).numpy() for x in outs7])
        e = np.abs(g - ref7).max() / (np.abs(ref7).max() + 1e-9)
        sync(); t0 = time.perf_counter()
        for _ in range(20): batched7(outs7)
        sync(); ms = 1e3 * (time.perf_counter() - t0) / 20
        print(f"  ON-DEVICE batched-7 (concat+reduce+1 readback) = {ms:.3f} ms  rel_err={e:.2e}  "
              f"{'OK' if e < 2e-2 else 'GATEFAIL'}")
    except Exception as ex:
        print(f"  ON-DEVICE batched-7 ERROR: {type(ex).__name__}: {str(ex)[:100]}")
finally:
    ttnn.close_device(dev)
