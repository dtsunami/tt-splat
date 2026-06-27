#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PROBE S2-dev increment 2 (silicon) — ttnn-computed AABB + SCALE the single-core counting sort to 1600px.

increment 1 proved the kernel @96px with host AABB. Here:
  (a) AABB/bucket computed in TTNN (the device S1 math) — the device-resident path, not host_assign;
  (b) SCALE to 1600px on a densified real scene (~32k Gaussians), packing the 5 AABB fields into ONE
      uint32/Gaussian (each <64 @1600 -> 6 bits) so histogram(487KB)+ginfo+sgid fit single-core L1.

Gates: (1) ttnn AABB == host AABB at scale (S1 re-check); (2) kernel sort per-tile gid SET == host
counting sort on the same AABB; + report device sort time. (50k needs sgid->DRAM, the S4 substrate.)
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "pathclear"))
from probe_S0_depthbucket import load_ply, project_cam, PLY, DATASET     # noqa: E402
from probe_S1_tile_assign import host_assign                              # noqa: E402
from probe_S2_dev_countsort import l1_u32, host_countsort, HOME           # noqa: E402
from train_tt import _load_colmap                                         # noqa: E402

TS = 32
D = 64
TARGET_N = 32000

KERNEL = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t ginfo_a = get_arg_val<uint32_t>(0);
    uint32_t N    = get_arg_val<uint32_t>(1);
    uint32_t ntx  = get_arg_val<uint32_t>(2);
    uint32_t nty  = get_arg_val<uint32_t>(3);
    uint32_t D    = get_arg_val<uint32_t>(4);
    uint32_t hist_a = get_arg_val<uint32_t>(5);
    uint32_t sgid_a = get_arg_val<uint32_t>(6);
    uint32_t rng_a  = get_arg_val<uint32_t>(7);
    uint32_t dbg_a  = get_arg_val<uint32_t>(8);
    volatile tt_l1_ptr uint32_t* gi   = (volatile tt_l1_ptr uint32_t*)ginfo_a;   // packed: 1 uint32/Gaussian
    volatile tt_l1_ptr uint32_t* hist = (volatile tt_l1_ptr uint32_t*)hist_a;
    volatile tt_l1_ptr uint32_t* sgid = (volatile tt_l1_ptr uint32_t*)sgid_a;
    volatile tt_l1_ptr uint32_t* rng  = (volatile tt_l1_ptr uint32_t*)rng_a;
    volatile tt_l1_ptr uint32_t* dbg  = (volatile tt_l1_ptr uint32_t*)dbg_a;
    uint32_t ntiles = ntx * nty, nkey = ntiles * D;
    for (uint32_t k = 0; k <= nkey; k++) hist[k] = 0;
    uint32_t tot = 0;
    for (uint32_t g = 0; g < N; g++) {
        uint32_t w = gi[g];
        uint32_t tx0 = w & 63, ty0 = (w >> 6) & 63, bw = (w >> 12) & 63, bh = (w >> 18) & 63, bk = (w >> 24) & 63;
        if (tx0 >= ntx || ty0 >= nty || bk >= D) continue;
        uint32_t dxm = (tx0 + bw <= ntx) ? bw : (ntx - tx0);
        uint32_t dym = (ty0 + bh <= nty) ? bh : (nty - ty0);
        for (uint32_t dy = 0; dy < dym; dy++) { uint32_t row = (ty0 + dy) * ntx;
            for (uint32_t dx = 0; dx < dxm; dx++) { uint32_t key = (row + tx0 + dx) * D + bk;
                if (key < nkey) { hist[key] += 1; tot++; } } }
    }
    uint32_t acc = 0;
    for (uint32_t k = 0; k < nkey; k++) { uint32_t h = hist[k]; hist[k] = acc; acc += h; }
    hist[nkey] = acc;
    for (uint32_t t = 0; t < ntiles; t++) { rng[t*2] = hist[t*D]; rng[t*2+1] = hist[(t+1)*D]; }
    for (uint32_t g = 0; g < N; g++) {
        uint32_t w = gi[g];
        uint32_t tx0 = w & 63, ty0 = (w >> 6) & 63, bw = (w >> 12) & 63, bh = (w >> 18) & 63, bk = (w >> 24) & 63;
        if (tx0 >= ntx || ty0 >= nty || bk >= D) continue;
        uint32_t dxm = (tx0 + bw <= ntx) ? bw : (ntx - tx0);
        uint32_t dym = (ty0 + bh <= nty) ? bh : (nty - ty0);
        for (uint32_t dy = 0; dy < dym; dy++) { uint32_t row = (ty0 + dy) * ntx;
            for (uint32_t dx = 0; dx < dxm; dx++) { uint32_t key = (row + tx0 + dx) * D + bk;
                if (key < nkey) { uint32_t p = hist[key]; hist[key] = p + 1; if (p < acc) sgid[p] = g; } } }
    }
    dbg[0] = tot; dbg[1] = N; dbg[2] = nkey; dbg[3] = acc;
}
"""


def densify_proj(arrs, target, rng):
    """Replicate projected (u,v,a,b,cc,zc) with pos+depth jitter to ~target Gaussians (same footprints)."""
    u, v, a, b, cc, zc = arrs
    K = max(1, int(round(target / len(u))))
    rx = 3.0 * np.sqrt(np.clip(cc / (a * cc - b ** 2 + 1e-12), .25, None))
    ry = 3.0 * np.sqrt(np.clip(a / (a * cc - b ** 2 + 1e-12), .25, None))
    zspan = zc.max() - zc.min() + 1e-9
    rep = lambda x: np.tile(x, K)
    n = len(u) * K
    return (rep(u) + rng.normal(0, 1, n) * rep(rx) * 0.5,
            rep(v) + rng.normal(0, 1, n) * rep(ry) * 0.5,
            rep(a), rep(b), rep(cc), rep(zc) + rng.normal(0, 1, n) * zspan * 0.04)


def ttnn_aabb(dev, u, v, a, b, cc, zc, W, H, ts, D):
    """Compute AABB(tx0,ty0,bw,bh)+depth bucket on DEVICE (ttnn S1 math); return 5 int arrays."""
    f = lambda x: ttnn.from_torch(torch.tensor(np.asarray(x), dtype=torch.float32).reshape(-1),
                                  dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    U, V, A, B, C, Z = f(u), f(v), f(a), f(b), f(cc), f(zc)
    ntx, nty = (W + ts - 1) // ts, (H + ts - 1) // ts
    inv = 1.0 / ts
    detc = ttnn.sub(ttnn.mul(A, C), ttnn.mul(B, B))
    rx = ttnn.mul(ttnn.sqrt(ttnn.clamp(ttnn.div(C, detc), 0.25, 1e30)), 3.0)
    ry = ttnn.mul(ttnn.sqrt(ttnn.clamp(ttnn.div(A, detc), 0.25, 1e30)), 3.0)
    tx0 = ttnn.clamp(ttnn.floor(ttnn.mul(ttnn.sub(U, rx), inv)), 0.0, float(ntx - 1))
    tx1 = ttnn.clamp(ttnn.floor(ttnn.mul(ttnn.add(U, rx), inv)), 0.0, float(ntx - 1))
    ty0 = ttnn.clamp(ttnn.floor(ttnn.mul(ttnn.sub(V, ry), inv)), 0.0, float(nty - 1))
    ty1 = ttnn.clamp(ttnn.floor(ttnn.mul(ttnn.add(V, ry), inv)), 0.0, float(nty - 1))
    zmin = float(zc.min()); zspan = float(zc.max() - zc.min() + 1e-9)
    bk = ttnn.clamp(ttnn.floor(ttnn.mul(ttnn.sub(Z, zmin), D / zspan)), 0.0, float(D - 1))
    n = len(np.asarray(u))
    g = lambda t: np.rint(ttnn.to_torch(t).flatten().numpy()[:n]).astype(np.int64)
    TX0, TX1, TY0, TY1, BK = g(tx0), g(tx1), g(ty0), g(ty1), g(bk)
    return TX0, TY0, (TX1 - TX0 + 1), (TY1 - TY0 + 1), BK


def main():
    P, npts = load_ply(PLY)
    cams, _, _ = _load_colmap(DATASET)
    rng = np.random.default_rng(0)
    dev = ttnn.open_device(device_id=0)
    try:
        LONG = 1600
        name, H, W, arrs = project_cam(P, cams[0], LONG)
        base = [arrs[0], arrs[1], arrs[2], arrs[3], arrs[4], arrs[6]]
        u, v, a, b, cc, zc = densify_proj(base, TARGET_N, rng)
        ntx, nty = (W + TS - 1) // TS, (H + TS - 1) // TS
        N = len(u)
        assert ntx <= 64 and nty <= 64 and D <= 64, "6-bit packing needs ntx,nty,D <= 64 (ok @1600)"

        # (a) AABB on device (ttnn) + parity vs host
        tx0, ty0, bw, bh, bk = ttnn_aabb(dev, u, v, a, b, cc, zc, W, H, TS, D)
        hb, _ = host_assign(u, v, a, b, cc, W, H, TS)
        zmin, zspan = zc.min(), zc.max() - zc.min() + 1e-9
        bk_h = np.clip(((zc - zmin) / zspan * D).astype(np.int64), 0, D - 1)
        aabb_match = (np.mean((tx0 == hb[0].astype(np.int64)) & (ty0 == hb[2].astype(np.int64))) * 100)
        ginfo_int = np.stack([tx0, ty0, bw, bh, bk], axis=1)
        packed = (tx0 | (ty0 << 6) | (bw << 12) | (bh << 18) | (bk << 24)).astype(np.int64)

        s_gid_h, ranges_h, total = host_countsort(ginfo_int, ntx, nty, D)
        ntiles, nkey = ntx * nty, ntx * nty * D
        kb = lambda b: b / 1024
        print(f"S2-dev inc2 @{LONG} ({W}x{H}, {ntiles} tiles) N={N} instances={total}")
        print(f"  L1: hist={kb((nkey+1)*4):.0f}KB ginfo={kb(N*4):.0f}KB sgid={kb(total*4):.0f}KB "
              f"-> sum={kb(((nkey+1)+N+total+ntiles*2)*4):.0f}KB / 1536KB")
        print(f"  (a) ttnn AABB == host AABB : {aabb_match:.2f}%  (bucket match {np.mean(bk==bk_h)*100:.2f}%)")

        gbuf = l1_u32(dev, N, packed)
        hbuf = l1_u32(dev, nkey + 1); sbuf = l1_u32(dev, total); rbuf = l1_u32(dev, ntiles * 2); dbuf = l1_u32(dev, 4)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
        rt = ttnn.RuntimeArgs()
        rt[HOME[0]][HOME[1]] = [gbuf.buffer_address(), N, ntx, nty, D, hbuf.buffer_address(),
                                sbuf.buffer_address(), rbuf.buffer_address(), dbuf.buffer_address()]
        k = ttnn.KernelDescriptor(kernel_source=KERNEL, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                  core_ranges=crs, compile_time_args=[], runtime_args=rt,
                                  config=ttnn.ReaderConfigDescriptor())
        prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
        ttnn.generic_op([gbuf, sbuf], prog)                      # warmup (JIT)
        t0 = time.perf_counter(); ttnn.generic_op([gbuf, sbuf], prog)
        dt = (time.perf_counter() - t0) * 1e3

        dbg = ttnn.to_torch(dbuf).flatten().numpy()[:4].astype(np.int64)
        s_gid_d = ttnn.to_torch(sbuf).flatten().numpy()[:total].astype(np.int64)
        ranges_d = ttnn.to_torch(rbuf).flatten().numpy()[:ntiles * 2].astype(np.int64).reshape(ntiles, 2)
        print(f"  kernel dbg: instances={dbg[0]}(host {total}) N={dbg[1]} acc={dbg[3]}")
        bad = checked = 0
        for t in range(ntiles):
            sh = set(s_gid_h[ranges_h[t, 0]:ranges_h[t, 1]].tolist())
            sd = set(s_gid_d[ranges_d[t, 0]:ranges_d[t, 1]].tolist())
            if sh:
                checked += 1; bad += (sh != sd)
        print(f"  (b) kernel sort per-tile SET : {checked-bad}/{checked} ({'PASS' if bad == 0 else f'{bad} MISMATCH'})")
        print(f"  device sort time           : {dt:.1f} ms (single core, {N} Gaussians @ {LONG}px)")
        ok = bad == 0 and dbg[0] == total and aabb_match > 99.0
        print(f"  -> {'INC2 PASS' if ok else 'INC2 CHECK'}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
