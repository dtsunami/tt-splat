#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PROBE S2-dev increment 1 (silicon) — the SINGLE-CORE counting-sort KERNEL.

Host computes per-Gaussian AABB+bucket (proven S1 math), packs into an L1 row-major int buffer; ONE core
does integer expansion + histogram + exclusive-scan + scatter (+ per-tile ranges) in local L1 — no NoC, no
contention. Gate vs a host numpy run of the SAME algorithm on the SAME ginfo. @96px.

HARDENED after a layout-mismatch hang wedged the card: (1) every per-Gaussian loop is BOUNDED to the grid
(<= ntx*nty iters/Gaussian) so a garbage read can't infinite-loop; (2) a dbg[] readback reports what the
kernel actually read (total instances + sample gi[] words) to confirm the raw L1 layout is linear.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "pathclear"))
from probe_S0_depthbucket import load_ply, project_cam, PLY, DATASET   # noqa: E402
from probe_S1_tile_assign import host_assign                            # noqa: E402
from train_tt import _load_colmap                                       # noqa: E402

TS = 32
D = 64
HOME = (0, 0)

KERNEL = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t ginfo_a = get_arg_val<uint32_t>(0);
    uint32_t N       = get_arg_val<uint32_t>(1);
    uint32_t ntx     = get_arg_val<uint32_t>(2);
    uint32_t nty     = get_arg_val<uint32_t>(3);
    uint32_t D       = get_arg_val<uint32_t>(4);
    uint32_t hist_a  = get_arg_val<uint32_t>(5);
    uint32_t sgid_a  = get_arg_val<uint32_t>(6);
    uint32_t rng_a   = get_arg_val<uint32_t>(7);
    uint32_t dbg_a   = get_arg_val<uint32_t>(8);
    volatile tt_l1_ptr uint32_t* gi   = (volatile tt_l1_ptr uint32_t*)ginfo_a;
    volatile tt_l1_ptr uint32_t* hist = (volatile tt_l1_ptr uint32_t*)hist_a;
    volatile tt_l1_ptr uint32_t* sgid = (volatile tt_l1_ptr uint32_t*)sgid_a;
    volatile tt_l1_ptr uint32_t* rng  = (volatile tt_l1_ptr uint32_t*)rng_a;
    volatile tt_l1_ptr uint32_t* dbg  = (volatile tt_l1_ptr uint32_t*)dbg_a;
    uint32_t ntiles = ntx * nty, nkey = ntiles * D;
    for (uint32_t k = 0; k <= nkey; k++) hist[k] = 0;
    uint32_t tot = 0;
    // pass 1: histogram (BOUNDED to the grid -> hang-proof)
    for (uint32_t g = 0; g < N; g++) {
        uint32_t tx0 = gi[g*5+0], ty0 = gi[g*5+1], bw = gi[g*5+2], bh = gi[g*5+3], bk = gi[g*5+4];
        if (tx0 >= ntx || ty0 >= nty || bk >= D) continue;
        uint32_t dxm = (tx0 + bw <= ntx) ? bw : (ntx - tx0);
        uint32_t dym = (ty0 + bh <= nty) ? bh : (nty - ty0);
        for (uint32_t dy = 0; dy < dym; dy++) { uint32_t row = (ty0 + dy) * ntx;
            for (uint32_t dx = 0; dx < dxm; dx++) { uint32_t key = (row + tx0 + dx) * D + bk;
                if (key < nkey) { hist[key] += 1; tot++; } } }
    }
    // exclusive scan -> base offsets; sentinel hist[nkey] = total
    uint32_t acc = 0;
    for (uint32_t k = 0; k < nkey; k++) { uint32_t h = hist[k]; hist[k] = acc; acc += h; }
    hist[nkey] = acc;
    for (uint32_t t = 0; t < ntiles; t++) { rng[t*2] = hist[t*D]; rng[t*2+1] = hist[(t+1)*D]; }
    // pass 2: scatter
    for (uint32_t g = 0; g < N; g++) {
        uint32_t tx0 = gi[g*5+0], ty0 = gi[g*5+1], bw = gi[g*5+2], bh = gi[g*5+3], bk = gi[g*5+4];
        if (tx0 >= ntx || ty0 >= nty || bk >= D) continue;
        uint32_t dxm = (tx0 + bw <= ntx) ? bw : (ntx - tx0);
        uint32_t dym = (ty0 + bh <= nty) ? bh : (nty - ty0);
        for (uint32_t dy = 0; dy < dym; dy++) { uint32_t row = (ty0 + dy) * ntx;
            for (uint32_t dx = 0; dx < dxm; dx++) { uint32_t key = (row + tx0 + dx) * D + bk;
                if (key < nkey) { uint32_t p = hist[key]; hist[key] = p + 1; if (p < acc) sgid[p] = g; } } }
    }
    dbg[0] = tot; dbg[1] = gi[0]; dbg[2] = gi[2]; dbg[3] = (N > 1) ? gi[7] : 0;   // readback: what the kernel saw
}
"""

pad32 = lambda m: ((m + 31) // 32) * 32


def l1_u32(dev, M, data=None):
    """Single-core (HOME) WIDTH-contiguous ROW_MAJOR L1 uint32 buffer of M elems -> linear raw bytes."""
    Mp = pad32(M)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(crs, [1, Mp], ttnn.ShardOrientation.ROW_MAJOR))
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, Mp]), ttnn.uint32,
                                              ttnn.ROW_MAJOR_LAYOUT, dev, mc)
    arr = np.zeros(Mp, np.int32); arr[:len(data)] = np.asarray(data, np.int32)
    return ttnn.from_torch(torch.from_numpy(arr).reshape(1, 1, 1, Mp), dtype=ttnn.uint32,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)


def host_countsort(ginfo, ntx, nty, D):
    tx0, ty0, bw, bh, bk = (ginfo[:, i] for i in range(5))
    counts = bw * bh
    gid = np.repeat(np.arange(len(ginfo)), counts)
    local = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    bwr = np.repeat(bw, counts)
    tx = np.repeat(tx0, counts) + local % bwr
    ty = np.repeat(ty0, counts) + local // bwr
    tile = ty * ntx + tx
    key = tile * D + np.repeat(bk, counts)
    order = np.argsort(key, kind="stable")
    s_gid, s_tile = gid[order], tile[order]
    ranges = np.zeros((ntx * nty, 2), np.int64)
    uniq, st = np.unique(s_tile, return_index=True)
    ranges[uniq, 0] = st; ranges[uniq, 1] = np.append(st[1:], len(s_tile))
    return s_gid, ranges, int(counts.sum())


def main():
    P, npts = load_ply(PLY)
    cams, _, _ = _load_colmap(DATASET)
    dev = ttnn.open_device(device_id=0)
    try:
        LONG = 96
        name, H, W, arrs = project_cam(P, cams[0], LONG)
        u, v, a, b, cc, zc = arrs[0], arrs[1], arrs[2], arrs[3], arrs[4], arrs[6]
        ntx, nty = (W + TS - 1) // TS, (H + TS - 1) // TS
        N = len(u)
        aabb, _ = host_assign(u, v, a, b, cc, W, H, TS)
        tx0, tx1, ty0, ty1 = aabb.astype(np.int64)
        bw, bh = tx1 - tx0 + 1, ty1 - ty0 + 1
        zmin, zmax = zc.min(), zc.max() + 1e-9
        bk = np.clip(((zc - zmin) / (zmax - zmin) * D).astype(np.int64), 0, D - 1)
        ginfo = np.stack([tx0, ty0, bw, bh, bk], axis=1).astype(np.int64)

        s_gid_h, ranges_h, total = host_countsort(ginfo, ntx, nty, D)
        ntiles, nkey = ntx * nty, ntx * nty * D
        print(f"S2-dev @{LONG} ({W}x{H}, {ntiles} tiles) N={N} total_instances={total} nkey={nkey}")

        gbuf = l1_u32(dev, N * 5, ginfo.reshape(-1))
        gback = ttnn.to_torch(gbuf).flatten().numpy()[:N * 5].astype(np.int64)
        print(f"  ginfo roundtrip (to_torch) : {'OK' if np.array_equal(gback, ginfo.reshape(-1)) else 'MISMATCH'}")
        hbuf = l1_u32(dev, nkey + 1); sbuf = l1_u32(dev, total); rbuf = l1_u32(dev, ntiles * 2); dbuf = l1_u32(dev, 4)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
        rt = ttnn.RuntimeArgs()
        rt[HOME[0]][HOME[1]] = [gbuf.buffer_address(), N, ntx, nty, D, hbuf.buffer_address(),
                                sbuf.buffer_address(), rbuf.buffer_address(), dbuf.buffer_address()]
        k = ttnn.KernelDescriptor(kernel_source=KERNEL, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                  core_ranges=crs, compile_time_args=[], runtime_args=rt,
                                  config=ttnn.ReaderConfigDescriptor())
        ttnn.generic_op([gbuf, sbuf], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[]))

        dbg = ttnn.to_torch(dbuf).flatten().numpy()[:4].astype(np.int64)
        print(f"  kernel readback dbg        : total={dbg[0]} (host {total})  gi[0]={dbg[1]}(={ginfo.reshape(-1)[0]}) "
              f"gi[2]={dbg[2]}(={ginfo.reshape(-1)[2]}) gi[7]={dbg[3]}(={ginfo.reshape(-1)[7]})")
        s_gid_d = ttnn.to_torch(sbuf).flatten().numpy()[:total].astype(np.int64)
        ranges_d = ttnn.to_torch(rbuf).flatten().numpy()[:ntiles * 2].astype(np.int64).reshape(ntiles, 2)

        bad = checked = 0
        for t in range(ntiles):
            sh = set(s_gid_h[ranges_h[t, 0]:ranges_h[t, 1]].tolist())
            sd = set(s_gid_d[ranges_d[t, 0]:ranges_d[t, 1]].tolist())
            if sh:
                checked += 1; bad += (sh != sd)
        print(f"  per-tile gid SET match     : {checked-bad}/{checked} ({'PASS' if bad == 0 else f'{bad} MISMATCH'})")
        print(f"  -> {'S2-DEV PASS' if bad == 0 and dbg[0] == total else 'S2-DEV FAIL'}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
