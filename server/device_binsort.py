#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Device counting-bucket bin/sort (campaign stage S2) — a drop-in for host `bin_and_sort`, on the Blackhole.

The whole sort is ONE counting sort on a composite key `tile_id*D + depth_bucket` (D=64, validated for
render parity in scratchpad/probe_S0_depthbucket.py): histogram -> exclusive-scan -> scatter, on a single
Tensix core (no NoC, no contention). AABB/bucket are computed on-device in ttnn (the S1 math). Proven on
silicon at 1600px/32k by scratchpad/probe_S2_dev_inc2.py (ttnn AABB==host 100%, per-tile sets 736/736).

CAPS (current single-core L1 form):
  - ntx, nty, D <= 64  (the 5 AABB fields pack into 6 bits each -> 1 uint32/Gaussian). OK to 1600px.
  - ~32k Gaussians @1600px (histogram 487KB + ginfo + sgid fit one core's 1.5MB L1). 50k needs sgid->DRAM
    (campaign stage S4's DRAM streaming substrate); the multi-core m2 owner-scatter is the millions path.

Returns the same shape as bin_and_sort: (s_gid, s_tile, ranges, ntx, nty, total) so it is a literal swap.
"""
from __future__ import annotations
import numpy as np
import torch
import ttnn

TS = 32
D_DEFAULT = 64
HOME = (0, 0)

_KERNEL = r"""
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
    volatile tt_l1_ptr uint32_t* gi   = (volatile tt_l1_ptr uint32_t*)ginfo_a;   // packed 1 uint32/Gaussian
    volatile tt_l1_ptr uint32_t* hist = (volatile tt_l1_ptr uint32_t*)hist_a;
    volatile tt_l1_ptr uint32_t* sgid = (volatile tt_l1_ptr uint32_t*)sgid_a;
    volatile tt_l1_ptr uint32_t* rng  = (volatile tt_l1_ptr uint32_t*)rng_a;
    uint32_t ntiles = ntx * nty, nkey = ntiles * D;
    for (uint32_t k = 0; k <= nkey; k++) hist[k] = 0;
    for (uint32_t g = 0; g < N; g++) {                                  // pass 1: histogram (grid-bounded)
        uint32_t w = gi[g];
        uint32_t tx0 = w & 63, ty0 = (w >> 6) & 63, bw = (w >> 12) & 63, bh = (w >> 18) & 63, bk = (w >> 24) & 63;
        if (tx0 >= ntx || ty0 >= nty || bk >= D) continue;
        uint32_t dxm = (tx0 + bw <= ntx) ? bw : (ntx - tx0);
        uint32_t dym = (ty0 + bh <= nty) ? bh : (nty - ty0);
        for (uint32_t dy = 0; dy < dym; dy++) { uint32_t row = (ty0 + dy) * ntx;
            for (uint32_t dx = 0; dx < dxm; dx++) { uint32_t key = (row + tx0 + dx) * D + bk;
                if (key < nkey) hist[key] += 1; } }
    }
    uint32_t acc = 0;                                                   // exclusive scan -> base offsets
    for (uint32_t k = 0; k < nkey; k++) { uint32_t h = hist[k]; hist[k] = acc; acc += h; }
    hist[nkey] = acc;
    for (uint32_t t = 0; t < ntiles; t++) { rng[t*2] = hist[t*D]; rng[t*2+1] = hist[(t+1)*D]; }
    for (uint32_t g = 0; g < N; g++) {                                  // pass 2: scatter gid
        uint32_t w = gi[g];
        uint32_t tx0 = w & 63, ty0 = (w >> 6) & 63, bw = (w >> 12) & 63, bh = (w >> 18) & 63, bk = (w >> 24) & 63;
        if (tx0 >= ntx || ty0 >= nty || bk >= D) continue;
        uint32_t dxm = (tx0 + bw <= ntx) ? bw : (ntx - tx0);
        uint32_t dym = (ty0 + bh <= nty) ? bh : (nty - ty0);
        for (uint32_t dy = 0; dy < dym; dy++) { uint32_t row = (ty0 + dy) * ntx;
            for (uint32_t dx = 0; dx < dxm; dx++) { uint32_t key = (row + tx0 + dx) * D + bk;
                if (key < nkey) { uint32_t p = hist[key]; hist[key] = p + 1; if (p < acc) sgid[p] = g; } } }
    }
}
"""

_pad32 = lambda m: ((m + 31) // 32) * 32


def _l1_u32(dev, M, data=None):
    """Single-core (HOME) WIDTH-contiguous ROW_MAJOR L1 uint32 buffer -> linear raw bytes for the kernel.
    (HEIGHT-shard [Mp,1] pads each row -> garbage reads -> hang/wedge; WIDTH [1,Mp] is linear.)"""
    Mp = _pad32(M)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(crs, [1, Mp], ttnn.ShardOrientation.ROW_MAJOR))
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, Mp]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev, mc)
    arr = np.zeros(Mp, np.int32); arr[:len(data)] = np.asarray(data, np.int32)
    return ttnn.from_torch(torch.from_numpy(arr).reshape(1, 1, 1, Mp), dtype=ttnn.uint32,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)


def _device_aabb(dev, cx, cy, ca, cb, cc, zc, Wp, Hp, ts, D):
    """AABB (tx0,ty0,bw,bh) + depth bucket on device (ttnn S1 math). Host arrays in -> packed uint32 out.
    (The AABB readback to pack on host is the only host hop here; on-device pack is an S4 optimization.)"""
    f = lambda x: ttnn.from_torch(torch.tensor(np.asarray(x), dtype=torch.float32).reshape(-1),
                                  dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    U, V, A, B, C, Z = f(cx), f(cy), f(ca), f(cb), f(cc), f(zc)
    ntx, nty = (Wp + ts - 1) // ts, (Hp + ts - 1) // ts
    inv = 1.0 / ts
    detc = ttnn.sub(ttnn.mul(A, C), ttnn.mul(B, B))
    rx = ttnn.mul(ttnn.sqrt(ttnn.clamp(ttnn.div(C, detc), 0.25, 1e30)), 3.0)
    ry = ttnn.mul(ttnn.sqrt(ttnn.clamp(ttnn.div(A, detc), 0.25, 1e30)), 3.0)
    tx0 = ttnn.clamp(ttnn.floor(ttnn.mul(ttnn.sub(U, rx), inv)), 0.0, float(ntx - 1))
    tx1 = ttnn.clamp(ttnn.floor(ttnn.mul(ttnn.add(U, rx), inv)), 0.0, float(ntx - 1))
    ty0 = ttnn.clamp(ttnn.floor(ttnn.mul(ttnn.sub(V, ry), inv)), 0.0, float(nty - 1))
    ty1 = ttnn.clamp(ttnn.floor(ttnn.mul(ttnn.add(V, ry), inv)), 0.0, float(nty - 1))
    zmin = float(np.min(zc)); zspan = float(np.max(zc) - np.min(zc) + 1e-9)
    bk = ttnn.clamp(ttnn.floor(ttnn.mul(ttnn.sub(Z, zmin), D / zspan)), 0.0, float(D - 1))
    n = len(np.asarray(cx))
    g = lambda t: np.rint(ttnn.to_torch(t).flatten().numpy()[:n]).astype(np.int64)
    TX0, TX1, TY0, TY1, BK = g(tx0), g(tx1), g(ty0), g(ty1), g(bk)
    bw, bh = (TX1 - TX0 + 1), (TY1 - TY0 + 1)
    packed = (TX0 | (TY0 << 6) | (bw << 12) | (bh << 18) | (BK << 24)).astype(np.int64)
    return packed, ntx, nty


def device_binsort(dev, cx, cy, ca, cb, cc, zc, Wp, Hp, ts=TS, D=D_DEFAULT):
    """On-device counting-bucket bin/sort. Drop-in for bin_and_sort (conic a,b,c in place of var_x,var_y).
    Returns (s_gid, s_tile, ranges, ntx, nty, total) — bin_and_sort's signature."""
    n = len(np.asarray(cx))
    packed, ntx, nty = _device_aabb(dev, cx, cy, ca, cb, cc, zc, Wp, Hp, ts, D)
    assert ntx <= 64 and nty <= 64 and D <= 64, f"6-bit packing needs ntx,nty,D<=64 (got {ntx},{nty},{D})"
    ntiles, nkey = ntx * nty, ntx * nty * D
    # total instances (host knows the packed counts; sizes the sgid buffer)
    bw = (packed >> 12) & 63; bh = (packed >> 18) & 63
    tx0 = packed & 63; ty0 = (packed >> 6) & 63
    dxm = np.minimum(bw, ntx - tx0); dym = np.minimum(bh, nty - ty0)
    valid = (tx0 < ntx) & (ty0 < nty)
    total = int((dxm * dym * valid).sum())

    gbuf = _l1_u32(dev, n, packed)
    hbuf = _l1_u32(dev, nkey + 1); sbuf = _l1_u32(dev, max(total, 1)); rbuf = _l1_u32(dev, ntiles * 2)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    rt = ttnn.RuntimeArgs()
    rt[HOME[0]][HOME[1]] = [gbuf.buffer_address(), n, ntx, nty, D,
                            hbuf.buffer_address(), sbuf.buffer_address(), rbuf.buffer_address()]
    k = ttnn.KernelDescriptor(kernel_source=_KERNEL, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=crs, compile_time_args=[], runtime_args=rt,
                              config=ttnn.ReaderConfigDescriptor())
    ttnn.generic_op([gbuf, sbuf], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[]))

    s_gid = ttnn.to_torch(sbuf).flatten().numpy()[:total].astype(np.int64)
    ranges = ttnn.to_torch(rbuf).flatten().numpy()[:ntiles * 2].astype(np.int64).reshape(ntiles, 2)
    for _t in (gbuf, hbuf, sbuf, rbuf):                                  # free HOME L1 before the raster reuses (0,0)
        _t.deallocate()
    s_tile = np.repeat(np.arange(ntiles), ranges[:, 1] - ranges[:, 0])    # tile per sorted instance
    return s_gid, s_tile, ranges, ntx, nty, total
