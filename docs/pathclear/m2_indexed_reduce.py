#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
M2c: per-Gaussian indexed reduce — the real backward scatter-add (acc[gid] += partial).
The home tile owns G Gaussians; it drains (gid, partial) pairs into its accumulator array.

Two strategies, measured on silicon (fixed-point int32, per the no-FPU finding):
  mode 0  indexed scatter   acc[gid[i]] += val[i]      (unsorted; random L1 access)
  mode 1  segmented reduce   sorted-by-gid: run-accumulate, flush on gid change (sequential)

Decides whether the inbox should be sorted by gid (→ enables the fast segmented path, but adds a
sort stage — Tensix-hard, x280 candidate) or scattered directly.
"""
import torch, ttnn

HOME = (0, 0)
G = 256          # Gaussians owned by this home tile
TOTAL = 4096     # incoming (gid, partial) pairs

DRAIN = r"""
#include "dataflow_api.h"
#include "risc_common.h"
void kernel_main() {
    uint32_t gid_a = get_arg_val<uint32_t>(0);
    uint32_t val_a = get_arg_val<uint32_t>(1);
    uint32_t acc_a = get_arg_val<uint32_t>(2);
    uint32_t total = get_arg_val<uint32_t>(3);
    uint32_t out_a = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr int* gid = (volatile tt_l1_ptr int*)gid_a;
    volatile tt_l1_ptr int* val = (volatile tt_l1_ptr int*)val_a;
    volatile tt_l1_ptr int* acc = (volatile tt_l1_ptr int*)acc_a;
    volatile tt_l1_ptr uint32_t* out = (volatile tt_l1_ptr uint32_t*)out_a;
    uint32_t t0 = get_timestamp_32b();
#if MODE==0
    for (uint32_t i = 0; i < total; i++) acc[gid[i]] += val[i];
#elif MODE==1
    int cur = gid[0], run = 0;
    for (uint32_t i = 0; i < total; i++) {
        int g = gid[i];
        if (g != cur) { acc[cur] += run; cur = g; run = 0; }
        run += val[i];
    }
    acc[cur] += run;
#elif MODE==2
    int cur = gid[0], run = 0; uint32_t i = 0;
    for (; i + 4 <= total; i += 4) {
        int g0=gid[i],g1=gid[i+1],g2=gid[i+2],g3=gid[i+3];   // 8 loads issued before use
        int v0=val[i],v1=val[i+1],v2=val[i+2],v3=val[i+3];
        if(g0!=cur){acc[cur]+=run;cur=g0;run=0;} run+=v0;
        if(g1!=cur){acc[cur]+=run;cur=g1;run=0;} run+=v1;
        if(g2!=cur){acc[cur]+=run;cur=g2;run=0;} run+=v2;
        if(g3!=cur){acc[cur]+=run;cur=g3;run=0;} run+=v3;
    }
    for (; i < total; i++){int g=gid[i];if(g!=cur){acc[cur]+=run;cur=g;run=0;}run+=val[i];}
    acc[cur] += run;
#endif
    uint32_t t1 = get_timestamp_32b();
    out[0] = t1 - t0;
}
"""

def l1_on(dev, core, h, w, dtype, layout, data=None):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*core), ttnn.CoreCoord(*core))])
    sh = ttnn.ShardSpec(crs, [h, w], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sh)
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, h, w]), dtype, layout, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, h, w), dtype=dtype, layout=layout, device=dev, memory_config=mc)

def run_mode(dev, mode, gids, vals):
    gid_t = l1_on(dev, HOME, 32, 128, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, gids)
    val_t = l1_on(dev, HOME, 32, 128, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, vals)
    acc_t = l1_on(dev, HOME, 16, 16, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.zeros(G, dtype=torch.int32))
    out_t = l1_on(dev, HOME, 1, 32, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.zeros(32, dtype=torch.int32))
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    rt = ttnn.RuntimeArgs()
    rt[HOME[0]][HOME[1]] = [gid_t.buffer_address(), val_t.buffer_address(),
                            acc_t.buffer_address(), TOTAL, out_t.buffer_address()]
    k = ttnn.KernelDescriptor(kernel_source=DRAIN, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=crs, compile_time_args=[], runtime_args=rt,
                              defines=[("MODE", str(mode))], config=ttnn.ReaderConfigDescriptor())
    ttnn.generic_op([gid_t, out_t], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[]))
    acc = ttnn.to_torch(acc_t).flatten()[:G].to(torch.int64)
    cyc = int(ttnn.to_torch(out_t).flatten()[0])
    return acc, cyc

def main():
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        gids = torch.randint(0, G, (TOTAL,), dtype=torch.int32)
        vals = torch.ones(TOTAL, dtype=torch.int32)
        golden = torch.bincount(gids.to(torch.int64), minlength=G)

        acc0, cyc0 = run_mode(dev, 0, gids, vals)                       # indexed scatter (unsorted)
        gs, _ = torch.sort(gids); vs = torch.ones(TOTAL, dtype=torch.int32)
        acc1, cyc1 = run_mode(dev, 1, gs.to(torch.int32), vs)          # segmented (sorted)
        acc2, cyc2 = run_mode(dev, 2, gs.to(torch.int32), vs)          # segmented ×4 prefetch

        ok0 = torch.equal(acc0, golden); ok1 = torch.equal(acc1, golden); ok2 = torch.equal(acc2, golden)
        print(f"G={G} total={TOTAL}")
        print(f"mode 0 indexed scatter   {'OK' if ok0 else 'MISMATCH'}  cycles={cyc0:7d}  cyc/elem={cyc0/TOTAL:.2f}")
        print(f"mode 1 segmented sorted  {'OK' if ok1 else 'MISMATCH'}  cycles={cyc1:7d}  cyc/elem={cyc1/TOTAL:.2f}")
        print(f"mode 2 segmented ×4 pref {'OK' if ok2 else 'MISMATCH'}  cycles={cyc2:7d}  cyc/elem={cyc2/TOTAL:.2f}")
        print("M2C_OK" if (ok0 and ok1 and ok2) else "M2C_FAIL")
    finally:
        ttnn.close_device(dev)

if __name__ == "__main__":
    main()
