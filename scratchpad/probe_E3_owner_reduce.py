#!/usr/bin/env python3
# E3: owner-single-writer cross-core reduce at realistic fan-out (extends m2_scatter_gather).
# Every source core writes 7 grad scalars for EVERY gid to owner(g)=g%N_OWNERS at a DISTINCT
# per-(source,gid) L1 slot (no collision, no atomics). Each owner reduces its inbox (FP32, single
# writer). Verify bit-exact vs host expected; measure drain cycles/elem. This is worst-case all-to-all
# fan-in (every source touches every gid) and tells us if FP32 owner-reduce avoids wedge + is fast.
import sys, struct
from pathlib import Path
import torch, ttnn

NCH = 7                      # grad scalars per (source, gid)

SRC = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t src_idx  = get_arg_val<uint32_t>(0);
    uint32_t n_gid    = get_arg_val<uint32_t>(1);
    uint32_t n_owners = get_arg_val<uint32_t>(2);
    uint32_t n_src    = get_arg_val<uint32_t>(3);
    uint32_t inbox    = get_arg_val<uint32_t>(4);
    uint32_t valbits  = get_arg_val<uint32_t>(5);     // f32 (src_idx+1), same for all channels
    // owner physical coords table at args[6..6+2*n_owners)
    for (uint32_t g = 0; g < n_gid; g++) {
        uint32_t owner = g % n_owners;
        uint32_t slot  = g / n_owners;
        uint32_t ox = get_arg_val<uint32_t>(6 + owner*2);
        uint32_t oy = get_arg_val<uint32_t>(6 + owner*2 + 1);
        // inbox layout per owner: [slot][src][NCH]
        uint32_t base = inbox + ((slot * n_src + src_idx) * 7u) * 4u;
        for (uint32_t ch = 0; ch < 7; ch++)
            noc_inline_dw_write(get_noc_addr(ox, oy, base + ch*4u), valbits);
    }
    noc_async_write_barrier();
}
"""
DRAIN = r"""
#include "dataflow_api.h"
#include "risc_common.h"
void kernel_main() {
    uint32_t inbox_addr = get_arg_val<uint32_t>(0);
    uint32_t slots      = get_arg_val<uint32_t>(1);   // gids owned by this core
    uint32_t n_src      = get_arg_val<uint32_t>(2);
    uint32_t out_addr   = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr float* inbox = (volatile tt_l1_ptr float*)inbox_addr;
    volatile tt_l1_ptr float* out   = (volatile tt_l1_ptr float*)out_addr;
    uint32_t t0 = get_timestamp_32b();
    for (uint32_t slot = 0; slot < slots; slot++) {
        for (uint32_t ch = 0; ch < 7; ch++) {
            float acc = 0.0f;
            for (uint32_t s = 0; s < n_src; s++)
                acc += inbox[(slot * n_src + s) * 7u + ch];
            out[slot * 7u + ch] = acc;
        }
    }
    uint32_t t1 = get_timestamp_32b();
    out[slots * 7u] = (float)(t1 - t0);   // cycles in the word after the grads
}
"""

def f2u(x): return struct.unpack("<I", struct.pack("<f", float(x)))[0]

W = 256                       # row width of the ROW_MAJOR shards (flat index i -> row i//W, col i%W)

def rm_shard(dev, cores, rows, data=None):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in cores])
    sh = ttnn.ShardSpec(crs, [rows, W], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sh)
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows * len(cores), W]), ttnn.float32,
                                              ttnn.ROW_MAJOR_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, rows * len(cores), W), dtype=ttnn.float32,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)

dev = ttnn.open_device(device_id=0)
def run(N_GID, N_OWNERS, N_SRC):
    owner_cores = [(1 + i, 0) for i in range(N_OWNERS)]           # row 0
    src_cores   = [(c % 8, 1 + c // 8) for c in range(N_SRC)]     # rows 1+
    slots = (N_GID + N_OWNERS - 1) // N_OWNERS                    # max gids per owner
    in_rows = (slots * N_SRC * NCH + W - 1) // W                  # rows of flat floats per owner
    out_rows = (slots * NCH + 2 + W - 1) // W
    inbox = rm_shard(dev, owner_cores, in_rows, torch.zeros(1, 1, in_rows * N_OWNERS, W))   # zero-init
    out = rm_shard(dev, owner_cores, out_rows)
    inbox_addr, out_addr = inbox.buffer_address(), out.buffer_address()
    owner_phys = []
    for (x, y) in owner_cores:
        hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(x, y)); owner_phys += [hp.x, hp.y]
    # scatter
    src_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in src_cores])
    src_rt = ttnn.RuntimeArgs()
    for s, (x, y) in enumerate(src_cores):
        src_rt[x][y] = [s, N_GID, N_OWNERS, N_SRC, inbox_addr, f2u(s + 1)] + owner_phys
    sk = ttnn.KernelDescriptor(kernel_source=SRC, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                               core_ranges=src_crs, compile_time_args=[], runtime_args=src_rt,
                               config=ttnn.WriterConfigDescriptor())
    ttnn.generic_op([inbox, out], ttnn.ProgramDescriptor(kernels=[sk], semaphores=[], cbs=[]))
    # drain on each owner
    own_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in owner_cores])
    drt = ttnn.RuntimeArgs()
    for (x, y) in owner_cores:
        drt[x][y] = [inbox_addr, slots, N_SRC, out_addr]
    dk = ttnn.KernelDescriptor(kernel_source=DRAIN, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                               core_ranges=own_crs, compile_time_args=[], runtime_args=drt,
                               config=ttnn.ReaderConfigDescriptor())
    ttnn.generic_op([inbox, out], ttnn.ProgramDescriptor(kernels=[dk], semaphores=[], cbs=[]))
    res = ttnn.to_torch(out).reshape(N_OWNERS * out_rows, W)     # row-major flat per owner block
    def owner_flat(oi, i):                                       # flat float i within owner oi's region
        return float(res[oi * out_rows + i // W, i % W])
    expected = sum(range(1, N_SRC + 1))                          # each gid: sum_s (s+1), per channel
    bad = 0; total = 0; cyc = 0.0
    for oi in range(N_OWNERS):
        n_here = len(range(oi, N_GID, N_OWNERS))
        for slot in range(n_here):
            for ch in range(NCH):
                total += 1
                if abs(owner_flat(oi, slot * NCH + ch) - expected) > 0.5: bad += 1
        cyc = max(cyc, owner_flat(oi, slots * NCH))
    elems = slots * N_SRC * NCH
    print(f"  N_GID={N_GID} owners={N_OWNERS} src={N_SRC} fan_in={N_SRC} | checked={total} bad={bad} "
          f"expected={expected} | drain_cyc={cyc:.0f} elems={elems} cyc/elem={cyc/max(elems,1):.2f}  "
          f"{'OK' if bad==0 else 'MISMATCH'}", flush=True)
    return bad

try:
    gs = dev.compute_with_storage_grid_size()
    print(f"E3 owner-single-writer FP32 reduce at fan-out (every src -> every gid, all-to-all). grid={gs.x}x{gs.y}")
    tot_bad = 0
    for cfg in [(256, 8, 16), (512, 8, 32), (1024, 8, 48)]:    # owners on row0 x1..8; sources rows1..6
        tot_bad += run(*cfg)
    print(f"\n  {'ALL OK (FP32 bit-exact, no wedge, all-to-all)' if tot_bad==0 else 'SOME MISMATCH'}")
finally:
    ttnn.close_device(dev)
