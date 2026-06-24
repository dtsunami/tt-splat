#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
M2b: contention-free cross-tile scatter-add — the backward gradient-accumulation design.

N source cores each write K partials into their OWN dedicated inbox region in a HOME tile's
L1 (NoC-visible, distinct slots => no collision, no atomics, no noc_accumulate, no wedge risk).
The home tile then drains+reduces all inboxes locally (single writer). This is the whole
scatter-add reframed into something Blackhole does natively.

  source core s writes value (s+1) to its K slots  ->  total = K * sum(1..N)
  home drains N*K floats, sums, reports cycles  ->  verify bit-exact + cycles/element (IPC).

Two generic_op calls (host barrier guarantees all writes land before the drain).
"""
import struct
import torch
import ttnn

HOME = (0, 0)
N_SRC = 8           # source cores (1,0)..(8,0)
K = 512             # partials per source
TOTAL = N_SRC * K   # drained elements

SRC_KERNEL = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t home_x   = get_arg_val<uint32_t>(0);
    uint32_t home_y   = get_arg_val<uint32_t>(1);
    uint32_t inbox    = get_arg_val<uint32_t>(2);
    uint32_t src_idx  = get_arg_val<uint32_t>(3);
    uint32_t k        = get_arg_val<uint32_t>(4);
    uint32_t valbits  = get_arg_val<uint32_t>(5);
    uint32_t base = inbox + src_idx * k * 4;     // this source's dedicated region
    for (uint32_t i = 0; i < k; i++) {
        noc_inline_dw_write(get_noc_addr(home_x, home_y, base + i * 4), valbits);
    }
    noc_async_write_barrier();
}
"""

DRAIN_KERNEL = r"""
#include "dataflow_api.h"
#include "risc_common.h"
void kernel_main() {
    uint32_t inbox_addr = get_arg_val<uint32_t>(0);
    uint32_t total      = get_arg_val<uint32_t>(1);
    uint32_t out_addr   = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr float* inbox = (volatile tt_l1_ptr float*)inbox_addr;
    volatile tt_l1_ptr float* out   = (volatile tt_l1_ptr float*)out_addr;
    uint32_t t0 = get_timestamp_32b();
    float acc = 0.0f;
    for (uint32_t i = 0; i < total; i++) acc += inbox[i];
    uint32_t t1 = get_timestamp_32b();
    out[0] = acc;
    out[1] = (float)(t1 - t0);
}
"""

def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]

def l1_on(dev, core, h, w):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*core), ttnn.CoreCoord(*core))])
    shard = ttnn.ShardSpec(crs, [h, w], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
    return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, h, w]), ttnn.float32,
                                          ttnn.TILE_LAYOUT, dev, mc), mc

def main():
    dev = ttnn.open_device(device_id=0)
    try:
        # home physical NoC coords (source kernels NoC-write here)
        hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(*HOME))
        home_x, home_y = hp.x, hp.y

        inbox, _ = l1_on(dev, HOME, 64, 64)    # 4096 floats of inbox on home
        out, _ = l1_on(dev, HOME, 32, 32)
        inbox_addr, out_addr = inbox.buffer_address(), out.buffer_address()

        # ---- call 1: scatter — each source writes (s+1) into its K slots ----
        src_cores = [(1 + i, 0) for i in range(N_SRC)]
        src_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
                                     for (x, y) in src_cores])
        src_rt = ttnn.RuntimeArgs()
        for s, (x, y) in enumerate(src_cores):
            src_rt[x][y] = [home_x, home_y, inbox_addr, s, K, f2u(float(s + 1))]
        src_k = ttnn.KernelDescriptor(
            kernel_source=SRC_KERNEL, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=src_crs, compile_time_args=[], runtime_args=src_rt,
            config=ttnn.WriterConfigDescriptor())
        ttnn.generic_op([inbox, out], ttnn.ProgramDescriptor(kernels=[src_k], semaphores=[], cbs=[]))

        # ---- call 2: drain+reduce on home (timed) ----
        home_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
        drain_rt = ttnn.RuntimeArgs()
        drain_rt[HOME[0]][HOME[1]] = [inbox_addr, TOTAL, out_addr]
        drain_k = ttnn.KernelDescriptor(
            kernel_source=DRAIN_KERNEL, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=home_crs, compile_time_args=[], runtime_args=drain_rt,
            config=ttnn.ReaderConfigDescriptor())
        ttnn.generic_op([inbox, out], ttnn.ProgramDescriptor(kernels=[drain_k], semaphores=[], cbs=[]))

        res = ttnn.to_torch(out)
        acc, cyc = float(res[0, 0, 0, 0]), float(res[0, 0, 0, 1])
        expected = K * sum(range(1, N_SRC + 1))
        cpe = cyc / TOTAL
        print(f"home=({home_x},{home_y}) sources={N_SRC} K={K} total={TOTAL}")
        print(f"acc={acc}  expected={expected}  {'OK' if abs(acc-expected)<0.5 else 'MISMATCH'}")
        print(f"drain cycles={cyc:.0f}  cycles/elem={cpe:.2f}")
        print("M2B_OK" if abs(acc - expected) < 0.5 else "M2B_FAIL")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
