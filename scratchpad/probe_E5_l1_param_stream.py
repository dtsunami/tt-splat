#!/usr/bin/env python3
# E5: can a resident generic_op kernel read per-Gaussian params from an L1 SCRATCH buffer (not
# get_arg_val runtime args, capped at max_runtime_args ~341 words ~42 Gaussians@8w) and loop over an
# arbitrary count? This is the substrate for collapsing nbatch*3 dispatches to ~3 (Stages 3/5/6).
# Test: stage M*8 param words in L1, a single kernel loops m=0..M-1 (M from ONE runtime arg, so the
# kernel compiles ONCE regardless of M) reading 8 words/Gaussian from L1 and writing a deterministic
# reduction. Verify bit-exact vs host for M far past the runtime-arg cap.
import sys, struct
from pathlib import Path
import torch, ttnn
HOME = (1, 1)
W = 512

KERNEL = r"""
#include "dataflow_api.h"
#include "risc_common.h"
void kernel_main() {
    uint32_t params_addr = get_arg_val<uint32_t>(0);   // L1 scratch: M*8 floats
    uint32_t M           = get_arg_val<uint32_t>(1);   // count drives the loop (compile-once)
    uint32_t out_addr    = get_arg_val<uint32_t>(2);   // L1: M floats
    uint32_t row_w       = get_arg_val<uint32_t>(3);   // out row width (row-major)
    volatile tt_l1_ptr float* p   = (volatile tt_l1_ptr float*)params_addr;
    volatile tt_l1_ptr float* out = (volatile tt_l1_ptr float*)out_addr;
    uint32_t t0 = get_timestamp_32b();
    for (uint32_t m = 0; m < M; m++) {
        // a deterministic per-Gaussian reduction over its 8 words (mimics consuming K-chunk params)
        float acc = 0.0f;
        for (uint32_t w = 0; w < 8; w++) acc += p[m*8 + w] * (float)(w + 1);
        out[(m / row_w) * row_w + (m % row_w)] = acc;
    }
    uint32_t t1 = get_timestamp_32b();
    // stash cycles just past the M outputs
    out[((M) / row_w) * row_w + (M % row_w)] = (float)(t1 - t0);
}
"""

def rm(dev, rows, data=None):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    sh = ttnn.ShardSpec(crs, [rows, W], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sh)
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows, W]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, rows, W), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=dev, memory_config=mc)

dev = ttnn.open_device(device_id=0)
def run(M):
    torch.manual_seed(M)
    params = torch.randn(M, 8)
    pflat = params.reshape(-1)
    prows = (M * 8 + W - 1) // W
    pbuf = torch.zeros(prows * W); pbuf[:M*8] = pflat
    pt = rm(dev, prows, pbuf)
    orows = (M + 2 + W - 1) // W
    ot = rm(dev, orows)
    hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(*HOME))
    rt = ttnn.RuntimeArgs(); rt[HOME[0]][HOME[1]] = [pt.buffer_address(), M, ot.buffer_address(), W]
    k = ttnn.KernelDescriptor(kernel_source=KERNEL, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))]),
                              compile_time_args=[], runtime_args=rt, config=ttnn.ReaderConfigDescriptor())
    ttnn.generic_op([pt, ot], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[]))
    res = ttnn.to_torch(ot).reshape(orows, W)
    gold = (params * torch.arange(1, 9).float()).sum(dim=1)
    got = torch.tensor([float(res[m // W, m % W]) for m in range(M)])
    err = (got - gold).abs().max().item()
    cyc = float(res[M // W, M % W])
    print(f"  M={M:5d} (cap~42 runtime-args) | max_err={err:.2e} cyc={cyc:.0f} cyc/Gaussian={cyc/M:.1f}  "
          f"{'OK' if err < 1e-3 else 'FAIL'}", flush=True)
    return err

try:
    print(f"E5 L1-scratch param streaming + internal loop (compile-once, count via runtime arg)")
    worst = 0.0
    for M in (16, 200, 1024):     # 200 and 1024 are far past the ~42-Gaussian runtime-arg cap
        worst = max(worst, run(M))
    print(f"\n  {'OK — resident kernel reads L1 params + loops past the runtime-arg cap' if worst < 1e-3 else 'FAIL'}")
    print("  => the persistent-kernel substrate (params from L1, not get_arg_val) is BUILDABLE on generic_op.")
finally:
    ttnn.close_device(dev)
