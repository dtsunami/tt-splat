#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
M2a-local: prove float accumulation in a tile's L1 from a baby RISC, via ttnn.generic_op
(custom kernel, authored inline). This is the foundation of the locality thesis: if a tile
can accumulate a stream locally and bit-exactly, the cross-core scatter-add wall is sidestepped
by giving each Gaussian a home tile.

Kernel: one data-movement RISC does  acc=0; for i in 0..N: acc += 1.0f;  in L1.
Host: read back, expect == N.
"""
import torch
import ttnn

CORE = (0, 0)
N = 4096

KERNEL = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t acc_addr = get_arg_val<uint32_t>(0);
    uint32_t n        = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr float* acc = (volatile tt_l1_ptr float*)(acc_addr);
    acc[0] = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        acc[0] += 1.0f;
    }
}
"""

def main():
    dev = ttnn.open_device(device_id=0)
    try:
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*CORE), ttnn.CoreCoord(*CORE))])
        # pin the accumulator tile to CORE's L1 so the kernel write and to_torch read the SAME core
        shard = ttnn.ShardSpec(core, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)
        l1_core = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
        inp = ttnn.from_torch(torch.zeros([1, 1, 32, 32]), dtype=ttnn.float32,
                              layout=ttnn.TILE_LAYOUT, device=dev, memory_config=l1_core)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 32, 32]), ttnn.float32, ttnn.TILE_LAYOUT, dev, l1_core,
        )

        rt = ttnn.RuntimeArgs()
        rt[CORE[0]][CORE[1]] = [out.buffer_address(), N]

        kdesc = ttnn.KernelDescriptor(
            kernel_source=KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=core,
            compile_time_args=[],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(kernels=[kdesc], semaphores=[], cbs=[])

        ttnn.generic_op([inp, out], prog)
        val = float(ttnn.to_torch(out)[0, 0, 0, 0])
        print(f"N={N}  acc[0]={val}  expected={N}")
        print("M2A_LOCAL_OK" if abs(val - N) < 0.5 else "M2A_LOCAL_FAIL")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
