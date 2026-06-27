#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""De-risk the fused-projection-backward kernel mechanism. FINDINGS (silicon):
 (1) RW register-file FAILS: pack a result into POOL slot s then copy_tile it back in a later
     instruction does NOT round-trip (packer->unpacker sync hazard when you bypass the CB push/pop
     FIFO handshake with explicit tile indices). So a generic auto-generated tile-VM interpreter with
     spill-to-L1 is OUT.  -> the earlier round-trip program (slot2=slot0*slot1; slot3=slot2+slot0)
     returned rel ~1.0 garbage.
 (2) dst-RESIDENT WORKS (the m17 shape): read inputs from a read-only CB (copy_tile by index is fine,
     no writeback), keep the working set in dst regs, pack only the final outputs. a*b+a is correct to
     bf16 (~1.5e-2 abs on [0,2]). fp32 needs fp32_dest_acc_en -> dst halves 16->8 regs.
 => fused projection backward must be a hand-structured / code-generated dst-resident fp32 kernel with
    recompute-based register allocation into <=8 regs (NO spill). Multi-session compiler effort.
This proto now validates path (2): OUT = a*b + a entirely in dst regs."""
import sys, struct
from pathlib import Path
sys.path.insert(0, str(Path.home()/"tt-splat"/"server"))
import numpy as np, torch, ttnn

HOME = (1, 1); TS = 32; NB = TS*TS*4; S = 8

READER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t aa=get_arg_val<uint32_t>(2), ab=get_arg_val<uint32_t>(3), nb=get_arg_val<uint32_t>(4);
    cb_reserve_back(0, 8);
    uint32_t base = get_write_ptr(0);
    noc_async_read(get_noc_addr(sx,sy, aa), base + 0*nb, nb);   // -> POOL slot 0
    noc_async_read(get_noc_addr(sx,sy, ab), base + 1*nb, nb);   // -> POOL slot 1
    noc_async_read_barrier();
    cb_push_back(0, 8);
}
"""

COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    cb_wait_front(0, 8);                       // POOL: 8 pages, inputs in slots 0,1
    cb_reserve_back(16, 1);                    // OUT
    init_sfpu(0, 16);
    mul_binary_tile_init();
    add_binary_tile_init();
    // dst-resident: read inputs (read-only CB), compute in dst regs, pack only the final output.
    tile_regs_acquire();
    copy_tile_init(0); copy_tile(0,0,0); copy_tile(0,1,1);   // d0=a (read-only), d1=b
    mul_binary_tile(0,1,2);                                  // d2 = a*b
    add_binary_tile(2,0,3);                                  // d3 = a*b + a   (d0 still = a)
    tile_regs_commit(); tile_regs_wait();
    pack_tile(3, 16, 0);                                     // final -> OUT
    tile_regs_release();
    cb_push_back(16, 1);
}
"""

WRITER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t ao=get_arg_val<uint32_t>(2), nb=get_arg_val<uint32_t>(3);
    cb_wait_front(16, 1);
    noc_async_write(get_read_ptr(16), get_noc_addr(sx,sy, ao), nb);
    noc_async_write_barrier();
    cb_pop_front(16, 1);
}
"""

def f2u(x): return struct.unpack("<I", struct.pack("<f", float(x)))[0]

def _l1(dev, data=None):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(crs, [TS, TS], ttnn.ShardOrientation.ROW_MAJOR))
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1,1,TS,TS]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1,1,TS,TS).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)

dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    a = torch.rand(TS, TS); b = torch.rand(TS, TS)
    ta, tb, tout = _l1(dev, a), _l1(dev, b), _l1(dev)
    hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(*HOME)); sx, sy = hp.x, hp.y
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    def rt(arr):
        r = ttnn.RuntimeArgs(); r[HOME[0]][HOME[1]] = arr; return r
    cbf = lambda i, d: ttnn.CBDescriptor(total_size=d*NB, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=NB)])
    ks = lambda s, arr, cfg, cta=[]: ttnn.KernelDescriptor(
        kernel_source=s, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=crs, runtime_args=rt(arr), compile_time_args=cta, config=cfg)
    prog = ttnn.ProgramDescriptor(kernels=[
        ks(READER, [sx, sy, ta.buffer_address(), tb.buffer_address(), NB], ttnn.ReaderConfigDescriptor()),
        ks(COMPUTE, [], ttnn.ComputeConfigDescriptor()),
        ks(WRITER, [sx, sy, tout.buffer_address(), NB], ttnn.WriterConfigDescriptor())],
        semaphores=[], cbs=[cbf(0, S), cbf(16, 1)])
    ttnn.generic_op([ta, tout], prog)
    got = ttnn.to_torch(tout).reshape(TS, TS)
    ref = a*b + a
    err = (got - ref).abs().max().item()
    print(f"max|got-ref| = {err:.3e}  (a*b+a, dst-resident; ~1.5e-2 = bf16 rounding, math correct)")
    print("DST_RESIDENT_OK" if err < 3e-2 else "DST_RESIDENT_FAIL")
finally:
    ttnn.close_device(dev)
