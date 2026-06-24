#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
M2b IPC study: the home drain/reduce is the scatter-add's serial bottleneck. Baseline was
~60 cyc/elem -> suspect SOFT-FLOAT (baby RISCs have no FPU). Compare reduce strategies on the
same 4096-elem L1 buffer, report cycles/element:

  mode 0: float scalar       acc += in[i]            (baseline)
  mode 1: int32 scalar       acc += ini[i]           (no FPU -> should be ~1-3 cyc)
  mode 2: float 4-accumulator unroll (hide soft-float latency via ILP)
  mode 3: int32 4-accumulator unroll

Conclusion drives whether gradient accumulation should be fixed-point (int) on the baby RISCs,
or pushed to the compute engine.
"""
import struct, torch, ttnn

HOME = (0, 0)
TOTAL = 4096

FILL = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t addr = get_arg_val<uint32_t>(0);
    uint32_t n    = get_arg_val<uint32_t>(1);
    uint32_t bits = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* p = (volatile tt_l1_ptr uint32_t*)addr;
    for (uint32_t i = 0; i < n; i++) p[i] = bits;   // int store, no float op
}
"""

DRAIN = r"""
#include "dataflow_api.h"
#include "risc_common.h"
void kernel_main() {
    uint32_t ia = get_arg_val<uint32_t>(0);
    uint32_t total = get_arg_val<uint32_t>(1);
    uint32_t oa = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr float* in  = (volatile tt_l1_ptr float*)ia;
    volatile tt_l1_ptr int*   ini = (volatile tt_l1_ptr int*)ia;
    volatile tt_l1_ptr float* out = (volatile tt_l1_ptr float*)oa;
    uint32_t t0 = get_timestamp_32b();
    float acc = 0.0f;
#if MODE==0
    for (uint32_t i = 0; i < total; i++) acc += in[i];
#elif MODE==1
    int s = 0; for (uint32_t i = 0; i < total; i++) s += ini[i]; acc = (float)s;
#elif MODE==2
    float a0=0,a1=0,a2=0,a3=0;
    for (uint32_t i=0;i<total;i+=4){a0+=in[i];a1+=in[i+1];a2+=in[i+2];a3+=in[i+3];}
    acc=(a0+a1)+(a2+a3);
#elif MODE==3
    int s0=0,s1=0,s2=0,s3=0;
    for (uint32_t i=0;i<total;i+=4){s0+=ini[i];s1+=ini[i+1];s2+=ini[i+2];s3+=ini[i+3];}
    acc=(float)((s0+s1)+(s2+s3));
#elif MODE==4
    int s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
    for (uint32_t i=0;i<total;i+=8){s0+=ini[i];s1+=ini[i+1];s2+=ini[i+2];s3+=ini[i+3];
        s4+=ini[i+4];s5+=ini[i+5];s6+=ini[i+6];s7+=ini[i+7];}
    acc=(float)(((s0+s1)+(s2+s3))+((s4+s5)+(s6+s7)));
#elif MODE==5
    int s[16]; for(int j=0;j<16;j++) s[j]=0;
    for (uint32_t i=0;i<total;i+=16){ for(int j=0;j<16;j++) s[j]+=ini[i+j]; }
    int t=0; for(int j=0;j<16;j++) t+=s[j]; acc=(float)t;
#endif
    uint32_t t1 = get_timestamp_32b();
    out[0] = acc; out[1] = (float)(t1 - t0);
}
"""

def f2u(x): return struct.unpack("<I", struct.pack("<f", x))[0]

def l1_on(dev, core, h, w):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*core), ttnn.CoreCoord(*core))])
    sh = ttnn.ShardSpec(crs, [h, w], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sh)
    return ttnn.allocate_tensor_on_device(ttnn.Shape([1,1,h,w]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)

def run_kernel(dev, src, core, rt_args, defines, io):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*core), ttnn.CoreCoord(*core))])
    rt = ttnn.RuntimeArgs(); rt[core[0]][core[1]] = rt_args
    k = ttnn.KernelDescriptor(kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                              core_ranges=crs, compile_time_args=[], runtime_args=rt,
                              defines=defines, config=ttnn.ReaderConfigDescriptor())
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[]))

def main():
    dev = ttnn.open_device(device_id=0)
    try:
        buf = l1_on(dev, HOME, 64, 64); out = l1_on(dev, HOME, 32, 32)
        ba, oa = buf.buffer_address(), out.buffer_address()
        names = {0:"float scalar",1:"int32 scalar",2:"float x4 unroll",3:"int32 x4 unroll",
                 4:"int32 x8 unroll",5:"int32 x16 unroll"}
        for mode in (0,1,2,3,4,5):
            bits = f2u(1.0) if mode in (0,2) else 1     # float 1.0 vs int 1
            run_kernel(dev, FILL, HOME, [ba, TOTAL, bits], [], [buf, out])
            run_kernel(dev, DRAIN, HOME, [ba, TOTAL, oa], [("MODE", str(mode))], [buf, out])
            res = ttnn.to_torch(out); acc, cyc = float(res[0,0,0,0]), float(res[0,0,0,1])
            print(f"mode {mode} {names[mode]:16s} acc={acc:9.1f} (exp {TOTAL}) cycles={cyc:7.0f}  cyc/elem={cyc/TOTAL:6.2f}")
    finally:
        ttnn.close_device(dev)

if __name__ == "__main__":
    main()
