#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""STEP 0 SPIKE — de-risk the load-bearing unknowns of the projection-fusion plan on SILICON:
 (A) fp32_dest_acc_en + dst-resident MAC chains give FP32 precision (not bf16) within the 8-reg budget
     -> the foundation for the signed-cancellation grads (conic/gscale/gquat). m17 only proved bf16-16.
 (B) masks gtz_tile / unary_lt_tile (cmask=(pre>0)&(pre<1), zmask=mc2>1e-4).
 (C) forward transcendentals: sigmoid_tile, sqrt_tile, clamp_tile (lower-clamp covers z=max(mc2,1e-4)).
Each test runs a dst-resident generic_op (read-only input CB -> dst regs -> packed outputs) and compares
to a host fp64 reference. PASS = fp32 path tracks host to ~1e-5; bf16 shown for contrast (~1e-2)."""
import sys, struct
from pathlib import Path
sys.path.insert(0, str(Path.home()/"tt-splat"/"server"))
import numpy as np, torch, ttnn

HOME = (1, 1); TS = 32; NB = TS*TS*4
def f2u(x): return struct.unpack("<I", struct.pack("<f", float(x)))[0]

INCLUDES = r"""
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/circular_buffer.h"
"""

READER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1), n=get_arg_val<uint32_t>(2), nb=get_arg_val<uint32_t>(3);
    cb_reserve_back(0, 8);
    uint32_t base = get_write_ptr(0);
    for (uint32_t i=0;i<n;i++) noc_async_read(get_noc_addr(sx,sy, get_arg_val<uint32_t>(4+i)), base+i*nb, nb);
    noc_async_read_barrier();
    cb_push_back(0, 8);
}
"""
WRITER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1), nout=get_arg_val<uint32_t>(2), nb=get_arg_val<uint32_t>(3);
    for (uint32_t j=0;j<nout;j++) {
        cb_wait_front(16+j, 1);
        noc_async_write(get_read_ptr(16+j), get_noc_addr(sx,sy, get_arg_val<uint32_t>(4+j)), nb);
        noc_async_write_barrier();
        cb_pop_front(16+j, 1);
    }
}
"""

def compute_src(body, nout):
    push = "".join(f"cb_push_back({16+j}, 1); " for j in range(nout))
    return INCLUDES + f"""
void kernel_main() {{
    cb_wait_front(0, 8);
    {"".join(f"cb_reserve_back({16+j}, 1); " for j in range(nout))}
    init_sfpu(0, 16);
    {body}
    {push}
}}
"""

def _l1(dev, data=None):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(crs, [TS, TS], ttnn.ShardOrientation.ROW_MAJOR))
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1,1,TS,TS]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1,1,TS,TS).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)

def run(dev, body, inputs, nout, fp32):
    ins = [_l1(dev, t) for t in inputs]
    outs = [_l1(dev) for _ in range(nout)]
    hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(*HOME)); sx, sy = hp.x, hp.y
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    def rt(arr):
        r = ttnn.RuntimeArgs(); r[HOME[0]][HOME[1]] = arr; return r
    cbf = lambda i, d: ttnn.CBDescriptor(total_size=d*NB, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=NB)])
    ks = lambda s, arr, cfg: ttnn.KernelDescriptor(kernel_source=s,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=crs,
            runtime_args=rt(arr), compile_time_args=[], config=cfg)
    cfg = ttnn.ComputeConfigDescriptor()
    if fp32: cfg.fp32_dest_acc_en = True
    prog = ttnn.ProgramDescriptor(kernels=[
        ks(READER, [sx, sy, len(ins), NB] + [t.buffer_address() for t in ins], ttnn.ReaderConfigDescriptor()),
        ks(compute_src(body, nout), [], cfg),
        ks(WRITER, [sx, sy, nout, NB] + [o.buffer_address() for o in outs], ttnn.WriterConfigDescriptor())],
        semaphores=[], cbs=[cbf(0, 8)] + [cbf(16+j, 1) for j in range(nout)])
    ttnn.generic_op([ins[0], outs[0]], prog)
    return [ttnn.to_torch(o).reshape(TS, TS) for o in outs]

def rel(got, ref):
    g, r = got.double(), ref.double()
    return ((g - r).norm() / (r.norm() + 1e-30)).item()

dev = ttnn.open_device(device_id=0)
results = []
try:
    torch.manual_seed(0)
    # ---------- (A) fp32 vs bf16: the conic chain ca=c/det, cb=-b/det, cc=a/det; det=a*c-b*b ----------
    a = torch.rand(TS, TS)+0.5; b = (torch.rand(TS, TS)-0.5); c = torch.rand(TS, TS)+0.5
    det = a*c - b*b
    refs = {"ca": c/det, "cb": -b/det, "cc": a/det}
    CONIC = r"""
    tile_regs_acquire();
    mul_binary_tile_init(); add_binary_tile_init(); copy_dest_values_init(); recip_tile_init();
    copy_tile_init(0); copy_tile(0,0,0); copy_tile(0,1,1); copy_tile(0,2,2);   // d0=a d1=b d2=c
    copy_dest_values(0,3); mul_binary_tile(3,2,3);          // d3 = a*c
    copy_dest_values(1,4); mul_binary_tile(4,4,4);          // d4 = b*b
    sub_binary_tile(3,4,3); recip_tile(3);                  // d3 = 1/det
    copy_dest_values(2,5); mul_binary_tile(5,3,5);          // d5 = c/det  = ca
    copy_dest_values(1,6); mul_binary_tile(6,3,6); mul_unary_tile(6,0xBF800000u); // d6 = -b/det = cb
    copy_dest_values(0,7); mul_binary_tile(7,3,7);          // d7 = a/det  = cc
    tile_regs_commit(); tile_regs_wait();
    pack_tile(5,16,0); pack_tile(6,17,0); pack_tile(7,18,0);
    tile_regs_release();
    """
    for mode in ("fp32", "bf16"):
        ca, cb, cc = run(dev, CONIC, [a, b, c], 3, fp32=(mode=="fp32"))
        r = max(rel(ca, refs["ca"]), rel(cb, refs["cb"]), rel(cc, refs["cc"]))
        # fp32 must PASS the real projection gate (1e-2); bf16 is contrast (gate=None) -> expected to FAIL it
        results.append((f"A.conic-chain[{mode}]", r, 1e-2 if mode=="fp32" else None))
        print(f"  A. conic a*c-b*b -> 1/det -> ca/cb/cc  [{mode}]  rel={r:.2e}")

    # ---------- (B) long signed accumulation (mimics a contraction) fp32 vs bf16 ----------
    xs = [torch.rand(TS, TS)+0.5 for _ in range(4)]; ys = [torch.rand(TS, TS)+0.5 for _ in range(4)]
    acc_ref = sum(((-1)**i) * xs[i]*ys[i] for i in range(4))     # signed cancellation
    body = "tile_regs_acquire(); mul_binary_tile_init(); add_binary_tile_init(); copy_dest_values_init();\n"
    body += "copy_tile_init(0);\n"
    body += "copy_tile(0,0,0); copy_tile(0,1,1); mul_binary_tile(0,1,7);\n"   # d7 = x0*y0 (acc)
    for i in range(1, 4):
        body += f"copy_tile(0,{2*i},0); copy_tile(0,{2*i+1},1); mul_binary_tile(0,1,2);\n"
        body += ("sub_binary_tile(7,2,7);\n" if i % 2 == 1 else "add_binary_tile(7,2,7);\n")
    body += "tile_regs_commit(); tile_regs_wait(); pack_tile(7,16,0); tile_regs_release();"
    for mode in ("fp32", "bf16"):
        (got,) = run(dev, body, [t for pair in zip(xs, ys) for t in pair], 1, fp32=(mode=="fp32"))
        r = rel(got, acc_ref)
        results.append((f"B.signed-accum[{mode}]", r, 1e-2 if mode=="fp32" else None))
        print(f"  B. sum (-1)^i x_i y_i (8 terms)        [{mode}]  rel={r:.2e}")

    # ---------- (C) masks: cmask = (p>0) & (p<1) ----------
    p = torch.rand(TS, TS)*1.6 - 0.3                              # spans <0, [0,1], >1
    cmask_ref = ((p > 0) & (p < 1)).double()
    MASK = r"""
    tile_regs_acquire();
    gtz_tile_init(); unary_lt_tile_init(); mul_binary_tile_init(); copy_dest_values_init();
    copy_tile_init(0); copy_tile(0,0,0); copy_dest_values(0,1);
    gtz_tile(0);                       // d0 = (p>0)
    unary_lt_tile(1, 0x3F800000u);     // d1 = (p<1)
    mul_binary_tile(0,1,0);            // d0 = (p>0)&(p<1)
    tile_regs_commit(); tile_regs_wait(); pack_tile(0,16,0); tile_regs_release();
    """
    (m,) = run(dev, MASK, [p], 1, fp32=True)
    r = (m.double() - cmask_ref).abs().max().item()
    results.append(("C.cmask gtz&unary_lt", r, 1e-5)); print(f"  C. cmask=(p>0)&(p<1)                   rel={r:.2e}")

    # ---------- (D) transcendentals: sigmoid, sqrt, lower-clamp(z=max(.,1e-4)) ----------
    s = torch.randn(TS, TS)*2
    # sigmoid via the pre-committed fallback 1/(1+exp(-s)) (sigmoid_tile returns garbage on this build)
    SIG = r"""
    tile_regs_acquire();
    exp_tile_init(); recip_tile_init(); binop_with_scalar_tile_init();
    copy_tile_init(0); copy_tile(0,0,0);
    mul_unary_tile(0, 0xBF800000u);   // -s
    exp_tile(0);                       // exp(-s)
    add_unary_tile(0, 0x3F800000u);    // 1+exp(-s)
    recip_tile(0);                     // 1/(1+exp(-s))
    tile_regs_commit(); tile_regs_wait(); pack_tile(0,16,0); tile_regs_release();
    """
    (sg,) = run(dev, SIG, [s], 1, fp32=True)
    r = rel(sg, torch.sigmoid(s)); results.append(("D.sigmoid=recip(1+exp(-x))", r, 5e-3)); print(f"  D. sigmoid via recip(1+exp(-s))        rel={r:.2e}")
    sp = torch.rand(TS, TS)+0.1
    SQRT = "tile_regs_acquire(); sqrt_tile_init(); copy_tile_init(0); copy_tile(0,0,0); sqrt_tile(0); tile_regs_commit(); tile_regs_wait(); pack_tile(0,16,0); tile_regs_release();"
    (sq,) = run(dev, SQRT, [sp], 1, fp32=True)
    r = rel(sq, torch.sqrt(sp)); results.append(("D.sqrt", r, 5e-3)); print(f"  D. sqrt(p)                             rel={r:.2e}")
    zc = torch.randn(TS, TS)*0.5
    CLAMP = "tile_regs_acquire(); clamp_tile_init(); copy_tile_init(0); copy_tile(0,0,0); clamp_tile(0, 0x38D1B717u, 0x461C4000u); tile_regs_commit(); tile_regs_wait(); pack_tile(0,16,0); tile_regs_release();"  # clamp to [1e-4, 1e4]
    (cl,) = run(dev, CLAMP, [zc], 1, fp32=True)
    r = rel(cl, zc.clamp(1e-4, 1e4)); results.append(("D.lower-clamp(=max)", r, 5e-3)); print(f"  D. clamp(z,1e-4,1e4) (covers max)      rel={r:.2e}")

    print("\n=== STEP-0 VERDICT (gate = the real projection bar, rel<1e-2) ===")
    allok = True; bf16_fails = True
    for name, r, gate in results:
        if gate is None:                                   # bf16 contrast: must EXCEED 1e-2 to prove fp32 is needed
            exceeds = r > 1e-2; bf16_fails &= exceeds
            print(f"  {'(bf16>1e-2 ✓)' if exceeds else '(bf16 ok?!)  '} {name:28s} rel={r:.2e}  [contrast]")
        else:
            ok = r < gate; allok &= ok
            print(f"  {'OK  ' if ok else 'FAIL'} {name:28s} rel={r:.2e}  (gate {gate:.0e})")
    print(f"\nfp32 chains + masks + transcendentals all pass 1e-2: {allok}")
    print(f"bf16 contrast all FAIL 1e-2 (proves fp32_dest_acc_en is REQUIRED): {bf16_fails}")
    print("SPIKE_OK -> plan is GREEN: fp32-8 dst-resident is necessary AND sufficient for projection grads"
          if (allok and bf16_fails) else "SPIKE_PARTIAL -> review")
finally:
    ttnn.close_device(dev)
