#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Device rasterizer scale-up (items 1+2): per-tile CULLING + unbounded N via BATCHED dispatches
with PERSISTENT L1 accumulators.

Each core keeps its tile's C (color) and T (transmittance) resident in L1. We dispatch the blend
kernel once per batch of B Gaussians; each core processes ITS OWN culled, depth-sorted list
(from M6 binning) batch-by-batch, reading C/T in and writing C/T back. So:
  - item 1 (cull): a tile only blends Gaussians whose 3σ AABB overlaps it.
  - item 2 (unbounded N): N = (#batches)·B, B is the only compile-time cap; #batches is host-looped.

Metaparams note: because the loop is host-orchestrated dispatches, continuous metaparams (lr/β/ε/
thresholds) are host-side and change live with NO kernel recompile (see design in the chat).

Validated vs a host render using the SAME per-tile lists (exact); telemetry reports actual
gaussian-blends (culled) and throughput.
"""
import struct, math, time, torch, ttnn
import numpy as np
from sfpu_blend import READER as _unused  # noqa  (kept for parity; we define our own below)
from bin_sort import bin_and_sort

B = 16          # Gaussians per dispatch (compile-time batch); N is unbounded via #batches
TS = 32


def f2u(x): return struct.unpack("<I", struct.pack("<f", float(x)))[0]
DUMMY = [0, 0, f2u(1.0), 0, f2u(1.0), 0, 0]   # op=0 -> alpha 0 -> no contribution (padding)

READER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t px=get_arg_val<uint32_t>(2), py=get_arg_val<uint32_t>(3);
    uint32_t ci=get_arg_val<uint32_t>(4), ti=get_arg_val<uint32_t>(5), nb=get_arg_val<uint32_t>(6);
    cb_reserve_back(0,1); noc_async_read(get_noc_addr(sx,sy,px), get_write_ptr(0), nb);
    cb_reserve_back(1,1); noc_async_read(get_noc_addr(sx,sy,py), get_write_ptr(1), nb);
    cb_reserve_back(2,1); noc_async_read(get_noc_addr(sx,sy,ci), get_write_ptr(2), nb);
    cb_reserve_back(3,1); noc_async_read(get_noc_addr(sx,sy,ti), get_write_ptr(3), nb);
    noc_async_read_barrier();
    cb_push_back(0,1); cb_push_back(1,1); cb_push_back(2,1); cb_push_back(3,1);
}
"""

COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
#define ONE 0x3F800000u
#define NEG1 0xBF800000u
#define NHALF 0xBF000000u
void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    init_sfpu(0, 16);
    binop_with_scalar_tile_init();
    cb_wait_front(0,1); cb_wait_front(1,1); cb_wait_front(2,1); cb_wait_front(3,1);
    cb_reserve_back(16,1); cb_reserve_back(17,1);
    tile_regs_acquire();
    copy_tile_init(2); copy_tile(2,0,0);     // C (accumulator) -> dst0
    copy_tile_init(3); copy_tile(3,0,1);     // T (transmittance) -> dst1
    for (uint32_t g = 0; g < B; g++) {
        uint32_t b = g*7;
        uint32_t cxb=get_arg_val<uint32_t>(b+0), cyb=get_arg_val<uint32_t>(b+1);
        uint32_t ab=get_arg_val<uint32_t>(b+2), twob=get_arg_val<uint32_t>(b+3);
        uint32_t cc=get_arg_val<uint32_t>(b+4), opb=get_arg_val<uint32_t>(b+5), colb=get_arg_val<uint32_t>(b+6);
        copy_tile_init(0); copy_tile(0,0,2); sub_unary_tile(2, cxb);
        copy_tile_init(1); copy_tile(1,0,3); sub_unary_tile(3, cyb);
        mul_binary_tile_init();
        mul_binary_tile(2,2,4); mul_unary_tile(4, ab);
        mul_binary_tile(3,3,5); mul_unary_tile(5, cc);
        mul_binary_tile(2,3,2); mul_unary_tile(2, twob);
        add_binary_tile_init();
        add_binary_tile(4,5,4); add_binary_tile(4,2,4);
        mul_unary_tile(4, NHALF);
        exp_tile_init(); exp_tile(4);
        mul_unary_tile(4, opb);
        mul_binary_tile_init(); mul_binary_tile(1,4,5);
        mul_unary_tile(5, colb);
        add_binary_tile_init(); add_binary_tile(0,5,0);     // C += T*alpha*col
        mul_unary_tile(4, NEG1); add_unary_tile(4, ONE);
        mul_binary_tile_init(); mul_binary_tile(1,4,1);     // T *= (1-alpha)
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, 16); pack_tile(1, 17);
    tile_regs_release();
    cb_push_back(16,1); cb_push_back(17,1);
    cb_pop_front(0,1); cb_pop_front(1,1); cb_pop_front(2,1); cb_pop_front(3,1);
}
"""

WRITER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t co=get_arg_val<uint32_t>(2), to=get_arg_val<uint32_t>(3), nb=get_arg_val<uint32_t>(4);
    cb_wait_front(16,1); cb_wait_front(17,1);
    noc_async_write(get_read_ptr(16), get_noc_addr(sx,sy,co), nb);
    noc_async_write(get_read_ptr(17), get_noc_addr(sx,sy,to), nb);
    noc_async_write_barrier();
    cb_pop_front(16,1); cb_pop_front(17,1);
}
"""


def scene(seed, W, H, N):
    g = torch.Generator().manual_seed(seed)
    cx = torch.rand(N, generator=g)*W; cy = torch.rand(N, generator=g)*H
    sx = 4 + torch.rand(N, generator=g)*6; sy = 4 + torch.rand(N, generator=g)*6
    th = torch.rand(N, generator=g)*math.pi
    op = 0.4 + torch.rand(N, generator=g)*0.4; col = 0.3 + torch.rand(N, generator=g)*0.6
    depth = torch.rand(N, generator=g)
    abc = []
    for i in range(N):
        ct, st = math.cos(th[i]), math.sin(th[i])
        R = torch.tensor([[ct, -st], [st, ct]])
        M = torch.inverse(R @ torch.diag(torch.tensor([sx[i]**2, sy[i]**2])) @ R.T)
        abc.append((float(M[0, 0]), float(M[0, 1]), float(M[1, 1])))
    return cx, cy, sx, sy, op, col, depth, abc


def golden_culled(cx, cy, op, col, abc, tile_lists, W, H, ntx):
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    C = torch.zeros(H, W); T = torch.ones(H, W)
    PX, PY = jj.float(), ii.float()
    for t, lst in enumerate(tile_lists):
        tx, ty = t % ntx, t // ntx
        ys, xs = slice(ty*TS, ty*TS+TS), slice(tx*TS, tx*TS+TS)
        c, tr = torch.zeros(TS, TS), torch.ones(TS, TS)
        px, py = PX[ys, xs], PY[ys, xs]
        for i in lst:
            a, b, cc = abc[i]; dx, dy = px-float(cx[i]), py-float(cy[i])
            al = (float(op[i])*torch.exp(-0.5*(a*dx*dx+2*b*dx*dy+cc*dy*dy))).clamp(max=0.99)
            c = c + tr*al*float(col[i]); tr = tr*(1-al)
        C[ys, xs] = c
    return C


def block_l1(dev, grid, W, H, data=None):
    sh = ttnn.ShardSpec(grid, [TS, TS], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, sh)
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, H, W]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, H, W).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        GX = GY = 6; W = H = GX*TS; N = 160          # N=160 >> B=16  -> proves unbounded N
        cx, cy, sx, sy, op, col, depth, abc = scene(2, W, H, N)
        s_gid, s_tile, ranges, ntx, nty, total = bin_and_sort(
            cx.numpy(), cy.numpy(), (sx**2).numpy(), (sy**2).numpy(), depth.numpy(), W, H, ts=TS)
        tile_lists = [s_gid[ranges[t, 0]:ranges[t, 1]].tolist() for t in range(ntx*nty)]
        max_count = max((len(l) for l in tile_lists), default=0)
        nbatch = (max_count + B - 1)//B
        actual_blends = sum(len(l) for l in tile_lists)
        print(f"{GX}x{GY} tiles, N={N}, dup={total/N:.1f}x, max gaussians/tile={max_count}, "
              f"batches={nbatch} (B={B}), culled blends={actual_blends} vs all-N={ntx*nty*N}")

        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX-1, GY-1))])
        ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        PXt = block_l1(dev, grid, W, H, jj.float()); PYt = block_l1(dev, grid, W, H, ii.float())
        acc = {"C": block_l1(dev, grid, W, H, torch.zeros(H, W)), "T": block_l1(dev, grid, W, H, torch.ones(H, W))}
        NB = TS*TS*4

        def compute_cfg():
            cfg = ttnn.ComputeConfigDescriptor()
            cfg.fp32_dest_acc_en = True; cfg.math_approx_mode = False   # precision over batched accum
            return cfg

        def params_for(t, d):
            lst = tile_lists[t][d*B:(d+1)*B]
            out = []
            for k in range(B):
                if k < len(lst):
                    i = lst[k]; a, b, c = abc[i]
                    out += [f2u(cx[i]), f2u(cy[i]), f2u(a), f2u(2*b), f2u(c), f2u(op[i]), f2u(col[i])]
                else:
                    out += DUMMY
            return out

        cbf = lambda idx: ttnn.CBDescriptor(total_size=2*NB, core_ranges=grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=NB)])
        coords = {}
        for gx in range(GX):
            for gy in range(GY):
                hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(gx, gy)); coords[(gx, gy)] = (hp.x, hp.y)

        def dispatch(d):
            rt_r, rt_c, rt_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for gx in range(GX):
                for gy in range(GY):
                    sx_, sy_ = coords[(gx, gy)]; t = gy*ntx + gx
                    rt_r[gx][gy] = [sx_, sy_, PXt.buffer_address(), PYt.buffer_address(),
                                    acc["C"].buffer_address(), acc["T"].buffer_address(), NB]
                    rt_c[gx][gy] = params_for(t, d)
                    rt_w[gx][gy] = [sx_, sy_, acc["C"].buffer_address(), acc["T"].buffer_address(), NB]
            mk = lambda src, rt, cfg, cta=[]: ttnn.KernelDescriptor(
                kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=grid, runtime_args=rt, compile_time_args=cta, config=cfg)
            prog = ttnn.ProgramDescriptor(kernels=[
                mk(READER, rt_r, ttnn.ReaderConfigDescriptor()),
                mk(COMPUTE, rt_c, compute_cfg(), [B]),
                mk(WRITER, rt_w, ttnn.WriterConfigDescriptor())], semaphores=[], cbs=[cbf(i) for i in (0, 1, 2, 3, 16, 17)])
            ttnn.generic_op([PXt, acc["C"]], prog)

        for d in range(nbatch): dispatch(d)              # warmup: JIT-compile the kernels
        acc["C"] = block_l1(dev, grid, W, H, torch.zeros(H, W))   # reset accumulators
        acc["T"] = block_l1(dev, grid, W, H, torch.ones(H, W))
        t0 = time.perf_counter()
        for d in range(nbatch): dispatch(d)
        _ = ttnn.to_torch(acc["C"])[0, 0, 0, 0]
        dt = time.perf_counter() - t0

        got = ttnn.to_torch(acc["C"]).reshape(H, W)
        gold = golden_culled(cx, cy, op, col, abc, tile_lists, W, H, ntx)
        mse = float(((got-gold)**2).mean()); psnr = 10*math.log10(float(gold.max())**2/max(mse, 1e-12))
        print(f"validate vs host-culled golden  MSE={mse:.3e}  PSNR={psnr:.1f} dB  -> {'OK' if mse < 1e-4 else 'FAIL'}")
        print(f"telemetry: {nbatch} dispatches, {dt*1e3:.1f} ms, {actual_blends/dt/1e6:.2f} Mblend/s (culled), "
              f"cull ratio {actual_blends/(ntx*nty*N):.2f} of all-N")
        print("SCALED_OK" if mse < 1e-4 else "SCALED_FAIL")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
