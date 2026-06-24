#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
M5 milestone 3: MULTI-TILE SFPU rasterizer — render a full image across a Tensix CORE GRID,
one 32x32 screen tile per core, in parallel. Reuses M5's single-tile blend kernel verbatim;
the pixel grids (PX,PY) and output are BLOCK-SHARDED over the grid so each core automatically
operates on its own tile (global pixel coords baked into its shard).

Validated vs CPU golden; THROUGHPUT TELEMETRY (host wall-clock) reported across grid sizes to
show it scales with cores: Mpixels/s and Mgaussian-blends/s, plus per-tile latency.
"""
import struct, math, time, torch, ttnn
from sfpu_blend import READER, COMPUTE, WRITER   # reuse the verified single-tile blend kernels

N = 32                                            # Gaussians (compile-time; all blended per tile)
TS = 32


def f2u(x): return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def scene(seed, W, H):
    g = torch.Generator().manual_seed(seed)
    cx = torch.rand(N, generator=g)*W; cy = torch.rand(N, generator=g)*H
    sx = 4 + torch.rand(N, generator=g)*8; sy = 4 + torch.rand(N, generator=g)*8
    th = torch.rand(N, generator=g)*math.pi
    op = 0.4 + torch.rand(N, generator=g)*0.4; col = 0.3 + torch.rand(N, generator=g)*0.6
    order = torch.argsort(torch.rand(N, generator=g)).tolist()
    abc = []
    for i in range(N):
        ct, st = math.cos(th[i]), math.sin(th[i])
        R = torch.tensor([[ct, -st], [st, ct]])
        M = torch.inverse(R @ torch.diag(torch.tensor([sx[i]**2, sy[i]**2])) @ R.T)
        abc.append((float(M[0, 0]), float(M[0, 1]), float(M[1, 1])))
    return cx, cy, op, col, order, abc


def golden(cx, cy, op, col, order, abc, W, H):
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    PX, PY = jj.float(), ii.float()
    C = torch.zeros(H, W); T = torch.ones(H, W)
    for i in order:
        a, b, c = abc[i]; dx, dy = PX-float(cx[i]), PY-float(cy[i])
        al = (float(op[i])*torch.exp(-0.5*(a*dx*dx+2*b*dx*dy+c*dy*dy))).clamp(max=0.99)
        C = C + T*al*float(col[i]); T = T*(1-al)
    return C


def block_l1(dev, grid, GX, GY, data=None):
    H, W = GY*TS, GX*TS
    sh = ttnn.ShardSpec(grid, [TS, TS], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, sh)
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, H, W]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, H, W).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)


def render_multitile(dev, GX, GY, cx, cy, op, col, order, abc, validate=False, reps=5):
    W, H = GX*TS, GY*TS
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX-1, GY-1))])
    PXt = block_l1(dev, grid, GX, GY, jj.float())     # global x-coords, one tile/core
    PYt = block_l1(dev, grid, GX, GY, ii.float())
    OUT = block_l1(dev, grid, GX, GY)
    NB = TS*TS*4

    params = []
    for i in order:
        a, b, c = abc[i]
        params += [f2u(cx[i]), f2u(cy[i]), f2u(a), f2u(2*b), f2u(c), f2u(op[i]), f2u(col[i])]

    rt_r, rt_c, rt_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for gx in range(GX):
        for gy in range(GY):
            hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(gx, gy)); sx, sy = hp.x, hp.y
            rt_r[gx][gy] = [sx, sy, PXt.buffer_address(), PYt.buffer_address(), NB]
            rt_c[gx][gy] = params
            rt_w[gx][gy] = [sx, sy, OUT.buffer_address(), NB]
    cbf = lambda idx: ttnn.CBDescriptor(total_size=2*NB, core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=NB)])
    mk = lambda src, rt, cfg, cta=[]: ttnn.KernelDescriptor(
        kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=grid, runtime_args=rt, compile_time_args=cta, config=cfg)
    prog = ttnn.ProgramDescriptor(kernels=[
        mk(READER, rt_r, ttnn.ReaderConfigDescriptor()),
        mk(COMPUTE, rt_c, ttnn.ComputeConfigDescriptor(), [N]),
        mk(WRITER, rt_w, ttnn.WriterConfigDescriptor())], semaphores=[], cbs=[cbf(0), cbf(1), cbf(16)])

    ttnn.generic_op([PXt, OUT], prog)                 # warmup (compile)
    res = None
    if validate:
        res = ttnn.to_torch(OUT).reshape(H, W)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); ttnn.generic_op([PXt, OUT], prog); _ = ttnn.to_torch(OUT)[0, 0, 0, 0]
        t.append(time.perf_counter() - t0)
    return res, sorted(t)[len(t)//2]                  # median wall-clock (incl. readback)


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        # validate at 4x4 tiles
        GX = GY = 4; W = H = GX*TS
        cx, cy, op, col, order, abc = scene(2, W, H)
        got, _ = render_multitile(dev, GX, GY, cx, cy, op, col, order, abc, validate=True)
        gold = golden(cx, cy, op, col, order, abc, W, H)
        mse = float(((got-gold)**2).mean()); psnr = 10*math.log10(float(gold.max())**2/max(mse, 1e-12))
        print(f"validate {GX}x{GY} tiles ({W}x{H})  MSE={mse:.3e}  PSNR={psnr:.1f} dB  -> "
              f"{'OK' if mse < 1e-4 else 'FAIL'}")

        # throughput scaling: more tiles = more cores in parallel
        print(f"\nthroughput (N={N} Gaussians/tile, median of 5):")
        print(f"  {'grid':>7} {'cores':>5} {'image':>9} {'ms':>7} {'Mpix/s':>9} {'Mblend/s':>9} {'us/tile':>8}")
        for g in (2, 4, 6, 8):
            W = H = g*TS; cx, cy, op, col, order, abc = scene(2, W, H)
            _, dt = render_multitile(dev, g, g, cx, cy, op, col, order, abc)
            tiles = g*g; pix = W*H; blends = tiles*N
            print(f"  {g}x{g:<5} {tiles:>5} {W}x{H:<5} {dt*1e3:7.2f} {pix/dt/1e6:9.1f} "
                  f"{blends/dt/1e6:9.2f} {dt/tiles*1e6:8.1f}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
