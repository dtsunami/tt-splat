#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PROTO R1 (silicon) — tile-BLOCK forward raster: process >grid tiles by LOOPING blocks of <=ncores tiles,
each block's tiles sharded row-major onto the usable grid (a TILE-LIST shard). The M14 READER/COMPUTE/
WRITER kernels are REUSED VERBATIM (1 tile/core); only the harness changes. This is the fix for the
"kernels on dispatch cores" crash (384px) AND the path to 1600px (1900 tiles / 110 cores = ~18 blocks).

De-risk on a deliberately SMALL grid to force multiple blocks cheaply: GX=GY=4 (16 cores), 8x8=64 tiles
-> 4 blocks. Gate: rendered image == host golden_culled (same per-tile Gaussian lists), bit-exact-ish.
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "pathclear"))
import sfpu_raster_scaled as M14                                          # noqa: E402
from sfpu_raster_scaled import scene, golden_culled, READER, COMPUTE, WRITER, B, TS, f2u, DUMMY   # noqa: E402
from bin_sort import bin_and_sort                                        # noqa: E402

GX = GY = 4                                                              # tiny grid -> forces blocking
NTX = NTY = 8                                                            # 8x8 = 64 tiles -> 4 blocks


def tileshard(dev, grid, ntiles_pad, data=None):
    """HEIGHT_SHARDED [TS,TS]/core on [ntiles_pad*TS, TS], TILE_LAYOUT — a row-major tile-LIST shard."""
    sh = ttnn.ShardSpec(grid, [TS, TS], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sh)
    shape = ttnn.Shape([1, 1, ntiles_pad * TS, TS])
    if data is None:
        return ttnn.allocate_tensor_on_device(shape, ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, ntiles_pad * TS, TS).float(), dtype=ttnn.float32,
                           layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


def tileblock_raster(dev, ntx, nty, tile_lists, cx, cy, a, b2, c, op, col, W, H):
    """Forward raster over ALL tiles by looping blocks of <=ncores tiles onto the GXxGY usable grid."""
    ncores = GX * GY
    ntiles = ntx * nty
    nblocks = (ntiles + ncores - 1) // ncores
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
    coords = {(gx, gy): (lambda hp: (hp.x, hp.y))(dev.worker_core_from_logical_core(ttnn.CoreCoord(gx, gy)))
              for gx in range(GX) for gy in range(GY)}
    NB = TS * TS * 4
    img = np.zeros((H, W), np.float64)

    cbf = lambda i: ttnn.CBDescriptor(total_size=2 * NB, core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=NB)])
    cfg = ttnn.ComputeConfigDescriptor(); cfg.fp32_dest_acc_en = True; cfg.math_approx_mode = False
    mk = lambda src, rt, cf, cta=[]: ttnn.KernelDescriptor(kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid, runtime_args=rt,
            compile_time_args=cta, config=cf)

    for k in range(nblocks):
        print(f"  block {k}/{nblocks}...", flush=True)
        bt = list(range(k * ncores, min((k + 1) * ncores, ntiles)))      # this block's global tile ids
        # tile-list pixel-coord shards (pad to ncores with tile 0's coords; padded cores get empty lists)
        PX = np.zeros((ncores * TS, TS), np.float32); PY = np.zeros((ncores * TS, TS), np.float32)
        for local, t in enumerate(bt):
            tx, ty = t % ntx, t // ntx
            PX[local * TS:(local + 1) * TS, :] = (np.arange(TS) + tx * TS)[None, :]
            PY[local * TS:(local + 1) * TS, :] = (np.arange(TS) + ty * TS)[:, None]
        PXt = tileshard(dev, grid, ncores, torch.from_numpy(PX))
        PYt = tileshard(dev, grid, ncores, torch.from_numpy(PY))
        accC = tileshard(dev, grid, ncores, torch.zeros(ncores * TS, TS))
        accT = tileshard(dev, grid, ncores, torch.ones(ncores * TS, TS))
        maxc = max((len(tile_lists[t]) for t in bt), default=0)
        nbatch = max(1, (maxc + B - 1) // B)

        def params_for(local, d):
            if local >= len(bt):
                return DUMMY * B
            lst = tile_lists[bt[local]][d * B:(d + 1) * B]
            out = []
            for kk in range(B):
                if kk < len(lst):
                    i = lst[kk]
                    out += [f2u(cx[i]), f2u(cy[i]), f2u(a[i]), f2u(2 * b2[i]), f2u(c[i]), f2u(op[i]), f2u(col[i])]
                else:
                    out += DUMMY
            return out

        for d in range(nbatch):
            rt_r, rt_c, rt_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for gx in range(GX):
                for gy in range(GY):
                    local = gy * GX + gx                                 # row-major core -> local tile
                    sx, sy = coords[(gx, gy)]
                    rt_r[gx][gy] = [sx, sy, PXt.buffer_address(), PYt.buffer_address(),
                                    accC.buffer_address(), accT.buffer_address(), NB]
                    rt_c[gx][gy] = params_for(local, d)
                    rt_w[gx][gy] = [sx, sy, accC.buffer_address(), accT.buffer_address(), NB]
            prog = ttnn.ProgramDescriptor(kernels=[
                mk(READER, rt_r, ttnn.ReaderConfigDescriptor()),
                mk(COMPUTE, rt_c, cfg, [B]),
                mk(WRITER, rt_w, ttnn.WriterConfigDescriptor())],
                semaphores=[], cbs=[cbf(i) for i in (0, 1, 2, 3, 16, 17)])
            ttnn.generic_op([PXt, accC], prog)
            if k == 0 and d == 0:
                print(f"    [block0 dispatch0 OK] nbatch={nbatch}", flush=True)

        back = ttnn.to_torch(accC).reshape(ncores, TS, TS).numpy()
        for local, t in enumerate(bt):
            tx, ty = t % ntx, t // ntx
            img[ty * TS:(ty + 1) * TS, tx * TS:(tx + 1) * TS] = back[local]
        for t in (PXt, PYt, accC, accT):
            t.deallocate()
    return img, nblocks


def main():
    W = H = NTX * TS
    Ng = 200
    cx, cy, sx, sy, op, col, depth, abc = scene(2, W, H, Ng)
    a = np.array([t[0] for t in abc]); b2 = np.array([t[1] for t in abc]); c = np.array([t[2] for t in abc])
    s_gid, _st, ranges, ntx, nty, _tot = bin_and_sort(cx.numpy(), cy.numpy(), (sx**2).numpy(),
                                                      (sy**2).numpy(), depth.numpy(), W, H, ts=TS)
    tile_lists = [s_gid[ranges[t, 0]:ranges[t, 1]].tolist() for t in range(ntx * nty)]

    dev = ttnn.open_device(device_id=0)
    try:
        img, nblocks = tileblock_raster(dev, ntx, nty, tile_lists, cx.numpy(), cy.numpy(), a, b2, c,
                                        op.numpy(), col.numpy(), W, H)
        gold = golden_culled(cx, cy, op, col, abc, tile_lists, W, H, ntx).numpy()
        err = np.abs(img - gold).max(); scl = np.abs(gold).max() + 1e-9
        print(f"tile-block fwd: {W}x{H} = {ntx}x{nty}={ntx*nty} tiles on {GX}x{GY}={GX*GY}-core grid "
              f"-> {nblocks} blocks")
        print(f"  rendered vs host golden_culled: max_abs={err:.3e} rel={err/scl:.3e}")
        print(f"  -> {'R1 PASS' if err/scl < 1e-2 else 'R1 FAIL'}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
