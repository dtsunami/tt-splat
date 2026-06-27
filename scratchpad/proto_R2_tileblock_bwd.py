#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PROTO R2 (silicon) — tile-BLOCK backward raster: the same R1 block-loop mechanism applied to the m17
fused backward. Loop blocks of <=ncores tiles, each block's tiles row-major TILE-LIST sharded onto the
usable grid; the m17 READER/COMPUTE/WRITER kernels are REUSED VERBATIM (1 tile/core, base stage = host
reduce); per-tile S/T recurrence across chunks unchanged; per-Gaussian grads host-accumulated across
chunks AND blocks. This is the >352px backward path (s4 in-kernel reduce stays the <=352px fast path).

De-risk on a 4x4=16-core grid with 8x8=64 tiles -> 4 blocks. GATE: grads == the verified
fused_backward_grid(stage="base") run on the full 8x8 grid (same scene/Tfin/gp).
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "pathclear"))
from sfpu_raster_scaled import scene, TS                                  # noqa: E402
from bin_sort import bin_and_sort                                        # noqa: E402
import fused_backward as FB                                              # noqa: E402
from fused_backward import READER, COMPUTE, WRITER, f2u, FUSED_K, _DUMMY_G, _NAMES, fused_backward_grid  # noqa: E402

GX = GY = 4
NTX = NTY = 8
_GEOM = ("cx", "cy", "a", "b", "c", "op")


def tileshard(dev, grid, ncores, stiles, data=None):
    """HEIGHT_SHARDED [stiles*TS, TS]/core on [ncores*stiles*TS, TS], TILE_LAYOUT — tile-list shard."""
    H = ncores * stiles * TS
    sh = ttnn.ShardSpec(grid, [stiles * TS, TS], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sh)
    shape = ttnn.Shape([1, 1, H, TS])
    if data is None:
        return ttnn.allocate_tensor_on_device(shape, ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, H, TS).float(), dtype=ttnn.float32,
                           layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


def host_forward_T(cx, cy, op, col, abc, tile_lists, W, H, ntx):
    """Final transmittance Tfin[H,W] (+ image), per-tile front-to-back — feeds the backward."""
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    PX, PY = jj.float(), ii.float()
    Tf = np.ones((H, W), np.float64)
    for t, lst in enumerate(tile_lists):
        tx, ty = t % ntx, t // ntx
        ys, xs = slice(ty * TS, ty * TS + TS), slice(tx * TS, tx * TS + TS)
        tr = torch.ones(TS, TS); px, py = PX[ys, xs], PY[ys, xs]
        for i in lst:
            a, b, cc = abc[i]; dx, dy = px - float(cx[i]), py - float(cy[i])
            al = (float(op[i]) * torch.exp(-0.5 * (a * dx * dx + 2 * b * dx * dy + cc * dy * dy))).clamp(max=0.99)
            tr = tr * (1 - al)
        Tf[ys, xs] = tr.numpy()
    return Tf


def tileblock_backward(dev, ntx, nty, tile_lists, cx, cy, a, b2, c, op, colv, Wp, Hp, gp, Tfin):
    """Backward over ALL tiles by looping blocks of <=ncores tiles (base m17 + host reduce)."""
    ncores = GX * GY
    ntiles = ntx * nty
    nblocks = (ntiles + ncores - 1) // ncores
    N = len(cx)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
    coords = {(gx, gy): (lambda hp: (hp.x, hp.y))(dev.worker_core_from_logical_core(ttnn.CoreCoord(gx, gy)))
              for gx in range(GX) for gy in range(GY)}
    NB = TS * TS * 4
    SHF = FUSED_K * TS
    geomg = {k: np.zeros(N) for k in _GEOM}
    colg = [np.zeros(N) for _ in range(3)]
    cbf = lambda i, d: ttnn.CBDescriptor(total_size=d * NB, core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=NB)])
    ks = lambda s, rt, cfg, cta=[]: ttnn.KernelDescriptor(kernel_source=s,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid, runtime_args=rt,
            compile_time_args=cta, config=cfg)

    def tchunk(t, d):
        lst = tile_lists[t][::-1][d * FUSED_K:(d + 1) * FUSED_K]
        return lst, lst + [None] * (FUSED_K - len(lst))

    for k in range(nblocks):
        bt = list(range(k * ncores, min((k + 1) * ncores, ntiles)))
        # tile-list pixel-coord shards for this block
        PX = np.zeros((ncores * TS, TS), np.float32); PY = np.zeros((ncores * TS, TS), np.float32)
        Tt0 = np.ones((ncores * TS, TS), np.float32)
        for local, t in enumerate(bt):
            tx, ty = t % ntx, t // ntx
            PX[local * TS:(local + 1) * TS, :] = (np.arange(TS) + tx * TS)[None, :]
            PY[local * TS:(local + 1) * TS, :] = (np.arange(TS) + ty * TS)[:, None]
            Tt0[local * TS:(local + 1) * TS, :] = Tfin[ty * TS:(ty + 1) * TS, tx * TS:(tx + 1) * TS]
        PXt = tileshard(dev, grid, ncores, 1, torch.from_numpy(PX))
        PYt = tileshard(dev, grid, ncores, 1, torch.from_numpy(PY))
        maxc = max((len(tile_lists[t]) for t in bt), default=0)
        nbatch = max(1, (maxc + FUSED_K - 1) // FUSED_K)
        outs = [tileshard(dev, grid, ncores, FUSED_K) for _ in range(7)]
        out_addrs = [o.buffer_address() for o in outs]

        for ch in range(3):
            dl = np.zeros((ncores * TS, TS), np.float32)
            for local, t in enumerate(bt):
                tx, ty = t % ntx, t // ntx
                dl[local * TS:(local + 1) * TS, :] = gp[ty * TS:(ty + 1) * TS, tx * TS:(tx + 1) * TS, ch]
            dLt = tileshard(dev, grid, ncores, 1, torch.from_numpy(dl))
            Tt = tileshard(dev, grid, ncores, 1, torch.from_numpy(Tt0.copy()))
            St = tileshard(dev, grid, ncores, 1, torch.zeros(ncores * TS, TS))
            for d in range(nbatch):
                Sout = tileshard(dev, grid, ncores, 1); Tout = tileshard(dev, grid, ncores, 1)
                rt_r, rt_c, rt_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for gx in range(GX):
                    for gy in range(GY):
                        local = gy * GX + gx
                        sx, sy = coords[(gx, gy)]
                        _, padded = tchunk(bt[local], d) if local < len(bt) else ([], [None] * FUSED_K)
                        cargs = []
                        for i in padded:
                            if i is None:
                                cargs += [f2u(_DUMMY_G["cx"]), f2u(_DUMMY_G["cy"]), f2u(_DUMMY_G["a"]),
                                          f2u(2 * _DUMMY_G["b"]), f2u(_DUMMY_G["c"]), f2u(_DUMMY_G["op"]),
                                          f2u(_DUMMY_G["col"]), f2u(_DUMMY_G["b"])]
                            else:
                                cargs += [f2u(cx[i]), f2u(cy[i]), f2u(a[i]), f2u(2 * b2[i]), f2u(c[i]),
                                          f2u(op[i]), f2u(colv[ch][i]), f2u(b2[i])]
                        rt_r[gx][gy] = [sx, sy, PXt.buffer_address(), PYt.buffer_address(), dLt.buffer_address(),
                                        Tt.buffer_address(), St.buffer_address(), NB, FUSED_K,
                                        Sout.buffer_address(), Tout.buffer_address()]
                        rt_c[gx][gy] = cargs
                        rt_w[gx][gy] = [sx, sy, FUSED_K, NB] + out_addrs
                prog = ttnn.ProgramDescriptor(kernels=[
                    ks(READER, rt_r, ttnn.ReaderConfigDescriptor()),
                    ks(COMPUTE, rt_c, ttnn.ComputeConfigDescriptor(), [FUSED_K]),
                    ks(WRITER, rt_w, ttnn.WriterConfigDescriptor())],
                    semaphores=[], cbs=[cbf(i, 2) for i in (0, 1, 2, 24, 25, 26, 27)] + [cbf(i, 3) for i in range(16, 23)])
                ttnn.generic_op([PXt, outs[0]], prog)
                # readback + host reduce (per Gaussian, this chunk) + accumulate
                hs = [ttnn.to_torch(o).reshape(ncores, FUSED_K, TS, TS).sum(dim=(2, 3)).numpy() for o in outs]
                for local, t in enumerate(bt):
                    lst, _ = tchunk(t, d)
                    if not lst:
                        continue
                    idx = np.asarray(lst); L = len(idx)
                    for gi_i, name in enumerate(_NAMES):
                        vals = hs[gi_i][local, :L]
                        (colg[ch] if name == "col" else geomg[name])[idx] += vals
                St, Tt = Sout, Tout
        for o in outs:
            o.deallocate()
        for t in (PXt, PYt):
            t.deallocate()
    return geomg, colg


def main():
    W = H = NTX * TS
    Ng = 150
    cx, cy, sx, sy, op, col, depth, abc = scene(3, W, H, Ng)
    a = np.array([t[0] for t in abc]); b2 = np.array([t[1] for t in abc]); c = np.array([t[2] for t in abc])
    s_gid, _st, ranges, ntx, nty, _tot = bin_and_sort(cx.numpy(), cy.numpy(), (sx**2).numpy(),
                                                      (sy**2).numpy(), depth.numpy(), W, H, ts=TS)
    tile_lists = [s_gid[ranges[t, 0]:ranges[t, 1]].tolist() for t in range(ntx * nty)]
    Tfin = host_forward_T(cx, cy, op, col, abc, tile_lists, W, H, ntx)
    rng = np.random.default_rng(0)
    gp = rng.standard_normal((H, W, 3)).astype(np.float64) * 0.01
    colv = [col.numpy(), col.numpy() * 0.7, col.numpy() * 1.3]

    dev = ttnn.open_device(device_id=0)
    try:
        cxn, cyn, opn = cx.numpy(), cy.numpy(), op.numpy()
        print(f"R2 backward tile-block: {ntx}x{nty}={ntx*nty} tiles on {GX}x{GY}={GX*GY}-core grid", flush=True)
        # reference: verified fused_backward_grid (base) on the FULL ntx x nty grid
        gref, cref = fused_backward_grid(dev, cxn, cyn, a, b2, c, opn, colv, tile_lists, ntx, nty, W, H,
                                         gp, Tfin, stage="base")
        print("  reference fused_backward_grid(base) done", flush=True)
        gblk, cblk = tileblock_backward(dev, ntx, nty, tile_lists, cxn, cyn, a, b2, c, opn, colv, W, H, gp, Tfin)
        print("  blocked backward done", flush=True)
        worst = 0.0
        for name in _GEOM:
            e = np.abs(gblk[name] - gref[name]).max(); s = np.abs(gref[name]).max() + 1e-9
            worst = max(worst, e / s)
        for ch in range(3):
            e = np.abs(cblk[ch] - cref[ch]).max(); s = np.abs(cref[ch]).max() + 1e-9
            worst = max(worst, e / s)
        print(f"  blocked vs fused_backward_grid(base): worst rel = {worst:.3e}")
        print(f"  -> {'R2 PASS' if worst < 1e-2 else 'R2 FAIL'}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
