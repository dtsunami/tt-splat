#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Phase 2b: reusable single-tile (32x32) FUSED BACKWARD — one generic_op kernel computes all 7 grads
for K Gaussians (reverse-blend, recip-reconstruct-T, suffix-S via CB recurrence). Verified vs torch
(K=2/K=4, err/scale 8e-3). Drop-in replacement for the O(N) ttnn-op backward in device_raster, at H=W=32.

  grads = fused_backward(dev, params_rev, dLdC, Tfinal)
    params_rev : list of K dicts {cx,cy,a,b,c,op,col} in REVERSE depth order (back-to-front)
    dLdC       : torch [32,32] upstream dL/dC for this channel
    Tfinal     : torch [32,32] final transmittance from the forward
    -> dict {cx,cy,a,b,c,op,col: np.ndarray[K]}  (same reverse order as params_rev)

Full kernel + verification: docs/pathclear/m17_fused_backward.py.
"""
from __future__ import annotations
import struct
import numpy as np
import torch
import ttnn

HOME = (1, 1)
TS = 32
FUSED_K = 16                          # fixed compile-time K (pad chunks with no-op dummies) -> 1 JIT compile
_DUMMY_G = {"cx": 0.0, "cy": 0.0, "a": 1.0, "b": 0.0, "c": 1.0, "op": 0.0, "col": 0.0}   # op=0 -> no contribution


def f2u(x): return struct.unpack("<I", struct.pack("<f", float(x)))[0]

# kernels (verbatim from the verified m17 K-loop kernel)
from pathlib import Path as _P
_m17 = (_P(__file__).resolve().parent.parent / "docs" / "pathclear" / "m17_fused_backward.py").read_text()
_g = {}
exec(compile(_m17, "m17", "exec"), _g)          # pulls in READER, COMPUTE, WRITER strings + f2u
READER, COMPUTE, WRITER = _g["READER"], _g["COMPUTE"], _g["WRITER"]

_NAMES = ["col", "op", "a", "c", "b", "cx", "cy"]   # CB pack order 16..22


def _l1(dev, data=None, nt=1):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(crs, [TS * nt, TS], ttnn.ShardOrientation.ROW_MAJOR))
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, TS * nt, TS]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, TS, TS).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)


def fused_backward(dev, params_rev, dLdC, Tfinal, ox=0, oy=0, S_init=None, return_state=False):
    """ox,oy = this tile's GLOBAL top-left pixel (so dx = (global x) - cx). dLdC/Tfinal/S_init are the
    tile's 32x32 sub-regions; Gaussian cx,cy are global. For chunking a dense tile, pass S_init (prev
    chunk's S) + Tfinal (prev chunk's T) and return_state=True to get the chunk's final S,T to thread on.
    Multi-tile = loop tiles + host-sum the per-Gaussian grads (cross-tile = host scalar sum).
    params_rev must be <= FUSED_K; padded with no-op dummies so K is a fixed compile-time constant."""
    nreal = len(params_rev)
    assert nreal <= FUSED_K, f"chunk {nreal} > FUSED_K {FUSED_K}"
    params_rev = list(params_rev) + [_DUMMY_G] * (FUSED_K - nreal)   # pad after real Gaussians (S/T unaffected)
    K = FUSED_K
    ii, jj = torch.meshgrid(torch.arange(TS), torch.arange(TS), indexing="ij")
    px, py = _l1(dev, (jj + ox).float()), _l1(dev, (ii + oy).float())
    dl = _l1(dev, dLdC.double())
    tf = _l1(dev, Tfinal.double())
    sz = _l1(dev, S_init.double() if S_init is not None else torch.zeros(TS, TS))
    s_out, t_out = _l1(dev), _l1(dev)                       # exported final S/T (chunk threading)
    outs = [_l1(dev, nt=K) for _ in range(7)]
    hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(*HOME)); sx, sy = hp.x, hp.y
    NB = TS * TS * 4
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME), ttnn.CoreCoord(*HOME))])

    def rt(arr):
        r = ttnn.RuntimeArgs(); r[HOME[0]][HOME[1]] = arr; return r
    cbf = lambda i, d: ttnn.CBDescriptor(total_size=d * NB, core_ranges=crs,
             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=NB)])
    ks = lambda s, arr, cfg, cta=[]: ttnn.KernelDescriptor(
        kernel_source=s, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=crs, runtime_args=rt(arr), compile_time_args=cta, config=cfg)
    cargs = []
    for q in params_rev:
        cargs += [f2u(q["cx"]), f2u(q["cy"]), f2u(q["a"]), f2u(2 * q["b"]), f2u(q["c"]),
                  f2u(q["op"]), f2u(q["col"]), f2u(q["b"])]
    oaddr = [o.buffer_address() for o in outs]
    cbs = [cbf(i, 2) for i in (0, 1, 2, 24, 25, 26, 27)] + [cbf(i, 3) for i in (16, 17, 18, 19, 20, 21, 22)]
    prog = ttnn.ProgramDescriptor(kernels=[
        ks(READER, [sx, sy, px.buffer_address(), py.buffer_address(), dl.buffer_address(),
                    tf.buffer_address(), sz.buffer_address(), NB, K,
                    s_out.buffer_address(), t_out.buffer_address()], ttnn.ReaderConfigDescriptor()),
        ks(COMPUTE, cargs, ttnn.ComputeConfigDescriptor(), [K]),
        ks(WRITER, [sx, sy, K, NB] + oaddr, ttnn.WriterConfigDescriptor())],
        semaphores=[], cbs=cbs)
    ttnn.generic_op([px, outs[0]], prog)
    grads = {}
    for i, n in enumerate(_NAMES):
        t = ttnn.to_torch(outs[i]).reshape(K, TS, TS)
        grads[n] = t.reshape(K, -1).sum(dim=1).numpy()          # host reduce per-Gaussian (exact)
    if return_state:
        S = ttnn.to_torch(s_out).reshape(TS, TS).clone()
        T = ttnn.to_torch(t_out).reshape(TS, TS).clone()
        return grads, S, T
    return grads
