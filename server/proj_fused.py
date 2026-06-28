#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Fused projection orchestrator (step #3, Trace-and-Pack). Traces a projection output-GROUP once via
TraceBackend, lowers the DAG to a dst-resident SFPU kernel (tilealloc.py), and launches it as ONE
generic_op — replacing the per-op ttnn dispatch swarm. Step 2 wires the COLOR/opacity backward group
(gop / gsh / gmean_color — the 18.4ms pure-mul/add half of Stage D).

Input naming (the trace/runtime contract): op_sig, up.{op,colR,colG,colB}, AC.pre.{c}, AC.wb.{k},
AC.{x,y,z,inv}, sh.{k}.{c}.  Outputs: gop, gsh.{k}.{c}, gmean_color.{d}.
"""
from __future__ import annotations
import numpy as np
import ttnn
from backend import TraceBackend, NumpyBackend, eval_dag
from device_project_backward import _color_op_backward
from tilealloc import lower, emit_cpp, partition

_HOME = (1, 1); _TS = 32; _NB = _TS * _TS * 4


def trace_color(deg):
    """Run _color_op_backward through TraceBackend with named-leaf inputs. Returns (nodes, outputs, inputs):
    nodes = the DAG; outputs = {semantic_name -> node_id}; inputs = ordered list of input names."""
    K = (deg + 1) ** 2
    tb = TraceBackend()
    up = {"op": tb.input("up.op"), "colR": tb.input("up.colR"),
          "colG": tb.input("up.colG"), "colB": tb.input("up.colB")}
    AC = {"pre": [tb.input(f"AC.pre.{c}") for c in range(3)],
          "wb": [None] + [tb.input(f"AC.wb.{k}") for k in range(1, K)],
          "x": tb.input("AC.x"), "y": tb.input("AC.y"), "z": tb.input("AC.z"), "inv": tb.input("AC.inv")}
    op_sig = tb.input("op_sig")
    shcol = lambda k, c: tb.input(f"sh.{k}.{c}")
    gop, gsh_t, gmean_color = _color_op_backward(tb, shcol, up, AC, op_sig, deg)
    outs = {"gop": gop.i}
    for c in range(3):
        for k in range(K):
            outs[f"gsh.{k}.{c}"] = gsh_t[(k, c)].i
    for d in range(3):
        outs[f"gmean_color.{d}"] = gmean_color[d].i
    return tb.nodes, outs, [n[1] for n in tb.nodes if n[0] == "input"]


def color_inputs(deg, src):
    """Build the {input_name -> value} dict from a source accessor `src(kind, *idx)`:
      src('op_sig'), src('up', key), src('pre', c), src('wb', k), src('dir', axis), src('inv'), src('sh', k, c)
    Returns a dict keyed by the canonical input names used in trace_color."""
    K = (deg + 1) ** 2
    d = {"op_sig": src("op_sig"), "AC.inv": src("inv"),
         "AC.x": src("dir", "x"), "AC.y": src("dir", "y"), "AC.z": src("dir", "z")}
    for key in ("op", "colR", "colG", "colB"):
        d[f"up.{key}"] = src("up", key)
    for c in range(3):
        d[f"AC.pre.{c}"] = src("pre", c)
    for k in range(1, K):
        d[f"AC.wb.{k}"] = src("wb", k)
    for c in range(3):
        for k in range(K):
            d[f"sh.{k}.{c}"] = src("sh", k, c)
    return d


# ============================================================================================
# Kernel build (trace -> lower -> emit) cached per deg, and the single-tile generic_op launcher.
# ============================================================================================
_READER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t addr=get_arg_val<uint32_t>(2), total=get_arg_val<uint32_t>(3), npg=get_arg_val<uint32_t>(4);
    cb_reserve_back(0, npg);
    noc_async_read(get_noc_addr(sx,sy, addr), get_write_ptr(0), total);
    noc_async_read_barrier();
    cb_push_back(0, npg);
}
"""
_WRITER = r"""
#include "dataflow_api.h"
void kernel_main() {
    uint32_t sx=get_arg_val<uint32_t>(0), sy=get_arg_val<uint32_t>(1);
    uint32_t addr=get_arg_val<uint32_t>(2), total=get_arg_val<uint32_t>(3), npg=get_arg_val<uint32_t>(4);
    cb_wait_front(16, npg);
    noc_async_write(get_read_ptr(16), get_noc_addr(sx,sy, addr), total);
    noc_async_write_barrier();
    cb_pop_front(16, npg);
}
"""
_CACHE = {}
_BUDGET = 1300                       # max instrs per JIT kernel (keeps each compile fast)

def build_color(deg):
    """trace -> lower (Sethi-Ullman, fp32-8) -> group-split into JIT-sized kernels. Cached per deg.
    Each kernel owns a contiguous range of the 52 output slots; recompute keeps every kernel <=6 regs."""
    if deg in _CACHE: return _CACHE[deg]
    nodes, outs, input_order = trace_color(deg)
    prog, in_slots, out_slots, maxreg = lower(nodes, outs, input_order)
    assert maxreg <= 8, f"color group needs {maxreg} fp32 dst regs (>8)"
    n_in, n_out = len(in_slots), len(out_slots)
    kernels = [{"src": emit_cpp(kp, n_in, ng), "off": off, "n": ng}
               for (kp, off, ng) in partition(prog, _BUDGET)]
    b = dict(nodes=nodes, outs=outs, input_order=input_order, prog=prog,
             in_slots=in_slots, out_slots=out_slots, maxreg=maxreg,
             n_in=n_in, n_out=n_out, kernels=kernels)
    _CACHE[deg] = b
    return b


def _l1(dev, data, nt):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*_HOME), ttnn.CoreCoord(*_HOME))])
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(crs, [_TS * nt, _TS], ttnn.ShardOrientation.ROW_MAJOR))
    if data is None:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, _TS * nt, _TS]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    return ttnn.from_torch(data.reshape(1, 1, _TS * nt, _TS), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)


def run_color_fused(dev, inputs, deg):
    """Single-tile (N<=1024) fused color/op backward. inputs: {name -> np[N]}. Returns {out_name -> np[N]}.
    One shared L1 input block + output block; one generic_op per group-split kernel (fp32-8 dst-resident)."""
    import torch
    b = build_color(deg)
    n_in, n_out, in_slots, out_slots = b["n_in"], b["n_out"], b["in_slots"], b["out_slots"]
    N = len(next(iter(inputs.values())))
    assert N <= 1024, f"single-tile path needs N<=1024 (got {N}); multi-tile loop is a follow-on"
    blk = np.zeros((n_in, _TS, _TS), np.float32)
    for name, slot in in_slots.items():
        pad = np.zeros(1024, np.float32); pad[:N] = inputs[name]; blk[slot] = pad.reshape(_TS, _TS)
    in_t = _l1(dev, torch.from_numpy(blk), n_in)
    out_t = _l1(dev, None, n_out)
    hp = dev.worker_core_from_logical_core(ttnn.CoreCoord(*_HOME)); sx, sy = hp.x, hp.y
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*_HOME), ttnn.CoreCoord(*_HOME))])
    def rt(arr):
        r = ttnn.RuntimeArgs(); r[_HOME[0]][_HOME[1]] = arr; return r
    def cbf(i, d): return ttnn.CBDescriptor(total_size=d * _NB, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=_NB)])
    ks = lambda s, arr, cfg: ttnn.KernelDescriptor(kernel_source=s,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=crs,
            runtime_args=rt(arr), compile_time_args=[], config=cfg)
    out_addr = out_t.buffer_address()
    for kern in b["kernels"]:                                 # one dispatch per group-split kernel
        ng, off = kern["n"], kern["off"]
        cfg = ttnn.ComputeConfigDescriptor(); cfg.fp32_dest_acc_en = True
        prog = ttnn.ProgramDescriptor(kernels=[
            ks(_READER, [sx, sy, in_t.buffer_address(), n_in * _NB, n_in], ttnn.ReaderConfigDescriptor()),
            ks(kern["src"], [], cfg),
            ks(_WRITER, [sx, sy, out_addr + off * _NB, ng * _NB, ng], ttnn.WriterConfigDescriptor())],
            semaphores=[], cbs=[cbf(0, n_in), cbf(16, ng)])
        ttnn.generic_op([in_t, out_t], prog)
    raw = ttnn.to_torch(out_t).reshape(n_out, _TS * _TS).numpy()
    return {name: raw[slot, :N].copy() for name, slot in out_slots.items()}


def color_backward_fused(dev, P, up, AC, deg):
    """Live drop-in for _color_op_backward via the fused kernel. Reads the device color-aux (AC + the cheap
    host-side op_sig/sh/up) into one numpy input set, runs the codegen'd kernel, and returns (gop, gsh_t,
    gmean_color) as ttnn [N] tensors so the rest of project_backward is unchanged. Host I/O (AC readback +
    output re-upload) is the current cost; on-device assembly is the 41x follow-on. Gated by TT_PROJ_FUSED."""
    import numpy as np, torch
    from device_project_backward import _t2t
    N = P["mean"].shape[0]; K = (deg + 1) ** 2
    def _tonp(t):                                                       # ttnn / torch / numpy -> numpy
        if isinstance(t, ttnn.Tensor): return ttnn.to_torch(t).numpy()
        if torch.is_tensor(t): return t.detach().cpu().numpy()
        return np.asarray(t)
    g = lambda t: ttnn.to_torch(t).flatten().numpy()[:N].astype(np.float64)
    shn = _tonp(P["sh"]).reshape(N, -1, 3)[:, :K, :].astype(np.float64)   # first K=(deg+1)^2 bands (SH-warmup safe)
    opl = _tonp(P["op"]).flatten()[:N]
    op_sig = 1.0 / (1.0 + np.exp(-opl))                                  # sigmoid host-side (no device read)
    up_np = {k: (g(up[k]) if isinstance(up[k], ttnn.Tensor) else np.asarray(up[k], np.float64)) for k in up}
    pre = [g(AC["pre"][c]) for c in range(3)]
    wb = [None] + [g(AC["wb"][k]) for k in range(1, K)]
    x, y, z, inv = g(AC["x"]), g(AC["y"]), g(AC["z"]), g(AC["inv"])
    def srcf(kind, *idx):
        if kind == "op_sig": return op_sig
        if kind == "inv": return inv
        if kind == "dir": return {"x": x, "y": y, "z": z}[idx[0]]
        if kind == "up": return up_np[idx[0]]
        if kind == "pre": return pre[idx[0]]
        if kind == "wb": return wb[idx[0]]
        return shn[:, idx[0], idx[1]]
    out = run_color_fused(dev, color_inputs(deg, srcf), deg)
    gop = _t2t(dev, out["gop"])
    gsh_t = {(k, c): _t2t(dev, out[f"gsh.{k}.{c}"]) for c in range(3) for k in range(K)}
    gmean_color = [_t2t(dev, out[f"gmean_color.{d}"]) for d in range(3)]
    return gop, gsh_t, gmean_color
