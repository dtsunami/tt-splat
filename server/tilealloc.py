#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Tile-compiler for the projection fusion (step #3). Lowers a TraceBackend DAG to a dst-RESIDENT SFPU
instruction stream and emits the m17-dialect kernel C++. Each output is computed by a tree-walk that
RECOMPUTES its producer cone into dst registers (no L1 spill — proven broken), packing only the final
result. Invariant: emit(node, r) computes `node` into reg r using only regs >= r, so a binary's left
operand in reg r survives while its right operand is built in reg r+1. Max live regs = the tree's
register number; asserted <= the dst budget (8 fp32 / 16 bf16).

A numpy register-file SIMULATOR replays the SAME instruction stream so the lowering is gated off-device
(simulate == eval_dag) before a single dispatch. Instruction tuples:
  ('copy', dst, in_slot)  ('mul'|'add'|'sub', dst, a, b)  ('smul'|'sadd', dst, k)
  ('gt'|'lt', dst, k)  ('pack', src, out_slot)
"""
from __future__ import annotations
import struct
import sys
def f2u(x): return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def lower(nodes, outs, input_order):
    """DAG -> (prog, in_slots{name->slot}, out_slots{name->slot}, maxreg). input_order/outs key order set
    the CB page layout the orchestrator must honor. Uses Sethi-Ullman ordering (evaluate the subtree that
    needs MORE registers first) to minimize peak dst-register pressure so the cone fits the fp32-8 budget."""
    in_slots = {name: i for i, name in enumerate(input_order)}
    out_names = list(outs.keys())
    out_slots = {name: i for i, name in enumerate(out_names)}
    sys.setrecursionlimit(1000000)

    su_memo = {}
    def su(nid):                                      # Sethi-Ullman number: regs needed for this cone
        if nid in su_memo: return su_memo[nid]
        t = nodes[nid]; op = t[0]
        if op == "input": v = 1
        elif op in ("smul", "sadd", "gt", "lt"): v = su(t[1])
        else:                                         # binary
            a, b = su(t[1]), su(t[2])
            v = a + 1 if a == b else max(a, b)
        su_memo[nid] = v; return v

    prog = []
    state = {"maxreg": 0}
    def emit(nid, r):
        if r > state["maxreg"]: state["maxreg"] = r
        t = nodes[nid]; op = t[0]
        if op == "input":
            prog.append(("copy", r, in_slots[t[1]]))
        elif op in ("smul", "sadd", "gt", "lt"):
            emit(t[1], r); prog.append((op, r, t[2]))
        else:                                         # binary: heavier child first into reg r
            a, b = t[1], t[2]
            left_first = su(a) >= su(b)
            (hi, lo) = (a, b) if left_first else (b, a)
            emit(hi, r); emit(lo, r + 1)
            if r + 1 > state["maxreg"]: state["maxreg"] = r + 1
            ra, rb = (r, r + 1) if left_first else (r + 1, r)   # physical regs holding (a, b)
            if op == "mul": prog.append(("mul", r, ra, rb))     # commutative
            elif op == "add": prog.append(("add", r, ra, rb))
            else: prog.append(("sub", r, ra, rb))               # a - b, result -> r

    for name in out_names:
        emit(outs[name], 0)
        prog.append(("pack", 0, out_slots[name]))
    return prog, in_slots, out_slots, state["maxreg"] + 1


def partition(prog, budget):
    """Split the per-output prog into KERNELS each <= `budget` instructions (group-split — keeps every
    emitted kernel small enough to JIT-compile fast; a single huge function is the only thing that doesn't).
    prog is in output order, so each kernel owns a CONTIGUOUS range of output slots. Returns a list of
    (kernel_prog_with_LOCAL_pack_slots, offset, n_out_local)."""
    segs = []; cur = []
    for ins in prog:
        cur.append(ins)
        if ins[0] == "pack":
            segs.append(cur); cur = []
    kernels = []; kprog = []; n = 0; off = 0; cnt = 0
    for seg in segs:
        if kprog and cnt + len(seg) > budget:
            kernels.append((kprog, off, n)); off += n; kprog = []; n = 0; cnt = 0
        kprog += seg[:-1] + [("pack", seg[-1][1], n)]   # remap pack to LOCAL slot
        n += 1; cnt += len(seg)
    if kprog: kernels.append((kprog, off, n))
    return kernels


def simulate(prog, inputs, in_slots, n_out):
    """Replay the emitted instruction stream in numpy fp64 (a register file). Gates the lowering against
    eval_dag with ZERO hardware. inputs: {name -> np.ndarray}."""
    import numpy as np
    slot_arr = {in_slots[name]: arr for name, arr in inputs.items()}
    regs = {}
    out = [None] * n_out
    for ins in prog:
        op = ins[0]
        if op == "copy": regs[ins[1]] = slot_arr[ins[2]].astype("float64").copy()
        elif op == "mul": regs[ins[1]] = regs[ins[2]] * regs[ins[3]]
        elif op == "add": regs[ins[1]] = regs[ins[2]] + regs[ins[3]]
        elif op == "sub": regs[ins[1]] = regs[ins[2]] - regs[ins[3]]
        elif op == "smul": regs[ins[1]] = regs[ins[1]] * ins[2]
        elif op == "sadd": regs[ins[1]] = regs[ins[1]] + ins[2]
        elif op == "gt": regs[ins[1]] = (regs[ins[1]] > ins[2]).astype("float64")
        elif op == "lt": regs[ins[1]] = (regs[ins[1]] < ins[2]).astype("float64")
        elif op == "pack": out[ins[2]] = regs[ins[1]].copy()
        else: raise ValueError(f"simulate: unknown {op}")
    return out


# ---- m17-dialect SFPU C++ emitter (Layer 3) --------------------------------------------------------
_INCLUDES = r"""
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"
#include "api/dataflow/circular_buffer.h"
"""

# one body line per instruction (POOL=input CB 0 read-only, OUT=CB 16). copy_tile reads input slot;
# binary ops act on dst regs; unary scalar ops are in-place; pack writes the output slot.
def _line(ins):
    op = ins[0]
    if op == "copy": return f"copy_tile(0,{ins[2]},{ins[1]});"
    if op == "mul":  return f"mul_binary_tile({ins[2]},{ins[3]},{ins[1]});"
    if op == "add":  return f"add_binary_tile({ins[2]},{ins[3]},{ins[1]});"
    if op == "sub":  return f"sub_binary_tile({ins[2]},{ins[3]},{ins[1]});"
    if op == "smul": return f"mul_unary_tile({ins[1]},{f2u(ins[2])}u);"
    if op == "sadd": return f"add_unary_tile({ins[1]},{f2u(ins[2])}u);"
    if op == "gt":   return f"unary_gt_tile({ins[1]},{f2u(ins[2])}u);"
    if op == "lt":   return f"unary_lt_tile({ins[1]},{f2u(ins[2])}u);"
    if op == "pack": return f"pack_tile({ins[1]},16,{ins[2]});"
    raise ValueError(op)


def emit_cpp(prog, n_in, n_out):
    """Emit the dst-resident compute kernel. Each OUTPUT gets its own tile_regs window (recompute its cone
    into dst regs, then commit/wait/pack/release) — packs must follow commit/wait and we can't hold all
    n_out results in <=8 regs. One tile (<=1024 Gaussians); the orchestrator loops tiles for larger N."""
    blocks = []
    seg = []
    for ins in prog:
        if ins[0] == "pack":
            lines = "\n        ".join(_line(i) for i in seg)
            blocks.append(
                "    tile_regs_acquire();\n        " + lines
                + "\n        tile_regs_commit(); tile_regs_wait();\n        "
                + f"pack_tile({ins[1]},16,{ins[2]});\n        tile_regs_release();")
            seg = []
        else:
            seg.append(ins)
    body = "\n".join(blocks)
    return _INCLUDES + f"""
void kernel_main() {{
    cb_wait_front(0, {n_in});
    cb_reserve_back(16, {n_out});
    init_sfpu(0, 16);
    mul_binary_tile_init(); add_binary_tile_init();
    binop_with_scalar_tile_init(); unary_gt_tile_init(); unary_lt_tile_init(); copy_dest_values_init();
{body}
    cb_push_back(16, {n_out});
}}
"""
