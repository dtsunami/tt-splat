#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Pluggable compute BACKEND for the projection forward/backward (the "Trace-and-Pack" refactor, step #3).

device_project.py and device_project_backward.py speak a small fixed elementwise vocabulary
(mul/add/sub/square/sqrt/recip/exp/sigmoid/clamp/maximum/gt/lt + neg). Routing that vocabulary through a
pluggable Backend object lets the SAME Python either:
  - dispatch live ttnn  (TtnnBackend, the DEFAULT -> byte-identical to the original, today's verified path)
  - record a dataflow DAG  (TraceBackend, added in step 2 -> feeds the tile-compiler / SFPU codegen)
  - interpret in fp64  (NumpyBackend, added in step 2 -> a zero-hardware oracle for the gates)

STEP 1 (this file) implements only Backend + TtnnBackend, threaded through both modules with TtnnBackend as
the default, so every existing caller and gate is unchanged. The two compute ops `mul`/`add` accept a tensor
OR a python scalar as the second operand (mirroring ttnn's broadcast overloads); the tracer in step 2 will
intern the scalar case as a const node. Structural ops (from_torch / slice / reshape / concat / to_torch) stay
direct ttnn — they are data-marshalling at the input/output boundary, not part of the fused DAG.
"""
from __future__ import annotations
import ttnn


class Backend:
    """Abstract elementwise op vocabulary. Operands are opaque tensor handles; the second arg of mul/add/sub
    may be a python float scalar (broadcast). Subclasses implement the ~14 methods below."""
    def mul(self, a, b): raise NotImplementedError
    def add(self, a, b): raise NotImplementedError
    def sub(self, a, b): raise NotImplementedError
    def neg(self, a): raise NotImplementedError
    def square(self, a): raise NotImplementedError
    def sqrt(self, a): raise NotImplementedError
    def recip(self, a): raise NotImplementedError
    def exp(self, a): raise NotImplementedError
    def sigmoid(self, a): raise NotImplementedError
    def clamp(self, a, lo, hi): raise NotImplementedError
    def maximum(self, a, b): raise NotImplementedError
    def gt(self, a, b): raise NotImplementedError
    def lt(self, a, b): raise NotImplementedError


class TtnnBackend(Backend):
    """Razor-thin delegation to ttnn — each method is the exact call the original code made, so the refactor
    is byte-identical. This is the default everywhere."""
    def mul(self, a, b): return ttnn.mul(a, b)
    def add(self, a, b): return ttnn.add(a, b)
    def sub(self, a, b): return ttnn.sub(a, b)
    def neg(self, a): return ttnn.mul(a, -1.0)
    def square(self, a): return ttnn.square(a)
    def sqrt(self, a): return ttnn.sqrt(a)
    def recip(self, a): return ttnn.reciprocal(a)
    def exp(self, a): return ttnn.exp(a)
    def sigmoid(self, a): return ttnn.sigmoid(a)
    def clamp(self, a, lo, hi): return ttnn.clamp(a, lo, hi)
    def maximum(self, a, b): return ttnn.maximum(a, b)
    def gt(self, a, b): return ttnn.gt(a, b)
    def lt(self, a, b): return ttnn.lt(a, b)


# Module-default backend: callers that pass backend=None get this, preserving today's behavior exactly.
DEFAULT = TtnnBackend()


def _is_scalar(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


class NumpyBackend(Backend):
    """fp64 numpy oracle — runs the SAME projection code on host numpy arrays (zero hardware). The
    correctness reference the tracer/codegen are gated against. Comparisons return float {0,1} masks."""
    def mul(self, a, b): return a * b
    def add(self, a, b): return a + b
    def sub(self, a, b): return a - b
    def neg(self, a): return -a
    def square(self, a): return a * a
    def sqrt(self, a): import numpy as np; return np.sqrt(a)
    def recip(self, a): return 1.0 / a
    def exp(self, a): import numpy as np; return np.exp(a)
    def sigmoid(self, a): import numpy as np; return 1.0 / (1.0 + np.exp(-a))
    def clamp(self, a, lo, hi): import numpy as np; return np.clip(a, lo, hi)
    def maximum(self, a, b): import numpy as np; return np.maximum(a, b)
    def gt(self, a, b): return (a > b).astype("float64")
    def lt(self, a, b): return (a < b).astype("float64")


class _H:
    """Opaque trace handle = a node id in the recorded DAG."""
    __slots__ = ("g", "i")
    def __init__(self, g, i): self.g = g; self.i = i


class TraceBackend(Backend):
    """Records the SAME projection code as a straight-line dataflow DAG (no hardware). Foreign scalars fold
    into smul/sadd nodes; tensor inputs are explicit named leaves via .input(name). Outputs are tagged by
    the harness. The DAG feeds the tile-compiler (tilealloc.py). Node tuples (all ids are ints):
      ('input', name) ('mul'|'add'|'sub', a, b) ('smul'|'sadd', a, k) ('gt'|'lt', a, k)
    Only the ops the COLOR/opacity backward needs are wired; square folds to mul(a,a). recip/exp/sqrt/etc
    raise until the geometry/forward groups are tackled (steps 3-5)."""
    def __init__(self):
        self.nodes = []
        self._input = {}
    def _add(self, t): self.nodes.append(t); return _H(self, len(self.nodes) - 1)
    def input(self, name):
        if name in self._input: return _H(self, self._input[name])
        h = self._add(("input", name)); self._input[name] = h.i; return h
    def _id(self, x):
        assert isinstance(x, _H), f"non-handle tensor operand in trace: {type(x)} (a foreign ttnn op leaked?)"
        return x.i
    def mul(self, a, b):
        if _is_scalar(b): return self._add(("smul", self._id(a), float(b)))
        if _is_scalar(a): return self._add(("smul", self._id(b), float(a)))
        return self._add(("mul", self._id(a), self._id(b)))
    def add(self, a, b):
        if _is_scalar(b): return self._add(("sadd", self._id(a), float(b)))
        if _is_scalar(a): return self._add(("sadd", self._id(b), float(a)))
        return self._add(("add", self._id(a), self._id(b)))
    def sub(self, a, b):
        if _is_scalar(b): return self._add(("sadd", self._id(a), -float(b)))
        return self._add(("sub", self._id(a), self._id(b)))
    def neg(self, a): return self._add(("smul", self._id(a), -1.0))
    def square(self, a): i = self._id(a); return self._add(("mul", i, i))
    def gt(self, a, b): return self._add(("gt", self._id(a), float(b)))
    def lt(self, a, b): return self._add(("lt", self._id(a), float(b)))
    def sqrt(self, a): raise NotImplementedError("sqrt not in color group")
    def recip(self, a): raise NotImplementedError("recip not in color group")
    def exp(self, a): raise NotImplementedError("exp not in color group")
    def sigmoid(self, a): raise NotImplementedError("sigmoid not in color group")
    def clamp(self, a, lo, hi): raise NotImplementedError("clamp not in color group")
    def maximum(self, a, b): raise NotImplementedError("maximum not in color group")


def eval_dag(nodes, inputs):
    """fp64 numpy interpreter of a TraceBackend DAG. inputs: {name -> np.ndarray}. Returns val list (per node).
    Validates trace fidelity (eval_dag == NumpyBackend == ttnn) before any codegen."""
    import numpy as np
    val = [None] * len(nodes)
    for i, t in enumerate(nodes):
        op = t[0]
        if op == "input": val[i] = inputs[t[1]]
        elif op == "mul": val[i] = val[t[1]] * val[t[2]]
        elif op == "add": val[i] = val[t[1]] + val[t[2]]
        elif op == "sub": val[i] = val[t[1]] - val[t[2]]
        elif op == "smul": val[i] = val[t[1]] * t[2]
        elif op == "sadd": val[i] = val[t[1]] + t[2]
        elif op == "gt": val[i] = (val[t[1]] > t[2]).astype("float64")
        elif op == "lt": val[i] = (val[t[1]] < t[2]).astype("float64")
        else: raise ValueError(f"eval_dag: unknown op {op}")
    return val
