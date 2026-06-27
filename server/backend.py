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
