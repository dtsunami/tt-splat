# Projection fusion (step #3) — architecture & plan

Decided by a 14-agent design workflow (6 architectures × 3 adversarial judges → synthesis), grounded in
the silicon constraints proved this session (`scratchpad/proto_tilevm.py`). Winner: **Trace-and-Pack**,
sequenced **easy-half-first**.

## The problem (measured, N=800, the real training scale)
B+D = 57.5 ms and is **~100% ttnn dispatch overhead** — ~570 tiny elementwise ops × ~0.11 ms (readback is
0.2 ms, noise). D = 37.8 ms splits: geometry contractions 19.9 ms (hard) + SH/color 18.4 ms (easy muls).
Per-Gaussian matrices vary (R from quat, J from mean) → `ttnn.matmul` doesn't apply. Only lever = kill
dispatch count via a fused custom kernel. Proven kernel constraints: dst-resident works; L1 spill (pack→slot
→copy back) FAILS (FIFO hazard); dst budget = 16 bf16 / **8 fp32** (`fp32_dest_acc_en`); signed-cancellation
grads need fp32.

## The architecture: Trace-and-Pack
`device_project.py` and `device_project_backward.py` already speak a **~17-op vocabulary**
(mul/add/sub/square/sqrt/recip/exp/sigmoid/clamp/max/gt/lt + scalar variants) through the `m()/ad()/s1()/
neg()` aliases and a few direct `ttnn.*` calls. Route that vocabulary through a pluggable **`Backend`**
object so the **exact same Python** either:
- dispatches live ttnn (`TtnnBackend`, the default → today's verified path, byte-identical), or
- records a straight-line **dataflow DAG** (`TraceBackend`), or
- interprets it in fp64 (`NumpyBackend`, a zero-hardware oracle).

A host-side **recompute register allocator** (`tilealloc.py`) schedules the DAG into ≤8 fp32 (or ≤16 bf16)
dst regs, partitioned into **per-output-group kernels**, and emits the proven m17 SFPU dialect. One
`generic_op` per group replaces ~570 dispatches. **Overflow policy = recompute-only** (re-derive a value's
producer cone, free at CB roots/consts); the only spill is the single proven m17 FIFO output-CB hand-off
(used for the shared `G = Rvᵀ gSC Rv`); fallback = group-split into an extra sequential dispatch. **Never**
the broken random-access spill.

**One source serves B and D.** The forward→backward **aux contract is generated, not hand-synced**: B's
trace marks its ~34 aux intermediates as outputs; D's trace references them as CB-root inputs. A math edit
in B automatically reshapes D's input list at trace time → the two files can't silently desync. This is the
durability win all three judges converged on.

## The refactor (file-level)
- **NEW `server/backend.py`** (~200 LOC): `Backend` ABC (17 ops + scalar variants); `TtnnBackend` (default,
  keeps every existing caller byte-identical); `TraceBackend` (node-ids, op tuples, const-interning by f2u,
  `input_tile`/`output`/aux-marking); `NumpyBackend` (fp64 oracle).
- **NEW `server/tilealloc.py`** (~250 LOC): topo-sort + DCE against tagged outputs; SSA-aware liveness
  (in-place `mul_unary` = new SSA value); greedy ≤8/≤16-reg allocator with recompute-on-overflow + group-split;
  m17-dialect SFPU C++ printer; **fp32 register-file SIMULATOR** that replays the emitted reg-stream (catches
  allocator bugs with zero hardware).
- **MODIFY `server/device_project_backward.py`** (~+60 LOC): thread `backend=None`; `m/ad/s1/neg` + direct
  `ttnn.*` become `B.*`; declare upstream grads + aux as `B.input_tile`, group outputs via `B.output`. Keep
  the full ttnn path as the golden/fallback `else`.
- **MODIFY `server/device_project.py`** (~+60 LOC): same threading; mark Rij/s2/SC/J/conic/qn/wb/pre as aux.
- **NEW `server/proj_fused.py`**: thin orchestrator — trace a group → allocate+emit → build CBs/runtime-args
  → launch via `fused_backward.py`'s `generic_op` pattern (reused, incl. grid-shard `_block` for N>1024);
  per-group `fp32_dest_acc_en` flag + per-call differential self-check + telemetry timer.
- **REUSE** unchanged: `fused_backward.py` (generic_op harness), `test_proj.py` (<5e-3), `test_proj_bwd.py`
  (<1e-2).

## Sequencing — every step ends at a silicon-verified gate
- **Step 0 — SPIKE — DONE ✅ GREEN (`scratchpad/proto_fp32_spike.py`, silicon).** Proved the load-bearing
  unknowns: (a) **fp32_dest_acc_en + 8-reg dst-resident MAC chains pass the 1e-2 gate** — conic chain
  (a*c−b*b→1/det→ca/cb/cc) rel **1.55e-3**, 8-term signed accumulation **1.14e-3**; the **bf16 contrast FAILS
  1e-2** (1.67e-2 / 1.22e-2) → `fp32_dest_acc_en` is necessary AND sufficient. (b) masks `gtz_tile` +
  `unary_lt_tile` → cmask exact (0.0). (c) `sqrt_tile` 2e-4, `clamp_tile` 4e-4 (lower-clamp covers
  z=max(·,1e-4)); `sigmoid_tile` returns garbage on this build → **fallback `1/(1+exp(−x))` = 1.7e-4** (wired).
  **WATCH-ITEM (new):** fp32-mode floor is ~1.5e-3 not 1e-6 — multiply inputs pass through 19-bit (tf32) SrcA/B
  even with fp32 dst-accum. Moderate chains pass 1e-2 with margin; **long geometry contraction chains
  (gscale/gquat) accumulate ~sqrt(nops)·1.5e-3 of error**, so the "compute G once + keep chains short via
  group-split" rule is a PRECISION safeguard, not just perf. Add a per-group readback micro-gate to catch drift.
- **Step 1 — backend refactor, byte-identical (~2 days).** Land `backend.py` (TtnnBackend default), thread `B`.
  Gate: test_proj + test_proj_bwd pass **byte-identically** (pure refactor).
- **Step 2 — SHIP THE EASY HALF (~3 days, FIRST REAL WIN).** TraceBackend + NumpyBackend + tilealloc on the
  **SH/color backward group** (gsh/gdir/gmean_color/gop — 18.4 ms, pure muls, bf16, fits 16 regs, ~0 recompute).
  Host gates first (NumpyBackend==ttnn, reg-sim==NumpyBackend — zero hardware), then silicon. Gate: color-half
  rel<1e-2; telemetry shows ~18.4 ms → <1 ms. Proves the whole substrate on a real sub-DAG.
- **Step 3 — geometry conic/mean chain (~3 days).** fp32-8, moderate recompute. Gate: those columns rel<1e-2.
- **Step 4 — hard contractions gscale/gquat (~1 week).** Compute shared `G = Rvᵀ gSC Rv` once, share via the
  proven FIFO output-CB (9–18 tiles co-live → G-share is mandatory, not optional). Gate: scale/quat rel<1e-2;
  D fully fused.
- **Step 5 — forward B (~3 days).** Same pipeline + transcendental opcodes (de-risked in Step 0). Gate:
  test_proj rel<5e-3. **Target: B+D 57.5 ms → ~3–5 ms.**

## Orchestrating the work
**Parallelize across output GROUPS off-device, serialize on silicon.** Each group (conic/gmean/gscale/gquat,
fwd geom/color) is an independent trace→allocate→emit→host-gate pipeline; the host oracles (NumpyBackend +
reg-simulator) need **no hardware**, so groups run embarrassingly parallel as workflow subagents. The single
Blackhole is the serialization point + human-in-the-loop boundary (NoC wedge hazard → `tt-smi -r 0`): host-
verified candidate kernels merge onto the device one group at a time behind the per-group flag, watching
telemetry + the differential self-check. Mixed-mode (some groups fused, rest on ttnn) is the steady rollout
state, so a regressing group falls back to ttnn instead of all-or-nothing.

## Honest risks (Step-0 must answer before committing)
1. ~~**fp32 8-reg arithmetic is UNPROVEN**~~ — **RESOLVED Step 0 ✅**: fp32-8 dst-resident chains pass 1e-2
   (1.5e-3); bf16 fails. Residual: the ~1.5e-3 tf32-input floor means long chains need the G-share/group-split
   to stay under 1e-2 (see Step-0 watch-item).
2. **Recompute blowup on geometry** — G feeds GR/gscale/gqn (9–18 co-live tiles); recompute-only could re-derive
   G's 3×3×3 cone at many use-points. Mitigation decided: compute G once + FIFO share + group-split budget cap.
3. **Allocator is trusted, load-bearing code** — a recompute scheduled across an in-place mutation silently
   returns stale data the loose 1e-2 gate can mask. Guardrails: SSA discipline + the fp32 reg-simulator
   (NumpyBackend == reg-sim == DAG) catches it with zero hardware before silicon.
4. Masks/transcendentals have zero silicon proof yet → Step 0 verifies each with a wired fallback.
