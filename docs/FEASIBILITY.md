# 3D Gaussian Splatting **Training** on Tenstorrent Blackhole — Feasibility

**Goal:** on-device 3DGS *training* (forward render + backward gradients + optimizer), not just inference.
**Target HW:** Blackhole (tt-1xx), host `ttstar`. Stack: tt-metal / ttnn / tt-train + bhtop NoC tooling.
**Date:** 2026-06-23. Repo: greenfield (`github.com/dtsunami/tt-splat`).

> Verdict in one line: **runnable, but it's a research project — ~80% custom-kernel work.** The optimizer/autograd
> plumbing exists; none of the splatting-specific pipeline does; and the backward pass hits the deepest hardware
> wall in the stack (cross-core float scatter-add). Forward rendering is hard-but-tractable; **full training
> feasibility rests almost entirely on solving gradient scatter-add.**

---

## 1. Pipeline mapped to Blackhole

| Stage | Maps? | Note / Wall |
|---|---|---|
| Project 3D→2D mean, 2D covariance (per-Gaussian 3×3/2×2) | Awkward | **Tile waste**: 32×32 matrix engine runs a 3×3 at ~⅛ throughput (`tech_reports/matrix_engine/matrix_engine.md:19`). Must batch N Gaussians into packed tiles. |
| Gaussian eval (`exp`), normalize quats/cov (`rsqrt`,`sqrt`,`recip`,`log`,`pow`) | ✅ clean | Blackhole SFPU has the full set (`tt_metal/.../sfpu/ckernel_sfpu_exp.h` etc). FP32 accum via `fp32_dest_acc_en`. Precision fine. |
| Tile assignment + **sort by (tile_id, depth)** | ✗ | **WALL 1.** No general/radix sort. Only `topk` (bitonic): bf16/bf8 + TILE-layout + 4D + tile-aligned-K; multi-core caps K≤64 (`ttnn/.../reduction/topk/topk_device_operation.cpp`). No histogram/bincount (only MOE `masked_bincount`), no group-by→variable-length lists. |
| Per-pixel front-to-back alpha blend | ✗ | **WALL 3.** Compute is hardwired 32×32 tiles, no per-pixel/row-major mode (`tt_metal/api/tt-metalium/constants.hpp:13`). Sequential saturation dependency + heavy divergence (0 vs 100+ Gaussians/tile). Reformulate tile-first. |
| **Backward: accumulate dL into shared Gaussian params** | ✗✗ | **WALL 2 — the killer.** Many pixels/cores scatter-add into the same Gaussian. No kernel-exposed atomic float-add; `ttnn::scatter_add` is per-core single-writer and rejects FP32-tiled. See §3. |
| Adam / optimizer step | ✅ (roll your own) | `moreh_adam` is **bf16-only AND returns wrong values in this build** (see §7) — use Adam built from ttnn primitives (fp32), validated §7. tt-train has autograd + optimizers but its Python bindings are **not built** here. |
| Adaptive density (clone/split/prune) | ✗ | **WALL 5.** Dynamic, data-dependent Gaussian count → reallocation. Doesn't fit static graphs; lands on host. |

---

## 2. Assets that already exist (don't rebuild)

- **tt-train** (`tt-metal/tt-train/`): real on-device training framework — autograd `Graph`/`GradFunction`
  (`sources/ttml/autograd/graph.hpp`), Adam/AdamW/SGD (`sources/ttml/optimizers/`), full loop examples
  (`sources/examples/linear_regression`, MNIST MLP, NanoGPT).
- **Backward ops**: ~70 eltwise `*_bw`, plus moreh matmul/linear/bmm/layernorm/softmax/sum/mean backward,
  embedding_backward, NLL loss backward.
- **Loss/reduction**: `mse_loss`, `l1_loss` on-device (`ttnn/.../loss/loss.hpp`); cumsum scan.
- **SFPU**: exp, sqrt, rsqrt, recip, sigmoid, tanh, gelu, log, pow on Blackhole. FP32 dest accumulation.
- **Custom-kernel model**: reader→compute→writer + circular buffers, NoC async read/write, well documented
  (`METALIUM_GUIDE.md`, `tt_metal/programming_examples/vecadd_multi_core/`).
- **No graphics code exists anywhere** (raster/splat/NeRF/alpha-blend) — confirmed. Built from scratch.

---

## 3. WALL 2 deep-dive: cross-core float scatter-add (the crux)

**The problem:** in backward, one Gaussian's gradient is written by many cores (it straddles multiple screen
tiles owned by different Tensix cores). *Within* one screen tile a single core owns the pixels → no atomic needed.
The cross-tile straddle is the irreducible scatter-add.

**The CAS-loop idiom (`__sync_bool_compare_and_swap` on a reinterpreted int*) is WRONG here.** Tensix hardware
atomics are scoped to a single tile's **cached L1 alias** (see `tt_metal/hw/inc/internal/debug/sanitize.h:295`,
`hw/inc/api/kernel_thread_globals.h:82`). A CAS aimed at a NoC-remote address is **not atomic between tiles** —
it compiles, runs, and silently loses updates. (The commonly-pasted CAS snippet is also buggy: reads `old_val`
once, never refreshes on failure → lost adds / infinite spin; should use `__sync_val_compare_and_swap`.)

**The correct primitive — native NoC FP32 accumulate (present on Blackhole):**
- `noc_accumulate(...)` — `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc.h:131`
- `#define NOC_AT_ACC_FP32 0x0` — `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:271`
- The NoC engine does the read-modify-write **at the destination** → many remote writers accumulate floats into
  one address with no coherence assumption. Right shape for scatter-add.
- **Caveats:** not exposed in high-level `dataflow_api` (must drive the NoC command buffer directly — bhtop
  injection territory); and BH requires "all 4 memory ports accept the transaction" for inline writes/atomics
  (`tt-1xx/blackhole/noc_nonblocking_api.h:454`) → finicky, hang-prone (see NoC hang hazard notes).

**Approaches, in recommended order:**
1. **Partial buffers + reduction (no atomics)** — each core writes its own gradient slot; second pass tree-reduces
   across cores. Deterministic, hang-safe. *Chosen — see below; VALIDATED.*
2. **Native NoC FP32 accumulate** — `noc_accumulate`; collapses scatter into the write but wedge-risky. Not needed.
3. **x280 coordinator** — fallback.

**✅ RESOLVED (M2, 2026-06-23) — the gate is cleared.** Approach 1, reframed via the locality thesis: give each
Gaussian a **home tile**; each source core `noc_inline_dw_write`s its partials into its **own dedicated inbox slot**
in the home's NoC-visible L1 (distinct slots → no collision); the home tile **drains+reduces locally** (single
writer). Validated on silicon (`pathclear/m2_scatter_gather.py`): 8 cores × 512 partials → bit-exact, **no atomics,
no `noc_accumulate`, no wedge.** Perf (`m2_drain_ipc.py`): baby RISCs have **no FPU** → float reduce is soft-float
(60 cyc/elem, doesn't unroll); **fixed-point int32 + ×8 unroll = 3.1 cyc/elem (~19×)** — so accumulate gradients in
int32, convert to float once for Adam. (×16 unroll regresses → register spill.) Remaining: per-Gaussian *indexed*
reduce (segmented, sorted-by-gid inbox) + integration into a real backward.

---

## 4. Heterogeneous angle: x280 for the irregular stages

Every wall (sort, binning, variable-length tile lists, densification, scatter coordination) is exactly what
Tensix is worst at and a general CPU is good at. Blackhole has 4 x280 (SiFive L2CPU, RVV) cores on the NoC.

- **x280**: tile-assignment, (tile_id, depth) sort, prefix-sum/segment boundaries, densification bookkeeping,
  scatter coordination.
- **Tensix**: batched projection math, per-tile rasterization, matmul-heavy backward, `moreh_adam` updates.
- **Caveats (measured):** only 4 cores; RVV gather/redsum slow (DLEN=256); x280↔Tensix share only via memory/NoC
  (register sharing / VCIX ruled out); avoid NoC hang hazard on L2CPU register access.
- The CRT/RNS quarter-square LUT integer matmul (already bit-exact on x280, "for splatting") is a candidate for
  the low-power projection path.

---

## 5. Recommended path

**De-risk in this order — prove the loop machinery before building the hard kernels:**

0. **Pathclear: existing-API proven flow. ✅ DONE — see §7.** M0 1D Gaussian fit + M1 2D anisotropic Gaussian→image
   fit, both forward + fp32 Adam on silicon, converged. The green eval+optimize spine is lit through the 2D conic.
1. **NoC FP32 atomic-add microbenchmark** — settle WALL 2 (see §3.2) on silicon.
2. **Forward-only, single low-res image** — batched projection (Tensix) + tile-first rasterizer + depth sort (x280),
   validated against a CPU golden splat render.
3. **End-to-end toy** — `moreh_adam` + hand-rolled backward for one param (opacity) on a toy scene.

**Bottom line:** forward inference rendering is hard-but-tractable; full *training* feasibility = solving the
gradient scatter-add (WALL 2). We have a real, unexplored hardware lead on it (`noc_accumulate` FP32).

---

## 7. Milestone 0 — VALIDATED ON SILICON (2026-06-23)

`pathclear/gaussian_fit.py` fits `y = A·exp(-½((x-μ)/σ)²)` to noisy samples. Result:

```
truth     A=2.0   mu=1.0   sigma=1.5
init      A=1.000 mu=0.000 sigma=1.000
step  1   loss=0.525197
step 300  loss=0.000105   A=2.000 mu=0.999 sigma=1.500   -> PATHCLEAR_OK
```

**On-device (fp32):** Gaussian forward (`sub/mul/square/exp`, SFPU) + Adam optimizer (built from
`mul/add/square/sqrt/div/sub`). **Host glue:** reducing 3 gradient scalars + grad-tile assembly.

**Proves:** SFPU `exp` is bit-exact on silicon; an fp32 training loop converges on Blackhole; we can build an
optimizer from primitives. This de-risks the *machinery*; the splatting-specific walls (§1, §3) remain.

**Milestone 1 — also VALIDATED (`pathclear/gaussian2d_image.py`):** 2D anisotropic (rotated) Gaussian fit to a
32×32 image, parametrized by the **conic Σ⁻¹** (a,b,c) — exactly what the rasterizer evaluates per pixel.
Recovered center `(15.99, 12.99)` vs truth `(16, 13)` and conic `(0.164,-0.149,0.282)` vs `(0.167,-0.153,0.286)`,
**PSNR 39.8 dB**. On-device: per-pixel forward (`sub/mul/square/exp`) + analytic 2D gradients (∂A,∂center,∂a,∂b,∂c)
+ fp32 Adam; host keeps a cheap PD projection on the conic. Side-by-side render: `pathclear/m1_target_vs_recovered.png`.
Still NO sort / bin / scatter — the green eval+optimize spine now reaches the 2D conic.

**Landmines found (real, not in the docs):**
- `ttnn.operations.moreh.adam` is **BFLOAT16-only** (`moreh_helper_functions.cpp:282`) **and numerically wrong in
  this build**: grad `[0.5,-0.5,2.0]` → `[-1.52, 3.52, 0.37]` (expect ~`[0.95,1.05,0.95]`); the `*_out` path
  corrupts lanes to ±1e18. Do not use it — roll Adam from primitives. (Firmware 19.11.0 > tested 19.5.0.)
- `ttnn.Tensor` has **no item assignment** (`t[:] = ...`) — rebind from op return values.
- tt-train (`ttml`) Python bindings are **not compiled** in this tree — the "train" API path needs a C++ build.

**Environment (reproducible):**
```
export TT_METAL_HOME=/home/starboy/comfy/tt-metal
/home/starboy/comfy/tt-metal/python_env/bin/python pathclear/gaussian_fit.py
```
The chip is single-owner: if open fails with TLB/hugepage errors, another process holds `/dev/tenstorrent/0`
(check `/proc/driver/tenstorrent/0/pids`) — kill it first.

## 6. Open experiments / unknowns

- [ ] Does `noc_accumulate` FP32 give bit-exact multi-writer sums without wedging NoC0? (microbench)
- [ ] topk/bitonic sort viability for per-tile depth ordering at 3DGS scale, or must it go to x280?
- [ ] Tile-packing scheme for batched 3×3 projection to beat the ⅛ matmul-throughput penalty.
- [ ] tt-train autograd: can we register a custom `GradFunction` for a splatting op, or is host orchestration needed?
- [ ] Densification (dynamic Gaussian count) — host round-trip cost per step.
