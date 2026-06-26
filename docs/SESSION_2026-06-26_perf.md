# Session wrap-up — Stage-A perf + the matmul-engine keystone (2026-06-26)

## What shipped (all on `main`, silicon-verified)

| commit | what | result |
|---|---|---|
| `1efbefc` | Stage 1: arg-pack-once + alloc-once | host-side Stage A win |
| `139b03e` | Stage 2: on-device per-tile reduce (`FB_S2`) | Stage A 145→60 ms (2.85×) |
| `bf0c396` | Stage 3: drain-once accumulator (`FB_S3`) | Stage A → 34 ms (**4.5× vs baseline**) |
| `bff0294` | Wire Stage 3 into training (`TT_FB_STAGE`, default `s3`) | grad-equiv 4.7e-4, converges identically |
| `9458657` | Probe: matmul-engine reduce holds fp32 in **bf16-dst** | 1.19e-2 worst (pool failed at 1.0) |
| `1207648` | Probe: SFPU arith + matmul reduce coexist in one kernel | 1.87e-3 |
| `aa349bd` | Re-fused single-tile kernel (arith + matmul reduce, 1 kernel) | K=2/K=4 6.89e-3 |

**Net:** Stage A (raster backward) went 320→70 ms in the live training loop (one env flag, correctness-safe).
The matmul-engine reformulation is de-risked end-to-end: the idle matrix engine accumulates fp32 well
enough to fold the reduce into the bf16 arithmetic kernel — the keystone for moving linear/reduction/
data-movement work off the SFPU.

## Hardware ground truth (this Blackhole)
- **110 usable Tensix cores** (11×10); we use **9** (one per 32×32 tile at 96px) → ~8% utilization.
- Compute = 5 RISCs/core: BRISC/NCRISC (NoC) + T0 unpack / T1 math (FPU+SFPU) / T2 pack, streaming via CBs.
- We are **SFPU-bound**; FPU/matrix engine (MVMUL/GMPOOL/GAPOOL/DOTPV) sits idle.
- Format conversion is **packer/unpacker-native** (UNPACR/PACR); transpose is native (TRNSPSRCA/B, SFPTRANSP)
  — not matmuls. → store params bf16 in DRAM, unpack→fp32 free.
- DRAM prefetcher exists (`ttnn.dram_prefetcher` / `start_tensor_prefetcher`, GCB + `dma_async_read`) — the
  millions-scale param-streaming substrate; reuse its primitives, not the matmul-shaped op.

## Amdahl analysis — the step is ~148 ms (Stage 3, N=1024/96px)
B=16 · raster_fwd=35 · A=70 · D=24 · C=3.

| stage | ms | % | step-× if →0 | technique | verdict |
|---|---|---|---|---|---|
| A raster bwd | 70 | 47% | 1.90× | Stages 1-3 done; matmul re-fuse residual | mostly done |
| raster fwd (M14) | 35 | 24% | 1.31× | in-kernel reduce + channel-parallel | **untouched, high ROI** |
| D proj bwd | 24 | 16% | 1.19× | projection-as-matmul | high ceiling |
| B proj fwd | 16 | 11% | 1.12× | projection-as-matmul (shared) | do with D |
| C Adam | 3 | 2% | **1.02×** | fuse + betas-as-const | **SKIP for perf** |

Discipline: **the bottleneck moves after every win — re-profile each time.** Don't optimize a stage below
its Amdahl share. The long-horizon wall is the **host-serial fraction** (bin/sort lexsort ≈ 6 s/step at
2.4M Gaussians) — no device-compute win crosses it; that's where on-device sort / x280 eventually matter.

## Next steps, Amdahl-ranked (the "1–5", pruned & sequenced)

1. **Color-channel parallelism (DO FIRST).** raster_fwd + A run R/G/B as 3 *serial* passes on 9 cores; the
   channels are independent → fan to ~27 of the 110 idle cores. Hits 105 ms / 71% of the step at once →
   up to ~1.9×. No new math; it's a work-decomposition restructure (`fused_backward_grid` channel loop →
   one dispatch over 3× cores or 3 concurrent; M14 forward likewise). **Highest ROI.**
2. **Re-profile.** After #1, projection (B+D=40 ms) is likely the largest.
3. **Projection-as-matmul (B+D).** The keystone's natural big target — 3×3 covariance/Jacobians are literally
   matmuls; moves 40 ms of dispatch-bound batched ttnn onto the idle FPU with fp32 accumulation.
4. **Finish the matmul re-fusion grid integration** (Stage A → 1 dispatch). Modest after #1 shrinks A; it's
   the proof-of-concept that de-risks #3. Single-tile already proven (`aa349bd`); next = wire `s4` stage into
   `fused_backward_grid` (no `outs[7]`; products are temp CBs; scalars → out_acc) + test_grid integrated gate.
5. **Fused Adam — SKIP for perf** (2% ceiling). Build only as the clean preload-constants example
   (betas as compile-time constants, m/v/p already resident, ~10 dispatches → 1).

## Open assumptions to verify as we build
- **Integrated precision of the matmul re-fusion** at grid magnitude — near-cancel microbench was 1.19e-2 vs
  test_grid's 1e-2 gate; gate on the *integrated* grad-check, fall back to fp32-dst reduce if marginal.
- **Channel parallelism core mapping** — `home(g)` / multi-channel grid must map through a runtime
  physical-core table (router-gap columns; `compute_with_storage_grid_size`), not the descriptor template.
- **A killed/hung kernel wedges the card** → `tt-smi -r 0`. Reader CB balance (S/T recurrence) must be exact.

## How to resume
- Train with Stage 3: `TT_DEVICE_RESIDENT=1 ttgs blackhole work/scene` (default `TT_FB_STAGE=s3`;
  `=base` to A/B). Gate: `scratchpad/test_grid.py` (set `FB_S2`/`FB_S3`), perf: `scratchpad/bench_S2.py`,
  full loop: `scratchpad/profile_resident.py` (`PN`/`PSZ`/`TT_FB_STAGE`).
- Keystone probes: `scratchpad/probe_matmul_reduce.py`, `proto_sfpu_matmul_coexist.py`, `proto_refuse.py`.
- Adam explainer for humans: `docs/adam_cartoon.html`.
