# Scale campaign — full-res (~1.6k) + millions of Gaussians

Goal: train at **~1.6k long-edge** (≈1,850 tiles @ 32px) and scale to **millions** of Gaussians on the
Blackhole. Decision (2026-06-27, Dan): **SHARED-CORE-FIRST** — build the two things *both* axes need
before scaling either. Builds on the closed device-resident loop + the in-flight perf work
(`PERF_LOCALITY_PLAN.md`, `PROJECTION_FUSION_PLAN.md`).

## The reframe — two axes, not two walls

Current default = **96px / ~1200 Gaussians**. The target moves two *independent* axes that stress
*different* walls:
- **Full resolution** (96px→1.6k = ~205× pixels; →native 4608×3456 = 1,728×): 9 tiles → ~1,850 (→15,552).
  TILE-COUNT-bound → explodes the **host-side tile loops** (bin/sort, Stage-A arg-pack/readback/scatter,
  fwd-raster dispatch) *even at today's N*.
- **Millions of Gaussians** (~1000× N): COUNT-bound → explodes projection (per-Gaussian B+D) and **capacity**.

### Wall map at the target
| Wall | Trigger | Today | Needs |
|---|---|---|---|
| **Capacity** (NEW) | millions | params+Adam+grad ≈ **944 B/Gaussian** → L1 fits only **~175k** | GDDR-resident params/Adam; L1 = hot window |
| **Bin/sort** | both (×tiles ×N ×rep) | host lexsort, **~6 s/step @ 2.4M** | on-device counting-bucket sort |
| **Arg-pack** | both | host f2u of every (Gaussian,tile) incidence into runtime args | GDDR param stream (reader/CB) |
| **Wall 2** raster-bwd scatter | full-res (×tiles) | L1 compute → **host** scatter | hash-home owner reduce (L1) |
| **Wall 1** projection B+D | millions (×N) | half-fused, DRAM round-trip | finish fusion + DRAM-stream params |
| **Densify/realloc** | millions | host prune only; no resident grow | structural realloc of GDDR tensors |

### The bombshell (ordering inversion)
Full Gaussian state (params + Adam m/v + grad ≈ 944 B) fits only **~175k** in L1 across ~110 cores.
**Millions is physically GDDR-resident, not optional.** This *inverts* the perf-plan ordering: the items
it deferred to "Stage-7 endgame" (**on-device bin/sort + GDDR tiering**) move to the **front of the
critical path**; the in-flight L1 work (projection fusion, owner-reduce) becomes the **per-step-speed**
layer that only matters *after* capacity + sort unlock the regime. Campaign shape:
**unlock the regime (capacity + sort) → then make each step fast (fusion + owner-reduce + full-res tiling).**

## The shared core = two coupled pieces, both reducing to PROVEN primitives

The host walls (bin/sort, arg-pack, grad-scatter) all exist for ONE reason: the per-Gaussian destination/
source slot lives in the **host-resident bin/sort index** (`tile_lists`). Move the index on-device and all
three host walls collapse together. Two coupled builds:

1. **On-device bin/sort (counting-bucket).** `bin_and_sort` (docs/pathclear/bin_sort.py) =
   (a) cull + conic→var + 3σ AABB tile-range + per-Gaussian counts [embarrassingly parallel ttnn];
   (b) instance expansion = **histogram(tile_id) → exclusive-scan → scatter** [counting sort on a bounded
   key]; (c) within-tile **depth** order [the only comparison-shaped part].
   - (b)'s histogram+scatter **IS the m2 owner-single-writer pattern** (`m2_scatter_gather.py`, PROVEN
     bit-exact FP32, no atomics). The scan is over ~1,850 tiles (tiny).
   - (c) bucketize depth (D bins) → counting sort within tile → *approximate* front-to-back, gated on
     **loss/render parity**, not bit-exactness (alpha-blend tolerates fine-bucket perturbation).
2. **GDDR-resident params + on-device param streaming.** Params already live in interleaved GDDR
   (`device_project.py:19` `from_torch` default). Add a reader/CB that **streams a tile's ≤K Gaussian
   params from a GDDR buffer indexed by the on-device sort output** — replacing the host f2u arg-pack.
   Reuses the **E5 `generic_op` streaming substrate** (PROVEN: streams L1 params past the runtime-arg cap).

**Net:** both pieces compose `m2_scatter_gather` + the E5 streaming substrate — both already on silicon.
The campaign is integration + parity-gating, not new-primitive invention.

## Probe ladder (each silicon-gated; S0 host-only, DECISIVE)
- **S0 — depth-bucketing parity (host) — DONE ✅ PASS (`scratchpad/probe_S0_depthbucket.py`).** Real
  trained `splat.ply`, exact-lexsort vs D-bucket order, conservative millions-proxy (real Gaussians
  replicated w/ pos+depth jitter to high cover; same-opacity = extra semi-transparent layers; GLOBAL
  buckets not per-tile). Result: unordered collapses to ~30 dB at 30× cover, but **D=64 holds ≥44 dB
  through 30× overlap** (15×→45.6, 30×→44.6), ~14 dB over unordered. → **on-device counting sort
  GREENLIT, bucket count D=64** (per-tile bucketing will do strictly better; D=32 marginal ~40 dB).
  Everything else reduces to proven primitives. ttnn has `floor`/`clamp`/`gt`/`lt`/`cumsum` (S1/S2 buildable).
- **S1 — device tile assignment (exact).** cull(zc>1e-4)+conic→var(var_x=cc/det,var_y=ca/det)+3σ AABB
  (tx0..ty1)+counts in batched ttnn. Gate: bit-exact vs `bin_and_sort` counts/AABB. (Check ttnn floor/clip.)
- **S2 — counting-bucket assemble (device).** histogram(tile_id)→scan→scatter instances; reuse
  `m2_scatter_gather` owner-scatter + a small device/host scan over ~1,850 tiles. Gate: per-tile instance
  multiset == host.
- **S3 — within-tile depth order (device).** D-bucket counting sort within each tile (uses S0's D). Gate:
  per-tile order ⇒ render parity (S0 bar) + the integrated grad gate.
- **S4 — GDDR param stream (device).** raster/backward reader streams a tile's ≤K params from a GDDR
  buffer indexed by S2/S3 output (vs host arg-pack). Reuse E5 substrate. Gate: bit-exact raster vs host-pack.
- **S5 — integrate.** device sort → GDDR stream → raster/backward; NO host `tile_lists`/arg-pack. Gate:
  loss parity at 96px → 1.6k (the `last_g3` grad-equivalence gate in device_resident.py).

## Then scale (after the shared core lands)
- **Full-res 1.6k** (~1,850 tiles): grid-shard fwd+bwd over ~110 cores (~17 tile-passes/dispatch); the host
  loops are already gone (S5). **PSU dI/dt guard mandatory** at full-grid 1.6k load (power_ramp RampController,
  see [[bh-psu-power-virus-reboot]]).
- **Millions** (N): (a) **densify/prune = structural realloc** of the GDDR-resident param/Adam/grad tensors
  (periodic, host-orchestrated cadence; the inner loop stays resident); (b) **Wall 1** — finish projection
  fusion (Steps 3-5) + DRAM-stream params (bf16 prefetch seam); (c) **Wall 2** — hash-home owner reduce
  (same `m2_scatter_gather` primitive as the bin/sort histogram → build once, use twice).

## Honest risks
1. **Depth-bucketing parity (S0)** — if D-bucket order degrades training, need finer buckets or exact
   within-tile sort (small per-tile counts make insertion-sort viable on sparse scenes). Gate before any device work.
2. **On-device scan** — exclusive-scan over tile counts; ~1,850 elements is tiny (host-or-device fine), but
   the instance-offset scan over N can be large → may stay a cheap host step initially (still removes the lexsort).
3. **Router-gap `home(g)`** — owner-scatter needs a runtime physical-core lookup table (grid has router-only
   column gaps), per probe E3.
4. **PSU dI/dt** at 1.6k full-grid load — continuous soft-start ramp + bg-thread telemetry/abort ([[bh-psu-power-virus-reboot]]).
5. **Single-owner card** — keep comfy/SDXL down; a SIGTERM-killed run wedges the card (`tt-smi -r 0`).
