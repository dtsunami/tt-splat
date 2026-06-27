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
- **S1 — device tile assignment (exact) — DONE ✅ PASS on silicon (`scratchpad/probe_S1_tile_assign.py`).**
  conic→var(detc guard + clip 0.25) + 3σ AABB(tx0..ty1) + counts in batched ttnn (fp32, interleaved DRAM —
  correctness layout; perf streaming is S4). Ops: `div`/`sqrt`/`floor`/`clamp`/`mul`/`sub`. Gate result:
  **AABB 100.00% bit-exact vs float64 host @96px AND @1.6k** (1,900 tiles), total instances exact
  (1045/4271, 0.000% off), 0 degenerate — the fp32 boundary off-by-ones never appeared on real data.
  Card opened/closed clean (no wedge).
- **S2+S3 — counting-bucket assemble + depth order — UNIFY into ONE counting sort on a COMPOSITE key**
  `key = tile_id*D + depth_bucket` (D=64). histogram(key)→exclusive-scan→scatter = `ttnn.cumsum` + scatter.
  - **HOST PROTO DONE ✅ PASS (`scratchpad/proto_S2_countsort.py`).** Per-tile gid SET identical to
    `bin_and_sort` (96px 9/9, **1600px 589/589 tiles**), instances exact (1045/4271), within-tile order
    monotonic-in-bucket. Algorithm correct + maps to device primitives.
  - **DEVICE PORT — increment 1 DONE ✅ PASS on silicon (`scratchpad/probe_S2_dev_countsort.py`).**
    SINGLE-CORE kernel: integer expand+histogram+exclusive-scan+scatter+ranges, all in one core's L1, no
    NoC/contention (multi-core `m2_scatter_gather` owner-scatter = MILLIONS-only, deferred). @96px: kernel
    instance count 1045==host, per-tile gid SET 9/9, dbg readback confirms linear L1. **HARD-WON GOTCHA
    (cost a wedge):** raw-L1 `from_torch` buffers for kernel pointer-access MUST be WIDTH-contiguous
    (`[1,1,1,Mp]`, WIDTH_SHARDED `[1,Mp]`, ROW_MAJOR) — HEIGHT-shard `[Mp,1]` pads each row → kernel reads
    garbage bw/bh → unbounded loop → HANG → SIGTERM wedges card (`tt-smi -r 0`). FIX = width-contiguous +
    grid-BOUNDED loops (`dxm=min(bw,ntx-tx0)`) make it hang-proof + `dbg[]` readback verifies layout.
  - **increment 2 DONE ✅ PASS on silicon (`scratchpad/probe_S2_dev_inc2.py`).** AABB/bucket computed in
    ttnn (device S1 math, not host) → packed 1 uint32/Gaussian (5×6-bit fields; ntx,nty,D ≤64 @1600) →
    single-core kernel sort at **1600px / 32k Gaussians** (171,422 instances). Gates: ttnn AABB == host
    **100.00%**, kernel instances == host, per-tile gid SET **736/736**, L1 1284/1536KB. So the device
    bin/sort runs END-TO-END at the 1600px target. **32k is the single-core L1 ceiling** (packed); 50k
    needs `sgid`→DRAM — folds into S4's DRAM streaming. Device-exec timing TBD (needs synchronize; host
    dispatch was async sub-ms).
- **S4 — GDDR param stream (device).** raster/backward reader streams a tile's ≤K params from a GDDR
  buffer indexed by S2/S3 output (vs host arg-pack). Reuse E5 substrate. Gate: bit-exact raster vs host-pack.
- **S5 — integrate.** device sort → GDDR stream → raster/backward; NO host `tile_lists`/arg-pack. Gate:
  loss parity at 96px → 1.6k (the `last_g3` grad-equivalence gate in device_resident.py).

## Then scale (after the shared core lands)
- **Full-res 1.6k** (~1,850 tiles): **⚠ FINDING (post-S2, decisive): the raster grid-shard assigns 1
  tile/core over an `ntx×nty` CoreRange and the worker grid is only ~110-130 cores → it physically caps at
  ~384px today** (render_device `_resources`, fused_backward `fused_backward_grid` both build a full
  `ntx×nty` grid; the `_resources_par` fit-guard only falls back on CHANNELS, never tiles). So 1600px (1900
  tiles) needs a **RASTER TILE-BLOCK LOOP** (process ≤grid tiles/dispatch, loop ~17 blocks) in BOTH the M14
  forward and the m17 backward — a THIRD workstream, neither S4 nor S5. **PSU dI/dt guard mandatory** at
  full-grid 1.6k load (power_ramp RampController, [[bh-psu-power-virus-reboot]]).

### Revised sequence to the 1600/50k live training run (post-S2)
S2 (device bin/sort) is DONE + packaged as **`server/device_binsort.py`** (drop-in for `bin_and_sort`).
Three remaining workstreams, in dependency order:
1. **Raster tile-block loop** (the 1600px gate; biggest, independent of S4/S5) — loop tile-blocks of
   ≤worker-grid; each block's tiles **row-major TILE-LIST shard** (HEIGHT_SHARDED [TS,TS]/core) onto the
   usable grid; **M14/m17 kernels REUSED VERBATIM** (1 tile/core), only the harness loops + gathers.
   - **R1 forward mechanism DONE ✅ PASS on silicon (`scratchpad/proto_R1_tileblock_fwd.py`).** 64 tiles on
     a 4×4=16-core grid → 4 blocks looped, rendered vs host golden **6.75e-4** (<1e-2). Kernel untouched.
   - **R2 backward mechanism DONE ✅ PASS on silicon (`scratchpad/proto_R2_tileblock_bwd.py`).** Same block
     loop on the m17 base backward (host reduce); m17 kernel + S/T recurrence reused verbatim; grads
     host-accumulated across chunks AND blocks. 8×8 tiles on 4×4 grid → 4 blocks, **bit-exact 1.795e-7** vs
     verified `fused_backward_grid(base)`. (s4 in-kernel reduce stays the ≤352px fast path; >352px uses base.)
   - **PRODUCTIONIZED + INTEGRATED — DONE ✅ PASS on silicon (R3, `scratchpad/test_R3_integration.py`).**
     `server/raster_blocked.py` = `raster_rgb_blocked` (R1 fwd, 3-channel) + `fused_backward_blocked` (R2
     bwd, base) + `needs_blocking(dev,ntx,nty)` (>352px or `TT_FORCE_BLOCKED=1`). `device_resident` branches:
     `blocked` skips `_resources`, routes both raster calls to the blocked path; ≤352px path byte-identical.
     R3: (1) force-blocked==normal @96px **1.72e-8** (bit-exact in the live loop); (2) **384px TRAINS**
     (device sort + auto tile-block, the config that used to FATAL on a dispatch core) loss 0.159→0.135.
     Run it: `TT_SIZE=384 TT_DEVICE_RESIDENT=1 TT_DEVICE_BINSORT=1 ttgs blackhole work/scene`. **1600px = the
     SAME blocked path** (more blocks; mechanism proven multi-block) — just slow until S4 kills the arg-pack.
2. **S5 integrate device_binsort — DONE ✅ PASS on silicon (`scratchpad/test_s5_binsort.py`).** Wired into
   `device_resident` behind `TT_DEVICE_BINSORT=1` (swap @device_resident.py:106). LIVE resident training on
   real corgi @**352px** (the worker grid is **11×10=110** cores → 1-tile/core raster maxes at 352px,
   confirming the cap): loss **0.151→0.120 (-20.7%)**, host-parity max curve diff **1.85e-4** (depth-bucket
   ≈ exact-lexsort, as S0 predicted). So the device bin/sort runs in the live loop, training descends. Free
   `device_binsort` L1 buffers before the raster reuses core (0,0).
3. **S4 — blocked-path speed — DONE ✅ PASS on silicon (`scratchpad/test_S4_argpack.py`).** Pragmatic S4
   (the runnable-validation version): (a) **vectorized arg-pack** in `raster_blocked` (f2u all params once via
   float32 bit-view → numpy gather per dispatch, bit-identical); (b) **ported the s4 in-kernel matmul-reduce
   into `fused_backward_blocked`** (drain-once, no per-chunk full-tile readback). Result @768px/16k/432-tile
   multi-block: backward **A 3571→453ms (7.9×)**, step **4137→966ms (4.3×)**, parity vs normal-s4 **0.00e+00**.
   (c) `TT_TARGET_POINTS` densify-up knob in train_tt (corgi seed is ~9.5k → replicate to 50k). **Full GDDR
   param streaming (zero host arg-pack) + `sgid`→DRAM (device sort >32k) = the production follow-on** — at 50k
   use host sort (`TT_DEVICE_BINSORT` unset); device_binsort caps at 32k.
Then the 50k@1600 loss-descending run = commit gate. A live training run at **≤384px with device_binsort**
(steps 2 only) is the nearest intermediate validation.
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
