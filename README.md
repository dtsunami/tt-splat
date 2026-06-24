# tt-splat

3D Gaussian Splatting **training** on Tenstorrent Blackhole — a self-contained pipeline with a browser
training dashboard (the `ttgs` CLI). Jump to [Install](#install) · [Quickstart](#quickstart).

- **[`docs/PROGRESS.md`](docs/PROGRESS.md) — START HERE: full milestone log (M0–M15), architecture,
  findings, and status.** Every algorithmic stage is proven on real Blackhole silicon or the right target.
- [`docs/ALGORITHM.md`](docs/ALGORITHM.md) + `algorithm.svg` — the pipeline map, annotated with current state.
- [`FEASIBILITY.md`](FEASIBILITY.md) — hardware/stack feasibility: pipeline→BH mapping, the walls
  (sort, **cross-core float scatter-add**, per-pixel alpha-blend), assets, and the path forward.
- [`docs/pathclear/`](docs/pathclear/) — minimal proven flows on real silicon, de-risking the machinery before
  the hard kernels.
  - `gaussian_fit.py` — **M0 (validated):** 1D Gaussian fit, on-device fp32 forward (SFPU `exp`) + Adam. `PATHCLEAR_OK`.
  - `gaussian2d_image.py` — **M1 (validated):** 2D anisotropic Gaussian → image fit via the conic (Σ⁻¹); per-pixel
    forward + analytic 2D grads + Adam on-device. PSNR 39.8 dB. `M1_OK`. Render: `m1_target_vs_recovered.png`.
  - `m2_local_accum.py` — **M2a-local (validated):** custom kernel via `ttnn.generic_op` — a baby RISC accumulates
    a stream of floats in L1 bit-exactly (4096 ✓). Proves the locality thesis's foundation + the custom-kernel vehicle.
  - `m2_scatter_gather.py` — **M2b (validated):** the backward scatter-add. N source cores `noc_inline_dw_write`
    partials into distinct inbox slots in a home tile's L1 → home drains+reduces. Bit-exact, **contention-free —
    no atomics, no `noc_accumulate`, no NoC wedge.** This is the gradient-accumulation design.
  - `m2_drain_ipc.py` — **IPC study:** baby RISCs have no FPU → float reduce = soft-float 60 cyc/elem; **fixed-point
    int32 + ×8 unroll = 3.1 cyc/elem (~19×).** Rule: accumulate gradients in int32, convert to float for Adam.
  - `m2_indexed_reduce.py` — **M2c (validated):** real backward `acc[gid]+=partial`. Sorted+segmented+×4-prefetch
    = 8.1 cyc/elem (vs 25 unsorted) → sort the inbox by gid.
  - `render_gaussians_2d.py` — **M3 forward rasterizer (validated):** 16 Gaussians, front→back α-blend on SFPU,
    **116 dB vs CPU golden.** `RASTER_OK`. Render: `render_golden_vs_device.png`.
  - `train2d_verify.py` — **confidence gate:** analytic blend-backward vs `torch.autograd`, all 7 params,
    **worst rel err 3.7e-16** (fp64). `VERIFY_OK`.
  - `train2d.py` — **M4 2D training loop (validated on silicon):** fwd render → MSE → backward → Adam, fitting a
    target. Gate: device grads == torch analytic (2e-7). **Loss 1.9e-3→1e-6, PSNR 53.6 dB.** `TRAIN2D_OK`.
    Render: `train2d_target_vs_fit.png`.
  - `sfpu_raster.py` — **M5 SFPU rasterizer (milestone 1, validated):** one Gaussian's alpha eval **fused into a
    single SFPU compute kernel** (conic + `exp_tile`, dst-register binary ops), reader→compute→writer via
    `generic_op`. 61 dB vs golden. `SFPU_RASTER_OK`. (perf path; next: blend loop + multi-tile.)
  - `bin_sort.py` — **M6 tile binning + depth sort (validated):** 3σ AABB → tile instances → sort by (tile,depth) →
    per-tile ranges, on a **general-purpose target** (host now; GPU-radix/x280 later). Verified vs brute force;
    host CPU ~6 M-inst/s (100k Gaussians = 77 ms, 1M = 0.9 s). `BIN_SORT_OK`. Resolves the two Tensix sort walls.
  - `sfpu_blend.py` — **M5 SFPU rasterizer (milestone 2, validated):** full front→back **blend loop fused in ONE
    compute kernel** (C/T accumulators persist in dst registers across N Gaussians). 41 dB vs golden. `SFPU_BLEND_OK`.
  - `sfpu_raster_multitile.py` — **M13 multi-tile device rasterizer:** full image across a Tensix **core grid**
    (1 tile/core, block-sharded), reusing M5's kernel. Validated 40.8 dB; **throughput scales with cores — 21→188
    Mpix/s (4→64 cores), 256² in 0.35 ms.** Telemetry: Mpix/s, Mblend/s, µs/tile.
  - `sfpu_raster_scaled.py` — **M14 culling + unbounded N:** batched dispatches with **persistent L1 C/T** accumulators;
    each core blends only its **culled** (M6-binned) Gaussians, batch-by-batch → **N unbounded** (B=16 the only compile
    cap). `fp32_dest_acc_en` → **74.7 dB**; cull 0.12× the blends. `SCALED_OK`. (Batched path is host-overhead-bound.)
  - `device_backward.py` — **M15 device backward (item 3 core, validated):** the reverse of the alpha-blend on device
    (suffix-color S, per-pixel grad products, per-Gaussian `ttnn.sum` reduce) → all 7 param grads, **matches host
    autograd to 2.5e-3** (the 0.2% is ttnn.sum's bf16 reduce; fp32 `reduce_tile` tightens it). `DEVICE_BWD_OK`.
  - `device_train_loop.py` — **M16 device training loop CLOSED:** integrated fwd render + M15 backward + Adam, all
    the per-pixel work on device, fitting a target. **Converges PSNR 17→72 dB.** `DEVICE_LOOP_OK`.
  - `train2d_densify.py` — **M7 densification (validated):** clone/split/prune on host (general-purpose target).
    Operators unit-verified; demonstrably helps — 3 blurry → 11 detailed Gaussians, **+18 dB** (24.6→43.0). `DENSIFY_OK`.
  - `train3d.py` — **M8 3DGS loop closed (validated):** 3D→2D EWA projection (mean + covariance Jacobian) + the 2D
    pipeline, multi-view training over synthetic cameras. Projection geometry verified; train 55.7 dB;
    **NOVEL-view PSNR 46.5 dB** (held-out pose → learned real 3D structure). `TRAIN3D_OK`.
  - `colmap_ingest.py` — **M9 COLMAP ingestion (validated vs canonical):** parse cameras/images/points3D → our
    `(Rv,tv,f,pp)` + Gaussian init. Checked against COLMAP's documented `qvec2rotmat` (1e-15) and **camera
    position = −Rᵀt** (1e-15); read-back cams reproduce renders (3e-32). `COLMAP_INGEST_OK`.
  - `prepare_data.py` — **M10 real-data front end:** accepts **`--video`** (ffmpeg frame-extract) **or `--images`**,
    runs COLMAP via pycolmap (no sudo) → text model (cameras/images/points3D).
  - `train_real.py` — **M10/M12 train 3DGS from COLMAP + images:** SH view-dependent color (`--sh 0..3`) + per-image
    masks (`--masks dir`, frames.json polygons) weighting the loss; inits from sparse points, held-out novel view,
    `--preview`. `--selftest` → `REAL_PIPE_OK` (SH train 50.9 dB; mask weighting machine-exact Δ=0).
- [`docs/`](docs/) — [`ALGORITHM.md`](docs/ALGORITHM.md) + `algorithm.svg`: the full training loop annotated with current state.

## Install

tt-splat installs into the **tt-metal `python_env`** venv — the one built by tt-metal's `./create_venv.sh`,
which already has `torch` + `ttnn` (and everything else the Blackhole path needs, so this installs nothing
extra and never touches the tt-metal torch build). The venv uses **`uv pip`**:

```bash
git clone https://github.com/dtsunami/tt-splat.git ~/tt-splat
cd ~/tt-splat
VIRTUAL_ENV=~/tt-metal/python_env uv pip install -e .   # registers the `ttgs` CLI
cp .env.example .env                                    # then edit TT_METAL_HOME etc.
```

The `ttgs` command lands at `~/tt-metal/python_env/bin/ttgs`. Put that dir on `PATH`, or prefix it explicitly
(`~/tt-metal/python_env/bin/ttgs …`). All examples below assume `ttgs` is on `PATH`.

> The optional host gsplat reference path (`ttgs train`/`run`) + viser viewer (`ttgs view`) need extra deps:
> `VIRTUAL_ENV=~/tt-metal/python_env uv pip install -e '.[reference]'`. **Not needed for `ttgs blackhole`** —
> and `gsplat` may pull a different `torch`, so prefer a separate venv for it.

`.env` is loaded automatically on every run (it walks up from the current directory). See
[`.env.example`](.env.example) for every variable; the key ones are `TT_METAL_HOME` / `TT_METAL_RUNTIME_ROOT`
(your tt-metal tree) and the host-render budget knobs `TT_MAX_POINTS` / `TT_SIZE`.

## Quickstart

```bash
# 1. Verify the box: tt-smi on PATH, /dev/tenstorrent/0 present, TT_METAL_HOME set, ttnn importable
ttgs info

# 2. Train on the bundled sample scene (the corgi capture in work/scene)
ttgs blackhole work/scene                           # → open http://localhost:7860/training

# 3. Train on your own data
ttgs blackhole /path/to/your/colmap-dataset --output work/my_out --steps 4000
```

`ttgs info` prints a **Tenstorrent Blackhole** panel (device + driver + runtime checks) — run it first; every
row should be green before you train. `ttgs setup` prints the full dependency guide.

**CLI entry points** (`ttgs --help` for all):

| command | purpose |
|---|---|
| `ttgs info` | system + Blackhole device status (run this first) |
| `ttgs setup` | dependency / install guide |
| `ttgs blackhole <dataset>` | **the main run** — TT training dashboard (Render\|GT\|Diff, prune/densify/clamp, live metrics) |
| `ttgs sfm` / `ttgs extract` | data prep (COLMAP poses / video→frames) for your own captures |
| `ttgs view <splat.ply>` | open a finished `.ply` in the viser viewer |

**Bring your own capture** (COLMAP via `ttgs sfm`, or the pathclear helper):
```bash
ttgs extract clip.mp4 --output runs/scene/frames     # video → frames (needs ffmpeg)
ttgs sfm runs/scene/frames --output runs/scene        # frames → COLMAP poses + sparse points (needs colmap)
ttgs blackhole runs/scene
```

### Under the hood / advanced

`ttgs blackhole` is a thin wrapper over [`server/serve_blackhole.py`](server/serve_blackhole.py), which stands up
the vendored `ttgs` FastAPI dashboard ([`ttgs/`](ttgs/), forked from arcgs; self-contained, no `PYTHONPATH` needed)
and routes the training stage to [`server/train_tt.py`](server/train_tt.py) — a drop-in `ttgs` training stage
(full `TrainingController` contract; **SH color** from `cfg.sh_degree` + **per-image masks** from frames.json;
writes standard 3DGS `splat.ply` — deg-3 = 3 f_dc + 45 f_rest).

Canonical env = the `~/tt-metal` tree (`v0.74-dev`) with its `python_env` venv. Custom kernels (`generic_op`)
need a one-time symlink the JIT include path omits on Blackhole:
`ln -sf api/dataflow/dataflow_api.h ~/tt-metal/tt_metal/hw/inc/dataflow_api.h`.

**Single-owner device:** if open fails with TLB/hugepage errors, kill whatever holds `/dev/tenstorrent/0`
(`/proc/driver/tenstorrent/0/pids`). NOTE on this host: comfy runs a live SDXL server on board p150 — the probe
avoids `import ttnn` so it won't contend; don't kill those PIDs blindly. Recover a wedged card with `tt-smi -r 0`.

The raw milestone scripts run directly too, e.g. `~/tt-metal/python_env/bin/python docs/pathclear/gaussian_fit.py`.

## Status

**The full 3DGS loop is closed and verified** — every stage proven on Blackhole silicon or the right general-purpose
target: M0–M1 fits, M2 scatter-add (3–8 cyc/elem), M3 forward raster (116 dB), **M4 2D training loop** (autograd-
verified, 53.6 dB), M5 SFPU eval+blend-loop kernel, M6 bin/sort (host, verified), M7 densification (+18 dB), and
**M8 3DGS** — 3D→2D EWA projection + multi-view training, **novel-view PSNR 46.7 dB**. No open feasibility questions.
Remaining is build-out/perf: in-loop device scatter-add, a scaled integrated device run, real-data ingestion
(video→ffmpeg→frames→COLMAP→cameras), and moving the projection onto the matrix engine. See `docs/ALGORITHM.md`.

Documentation — docs/PROGRESS.md
A complete single-source-of-truth progress log:

The arc (feasibility → 2D → 3D → real data → arcgs dashboard → device kernels)
Milestone table M0–M16 — what, result, verification, script (every one runnable with an *_OK check)
Architecture — the kernel-differentiation table (which engine does what) + the metaparam-update design
Hard-won findings — moreh_adam broken, no-FPU/fixed-point, the CAS-is-wrong scatter-add insight, the generic_op symlink, fp32_dest_acc_en, ttnn.sum bf16, COLMAP canonical −Rᵀt, the SDXL device-sharing hazard
Proven-vs-remaining + env/run instructions
The cartoon (M15 backward + device-path callout), README (now points to PROGRESS.md first), and memory are all synced.

The loop is closed — device_train_loop.py (M16)
The integrated device training loop runs on Blackhole and converges: PSNR 17 → 72 dB (loss → 0):

forward blend → loss → M15 device backward (reverse pass + reduction) → Adam → params
the per-pixel render + backward + reduction all on device; Adam on the device-reduced gradients
the bf16-reduce 0.2% gradient error doesn't block convergence — Adam normalizes it
Where this lands the whole project
From "is 3DGS training even feasible on Blackhole?" to a device-resident training loop closed and converging on real silicon — 22 runnable, self-checking scripts, M0–M16, every algorithmic stage proven:

forward (188 Mpix/s, multi-tile, culled, unbounded N) · backward (autograd-verified) · scatter-add (3 cyc/elem) · Adam · binning/sort · densification · 3D + novel view · COLMAP ingest · arcgs dashboard on Blackhole · and now the full loop closed on device.

No feasibility unknowns remain. What's left is performance and scale: fuse fwd+bwd into the custom SFPU kernels, swap to fp32 reduce_tile, wire M2 scatter-add for many-Gaussian/multi-tile, and drop it behind train_tt's render_device/backward_device hooks so the arcgs dashboard drives Blackhole-accelerated training end to end — on your actual corgi capture.

That's the loop closed. Hell of a run. 🚀