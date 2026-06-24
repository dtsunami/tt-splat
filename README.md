# tt-splat

3D Gaussian Splatting **training** on Tenstorrent Blackhole.

- [`FEASIBILITY.md`](FEASIBILITY.md) — hardware/stack feasibility: pipeline→BH mapping, the walls
  (sort, **cross-core float scatter-add**, per-pixel alpha-blend), assets, and the path forward.
- [`pathclear/`](pathclear/) — minimal proven flows on real silicon, de-risking the machinery before
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

## Run

```bash
export TT_METAL_HOME=~/tt-metal TT_METAL_RUNTIME_ROOT=~/tt-metal
~/tt-metal/python_env/bin/python pathclear/gaussian_fit.py
```

**arcgs dashboard on Blackhole** ([`server/`](server/)) — reuses the arcgs FastAPI dashboard (imported as a
library; `~/arcgs` untouched), routing the training stage to the TT pipeline:
```bash
PYTHONPATH=/home/starboy/arcgs ~/tt-metal/python_env/bin/python \
  server/serve_blackhole.py --dataset work/scene --output work/tt_out   # → http://localhost:7860/training
```
`server/train_tt.py` is a drop-in arcgs training stage (full `TrainingController` contract: live Render|GT|metrics,
prune/densify/clamp/pause commands; **SH color** from `cfg.sh_degree` + **per-image masks** from frames.json; writes
standard 3DGS `splat.ply` — deg-3 = 3 f_dc + 45 f_rest). Verified in-process (all endpoints 200) + driving a real
controller on the corgi data. NOTE: comfy runs a live SDXL server on the device (board p150) — don't kill
`/proc/driver/tenstorrent/0/pids` blindly; the probe avoids `import ttnn` to not contend.

**Train on your own data** (pycolmap installed in the venv; ffmpeg only for video):
```bash
PY=~/tt-metal/python_env/bin/python
# images (no sudo): prefer this for a bad video
$PY pathclear/prepare_data.py --images /path/to/photos --out runs/scene
$PY pathclear/train_real.py  --model runs/scene/sparse/0 --images /path/to/photos --size 96 --preview out.png
# OR video (needs: sudo apt install ffmpeg):
$PY pathclear/prepare_data.py --video clip.mp4 --out runs/scene --every 10
```

Canonical env = the `~/tt-metal` tree built at `v0.74-dev` with its `python_env` venv (`./create_venv.sh`,
includes torch + ttnn). Custom kernels (`generic_op`) need a one-time symlink that the JIT include path omits on
Blackhole: `ln -sf api/dataflow/dataflow_api.h ~/tt-metal/tt_metal/hw/inc/dataflow_api.h`.

Single-owner device: if open fails with TLB/hugepage errors, kill whatever holds `/dev/tenstorrent/0`
(`/proc/driver/tenstorrent/0/pids`).

## Status

**The full 3DGS loop is closed and verified** — every stage proven on Blackhole silicon or the right general-purpose
target: M0–M1 fits, M2 scatter-add (3–8 cyc/elem), M3 forward raster (116 dB), **M4 2D training loop** (autograd-
verified, 53.6 dB), M5 SFPU eval+blend-loop kernel, M6 bin/sort (host, verified), M7 densification (+18 dB), and
**M8 3DGS** — 3D→2D EWA projection + multi-view training, **novel-view PSNR 46.7 dB**. No open feasibility questions.
Remaining is build-out/perf: in-loop device scatter-add, a scaled integrated device run, real-data ingestion
(video→ffmpeg→frames→COLMAP→cameras), and moving the projection onto the matrix engine. See `docs/ALGORITHM.md`.
