# tt-splat — Progress Log

**3D Gaussian Splatting training on Tenstorrent Blackhole** — the Tenstorrent backend for the
[`arcgs`](../../arcgs) pipeline. This document is the single source of truth for what's been
built and proven. Companion docs: [`FEASIBILITY.md`](../FEASIBILITY.md) (original analysis),
[`ALGORITHM.md`](ALGORITHM.md) + `algorithm.svg` (annotated pipeline map), [`README.md`](../README.md).

**Status (2026-06-27): the full device-resident training loop is closed and running at full resolution
on silicon.** Every algorithmic stage (M0–M16 below) is proven; beyond them, the integrated loop
(projection → M14 raster → fused backward → analytic projection backward → Adam) runs entirely on the
Blackhole with params + optimizer resident (no host autograd), drives the live dashboard, and now trains at
**1600px / 50k Gaussians (1.4k+ iterations stable)** via an on-device counting bin/sort and a tile-block
raster. The scale-up plans of record are [`SCALE_CAMPAIGN_PLAN.md`](SCALE_CAMPAIGN_PLAN.md) (full-res +
millions) and [`PROJECTION_FUSION_PLAN.md`](PROJECTION_FUSION_PLAN.md). No open feasibility questions
remain — what's left is perf/scale (GDDR streaming, power-ramp guard, on-device densify, on-die sort).

---

## The arc

```
feasibility  →  1D/2D fit  →  scatter-add gate  →  2D training loop  →  3D + novel view
   →  real data (COLMAP)  →  arcgs dashboard on Blackhole  →  device-kernel scale-up
```

Started as "is 3DGS training even feasible on Blackhole?" — the answer is now a verified **yes**,
end to end, including a render of an actual corgi capture and a device backward matching autograd.

---

## Milestones

Every milestone is a runnable script under [`../pathclear/`](../pathclear/) (or [`../server/`](../server/))
with an `*_OK` self-check.

| # | What | Result (on silicon unless noted) | Script |
|---|---|---|---|
| **M0** | 1D Gaussian fit: fp32 forward (SFPU `exp`) + Adam from primitives | loss 0.525→1e-4 | `gaussian_fit.py` |
| **M1** | 2D anisotropic Gaussian→image via conic (Σ⁻¹) + analytic 2D grads | 39.8 dB | `gaussian2d_image.py` |
| **M2** | **Scatter-add gate** — contention-free home-tile inbox→drain | bit-exact; fixed-pt int32 **3.1 cyc/elem** (×8 unroll), indexed reduce 8.1 | `m2_*.py` |
| **M3** | Forward rasterizer (ttnn ops): 16-Gaussian front→back α-blend | 116 dB vs golden | `render_gaussians_2d.py` |
| **M4** | **2D training loop** — fwd→loss→backward→Adam | autograd-verified **3.7e-16**; device==math **2e-7**; **53.6 dB** | `train2d_verify.py`, `train2d.py` |
| **M5** | SFPU eval kernel (61 dB) + **fused blend-loop kernel** (one dispatch) | 41 dB, C/T in dst regs | `sfpu_raster.py`, `sfpu_blend.py` |
| **M6** | Tile binning + depth sort on a general-purpose target (host now) | verified vs brute force; **6 M-inst/s** | `bin_sort.py` |
| **M7** | Densification (clone/split/prune) | operators verified; **+18 dB** (3→11 Gaussians) | `train2d_densify.py` |
| **M8** | **3DGS loop closed** — EWA projection + multi-view training | **novel-view 46.7 dB** (held-out pose) | `train3d.py` |
| **M9** | COLMAP ingestion verified **vs canonical** | `qvec2rotmat` 1e-15, camera pos `−Rᵀt` 1e-15 | `colmap_ingest.py` |
| **M10** | Real-data pipeline (video/images→COLMAP→train), RGB | `REAL_PIPE_RGB_OK`, novel 44.1 dB | `prepare_data.py`, `train_real.py` |
| **M11** | **arcgs dashboard on Blackhole** driving the TT backend | endpoints 200; live controller verified | `server/train_tt.py`, `server/serve_blackhole.py` |
| **M12** | SH view-dependent color (deg 0–3) + per-image masks→loss | SH 50.9 dB; mask weighting exact (Δ=0) | `train_real.py` |
| **M13** | **Multi-tile device rasterizer** across the Tensix core grid | **21→188 Mpix/s (4→64 cores)**; 256² in 0.35 ms | `sfpu_raster_multitile.py` |
| **M14** | Culling + **unbounded N** (batched dispatch, persistent L1 C/T) | **74.7 dB** (fp32 dest); cull 0.12× blends | `sfpu_raster_scaled.py` |
| **M15** | **Device backward** of the alpha-blend (reverse pass on device) | matches autograd **2.5e-3** (bf16-reduce limited) | `device_backward.py` |
| **M16** | **Device training loop closed** — fwd + M15 backward + Adam, on device | converges **PSNR 17→72 dB** (loss→0) | `device_train_loop.py` |

Plus: a **real corgi capture** ran end-to-end through COLMAP → cameras (9 imgs, 9544 pts) →
geometry verified by point-cloud overlay on the photo (blanket stripes align).

---

## Architecture — kernel differentiation per stage

The deep lesson: **baby RISC-V cores are integer/control engines (no FPU); the compute trio
(TRISC→SFPU/matrix) is the float horsepower; the host/x280 handle the irregular work.** Each
stage lands on the engine it fits:

| Stage | Engine | Locality / target |
|---|---|---|
| Project 3D→2D (EWA) | matrix/FPU | Gaussians L1-resident per gid-partition |
| Bin + sort | host / x280 | general-purpose target (M6) |
| Gaussian eval + α-blend | **SFPU**, tile-vectorized | screen-tile resident (M5/M13/M14) |
| Loss, ∂L/∂image | SFPU eltwise | per-tile |
| Backward blend | **SFPU** (reverse) | screen-tile resident (M15) |
| Scatter-add | **baby RISC, fixed-point int** | home-tile inbox→drain (M2) |
| Adam | SFPU/FPU | params L1-resident (M0) |
| Densification | host | dynamic count (M7) |

**Metaparam updates (live, no recompile):** the training loop is host-orchestrated dispatches,
so continuous metaparams (lr, β, ε, thresholds, λ) are host-side values applied per dispatch —
changed live with **zero kernel recompile**. Structural changes (prune/densify → Gaussian count)
rebuild buffers between dispatches (rare). The hard rule: **never recompile the SFPU kernel for a
hyperparam** (JIT compile ≈ 500 ms). arcgs's `update_config` maps straight onto tier-1.

---

## Hard-won findings (the non-obvious stuff)

- **`moreh_adam` is bf16-only AND numerically broken** in this build → roll Adam from ttnn primitives (M0).
- **Baby RISCs have no FPU** → scalar float = soft-float (60 cyc/elem, doesn't unroll). Accumulate
  gradients in **fixed-point int32** → 3.1 cyc/elem (~19×) (M2 IPC study).
- **The CAS-loop atomic-float idiom is wrong on Tensix** (atomics are local-L1-only, not cross-tile).
  The contention-free **home-tile inbox→drain** sidesteps it entirely — no atomics, no NoC wedge (M2).
- **`generic_op` custom kernels are `skip_for_blackhole`** — the JIT omits one include path; fix is a
  one-line symlink: `ln -sf api/dataflow/dataflow_api.h ~/tt-metal/tt_metal/hw/inc/dataflow_api.h`.
- **`fp32_dest_acc_en`** is needed for SFPU precision across batched accumulation (36 → 74 dB) (M14).
- **`ttnn.sum` uses bf16 accumulation** (~0.1–1% on large reductions); fp32 = custom
  `reduce_tile<SUM, REDUCE_SCALAR, enforce_fp32_accumulation>` (M15).
- **COLMAP canonical conventions**: world→cam `qvec`(w,x,y,z)+`tvec`; **camera position = −Rᵀt**;
  our `quat_to_rot` == COLMAP `qvec2rotmat` (1e-15). A self-round-trip hid an improper-rotation bug
  in `look_at` (det = −1) that only the canonical check caught (M9).
- **Device sharing hazard**: `~/comfy` runs a live SDXL inference server on the chip (board p150,
  port 8000). **Never kill `/proc/driver/tenstorrent/0/pids` without identifying them**; the TT
  probe avoids `import ttnn` (which inits the cluster and contends).

---

## What's proven vs what remains

**Proven on silicon** (no feasibility risk): SFPU math, fp32 training loop, 2D + 3D training with
novel-view synthesis, scatter-add, multi-tile rasterizer (188 Mpix/s), culling + unbounded N,
**device backward (autograd-verified)**, COLMAP ingestion, the arcgs dashboard driving the TT backend.

**The full device-resident loop is closed AND scaled** — project → raster → fused backward → analytic
projection backward → Adam, all resident; it drives the dashboard and trains **1600px / 50k Gaussians
(1.4k+ iterations stable)**. Landed since M16: the fused m17 backward, projection fusion, the on-device
counting **bin/sort** (`server/device_binsort.py`), and the **tile-block raster** (any resolution,
`server/raster_blocked.py`) with a vectorized arg-pack + s4 in-kernel reduce. **Remaining — perf/scale,
not research** (plan of record: [`SCALE_CAMPAIGN_PLAN.md`](SCALE_CAMPAIGN_PLAN.md)):
1. Full **GDDR param streaming** (zero host arg-pack) + `sgid`→DRAM (on-device sort past ~32k).
2. **Power-ramp guard** wired into the resident loop (PSU dI/dt at full-grid 1600px — `power_ramp.py`).
3. **On-device densification** + hash-home owner-reduce (grads resident A→D→C) toward millions.
4. On-die counting/radix **sort** for >1M Gaussians; GDDR owner-partitioned tiering past the L1 ceiling.

---

## Environment & running

Canonical env: the `~/tt-metal` tree (`v0.74-dev`) + its `python_env` venv (`./create_venv.sh`).
```bash
export TT_METAL_HOME=~/tt-metal TT_METAL_RUNTIME_ROOT=~/tt-metal
~/tt-metal/python_env/bin/python pathclear/<script>.py      # device scripts need the symlink (above)
```
Train on real data:
```bash
PY=~/tt-metal/python_env/bin/python
$PY pathclear/prepare_data.py --images /path/to/photos --out runs/scene     # or --video clip.mp4 (needs ffmpeg)
$PY pathclear/train_real.py  --model runs/scene/sparse/0 --images /path/to/photos --size 96 --sh 3 --preview out.png
```
arcgs dashboard on Blackhole:
```bash
PYTHONPATH=/home/starboy/arcgs $PY server/serve_blackhole.py --dataset work/scene --output work/tt_out
# → http://localhost:7860/training
```

Single-owner device: if a run fails with TLB/hugepage errors, another process holds
`/dev/tenstorrent/0` (check `/proc/driver/tenstorrent/0/pids` — and identify before killing; the
SDXL server lives there).
