# Closing the gsplat training-recipe gaps — plan of record

The render/backward primitives are at parity with gsplat; the gaps are all in the **training recipe**.
This session closed the two biggest cheap ones; this doc plans the rest, **pose-opt first**.

## Done this session (validated on the real corgi)
- ✅ **L1 + D-SSIM loss** (`server/loss.py`, wired into both train paths; `λ dssim` knob live). Was pure MSE.
  `dL/dimage` via autograd — correct by construction. PSNR climbs 8.6→11.0 on a 24-step resident run.
- ✅ **Exponential LR decay on means** (`TT_LR_DECAY`, default →1% over the run; both paths). gsplat-style.

## Validation discipline (the lesson from the densify-fog incident)
Every item below ships only when it **demonstrably helps (or doesn't hurt) PSNR on the real corgi** — not
"it runs." Where there's a hand-derived gradient, **finite-difference grad-check it first**.

---

## #1 — Camera pose optimization (PRIORITY) — gsplat `pose_opt`

**Goal.** Make the camera extrinsics trainable to correct COLMAP pose error (ghosting/double-images), and
expose the same handle to the game ("grab the ghosting camera and nudge"). gsplat ships this as `pose_opt`
(`pose_opt_lr≈1e-5`, reg `≈1e-6`); tt-splat fixes poses from COLMAP today.

**Key tractability insight — it's HOST-side, no device-kernel change.** The cameras are host params, and the
gradient inputs are already read back:
- The resident loop already has, per Gaussian, **`dL/du`, `dL/dv`** = `grads2d["cx"]`, `grads2d["cy"]`
  (`server/device_resident.py`, Stage A output), and **`u, v, zc`** (read back in Stage B, `device_resident.py`
  `g(u_t)/g(v_t)/g(zc_t)`). From `u,v,zc` + intrinsics you reconstruct camera-space `mc = (xc,yc,zc)`.
- So `dL/d(pose)` is a **host-side sum**: `Σ_g J_gᵀ · [dL/du_g, dL/dv_g]`, where `J_g` (2×6) is the analytic
  pose Jacobian of the perspective projection. No `device_project_backward` change needed for the MVP.

**The analytic Jacobian** (per Gaussian, camera-space `mc=(xc,yc,zc)`, focal `fx,fy`):
- perspective: `∂u/∂mc = [fx/zc, 0, -fx·xc/zc²]`, `∂v/∂mc = [0, fy/zc, -fy·yc/zc²]`.
- translation: `∂mc/∂t = I` → `∂(u,v)/∂t` = the rows above.
- rotation (so(3) tangent ω): `∂mc/∂ω = -[mc]×` (skew) → chain with `∂(u,v)/∂mc`.
- Stack → `J_g` is 2×6 `[∂(u,v)/∂(ω,t)]`. (MVP uses the mean-projection term, which dominates; the
  conic-vs-pose coupling is a second-order refinement to add later.)

**Implementation.**
1. Per-camera 6-DoF correction `δ=(ω,t)` (init 0), stored host-side; apply as `R'=Exp(ω)·R_colmap`,
   `t'=t_colmap+t` when building the cam tuple each step.
2. Resident path: after Stage A, compute `dL/dδ` host-side from `grads2d` + reconstructed `mc` (above);
   accumulate per camera; step a small **host Adam** (lr~1e-3 on ω/t in these units; tune) with weight decay.
   **Host path: nearly free** — make `R,t` torch leaves and let autograd + the existing optimizer handle it.
3. Wire the interactive hook: a `pose_nudge` command (camera_name, δ) for the game's "grab the camera."
4. Knobs: `TT_POSE_OPT` (default off), `pose_opt_lr`, `pose_opt_reg`; only optimize after a warm-up
   (e.g. `pose_opt_from≈500`, like gsplat) so geometry settles first.

**Validation (gate).** (a) finite-difference grad-check `dL/dδ` vs a numerical perturbation of one camera's
pose on a fixed scene; (b) inject a known pose error into one corgi camera, confirm pose-opt **drives it back
toward zero** and PSNR for that view climbs above the fixed-pose baseline.

**Files.** `server/device_resident.py` (host-side pose-grad after Stage A), `server/train_tt.py` (pose state +
host Adam + the `pose_nudge`/`TT_POSE_OPT` wiring), `docs/pathclear/train_real.py` (host-path autograd poses).

---

## #2 — Progressive-SH warmup

Ramp the **effective** SH degree `0→deg` over the run (gsplat `sh_degree_interval≈1000`) so high-freq
view-dependent colour isn't fit before geometry settles. **First verify** `device_project.project_color` and
`device_project_backward.project_backward` accept an effective degree `< K` (the `sh` tensor stays full-size;
only the first `(deg_eff+1)²` bands contribute). Then thread `deg_eff = min(deg, nstep//sh_interval)` through
forward+backward in `device_resident.step`. Knob `TT_SH_WARMUP` (steps/band). **Gate:** converges; final PSNR
≥ no-warmup on a longer (~2–3k-step) run (the benefit only shows past the first band bump).

## #3 — Anti-aliasing (Mip-Splatting opacity compensation)

tt-splat applies the fixed `0.3·I` EWA dilation but not the AA opacity factor. Multiply each Gaussian's
opacity by `sqrt(det(Σ2D_raw) / det(Σ2D_dilated))` (≤1, shrinks sub-pixel splats). Add it where the 2D conic
is formed: `server/device_project.py:project_geom` (device) and `docs/pathclear/train_real.py:project_general`
(host) — both already compute `Σ2D` with and without the `+0.3I`, so it's one extra scalar per Gaussian into
`op`. Knob `TT_AA` (default off). **Gate:** PSNR unchanged at the training resolution; render a 2× and 0.5×
view and confirm reduced aliasing (the AA benefit is multi-scale).

## #4 — Scene-scale-aware LR

Compute `scene_scale` once at init (radius of the camera centres `-Rᵀt` about their centroid) and scale the
**mean** LR by it so the same config transfers across captures of different extent. Apply **conservatively**
(normalize so the corgi's tuned `.01` is ~unchanged) since it can't be validated on one scene. `server/train_tt.py`
init. **Gate:** corgi PSNR unchanged; document the cross-scene intent.

## #5 — Densify PSNR-gate (finish the safeguarded auto-densify)

Auto-densify is wired + default-OFF after it fogged; the safeguards (scale-prune in `densify_3d`,
opacity-reset cadence) are in but **not outcome-validated**. Run baseline vs `TT_DENSIFY=1` on a longer corgi
run (≥2k steps), confirm densify **raises** final PSNR (not fogs). Tune `densify_from` (later warm-up),
`prune_big_mult`, `opacity_reset_every`. Only then flip the default / recommend it. The **interactive bubble
gun + eraser already sidestep this** (human-driven density control) and are validated.

---

## Sequencing
**#1 pose-opt** (priority; host-side, grad-checked) → **#2 progressive-SH** + **#4 scene-scale** (cheap
schedules) → **#3 AA** (multi-scale) → **#5 densify-gate** (finish the auto path). Cross-ref the broader
roadmap: [`ROADMAP.md`](ROADMAP.md) (NOW/NEXT/LATER) and the game-engine sibling [`GAME_ENGINE_PLAN.md`](GAME_ENGINE_PLAN.md)
(pose-opt is also the game's interactive-camera feature).
