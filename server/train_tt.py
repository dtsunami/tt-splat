#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
ttgs training stage for the TENSTORRENT/BLACKHOLE backend.

Drop-in for ttgs.stages.train.run — same signature + same TrainingController contract,
so the existing ttgs FastAPI dashboard (Render|GT|Diff, prune/densify/clamp controls,
pause/stop, live metrics) drives our tt-splat pipeline unchanged.

v1 runs the host-reference render+optimize (the verified tt-splat math; device kernels —
SFPU blend-loop M5, scatter-add M2 — are validated separately and slot in as the perf path).
Loads the COLMAP model via pycolmap (binary or text), inits Gaussians from sparse points,
writes a standard 3DGS splat.ply that ttgs export/view consume.
"""
from __future__ import annotations
import sys, math, os
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "pathclear"))
from train_real import (project_general, render, init_from_points, load_image, load_mask,
                         tensors, sh_dim, C0)  # verified SH render + masks
sys.path.insert(0, str(Path(__file__).resolve().parent))          # server/ (loss, densify, device_*)
from loss import image_loss                                       # L1 + D-SSIM training loss
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # vendored ttgs package
from ttgs.viewer.dashboard import build_update

HOST_MAX_POINTS = int(os.environ.get("TT_MAX_POINTS", "1200"))   # host-render budget
HOST_SIZE = int(os.environ.get("TT_SIZE", "96"))
USE_DEVICE_RENDER = int(os.environ.get("TT_DEVICE_RENDER", "0"))  # render dashboard frame ON the Blackhole (M14)
USE_DEVICE_TRAIN = int(os.environ.get("TT_DEVICE_TRAIN", "0"))    # fwd+bwd GRADIENTS on the Blackhole (Phase 2a)
USE_DEVICE_RESIDENT = int(os.environ.get("TT_DEVICE_RESIDENT", "0"))  # FULL device-resident loop (B/raster/A/D/C + Adam on-device)
_render_device = None                                             # lazy: imports ttnn only when enabled
_render_train = None

if USE_DEVICE_TRAIN:    # M14 fused forward + culled chunked fused backward → scales past the old 16/64 cap
    print(f"device TRAIN enabled — M14 fused forward + culled fused backward, "
          f"{HOST_MAX_POINTS} Gaussians @ {HOST_SIZE}px (backward is host-tile-looped; lower if too slow)")


def _train_render(P, cam, H, W, PX, PY):
    """Training render feeding the loss. ON-DEVICE differentiable fwd+bwd (M16 bridge) when
    TT_DEVICE_TRAIN=1, else host autograd. Falls back to host (loudly) on device error."""
    global _render_train, USE_DEVICE_TRAIN
    if USE_DEVICE_TRAIN:
        try:
            if _render_train is None:
                from device_raster import render_train as _rt
                _render_train = _rt
                print("device TRAIN: ON — fwd+bwd gradients on the Blackhole (correctness path)")
            return _render_train(P, cam, H, W)        # [H,W,3] float32, differentiable wrt P
        except Exception as exc:
            print(f"device train failed ({type(exc).__name__}: {exc}) — falling back to host autograd")
            USE_DEVICE_TRAIN = 0
    return render(P, cam, H, W, PX, PY, aa=bool(AA_ON))   # host float64, differentiable (+#3 AA)


def _display_render(P, cam, H, W, PX, PY):
    """Dashboard display frame. Renders ON-DEVICE (M14 rasterizer) when TT_DEVICE_RENDER=1, else host.
    Falls back to host on any device error so the dashboard never breaks. Training math is unaffected
    either way (this is the no-grad display path; gradients still use the host render)."""
    global _render_device, USE_DEVICE_RENDER
    if USE_DEVICE_RENDER:
        try:
            if _render_device is None:
                from render_device import render_device as _rd   # server/ already on sys.path
                _render_device = _rd
                print("device render: ON — dashboard frames rendered on the Blackhole (M14)")
            return _render_device(P, cam, H, W)
        except Exception as exc:
            print(f"device render failed ({type(exc).__name__}: {exc}) — falling back to host render")
            USE_DEVICE_RENDER = 0
    return render(P, cam, H, W, PX, PY, aa=bool(AA_ON)).clamp(0, 1).float().cpu().numpy()


def _load_colmap(dataset_dir: Path):
    import pycolmap
    model = dataset_dir / "sparse" / "0"
    if not model.exists():
        model = dataset_dir / "sparse"
    rec = pycolmap.Reconstruction(str(model))
    cams, names = [], []
    for _id, img in rec.images.items():
        cam = rec.cameras[img.camera_id]
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        Rv = torch.tensor(np.array(cfw.rotation.matrix()), dtype=torch.float64)
        tv = torch.tensor(np.array(cfw.translation), dtype=torch.float64)
        cams.append((Rv, tv, cam.focal_length_x, cam.focal_length_y,
                     cam.principal_point_x, cam.principal_point_y, img.name))
    xyz = torch.tensor(np.array([p.xyz for p in rec.points3D.values()]), dtype=torch.float64)
    rgb = torch.tensor(np.array([p.color for p in rec.points3D.values()]), dtype=torch.float64)
    return cams, xyz, rgb


def _write_ply(path: Path, P):
    """Standard 3DGS PLY with SH: f_dc (band 0) + f_rest (bands 1..) in channel-major order."""
    with torch.no_grad():
        xyz = P["mean"].detach().cpu().numpy().astype(np.float32)
        sh = P["sh"].detach()                                  # [N,K,3]
        f_dc = sh[:, 0].cpu().numpy().astype(np.float32)       # [N,3]
        rest = sh[:, 1:]                                       # [N,K-1,3]
        f_rest = rest.permute(0, 2, 1).reshape(rest.shape[0], -1).cpu().numpy().astype(np.float32)  # channel-major
        opacity = P["op"].detach().cpu().numpy().astype(np.float32).reshape(-1, 1)  # logit (3DGS raw)
        scale = P["scale"].detach().cpu().numpy().astype(np.float32)                # log-scale
        q = P["quat"].detach()
        q = (q / q.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)     # wxyz
    N = xyz.shape[0]; nrest = f_rest.shape[1]
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    props += [f"f_rest_{i}" for i in range(nrest)]
    props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {N}"] + \
          [f"property float {p}" for p in props] + ["end_header"]
    rows = np.concatenate([xyz, np.zeros((N, 3), np.float32), f_dc, f_rest, opacity, scale, q],
                          axis=1).astype(np.float32)
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode()); f.write(rows.tobytes())


_LIVE_CFG_INT = ("iterations", "dashboard_every", "snapshot_every", "save_every",
                 "densify_from", "densify_until", "densify_every", "opacity_reset_every",
                 "pose_opt", "pose_opt_from",                   # #1 pose-opt enable + warm-up (live)
                 "sh_warmup", "aa", "scene_scale_lr",           # #2/#3/#4 (stored live; applied at run start)
                 "densify")                                     # #5 auto-densify on/off (live)
_LIVE_CFG_FLOAT = ("densify_grad_threshold", "lambda_dssim",
                   "pose_opt_lr", "pose_opt_reg")               # #1 pose-opt lr + reg (live)


def _apply_config(cfg, cmd, dashboard=None):
    """Apply the recognized update_config fields the TT loop honors live (training schedule + densify +
    loss/opacity-reset cadence + pose-opt). Both train paths honor these live now."""
    changed = {}
    for k in _LIVE_CFG_INT:
        if k in cmd:
            try:
                changed[k] = int(cmd[k]); setattr(cfg, k, changed[k])
            except (TypeError, ValueError):
                pass
    for k in _LIVE_CFG_FLOAT:
        if k in cmd:
            try:
                changed[k] = float(cmd[k]); setattr(cfg, k, changed[k])
            except (TypeError, ValueError):
                pass
    if "snapshot_every" in changed and dashboard is not None:
        dashboard._snapshot_every = changed["snapshot_every"]   # so live snapshot-interval edits take effect
    if changed and dashboard is not None:                       # re-publish so GET /config (the chips) tracks live edits
        try:
            from dataclasses import asdict
            dashboard.set_config(asdict(cfg))
        except Exception:
            pass
    return changed


LR_DECAY = float(os.environ.get("TT_LR_DECAY", "0.01"))      # exp-decay the mean LR to this fraction over the run (>=1 = off)
DENSIFY_ON = int(os.environ.get("TT_DENSIFY", "1"))          # adaptive clone/split/prune — cfg-driven (default ON)
DENSIFY_MAX = int(os.environ.get("TT_DENSIFY_MAX", "100000"))  # hard cap on the Gaussian count

# ---- gsplat training-recipe gaps (docs/RECIPE_GAPS_PLAN.md). All default to the prior behavior. ----
# #1 pose-opt (gsplat pose_opt): now LIVE cfg fields (pose_opt / pose_opt_lr / pose_opt_reg / pose_opt_from)
# tunable from the dashboard. TT_POSE_OPT* env vars still seed the initial cfg — see _seed_cfg_from_env.
# #2/#3/#4 now default ON and are cfg fields (sh_warmup / aa / scene_scale_lr); these module globals are the
# import-time defaults (env overrides) and get re-set from cfg at run start (_seed_cfg_from_env + override).
SH_WARMUP = int(os.environ.get("TT_SH_WARMUP", "1000"))     # #2 steps per SH band (0 = full degree from step 1)
AA_ON = int(os.environ.get("TT_AA", "1"))                   # #3 Mip-Splatting anti-alias opacity compensation
SCENE_SCALE_LR = int(os.environ.get("TT_SCENE_SCALE_LR", "1"))   # #4 scale mean LR by scene extent
SCENE_SCALE_REF = float(os.environ.get("TT_SCENE_SCALE_REF", "3.886028"))  # ref extent (corgi) -> factor 1.0
# NOTE: auto-densify is EXPERIMENTAL — without the paired opacity-reset cadence + scale-prune that real
# 3DGS uses, clone/split accumulates floaters into fog. Default OFF; opt in with TT_DENSIFY=1 or the manual
# Densify button. Proper fix (opacity-reset pairing + scale prune + PSNR-on-real-data gate) is pending.


def _densify_window(cfg):
    """(from, until, every) for the auto-densify schedule, from cfg with 3DGS-ish defaults."""
    return (int(getattr(cfg, "densify_from", 500)),
            int(getattr(cfg, "densify_until", 15000)),
            max(1, int(getattr(cfg, "densify_every", 100))))


def _seed_cfg_from_env(cfg):
    """Seed the recipe-gap cfg fields from their TT_* env vars when explicitly set, so an env-launched
    run shows the right values in the dashboard AND stays editable from there. An unset env leaves the
    cfg/TOML default untouched (so the new defaults — pose-opt off; SH-warmup/AA/scene-scale ON — hold)."""
    for name, env, cast in (("pose_opt", "TT_POSE_OPT", int), ("pose_opt_lr", "TT_POSE_OPT_LR", float),
                            ("pose_opt_reg", "TT_POSE_OPT_REG", float), ("pose_opt_from", "TT_POSE_OPT_FROM", int),
                            ("sh_warmup", "TT_SH_WARMUP", int), ("aa", "TT_AA", int),       # #2 / #3
                            ("scene_scale_lr", "TT_SCENE_SCALE_LR", int),                   # #4
                            ("densify", "TT_DENSIFY", int)):                                # #5 auto-densify
        if env in os.environ:
            try: setattr(cfg, name, cast(os.environ[env]))
            except (TypeError, ValueError): pass


def _sync_pose(pose, cfg, n_views):
    """(Re)build / retune the per-camera PoseOptimizer from the live cfg.pose_opt* fields.
    Returns (pose_or_None, pose_from). Toggling pose_opt off drops corrections; toggling on (re)builds
    from zero; lr/reg edits retune in place. Shared by both train paths."""
    on  = int(getattr(cfg, "pose_opt", 0) or 0)
    lr  = float(getattr(cfg, "pose_opt_lr", 1e-3) or 1e-3)
    reg = float(getattr(cfg, "pose_opt_reg", 1e-4) or 0.0)
    frm = int(getattr(cfg, "pose_opt_from", 0) or 0)
    if not on:
        return None, frm
    if pose is None:
        from pose_opt import PoseOptimizer
        pose = PoseOptimizer(n_views, lr=lr, reg=reg)
    else:                                                       # retune in place (keep accumulated δ)
        pose.reg = reg
        for g in pose.opt.param_groups: g["lr"] = lr
    return pose, frm


def _densify_params(P, gpos, gacc, cfg, tensors_fn):
    """Run clone/split/prune on host P (torch leaves) using the accumulated positional grad.
    Returns (new re-leafed P, stats). Caller rebuilds the optimizer + resets the accumulator."""
    from densify import densify_3d
    g = (gpos / max(int(gacc), 1)).float()
    newP, st = densify_3d({k: P[k].detach() for k in P}, g,
                          grad_threshold=float(getattr(cfg, "densify_grad_threshold", 0.0) or 0.0),
                          n_max=DENSIFY_MAX)
    for k in tensors_fn(newP):
        newP[k] = newP[k].requires_grad_()
    return newP, st


def _project_uvz(P, cam):
    """Project current Gaussians to screen (u,v) + camera depth zc for the given camera. Numpy out."""
    Rv, tv, fx, fy, ppx, ppy = cam[:6]
    u, v, zc, _ = project_general(P, Rv, tv, fx, fy, ppx, ppy)
    return (u.detach().cpu().numpy().astype(np.float64),
            v.detach().cpu().numpy().astype(np.float64),
            zc.detach().cpu().numpy().astype(np.float64))


def _unproject_spawn(u, v, zc, cam, points, gt, n_per=3, brush=4.0):
    """Bubble gun: turn screen click points into NEW world-space Gaussian means + colours.
    Depth at each point = median camera-depth of nearby existing splats (local surface), else the
    scene-median depth (empty region -> SfM-seeding). Colour = the ground-truth photo at that pixel."""
    Rv, tv, fx, fy, ppx, ppy = cam[:6]
    Rv = np.asarray(Rv, np.float64); tv = np.asarray(tv, np.float64).reshape(3)
    fx, fy, ppx, ppy = float(fx), float(fy), float(ppx), float(ppy)
    g = gt.detach().cpu().numpy() if torch.is_tensor(gt) else np.asarray(gt)
    H, W = g.shape[:2]
    valid = zc > 1e-4
    zmed = float(np.median(zc[valid])) if valid.any() else 2.0
    R2 = (brush * 4.0) ** 2
    means, cols = [], []
    for p in points:
        px, py = float(p[0]), float(p[1])
        near = valid & ((u - px) ** 2 + (v - py) ** 2 < R2)
        z = float(np.median(zc[near])) if near.any() else zmed
        for _ in range(max(1, int(n_per))):
            jx, jy = px + np.random.randn() * brush, py + np.random.randn() * brush
            xc, yc = (jx - ppx) * z / fx, (jy - ppy) * z / fy
            means.append(Rv.T @ (np.array([xc, yc, z]) - tv))          # camera -> world
            iy, ix = int(np.clip(jy, 0, H - 1)), int(np.clip(jx, 0, W - 1))
            cols.append(np.asarray(g[iy, ix], np.float64).reshape(-1)[:3])
    return np.asarray(means, np.float64), np.asarray(cols, np.float64)


def _select_region(u, v, zc, points, brush):
    """Boolean mask of Gaussians whose screen projection falls under any brush point (eraser stroke)."""
    valid = zc > 1e-4
    sel = np.zeros(u.shape[0], dtype=bool)
    R2 = float(brush) ** 2
    for p in points:
        px, py = float(p[0]), float(p[1])
        sel |= valid & ((u - px) ** 2 + (v - py) ** 2 < R2)
    return sel


def run(dataset_dir, output_dir, cfg, backend, resume=False, viewer_port=None,
        dashboard=None, masks_dir=None, excluded=None) -> Path:
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = dataset_dir / "images"
    if not img_dir.exists(): img_dir = dataset_dir

    cams, xyz, rgb = _load_colmap(dataset_dir)
    excluded = excluded or set()
    if HOST_MAX_POINTS and xyz.shape[0] > HOST_MAX_POINTS:
        idx = torch.randperm(xyz.shape[0])[:HOST_MAX_POINTS]; xyz, rgb = xyz[idx], rgb[idx]
    _tgt = int(os.environ.get("TT_TARGET_POINTS", "0"))   # densify seed UP to N (jittered replication) for scale tests
    if _tgt > xyz.shape[0]:
        rep = (_tgt + xyz.shape[0] - 1) // xyz.shape[0]
        jit = torch.randn(xyz.shape[0] * rep, 3) * float(xyz.std(0).mean()) * 0.01
        xyz = (xyz.repeat(rep, 1) + jit)[:_tgt]; rgb = rgb.repeat(rep, 1)[:_tgt]
        print(f"densified seed -> {xyz.shape[0]} Gaussians (TT_TARGET_POINTS={_tgt})")

    # load + downscale images; scale intrinsics; load per-image masks (frames.json -> PNGs)
    views, targets, masks = [], [], []
    for c in cams:
        if c[6] in excluded: continue
        p = img_dir / c[6]
        if not p.exists(): continue
        img, s = load_image(str(p), HOST_SIZE); H, W, _ = img.shape
        views.append((c[0], c[1], c[2]*s, c[3]*s, c[4]*s, c[5]*s, c[6])); targets.append(img)
        mp = (Path(masks_dir) / (Path(c[6]).stem + ".png")) if masks_dir else None
        masks.append(load_mask(str(mp), H, W) if mp else None)
    if not views:
        raise RuntimeError(f"no images found under {img_dir}")
    H, W, _ = targets[0].shape
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    PX, PY = jj.double(), ii.double()

    name2idx = {v[6]: i for i, v in enumerate(views)}
    def _reload_mask(nm):
        """Re-read one image's mask from disk so live mask edits reach the next training step."""
        if not masks_dir or not nm:
            return
        i = name2idx.get(nm)
        if i is None:
            for k, j in name2idx.items():
                if Path(k).name == Path(nm).name:
                    i = j; break
        if i is None:
            return
        mp = Path(masks_dir) / (Path(nm).stem + ".png")
        masks[i] = load_mask(str(mp), H, W) if mp.exists() else None

    P = init_from_points(xyz, rgb, sh_degree=int(getattr(cfg, "sh_degree", 3)))
    for k in tensors(P): P[k].requires_grad_()
    lr = {"mean": .01, "scale": .01, "quat": .01, "op": .02, "sh": .01}
    _seed_cfg_from_env(cfg)      # env (TT_*) seeds the live cfg; the dashboard can retune from here
    global SH_WARMUP, AA_ON, SCENE_SCALE_LR, DENSIFY_ON  # #2/#3/#4/#5 are cfg-driven now; override the import-time
    SH_WARMUP = int(getattr(cfg, "sh_warmup", SH_WARMUP) or 0)         # globals from this run's cfg so every
    AA_ON = int(getattr(cfg, "aa", AA_ON) or 0)                        # use-site (incl. _train_render) sees it
    SCENE_SCALE_LR = int(getattr(cfg, "scene_scale_lr", SCENE_SCALE_LR) or 0)
    DENSIFY_ON = int(getattr(cfg, "densify", DENSIFY_ON) or 0)         # #5 auto-densify (live-toggleable below)
    if SCENE_SCALE_LR:           # #4 scale the mean LR by scene extent so one config transfers across captures
        _ctrs = np.stack([(-np.asarray(v[0], np.float64).T @ np.asarray(v[1], np.float64).reshape(3))
                          for v in views])
        _scale = float(np.linalg.norm(_ctrs - _ctrs.mean(0), axis=1).mean())   # radius of cam centres
        _factor = _scale / SCENE_SCALE_REF if SCENE_SCALE_REF > 0 else 1.0
        lr["mean"] = lr["mean"] * _factor
        print(f"scene-scale LR: scene_scale={_scale:.4f} ref={SCENE_SCALE_REF:.4f} -> mean LR x{_factor:.3f} "
              f"= {lr['mean']:.5f}")
    def make_opt(): return torch.optim.Adam([{"params": [P[k]], "lr": lr[k]} for k in tensors(P)])
    opt = make_opt()
    pose, _pose_from = _sync_pose(None, cfg, len(views))   # #1 trainable per-camera 6-DoF extrinsic correction
    if pose is not None:
        print(f"camera pose-opt: ON — {len(views)} cams, lr={getattr(cfg,'pose_opt_lr')} "
              f"reg={getattr(cfg,'pose_opt_reg')} from step {_pose_from}")
    psnr = lambda a, b: 10*math.log10(max(float(b.max()), 1e-6)**2/max(float(((a-b)**2).mean()), 1e-12))
    if dashboard is not None:
        from dataclasses import asdict
        try: dashboard.set_config(asdict(cfg))
        except Exception: pass

    total = int(getattr(cfg, "iterations", 1000))
    focus = None; ply = output_dir / "splat.ply"

    if USE_DEVICE_RESIDENT:    # full device-resident loop: B/raster/A/D/C + Adam on the Blackhole, per-stage telemetry
        from render_device import _device
        from device_resident import DeviceResidentTrainer
        dev = _device()
        trainer = DeviceResidentTrainer(dev, P, lr=lr, deg=int(getattr(cfg, "sh_degree", 3)),
                                        lambda_dssim=float(getattr(cfg, "lambda_dssim", 0.2) or 0.0),
                                        sh_interval=SH_WARMUP, aa=bool(AA_ON))
        ngauss = P["mean"].shape[0]
        print(f"device RESIDENT train: ON — B/raster/A/D/C + Adam on the Blackhole, params resident "
              f"({ngauss} Gaussians @ {HOST_SIZE}px); per-stage timings stream to the dashboard")
        if SH_WARMUP > 0:
            print(f"progressive-SH warmup: ON — +1 band every {SH_WARMUP} steps up to deg {trainer.deg}")
        if AA_ON:
            print("anti-aliasing: ON — Mip-Splatting opacity compensation (sub-pixel splats shrink)")
        _df, _du, _de = _densify_window(cfg)
        _dgt = float(getattr(cfg, "densify_grad_threshold", 0.0) or 0.0)
        _ore = int(getattr(cfg, "opacity_reset_every", 0) or 0)   # floater-cull cadence (safeguard)
        if DENSIFY_ON:
            print(f"densification ON (resident): from {_df} until {_du} every {_de}, cap {DENSIFY_MAX}, "
                  f"opacity-reset {_ore}")
        _lrm0 = trainer.adam.lr["mean"]                            # base mean LR for exp decay
        for step in range(1, total + 1):
            every = max(1, int(getattr(cfg, "dashboard_every", 25)))
            trainer.lambda_dssim = float(getattr(cfg, "lambda_dssim", 0.2) or 0.0)   # live D-SSIM weight
            if LR_DECAY < 1.0:                                     # exp LR decay on means (gsplat-style)
                trainer.adam.lr["mean"] = _lrm0 * (LR_DECAY ** (step / max(total, 1)))
            if dashboard is not None:
                if dashboard.should_stop: break
                dashboard.wait_if_paused()
                for cmd in dashboard.drain_commands():
                    t = cmd.get("type"); detail = ""
                    if t == "focus_camera":
                        focus = cmd.get("camera_name")
                    elif t == "save":
                        _write_ply(ply, trainer.params_host())
                    elif t == "update_config":
                        ch = _apply_config(cfg, cmd, dashboard)
                        if "iterations" in ch: total = ch["iterations"]
                        if any(k.startswith("densify") for k in ch):
                            _df, _du, _de = _densify_window(cfg); _dgt = float(getattr(cfg, "densify_grad_threshold", 0.0) or 0.0)
                        if "opacity_reset_every" in ch: _ore = int(getattr(cfg, "opacity_reset_every", 0) or 0)
                        if "densify" in ch: DENSIFY_ON = int(getattr(cfg, "densify", 0) or 0)   # #5 live on/off
                        if any(k.startswith("pose_opt") for k in ch):   # #1 (re)build/retune pose-opt live
                            pose, _pose_from = _sync_pose(pose, cfg, len(views))
                        detail = ",".join(f"{k}={v}" for k, v in ch.items())
                    elif t == "pose_nudge":                        # interactive "grab the camera"
                        _ix = name2idx.get(cmd.get("camera_name"))
                        if pose is not None and _ix is not None:
                            pose.nudge(_ix, cmd.get("omega", (0, 0, 0)), cmd.get("trans", (0, 0, 0)))
                            detail = f"{cmd.get('camera_name')} dω={cmd.get('omega')} dt={cmd.get('trans')}"
                    elif t == "reload_masks":
                        _reload_mask(cmd.get("image_name")); detail = cmd.get("image_name", "")
                    elif t == "prune":
                        ngauss = trainer.prune(float(cmd.get("threshold", 0.005))); detail = f"-> {ngauss}"
                    elif t == "reset_opacities":
                        trainer.reset_opacities()
                    elif t == "clamp_scale":
                        trainer.clamp_scale(float(cmd.get("max_log_scale", 2.5)))
                    elif t == "set_lr":
                        trainer.set_lr(float(cmd.get("lr_factor", 1.0)))
                    elif t == "densify_now":
                        _st = trainer.densify(grad_threshold=_dgt, n_max=DENSIFY_MAX); ngauss = trainer.N
                        detail = f"clone={_st['clone']} split={_st['split']} prune={_st['prune']} -> {ngauss}"
                        print(f"  [resident densify_now] {detail}")
                    elif t == "splat_spawn":                       # bubble gun
                        _ix = name2idx.get(cmd.get("camera_name"))
                        if _ix is not None and cmd.get("points"):
                            _u, _v, _zc = _project_uvz(trainer.params_host(), views[_ix])
                            _mns, _cls = _unproject_spawn(_u, _v, _zc, views[_ix], cmd["points"], targets[_ix],
                                                          int(cmd.get("n_per", 3)), float(cmd.get("brush", 4.0)))
                            if len(_mns):
                                ngauss = trainer.spawn(_mns, _cls); detail = f"+{len(_mns)} -> {ngauss}"
                                print(f"  [resident bubble-gun] +{len(_mns)} -> {ngauss}")
                    elif t == "cull_region":                       # bubble gun: eraser
                        _ix = name2idx.get(cmd.get("camera_name"))
                        if _ix is not None and cmd.get("points"):
                            _u, _v, _zc = _project_uvz(trainer.params_host(), views[_ix])
                            _sel = _select_region(_u, _v, _zc, cmd["points"], float(cmd.get("brush", 6.0)))
                            if _sel.any():
                                ngauss = trainer.cull(~_sel); detail = f"-{int(_sel.sum())} -> {ngauss}"
                                print(f"  [resident eraser] -{int(_sel.sum())} -> {ngauss}")
                    dashboard.log_command(t, step, detail, {"n_gaussians": ngauss})

            vi = (step - 1) % len(views)
            if focus is not None:
                cand = [i for i, v in enumerate(views) if v[6] == focus]; vi = cand[0] if cand else vi
            cam, gt, pm = views[vi], targets[vi], masks[vi]
            mask = (pm.detach().cpu().numpy() if torch.is_tensor(pm) else np.asarray(pm)) if pm is not None else None
            gmask = dashboard.get_mask() if dashboard is not None else None
            if gmask is not None and np.asarray(gmask).shape[:2] == (H, W):
                gm = np.asarray(gmask, np.float64); mask = gm if mask is None else mask * gm
            cam_use = pose.corrected_cam(vi, cam) if pose is not None else cam   # #1 apply trainable extrinsics
            loss, img = trainer.step(cam_use, gt, mask)   # all param math on device; (loss, np[H,W,3])
            if pose is not None and step >= _pose_from and trainer.last_screen is not None:
                pose.resident_grad(vi, trainer.last_screen, cam_use)   # analytic dL/dδ from device 2D grads
                pose.step()
            if DENSIFY_ON and _df <= step <= _du and step % _de == 0:
                _st = trainer.densify(grad_threshold=_dgt, n_max=DENSIFY_MAX); ngauss = trainer.N
                print(f"  [densify auto@{step}] clone={_st['clone']} split={_st['split']} "
                      f"prune={_st['prune']} N {_st['n_before']}->{_st['n_after']}", flush=True)
            if DENSIFY_ON and _ore > 0 and _df < step <= _du and step % _ore == 0:
                trainer.reset_opacities()                    # safeguard: periodic floater cull
                print(f"  [opacity reset @{step}]", flush=True)

            if dashboard is not None and (step == 1 or step % every == 0):
                g = (gt.detach().cpu().numpy() if torch.is_tensor(gt) else np.asarray(gt)).astype(np.float32)
                g = np.clip(g, 0, 1); r = np.clip(img, 0, 1)
                _pc = None
                if pose is not None:                         # #1 surface this camera's pose correction magnitude
                    _dw, _dt = pose.magnitude(vi); _pc = {"dw_deg": math.degrees(_dw), "dt": _dt}
                dashboard.push_update(build_update(
                    step=step, total_steps=total, loss=loss, n_gaussians=ngauss,
                    camera_name=cam[6], render=r, gt=g, is_paused=dashboard.is_paused,
                    focus_camera=focus, l1=float(abs(r - g).mean()), mse=float(((r - g) ** 2).mean()),
                    psnr=psnr(torch.tensor(r), torch.tensor(g)), stage_timings=trainer.step_log[-1],
                    pose_corr=_pc))

            _sv = int(getattr(cfg, "save_every", 0) or 0)
            if _sv > 0 and step % _sv == 0:
                _write_ply(ply, trainer.params_host())
        _write_ply(ply, trainer.params_host())
        if getattr(trainer, "profiler", None) is not None:
            trainer.profiler.close()                 # teardown: shrink the profiler CSV back to its header
        print(f"device RESIDENT train done — {trainer.report()}")
        return ply

    # ---- host-path adaptive densification (clone/split/prune) ----
    _df, _du, _de = _densify_window(cfg)
    _dgt = float(getattr(cfg, "densify_grad_threshold", 0.0) or 0.0)
    _ore = int(getattr(cfg, "opacity_reset_every", 0) or 0)      # floater-cull cadence (safeguard)
    _deg_full = int(P["deg"])                                    # #2 full SH degree (P["deg"] ramps under warmup)
    gpos = torch.zeros(P["mean"].shape[0], dtype=torch.float64)   # accumulated positional grad (densify signal)
    gacc = 0
    if DENSIFY_ON:
        print(f"densification ON (host): from {_df} until {_du} every {_de}, cap {DENSIFY_MAX}, opacity-reset {_ore}")
    if SH_WARMUP > 0:
        print(f"progressive-SH warmup (host): ON — +1 band every {SH_WARMUP} steps up to deg {_deg_full}")
    if AA_ON:
        print("anti-aliasing (host): ON — Mip-Splatting opacity compensation")

    for step in range(1, total + 1):
        if LR_DECAY < 1.0:                                       # exp LR decay on means (gsplat-style)
            for _g in opt.param_groups:
                if _g["params"] and _g["params"][0] is P["mean"]:
                    _g["lr"] = lr["mean"] * (LR_DECAY ** (step / max(total, 1))); break
        if dashboard is not None:
            if dashboard.should_stop: break
            dashboard.wait_if_paused()
            for cmd in dashboard.drain_commands():
                t = cmd.get("type"); rebuilt = False; detail = ""
                with torch.no_grad():
                    if t == "reset_opacities":
                        P["op"].fill_(float(torch.logit(torch.tensor(0.01)))); rebuilt = True
                    elif t == "prune":
                        keep = torch.sigmoid(P["op"]) > float(cmd.get("threshold", 0.02))
                        if keep.any():
                            for k in tensors(P): P[k] = P[k][keep].clone().requires_grad_()
                            rebuilt = True
                        detail = f"-> {int(P['mean'].shape[0])}"
                    elif t == "clamp_scale":
                        P["scale"].clamp_(max=float(cmd.get("max_log_scale", 2.0)))
                    elif t == "set_lr":
                        for g in opt.param_groups: g["lr"] *= float(cmd.get("lr_factor", 1.0))
                    elif t == "focus_camera":
                        focus = cmd.get("camera_name")
                    elif t == "save":
                        _write_ply(ply, P)
                    elif t == "reload_masks":
                        _reload_mask(cmd.get("image_name")); detail = cmd.get("image_name", "")
                    elif t == "update_config":
                        ch = _apply_config(cfg, cmd, dashboard)
                        if "iterations" in ch: total = ch["iterations"]
                        if any(k.startswith("densify") for k in ch):
                            _df, _du, _de = _densify_window(cfg); _dgt = float(getattr(cfg, "densify_grad_threshold", 0.0) or 0.0)
                        if "opacity_reset_every" in ch: _ore = int(getattr(cfg, "opacity_reset_every", 0) or 0)
                        if "densify" in ch: DENSIFY_ON = int(getattr(cfg, "densify", 0) or 0)   # #5 live on/off
                        if any(k.startswith("pose_opt") for k in ch):   # #1 (re)build/retune pose-opt live
                            pose, _pose_from = _sync_pose(pose, cfg, len(views))
                        detail = ",".join(f"{k}={v}" for k, v in ch.items())
                    elif t == "pose_nudge":                        # interactive "grab the camera"
                        _ix = name2idx.get(cmd.get("camera_name"))
                        if pose is not None and _ix is not None:
                            pose.nudge(_ix, cmd.get("omega", (0, 0, 0)), cmd.get("trans", (0, 0, 0)))
                            detail = f"{cmd.get('camera_name')} dω={cmd.get('omega')} dt={cmd.get('trans')}"
                    elif t == "densify_now":
                        P, _st = _densify_params(P, gpos, gacc, cfg, tensors)
                        opt = make_opt(); gpos = torch.zeros(P["mean"].shape[0], dtype=torch.float64); gacc = 0
                        detail = f"clone={_st['clone']} split={_st['split']} prune={_st['prune']} -> {_st['n_after']}"
                        print(f"  [host densify_now] {detail}")
                    elif t == "splat_spawn":                       # bubble gun
                        _ix = name2idx.get(cmd.get("camera_name"))
                        if _ix is not None and cmd.get("points"):
                            _u, _v, _zc = _project_uvz({k: P[k].detach() for k in P}, views[_ix])
                            _mns, _cls = _unproject_spawn(_u, _v, _zc, views[_ix], cmd["points"], targets[_ix],
                                                          int(cmd.get("n_per", 3)), float(cmd.get("brush", 4.0)))
                            if len(_mns):
                                from densify import spawn_gaussians
                                P = spawn_gaussians({k: P[k].detach() for k in P},
                                                    torch.as_tensor(_mns), torch.as_tensor(_cls))
                                for k in tensors(P): P[k] = P[k].requires_grad_()
                                rebuilt = True; detail = f"+{len(_mns)} -> {P['mean'].shape[0]}"
                    elif t == "cull_region":                       # bubble gun: eraser
                        _ix = name2idx.get(cmd.get("camera_name"))
                        if _ix is not None and cmd.get("points"):
                            _u, _v, _zc = _project_uvz({k: P[k].detach() for k in P}, views[_ix])
                            _sel = _select_region(_u, _v, _zc, cmd["points"], float(cmd.get("brush", 6.0)))
                            if _sel.any() and int((~_sel).sum()) > 0:
                                _keep = torch.as_tensor(~_sel)
                                for k in tensors(P): P[k] = P[k][_keep].clone().requires_grad_()
                                rebuilt = True; detail = f"-{int(_sel.sum())} -> {P['mean'].shape[0]}"
                if rebuilt:
                    opt = make_opt()
                    gpos = torch.zeros(P["mean"].shape[0], dtype=torch.float64); gacc = 0   # N changed
                dashboard.log_command(t, step, detail, {"n_gaussians": P["mean"].shape[0]})

        vi = (step - 1) % len(views)
        if focus is not None:
            cand = [i for i, v in enumerate(views) if v[6] == focus]; vi = cand[0] if cand else vi
        cam, gt, pm = views[vi], targets[vi], masks[vi]
        if SH_WARMUP > 0:                                       # #2 ramp the effective SH degree 0->full
            P["deg"] = min(_deg_full, (step - 1) // SH_WARMUP)
        pose_active = pose is not None and step >= _pose_from   # #1 trainable extrinsics (autograd through cam)
        cam_use = pose.corrected_cam(vi, cam, differentiable=pose_active) if pose is not None else cam
        img = _train_render(P, cam_use, H, W, PX, PY)         # host autograd OR device fwd+bwd
        wmask = (pm if torch.is_tensor(pm) else torch.tensor(pm, dtype=torch.float64)) if pm is not None else None
        gmask = dashboard.get_mask() if dashboard is not None else None
        if gmask is not None:                                   # global mask (dashboard painter)
            m = torch.tensor(gmask, dtype=torch.float64)
            if m.shape[:2] == img.shape[:2]: wmask = m if wmask is None else wmask * m
        _lam = float(getattr(cfg, "lambda_dssim", 0.2) or 0.0)
        loss = image_loss(img, gt.to(img.dtype),                # L1 + D-SSIM (was pure MSE)
                          wmask.to(img.dtype) if wmask is not None else None, _lam)
        opt.zero_grad()
        if pose_active: pose.zero_grad()                        # clear δ.grad before this step's backward
        loss.backward(); opt.step()
        if pose_active: pose.step()                             # #1 step the per-camera 6-DoF correction
        with torch.no_grad():
            if P["mean"].grad is not None:
                gpos += (P["mean"].grad.detach().double() ** 2).sum(dim=1).sqrt()   # ‖∂L/∂mean‖ (3D proxy)
                gacc += 1
        if DENSIFY_ON and _df <= step <= _du and step % _de == 0:
            P, _st = _densify_params(P, gpos, gacc, cfg, tensors); opt = make_opt()
            gpos = torch.zeros(P["mean"].shape[0], dtype=torch.float64); gacc = 0
            print(f"  [densify auto@{step}] clone={_st['clone']} split={_st['split']} "
                  f"prune={_st['prune']} N {_st['n_before']}->{_st['n_after']}", flush=True)
        if DENSIFY_ON and _ore > 0 and _df < step <= _du and step % _ore == 0:
            with torch.no_grad():
                P["op"].fill_(float(torch.logit(torch.tensor(0.01))))   # safeguard: periodic floater cull
            opt = make_opt(); print(f"  [opacity reset @{step}]", flush=True)

        if dashboard is not None and (step == 1 or step % max(1, getattr(cfg, "dashboard_every", 25)) == 0):
            with torch.no_grad():
                # device-train already rendered on-device this step — reuse it (no second render)
                r = (img.detach().clamp(0, 1).float().cpu().numpy() if USE_DEVICE_TRAIN
                     else _display_render(P, cam_use, H, W, PX, PY))
                g = gt.clamp(0, 1).float().cpu().numpy()
                l = float(loss.detach())
            _pc = None
            if pose is not None:                             # #1 surface this camera's pose correction magnitude
                _dw, _dt = pose.magnitude(vi); _pc = {"dw_deg": math.degrees(_dw), "dt": _dt}
            dashboard.push_update(build_update(
                step=step, total_steps=total, loss=l, n_gaussians=P["mean"].shape[0],
                camera_name=cam[6], render=r, gt=g, is_paused=dashboard.is_paused,
                focus_camera=focus, l1=float(abs(r - g).mean()), mse=float(((r - g) ** 2).mean()),
                psnr=psnr(torch.tensor(r), torch.tensor(g)), pose_corr=_pc))

        _sv = int(getattr(cfg, "save_every", 0) or 0)
        if _sv > 0 and step % _sv == 0:
            _write_ply(ply, P)
    if SH_WARMUP > 0:
        P["deg"] = _deg_full                                    # restore full degree for export
    _write_ply(ply, P)
    return ply


if __name__ == "__main__":
    import argparse
    from ttgs.config import TrainConfig
    from ttgs.viewer.dashboard import TrainingController
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--output", default="work/tt_out")
    ap.add_argument("--steps", type=int, default=40)
    a = ap.parse_args()
    cfg = TrainConfig(); cfg.iterations = a.steps; cfg.dashboard_every = 10
    ctrl = TrainingController(output_dir=Path(a.output))
    # smoke: queue a prune mid-run to exercise the command path
    out = run(Path(a.dataset), Path(a.output), cfg, backend=None, dashboard=ctrl)
    hist = ctrl.get_history()
    print(f"TRAIN_TT wrote {out} ; dashboard updates={len(hist)} ; "
          f"last={hist[-1].get('step') if hist else None}")
