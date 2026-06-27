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
    return render(P, cam, H, W, PX, PY)               # host float64, differentiable


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
    return render(P, cam, H, W, PX, PY).clamp(0, 1).float().cpu().numpy()


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

    P = init_from_points(xyz, rgb, sh_degree=int(getattr(cfg, "sh_degree", 3)))
    for k in tensors(P): P[k].requires_grad_()
    lr = {"mean": .01, "scale": .01, "quat": .01, "op": .02, "sh": .01}
    def make_opt(): return torch.optim.Adam([{"params": [P[k]], "lr": lr[k]} for k in tensors(P)])
    opt = make_opt()
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
        trainer = DeviceResidentTrainer(dev, P, lr=lr, deg=int(getattr(cfg, "sh_degree", 3)))
        ngauss = P["mean"].shape[0]
        print(f"device RESIDENT train: ON — B/raster/A/D/C + Adam on the Blackhole, params resident "
              f"({ngauss} Gaussians @ {HOST_SIZE}px); per-stage timings stream to the dashboard")
        every = max(1, getattr(cfg, "dashboard_every", 25))
        for step in range(1, total + 1):
            if dashboard is not None:
                if dashboard.should_stop: break
                dashboard.wait_if_paused()
                for cmd in dashboard.drain_commands():
                    t = cmd.get("type")
                    if t == "focus_camera":
                        focus = cmd.get("camera_name")
                    elif t == "save":
                        _write_ply(ply, trainer.params_host())
                    elif t == "update_config" and "iterations" in cmd:
                        total = int(cmd["iterations"])
                    elif t in ("prune", "reset_opacities", "clamp_scale", "set_lr"):
                        print(f"  [resident] command '{t}' not supported in device-resident mode (skipped)")
                    dashboard.log_command(t, step, "", {"n_gaussians": ngauss})

            vi = (step - 1) % len(views)
            if focus is not None:
                cand = [i for i, v in enumerate(views) if v[6] == focus]; vi = cand[0] if cand else vi
            cam, gt, pm = views[vi], targets[vi], masks[vi]
            mask = (pm.detach().cpu().numpy() if torch.is_tensor(pm) else np.asarray(pm)) if pm is not None else None
            gmask = dashboard.get_mask() if dashboard is not None else None
            if gmask is not None and np.asarray(gmask).shape[:2] == (H, W):
                gm = np.asarray(gmask, np.float64); mask = gm if mask is None else mask * gm
            loss, img = trainer.step(cam, gt, mask)       # all param math on device; (loss, np[H,W,3])

            if dashboard is not None and (step == 1 or step % every == 0):
                g = (gt.detach().cpu().numpy() if torch.is_tensor(gt) else np.asarray(gt)).astype(np.float32)
                g = np.clip(g, 0, 1); r = np.clip(img, 0, 1)
                dashboard.push_update(build_update(
                    step=step, total_steps=total, loss=loss, n_gaussians=ngauss,
                    camera_name=cam[6], render=r, gt=g, is_paused=dashboard.is_paused,
                    focus_camera=focus, l1=float(abs(r - g).mean()), mse=loss,
                    ssim=psnr(torch.tensor(r), torch.tensor(g)), stage_timings=trainer.step_log[-1]))
        _write_ply(ply, trainer.params_host())
        print(f"device RESIDENT train done — {trainer.report()}")
        return ply

    for step in range(1, total + 1):
        if dashboard is not None:
            if dashboard.should_stop: break
            dashboard.wait_if_paused()
            for cmd in dashboard.drain_commands():
                t = cmd.get("type"); rebuilt = False
                with torch.no_grad():
                    if t == "reset_opacities":
                        P["op"].fill_(float(torch.logit(torch.tensor(0.01)))); rebuilt = True
                    elif t == "prune":
                        keep = torch.sigmoid(P["op"]) > float(cmd.get("threshold", 0.02))
                        if keep.any():
                            for k in tensors(P): P[k] = P[k][keep].clone().requires_grad_()
                            rebuilt = True
                    elif t == "clamp_scale":
                        P["scale"].clamp_(max=float(cmd.get("max_log_scale", 2.0)))
                    elif t == "set_lr":
                        for g in opt.param_groups: g["lr"] *= float(cmd.get("lr_factor", 1.0))
                    elif t == "focus_camera":
                        focus = cmd.get("camera_name")
                    elif t == "save":
                        _write_ply(ply, P)
                    elif t == "update_config":
                        if "iterations" in cmd: total = int(cmd["iterations"])
                if rebuilt: opt = make_opt()
                dashboard.log_command(t, step, "", {"n_gaussians": P["mean"].shape[0]})

        vi = (step - 1) % len(views)
        if focus is not None:
            cand = [i for i, v in enumerate(views) if v[6] == focus]; vi = cand[0] if cand else vi
        cam, gt, pm = views[vi], targets[vi], masks[vi]
        img = _train_render(P, cam, H, W, PX, PY)              # host autograd OR device fwd+bwd
        diff = (img - gt)**2                                   # torch promotes f32(device)/f64(gt) → f64
        if pm is not None: diff = diff * pm[..., None]          # per-image mask (frames.json)
        gmask = dashboard.get_mask() if dashboard is not None else None
        if gmask is not None:                                   # global mask (dashboard painter)
            m = torch.tensor(gmask, dtype=torch.float64)
            if m.shape[:2] == diff.shape[:2]: diff = diff * m[..., None]
        loss = diff.mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if dashboard is not None and (step == 1 or step % max(1, getattr(cfg, "dashboard_every", 25)) == 0):
            with torch.no_grad():
                # device-train already rendered on-device this step — reuse it (no second render)
                r = (img.detach().clamp(0, 1).float().cpu().numpy() if USE_DEVICE_TRAIN
                     else _display_render(P, cam, H, W, PX, PY))
                g = gt.clamp(0, 1).float().cpu().numpy()
                l = float(loss.detach())
            dashboard.push_update(build_update(
                step=step, total_steps=total, loss=l, n_gaussians=P["mean"].shape[0],
                camera_name=cam[6], render=r, gt=g, is_paused=dashboard.is_paused,
                focus_camera=focus, l1=float(abs(r - g).mean()), mse=l,
                ssim=psnr(torch.tensor(r), torch.tensor(g))))
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
