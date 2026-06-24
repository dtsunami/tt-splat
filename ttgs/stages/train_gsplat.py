"""Stage 3 (gsplat backend): Pure-Python 3DGS training using gsplat + PyTorch.

Runs natively on Intel Arc XPU via torch.device("xpu"). No external binary
or CUDA extension required — gsplat falls back to Python ops automatically.

Pipeline:
  1. Load COLMAP cameras.bin / images.bin / points3D.bin
  2. Initialise Gaussians from SfM point cloud
  3. Training loop: rasterize → L1+SSIM loss → backward → densify
  4. Save standard 3DGS .ply (compatible with export stage)
"""

from __future__ import annotations

import math
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from ttgs.backend.detect import Backend, BackendInfo
from ttgs.config import TrainConfig

console = Console()

# ─── COLMAP binary format constants ──────────────────────────────────────────

# model_id → (name, num_params)
_COLMAP_CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),   # f, cx, cy
    1: ("PINHOLE", 4),          # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL", 4),    # f, cx, cy, k
    3: ("RADIAL", 5),           # f, cx, cy, k1, k2
    4: ("OPENCV", 8),           # fx, fy, cx, cy, k1, k2, p1, p2
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Camera:
    camera_id: int
    model_id: int
    width: int
    height: int
    params: tuple  # (fx, fy, cx, cy) or similar


@dataclass
class TrainCamera:
    """Fully resolved camera ready for rasterization."""
    image_path: Path
    viewmat: torch.Tensor   # (4, 4) world-to-camera, float32
    K: torch.Tensor         # (3, 3) intrinsics, float32
    W: int
    H: int


# ─── COLMAP binary loaders ────────────────────────────────────────────────────

def _load_cameras_bin(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id = struct.unpack("<I", f.read(4))[0]      # uint32
            model_id = struct.unpack("<i", f.read(4))[0]    # int32
            width = struct.unpack("<Q", f.read(8))[0]        # uint64
            height = struct.unpack("<Q", f.read(8))[0]       # uint64
            if model_id not in _COLMAP_CAMERA_MODELS:
                raise ValueError(f"Unknown COLMAP camera model id: {model_id}")
            num_params = _COLMAP_CAMERA_MODELS[model_id][1]
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = Camera(cam_id, model_id, width, height, params)
    return cameras


def _load_images_bin(path: Path) -> list[tuple]:
    """Returns list of (image_id, qvec, tvec, camera_id, name)."""
    images = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)))   # qw qx qy qz
            tvec = np.array(struct.unpack("<3d", f.read(24)))
            camera_id = struct.unpack("<I", f.read(4))[0]
            # Read null-terminated name
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8", errors="replace")
            # Skip 2D point observations
            (num_pts2d,) = struct.unpack("<Q", f.read(8))
            f.read(num_pts2d * (8 + 8 + 8))  # x(f64) y(f64) pt3d_id(i64)
            images.append((image_id, qvec, tvec, camera_id, name))
    return images


def _load_points3d_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (xyz float64 [N,3], rgb uint8 [N,3])."""
    xyz_list = []
    rgb_list = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _pt_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            _error = struct.unpack("<d", f.read(8))[0]
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.read(track_len * 8)  # image_id(u32) + pt2d_idx(u32) each
            xyz_list.append(xyz)
            rgb_list.append(rgb)
    xyz = np.array(xyz_list, dtype=np.float64)
    rgb = np.array(rgb_list, dtype=np.uint8)
    return xyz, rgb


def _qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion (qw, qx, qy, qz) to 3×3 rotation matrix."""
    qw, qx, qy, qz = qvec / np.linalg.norm(qvec)
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def _camera_to_K(cam: Camera) -> np.ndarray:
    """Build 3×3 intrinsic matrix from COLMAP camera params."""
    p = cam.params
    model = _COLMAP_CAMERA_MODELS[cam.model_id][0]
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif model in ("SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    elif model in ("OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        # Fallback: treat first param as focal if single, first two as fx/fy
        fx = p[0]
        fy = p[1] if len(p) > 3 else p[0]
        cx = p[2] if len(p) > 2 else cam.width / 2
        cy = p[3] if len(p) > 3 else cam.height / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _load_colmap(dataset_dir: Path, images_dir: Path) -> list[TrainCamera]:
    """Load COLMAP sparse reconstruction and return a list of TrainCameras."""
    sparse = dataset_dir / "sparse"
    if not sparse.is_dir():
        sparse = dataset_dir  # dataset_dir IS the sparse dir

    cameras_bin = sparse / "cameras.bin"
    images_bin = sparse / "images.bin"
    if not cameras_bin.exists() or not images_bin.exists():
        raise FileNotFoundError(
            f"COLMAP binary files not found in {sparse}. "
            "Expected cameras.bin and images.bin."
        )

    cameras = _load_cameras_bin(cameras_bin)
    image_records = _load_images_bin(images_bin)

    train_cameras: list[TrainCamera] = []
    missing = 0
    for _image_id, qvec, tvec, cam_id, name in image_records:
        img_path = images_dir / name
        if not img_path.exists():
            # Try basename only (some datasets nest in subdirs)
            img_path = images_dir / Path(name).name
        if not img_path.exists():
            missing += 1
            continue

        cam = cameras[cam_id]
        R = _qvec2rotmat(qvec)
        # viewmat: world-to-camera [R|t] as 4×4
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R.astype(np.float32)
        viewmat[:3, 3] = tvec.astype(np.float32)

        K = _camera_to_K(cam).astype(np.float32)

        train_cameras.append(TrainCamera(
            image_path=img_path,
            viewmat=torch.from_numpy(viewmat),
            K=torch.from_numpy(K),
            W=cam.width,
            H=cam.height,
        ))

    if missing:
        console.print(f"[yellow]train[/] {missing} images referenced in images.bin not found in {images_dir}")

    return train_cameras


# ─── Gaussian initialization ─────────────────────────────────────────────────

_SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))


def _init_gaussians(
    xyz: np.ndarray,
    rgb: np.ndarray,
    sh_degree: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Initialise Gaussian parameters from SfM point cloud.

    Returns a dict of raw (pre-activation) parameter tensors:
      means:     (N, 3)  world positions
      scales:    (N, 3)  log(scale), uniform in all directions
      quats:     (N, 4)  normalised quaternions (1, 0, 0, 0)
      opacities: (N,)    logit(initial_opacity)
      sh0:       (N, 1, 3)  DC SH coefficients (from point colours)
      shN:       (N, K-1, 3) higher-order SH coefficients, zeroed
    """
    N = len(xyz)
    means_t = torch.from_numpy(xyz.astype(np.float32)).to(device)

    # Scale: initialise each Gaussian to the avg distance to its 3 nearest
    # neighbours. Use random subsample for kNN if point cloud is huge.
    sample_for_knn = means_t
    max_knn_pts = 50_000
    if N > max_knn_pts:
        idx = torch.randperm(N, device=device)[:max_knn_pts]
        sample_for_knn = means_t[idx]

    dists = torch.cdist(sample_for_knn, sample_for_knn)   # (M, M)
    dists.fill_diagonal_(float("inf"))
    k = min(3, dists.shape[0] - 1)
    knn_dists, _ = dists.topk(k, dim=1, largest=False)    # (M, k)
    avg_dist = knn_dists.mean().item()
    avg_dist = max(avg_dist, 1e-4)

    scales = torch.full((N, 3), math.log(avg_dist), device=device)

    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = 1.0  # w=1, identity rotation

    # Initial opacity: sigmoid^{-1}(0.1) ≈ -2.197
    opacities = torch.full((N,), math.log(0.1 / 0.9), device=device)

    # DC SH from RGB: colour = 0.5 + C0 * sh_dc  →  sh_dc = (colour - 0.5) / C0
    colour_f = torch.from_numpy(rgb.astype(np.float32) / 255.0).to(device)
    sh_dc = (colour_f - 0.5) / _SH_C0     # (N, 3)
    sh0 = sh_dc.unsqueeze(1)               # (N, 1, 3)

    K = (sh_degree + 1) ** 2
    shN = torch.zeros(N, K - 1, 3, device=device)

    return {
        "means":     means_t,
        "scales":    scales,
        "quats":     quats,
        "opacities": opacities,
        "sh0":       sh0,
        "shN":       shN,
    }


# ─── Loss functions ───────────────────────────────────────────────────────────

def _gaussian_kernel_1d(window_size: int, sigma: float) -> torch.Tensor:
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-x * x / (2 * sigma * sigma))
    return g / g.sum()


def _ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Structural Similarity Index. Inputs: (H, W, 3) float32 [0, 1]."""
    # → (1, 3, H, W)
    pred = pred.permute(2, 0, 1).unsqueeze(0)
    gt = gt.permute(2, 0, 1).unsqueeze(0)
    C = 3
    pad = window_size // 2

    k1d = _gaussian_kernel_1d(window_size, 1.5).to(pred.device)
    k2d = (k1d.unsqueeze(0) * k1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # (1,1,W,W)
    window = k2d.expand(C, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(pred, window, padding=pad, groups=C)
    mu2 = F.conv2d(gt,   window, padding=pad, groups=C)
    mu1_sq, mu2_sq = mu1 * mu1, mu2 * mu2
    mu12   = mu1 * mu2

    sig1_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu1_sq
    sig2_sq = F.conv2d(gt   * gt,   window, padding=pad, groups=C) - mu2_sq
    sig12   = F.conv2d(pred * gt,   window, padding=pad, groups=C) - mu12

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * sig12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2))
    return ssim_map.mean()


def _compute_loss(
    render: torch.Tensor,
    gt: torch.Tensor,
    lambda_dssim: float,
) -> torch.Tensor:
    """L1 + SSIM loss.  render, gt: (H, W, 3) float32 in [0, 1]."""
    l1 = (render - gt).abs().mean()
    if lambda_dssim > 0.0:
        s = 1.0 - _ssim(render, gt)
        return (1.0 - lambda_dssim) * l1 + lambda_dssim * s
    return l1


# ─── Image loading ────────────────────────────────────────────────────────────

def _load_image(path: Path, W: int, H: int, device: torch.device) -> torch.Tensor:
    """Load image as (H, W, 3) float32 [0, 1] tensor on device."""
    img = Image.open(path).convert("RGB")
    if img.width != W or img.height != H:
        img = img.resize((W, H), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
    return torch.from_numpy(arr).to(device)


# ─── PLY saving ───────────────────────────────────────────────────────────────

def _save_ply(
    path: Path,
    means: torch.Tensor,      # (N, 3)
    scales: torch.Tensor,     # (N, 3)  log-space
    quats: torch.Tensor,      # (N, 4) normalised (w, x, y, z)
    opacities: torch.Tensor,  # (N,)  logit-space
    sh0: torch.Tensor,        # (N, 1, 3)
    shN: torch.Tensor,        # (N, K-1, 3)
) -> None:
    """Write a standard 3DGS .ply file (binary little-endian)."""
    N = means.shape[0]
    K_rest = shN.shape[1]  # (sh_degree+1)^2 - 1

    # Move to CPU numpy
    means_np    = means.detach().float().cpu().numpy()
    scales_np   = scales.detach().float().cpu().numpy()
    quats_np    = quats.detach().float().cpu().numpy()
    opas_np     = opacities.detach().float().cpu().numpy()
    sh0_np      = sh0.detach().float().cpu().numpy().reshape(N, 3)   # (N,3)
    shN_np      = shN.detach().float().cpu().numpy().reshape(N, K_rest * 3)

    # Normalise quaternions
    norms = np.linalg.norm(quats_np, axis=1, keepdims=True)
    quats_np = quats_np / np.where(norms == 0, 1.0, norms)

    # Build property list
    properties = ["x", "y", "z", "nx", "ny", "nz"]
    properties += ["f_dc_0", "f_dc_1", "f_dc_2"]
    properties += [f"f_rest_{i}" for i in range(K_rest * 3)]
    properties += ["opacity"]
    properties += ["scale_0", "scale_1", "scale_2"]
    properties += ["rot_0", "rot_1", "rot_2", "rot_3"]

    # Pack all data: (N, num_props) float32
    normals = np.zeros((N, 3), dtype=np.float32)
    data = np.concatenate([
        means_np,                        # x y z
        normals,                         # nx ny nz (zeros)
        sh0_np,                          # f_dc_0 f_dc_1 f_dc_2
        shN_np,                          # f_rest_0 … f_rest_N
        opas_np[:, None],                # opacity (logit, stored raw)
        scales_np,                       # scale_0 scale_1 scale_2 (log, stored raw)
        quats_np,                        # rot_0 rot_1 rot_2 rot_3
    ], axis=1).astype(np.float32)

    # Write PLY header + binary body
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
    ]
    for prop in properties:
        header_lines.append(f"property float {prop}")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


# ─── Learning rate helpers ────────────────────────────────────────────────────

def _make_optimizers(
    params: dict[str, torch.Tensor],
    total_steps: int,
) -> dict[str, torch.optim.Adam]:
    """Create one Adam optimiser per parameter group with standard 3DGS learning rates."""
    lr_map = {
        "means":     1.6e-4,
        "scales":    5e-3,
        "quats":     1e-3,
        "opacities": 5e-2,
        "sh0":       2.5e-3,
        "shN":       2.5e-3 / 20.0,
    }
    return {
        k: torch.optim.Adam([v], lr=lr_map[k], eps=1e-15)
        for k, v in params.items()
    }


def _update_pos_lr(
    optimizer: torch.optim.Adam,
    step: int,
    total_steps: int,
    start_lr: float = 1.6e-4,
    end_lr: float = 1.6e-6,
) -> None:
    """Exponential decay for Gaussian positions."""
    t = step / max(total_steps, 1)
    lr = start_lr * (end_lr / start_lr) ** t
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ─── Densification (uses gsplat DefaultStrategy) ─────────────────────────────

def _make_strategy(cfg: TrainConfig):
    """Create gsplat DefaultStrategy with TrainConfig densification settings."""
    try:
        from gsplat.strategy import DefaultStrategy
        return DefaultStrategy(
            prune_opa=0.005,
            grow_grad2d=cfg.densify_grad_threshold,
            grow_scale3d=0.01,
            prune_scale3d=0.1,
            refine_start_iter=cfg.densify_from,
            refine_stop_iter=cfg.densify_until,
            refine_every=cfg.densify_every,
            absgrad=True,
            verbose=False,
        )
    except ImportError:
        return None


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _save_checkpoint(
    path: Path,
    step: int,
    params: dict[str, torch.Tensor],
    optimizers: dict[str, torch.optim.Adam],
) -> None:
    torch.save({
        "step": step,
        "params": {k: v.detach().cpu() for k, v in params.items()},
        "opt_states": {k: opt.state_dict() for k, opt in optimizers.items()},
    }, path)


def _load_checkpoint(
    path: Path,
    params: dict[str, torch.Tensor],
    optimizers: dict[str, torch.optim.Adam],
    device: torch.device,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for k, v in ckpt["params"].items():
        if k in params:
            params[k].data = v.to(device)
    for k, state in ckpt["opt_states"].items():
        if k in optimizers:
            try:
                optimizers[k].load_state_dict(state)
            except Exception:
                pass  # shape mismatch after densification — skip
    return ckpt["step"]


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(
    dataset_dir: Path,
    output_dir: Path,
    cfg: TrainConfig,
    backend: BackendInfo,
    resume: bool = False,
    viewer_port: int | None = None,
    dashboard=None,  # ttgs.viewer.dashboard.TrainingController | None
    masks_dir: Path | None = None,
    excluded: set[str] | None = None,
) -> Path:
    """Run gsplat 3DGS training.

    Args:
        dataset_dir:  COLMAP undistorted dataset dir (contains sparse/ + images/).
        output_dir:   Destination for splat.ply and checkpoints.
        cfg:          TrainConfig parameters.
        backend:      Compute backend (xpu / cuda / directml / cpu).
        resume:       If True, try to resume from output_dir/checkpoint.pt.
        dashboard:    TrainingController from DashboardServer — enables the
                      interactive FastAPI dashboard (masks, prune, pause, etc.)

    Returns:
        Path to the written splat.ply.
    """
    try:
        from gsplat import rasterization
    except ImportError:
        raise RuntimeError(
            "gsplat is not installed. Install it with:\n"
            "  pip install gsplat\n"
            "Then re-run training."
        )

    # --- Device selection ---
    if backend.backend == Backend.XPU:
        device = torch.device("xpu")
    elif backend.backend == Backend.CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if backend.backend == Backend.DIRECTML:
            console.print(
                "[yellow]train[/] DirectML is not supported by gsplat; "
                "falling back to [bold]CPU[/] mode."
            )

    console.print(
        f"[bold cyan]train[/] gsplat on [green]{backend.device_name}[/] "
        f"([green]{device}[/]) — {cfg.iterations:,} iterations"
    )

    # --- Locate dataset directories ---
    # image_undistorter output: <dataset_dir>/sparse/ + <dataset_dir>/images/
    sparse_dir = dataset_dir / "sparse"
    images_dir = dataset_dir / "images"
    if not sparse_dir.is_dir():
        sparse_dir = dataset_dir
        images_dir = dataset_dir.parent / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Images directory not found at {images_dir}. "
            "Expected COLMAP undistorted layout: sparse/ + images/."
        )

    # --- Load COLMAP data ---
    console.print("[dim]loading COLMAP data...[/]")
    train_cameras = _load_colmap(dataset_dir, images_dir)
    if not train_cameras:
        raise RuntimeError(
            "No training cameras loaded. "
            "Check that images exist in " + str(images_dir)
        )

    # Filter excluded images
    if excluded:
        before = len(train_cameras)
        train_cameras = [c for c in train_cameras if c.image_path.name not in excluded]
        n_excl = before - len(train_cameras)
        if n_excl:
            console.print(f"[dim]excluded {n_excl} images → {len(train_cameras)} remain[/]")
        if not train_cameras:
            raise RuntimeError("All cameras excluded — nothing to train on.")

    console.print(f"[dim]loaded {len(train_cameras)} cameras[/]")

    # Per-image mask cache  (masks_dir/{stem}.png → float32 [0,1])
    # Only caches successfully loaded masks.  Missing masks are re-checked
    # each time so that masks created mid-training are picked up after a
    # reload_masks command clears the cache entry.
    _img_mask_cache: dict[str, np.ndarray] = {}

    def _get_img_mask(image_name: str) -> np.ndarray | None:
        if image_name in _img_mask_cache:
            return _img_mask_cache[image_name]
        if masks_dir is None or not masks_dir.exists():
            return None
        p = masks_dir / (Path(image_name).stem + ".png")
        if p.exists():
            mask = np.array(
                Image.open(p).convert("L"), dtype=np.float32
            ) / 255.0
            _img_mask_cache[image_name] = mask
            return mask
        return None

    # --- Load SfM point cloud ---
    points3d_bin = sparse_dir / "points3D.bin"
    if points3d_bin.exists():
        xyz, rgb = _load_points3d_bin(points3d_bin)
        console.print(f"[dim]SfM point cloud: {len(xyz):,} points[/]")
    else:
        # Fallback: place a single Gaussian at scene centre
        console.print("[yellow]train[/] points3D.bin not found — initialising with random cloud")
        xyz = np.random.randn(1000, 3).astype(np.float64)
        rgb = np.full((1000, 3), 128, dtype=np.uint8)

    # --- Initialise Gaussians ---
    console.print("[dim]initialising Gaussians...[/]")
    raw = _init_gaussians(xyz, rgb, cfg.sh_degree, device)
    # Wrap as nn.Parameters for gradient tracking
    params: dict[str, torch.nn.Parameter] = {
        k: torch.nn.Parameter(v.clone()) for k, v in raw.items()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_ply = output_dir / "splat.ply"
    ckpt_path = output_dir / "checkpoint.pt"

    # --- Optimisers ---
    optimizers = _make_optimizers(params, cfg.iterations)

    # --- Resume ---
    start_step = 0
    if resume and ckpt_path.exists():
        console.print(f"[dim]resuming from {ckpt_path}[/]")
        start_step = _load_checkpoint(ckpt_path, params, optimizers, device)
        console.print(f"[dim]resumed at step {start_step:,}[/]")
    elif resume:
        console.print("[yellow]train[/] --resume requested but no checkpoint found — starting fresh")

    # --- Densification strategy ---
    strategy = _make_strategy(cfg)
    strategy_state: dict[str, Any] = {}
    if strategy is not None:
        try:
            strategy_state = strategy.initialize_state(scene_scale=1.0)
            strategy.check_sanity(params, optimizers)
        except Exception as exc:
            console.print(f"[yellow]train[/] DefaultStrategy init failed ({exc}) — densification disabled")
            strategy = None

    # --- Progress display ---
    progress = Progress(
        TextColumn("[cyan]train[/]"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]loss: {task.fields[loss]}  N: {task.fields[n_gauss]}[/]"),
        console=console,
    )
    task_id = progress.add_task(
        "training",
        total=cfg.iterations,
        completed=start_step,
        loss="—",
        n_gauss=f"{params['means'].shape[0]:,}",
    )

    last_loss_val = float("nan")
    _focus_camera: str | None = None     # set by focus_camera command
    _force_densify: bool      = False    # set by densify_now command
    _dashboard_every = getattr(cfg, "dashboard_every", 25)
    _snapshot_every = getattr(cfg, "snapshot_every", 0)
    _max_iterations = cfg.iterations     # mutable — updated by update_config command

    # Push initial config to dashboard
    if dashboard is not None:
        from dataclasses import asdict
        dashboard.set_config(asdict(cfg))
        dashboard._snapshot_every = _snapshot_every

    interrupted = False
    step = start_step
    with Live(progress, console=console, refresh_per_second=4):
      try:
        while step < _max_iterations:

            # ── Check for stop signal from dashboard ──
            if dashboard is not None and dashboard.should_stop:
                interrupted = True
                console.print("\n[yellow]train[/] stop requested — saving current state…")
                _save_checkpoint(ckpt_path, step, params, optimizers)
                break

            # ── Drain commands at top of step so they affect this iteration ──
            _params_mutated = False  # set True if param count changes (prune)
            if dashboard is not None:
                for cmd in dashboard.drain_commands():
                    ctype = cmd.get("type")
                    _cmd_detail = ""
                    if ctype == "prune":
                        thresh = float(cmd.get("threshold", 0.005))
                        with torch.no_grad():
                            keep = torch.sigmoid(params["opacities"]) > thresh
                            n_rm = (~keep).sum().item()
                            for k in params:
                                params[k].data = params[k].data[keep]
                            for opt in optimizers.values():
                                opt.state.clear()
                        _cmd_detail = f"removed {n_rm:,}, {params['means'].shape[0]:,} remain (thresh={thresh:.3f})"
                        _params_mutated = True
                        console.print(f"[yellow]dashboard[/] pruned {_cmd_detail}")
                    elif ctype == "reset_opacities":
                        with torch.no_grad():
                            reset_val = math.log(0.01 / 0.99)
                            params["opacities"].data.fill_(reset_val)
                        _cmd_detail = "all opacities → 0.01"
                        console.print("[yellow]dashboard[/] opacities reset")
                    elif ctype == "clamp_scale":
                        max_log = float(cmd.get("max_log_scale", 2.5))
                        with torch.no_grad():
                            params["scales"].data.clamp_(max=max_log)
                        n = params["means"].shape[0]
                        _cmd_detail = f"max_log_scale={max_log}, {n:,} Gaussians"
                        console.print(f"[yellow]dashboard[/] scales clamped to {max_log} ({n:,} Gaussians)")
                    elif ctype == "set_lr":
                        factor = float(cmd.get("lr_factor", 1.0))
                        for opt in optimizers.values():
                            for pg in opt.param_groups:
                                pg["lr"] *= factor
                        _cmd_detail = f"\u00d7{factor}"
                        console.print(f"[yellow]dashboard[/] LR scaled \u00d7{factor}")
                    elif ctype == "focus_camera":
                        _focus_camera = cmd.get("camera_name")
                        _cmd_detail = _focus_camera or "cleared"
                        if _focus_camera:
                            console.print(f"[yellow]dashboard[/] focus_camera → '{_focus_camera}'")
                        else:
                            console.print("[yellow]dashboard[/] focus_camera cleared")
                    elif ctype == "densify_now":
                        _force_densify = True
                        _cmd_detail = "next step"
                        console.print("[yellow]dashboard[/] densify_now requested")
                    elif ctype == "reload_masks":
                        img_name = cmd.get("image_name")
                        if img_name:
                            _img_mask_cache.pop(img_name, None)
                            _cmd_detail = img_name
                            console.print(f"[yellow]dashboard[/] mask reloaded: {img_name}")
                        else:
                            _img_mask_cache.clear()
                            _cmd_detail = "all"
                            console.print(f"[yellow]dashboard[/] all masks reloaded")
                    elif ctype == "save":
                        _save_checkpoint(ckpt_path, step, params, optimizers)
                        _cmd_detail = f"step {step:,}"
                        console.print(f"[yellow]dashboard[/] checkpoint saved at step {step:,}")
                    elif ctype == "update_config":
                        _cfg_fields = {
                            "iterations", "save_every", "snapshot_every",
                            "dashboard_every", "lambda_dssim", "densify_from",
                            "densify_until", "densify_every",
                            "densify_grad_threshold", "opacity_reset_every",
                            "log_every",
                        }
                        changes = []
                        for k, v in cmd.items():
                            if k in _cfg_fields and hasattr(cfg, k):
                                old = getattr(cfg, k)
                                setattr(cfg, k, type(old)(v))
                                changes.append(f"{k}: {old}\u2192{getattr(cfg, k)}")
                        if "iterations" in cmd:
                            _max_iterations = cfg.iterations
                            progress.update(task_id, total=_max_iterations)
                        if "dashboard_every" in cmd:
                            _dashboard_every = cfg.dashboard_every
                        if "snapshot_every" in cmd:
                            _snapshot_every = cfg.snapshot_every
                            dashboard._snapshot_every = _snapshot_every
                        from dataclasses import asdict
                        dashboard.set_config(asdict(cfg))
                        _cmd_detail = ", ".join(changes) if changes else "no changes"
                        if changes:
                            console.print(
                                f"[yellow]dashboard[/] config updated: "
                                + ", ".join(changes)
                            )

                    # Log the command with current training stats
                    if ctype:
                        dashboard.log_command(
                            ctype, step, _cmd_detail,
                            stats={
                                "loss": last_loss_val,
                                "n_gaussians": params["means"].shape[0],
                            },
                        )

            # Skip forward/backward if prune changed param shapes this step —
            # the autograd graph would reference stale tensor shapes.
            if _params_mutated:
                step += 1
                continue

            # Camera selection — respect focus if set
            if _focus_camera:
                matching = [c for c in train_cameras
                            if c.image_path.name == _focus_camera]
                cam = random.choice(matching) if matching else random.choice(train_cameras)
            else:
                cam = random.choice(train_cameras)

            # Move camera tensors to device (avoid storing them all on GPU)
            viewmat = cam.viewmat.to(device).unsqueeze(0)   # (1, 4, 4)
            K_mat   = cam.K.to(device).unsqueeze(0)         # (1, 3, 3)

            # Current SH degree (progressive: increase every 1000 steps)
            sh_degree_cur = min(cfg.sh_degree, step // 1000)

            # Assemble SH coefficients: (N, K, 3)
            sh_coeffs = torch.cat([params["sh0"], params["shN"]], dim=1)

            # Rasterize
            renders, _alphas, info = rasterization(
                means=params["means"],
                quats=F.normalize(params["quats"], dim=-1),
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=sh_coeffs,
                viewmats=viewmat,
                Ks=K_mat,
                width=cam.W,
                height=cam.H,
                sh_degree=sh_degree_cur,
                packed=False,
                absgrad=True,
            )

            render_img = renders[0]  # (H, W, 3)

            # Ground truth
            gt_img = _load_image(cam.image_path, cam.W, cam.H, device)

            # Loss mask: combine per-image mask with dashboard global mask
            _mask_np = _get_img_mask(cam.image_path.name)
            if dashboard is not None:
                _global = dashboard.get_mask()
                if _global is not None:
                    _mask_np = np.minimum(_mask_np, _global) if _mask_np is not None else _global

            if _mask_np is not None:
                import torch.nn.functional as _F
                _m = torch.from_numpy(_mask_np).to(device)
                # Resize mask to current camera resolution
                _m = _F.interpolate(
                    _m.unsqueeze(0).unsqueeze(0), size=(cam.H, cam.W), mode="nearest"
                ).squeeze().unsqueeze(-1)  # (H, W, 1)
                # Masked L1
                diff = (render_img - gt_img).abs() * _m
                loss = diff.sum() / (_m.sum() * 3 + 1e-8)
                if cfg.lambda_dssim > 0.0:
                    # Mask SSIM: replace masked pixels with GT so they
                    # contribute zero SSIM loss (render == GT → SSIM=1).
                    masked_render = render_img * _m + gt_img * (1.0 - _m)
                    ssim_val = _ssim(masked_render, gt_img)
                    loss = (1.0 - cfg.lambda_dssim) * loss + cfg.lambda_dssim * (1.0 - ssim_val)
            else:
                loss = _compute_loss(render_img, gt_img, cfg.lambda_dssim)

            # Zero gradients
            for opt in optimizers.values():
                opt.zero_grad()

            # Pre-backward hook (accumulates gradient stats for densification)
            if strategy is not None:
                try:
                    strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
                except Exception:
                    pass

            loss.backward()

            # Optimiser step
            _update_pos_lr(optimizers["means"], step, _max_iterations)
            for opt in optimizers.values():
                opt.step()

            # Post-backward hook (densification / pruning)
            if strategy is not None:
                try:
                    strategy.step_post_backward(
                        params, optimizers, strategy_state, step, info, packed=False
                    )
                except Exception:
                    pass

            # Forced densification (from dashboard densify_now command)
            if _force_densify and strategy is not None:
                try:
                    # Use a step value inside the densify window so the strategy runs
                    force_step = cfg.densify_from + (
                        cfg.densify_every - (step % cfg.densify_every)
                    ) % cfg.densify_every
                    strategy.step_post_backward(
                        params, optimizers, strategy_state, force_step, info, packed=False
                    )
                    console.print(
                        f"[yellow]dashboard[/] forced densification "
                        f"→ {params['means'].shape[0]:,} Gaussians"
                    )
                except Exception as _e:
                    console.print(f"[yellow]dashboard[/] densify_now failed: {_e}")
                _force_densify = False

            # Opacity reset
            if cfg.opacity_reset_every > 0 and step > 0 and step % cfg.opacity_reset_every == 0:
                with torch.no_grad():
                    reset_val = math.log(0.01 / 0.99)  # sigmoid^{-1}(0.01)
                    params["opacities"].data.clamp_(max=reset_val)

            # Logging
            if step % cfg.log_every == 0 or step == _max_iterations - 1:
                last_loss_val = loss.item()

            # Dashboard: pause check + snapshot push
            if dashboard is not None:
                dashboard.wait_if_paused()

                # First step always pushes; then every dashboard_every steps
                _push = (step == start_step) or (step % _dashboard_every == 0)
                if _push:
                    try:
                        from ttgs.viewer.dashboard import build_update
                        with torch.no_grad():
                            _r = render_img.detach()
                            _g = gt_img.detach()
                            _l1_val  = (_r - _g).abs().mean().item()
                            _mse_val = ((_r - _g) ** 2).mean().item()
                            _ssim_val = (1.0 - _ssim(_r, _g)).item() if cfg.lambda_dssim > 0 else 0.0
                        dashboard.push_update(build_update(
                            step=step,
                            total_steps=_max_iterations,
                            loss=loss.item(),
                            l1=_l1_val,
                            ssim=_ssim_val,
                            mse=_mse_val,
                            n_gaussians=params["means"].shape[0],
                            camera_name=cam.image_path.name,
                            render=_r.float().cpu().numpy(),
                            gt=_g.float().cpu().numpy(),
                            is_paused=dashboard.is_paused,
                            focus_camera=_focus_camera,
                        ))
                    except Exception:
                        pass

            # Checkpoint
            if cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0:
                _save_checkpoint(ckpt_path, step, params, optimizers)

            # Update progress bar
            progress.update(
                task_id,
                total=_max_iterations,
                completed=step + 1,
                loss=f"{last_loss_val:.4f}",
                n_gauss=f"{params['means'].shape[0]:,}",
            )

            step += 1

      except KeyboardInterrupt:
        interrupted = True
        console.print("\n[yellow]train[/] interrupted — saving current state…")
        _save_checkpoint(ckpt_path, step, params, optimizers)

    console.print("[dim]saving splat.ply...[/]")
    _save_ply(
        output_ply,
        params["means"],
        params["scales"],
        F.normalize(params["quats"], dim=-1),
        params["opacities"],
        params["sh0"],
        params["shN"],
    )

    n_gaussians = params["means"].shape[0]
    size_mb = output_ply.stat().st_size / 1024**2
    steps_done = step if not interrupted else step
    label = "[yellow]interrupted[/]" if interrupted else "[bold green]done[/]"
    console.print(
        f"[bold cyan]train[/] {label} — {steps_done:,}/{_max_iterations:,} steps, "
        f"loss: [yellow]{last_loss_val:.4f}[/], "
        f"{n_gaussians:,} Gaussians → [cyan]{output_ply}[/] ({size_mb:.1f} MB)"
    )
    if interrupted:
        console.print(
            f"[dim]resume with:[/] ttgs train {dataset_dir} --resume"
        )
    return output_ply
