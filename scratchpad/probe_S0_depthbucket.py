#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PROBE S0 (host, no device) — DECISIVE de-risk for the on-device counting-bucket bin/sort.

Question: does replacing the exact within-tile DEPTH lexsort with a COUNTING-BUCKET order (quantize
depth into D bins, no ordering inside a bin) hold render parity? If yes, the on-device bin/sort is
greenlit (everything else reduces to already-proven primitives: m2 owner-scatter + E5 streaming).

Method: load the REAL trained model (work/tt_out/splat.ply), project from a real camera, then alpha-blend
front-to-back TWICE with identical math, differing ONLY in Gaussian order:
  (a) EXACT   = sort by depth                          (the lexsort secondary key)
  (b) BUCKET  = stable-sort by depth-bucket index      (models the counting sort: arbitrary within bin)
PSNR(a,b) per D. Reference: PSNR(exact, RANDOM order) = how much ordering matters at all here.
Gate: smallest D with PSNR > 40 dB is the on-device bucket count.
"""
import sys, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "pathclear"))
from train_tt import _load_colmap                       # noqa: E402
from train_real import project_general, sh_eval         # noqa: E402

PLY = Path(__file__).resolve().parent.parent / "work" / "tt_out" / "splat.ply"
DATASET = Path(__file__).resolve().parent.parent / "work" / "scene2"
DEG = 3


def load_ply(path):
    """Parse the 3DGS PLY written by train_tt._write_ply -> P dict (mean, scale[log], quat, op[logit], sh)."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        fmt = f.readline().strip()
        n = int(f.readline().split()[-1])
        props = []
        while True:
            ln = f.readline().strip()
            if ln == b"end_header":
                break
            if ln.startswith(b"property"):
                props.append(ln.split()[-1].decode())
        data = np.frombuffer(f.read(n * len(props) * 4), dtype="<f4").reshape(n, len(props)).copy()
    col = {p: i for i, p in enumerate(props)}
    mean = data[:, [col["x"], col["y"], col["z"]]]
    op = data[:, col["opacity"]]
    scale = data[:, [col[f"scale_{i}"] for i in range(3)]]
    quat = data[:, [col[f"rot_{i}"] for i in range(4)]]
    f_dc = data[:, [col[f"f_dc_{i}"] for i in range(3)]]                       # [N,3] band 0
    nrest = sum(p.startswith("f_rest_") for p in props)
    K = 1 + nrest // 3
    sh = np.zeros((n, K, 3), np.float32)
    sh[:, 0, :] = f_dc
    if nrest:
        rest = data[:, [col[f"f_rest_{i}"] for i in range(nrest)]]            # channel-major [N, 3*(K-1)]
        sh[:, 1:, :] = rest.reshape(n, 3, K - 1).transpose(0, 2, 1)           # invert write's permute
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    return {"mean": t(mean), "scale": t(scale), "quat": t(quat), "op": t(op), "sh": t(sh), "deg": DEG}, n


def blend(order, u, v, a, b, c, op, col, H, W):
    """Front-to-back alpha blend over Gaussians in `order` (vectorized over pixels)."""
    C = np.zeros((H, W, 3)); T = np.ones((H, W))
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    for i in order:
        dx = xs - u[i]; dy = ys - v[i]
        sigma = 0.5 * (a[i] * dx * dx + c[i] * dy * dy) + b[i] * dx * dy     # gsplat conic power
        alpha = np.clip(op[i] * np.exp(-sigma), 0.0, 0.999)
        w = T * alpha
        C += w[..., None] * col[i]
        T *= (1.0 - alpha)
    return C


def psnr(x, y):
    mse = float(((x - y) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10 * math.log10(1.0 / mse)


def sweep(tag, u, v, a, b, cc, op, zc, col, H, W, rng):
    """D-sweep: PSNR(exact-depth-order vs D-bucket-order) + random-order reference."""
    rx = 3.0 * np.sqrt(np.clip(cc / (a * cc - b ** 2 + 1e-12), .25, None))
    cover = float(np.mean(np.pi * rx * rx)) * u.size / (H * W)
    order_exact = np.argsort(zc)
    ref = blend(order_exact, u, v, a, b, cc, op, col, H, W)
    rand = blend(rng.permutation(u.size), u, v, a, b, cc, op, col, H, W)
    zmin, zmax = zc.min(), zc.max() + 1e-9
    line = []
    for D in (8, 16, 32, 64, 128):
        bucket = np.clip(((zc - zmin) / (zmax - zmin) * D).astype(np.int64), 0, D - 1)
        order_b = np.argsort(bucket, kind="stable")                  # counting sort: stable within bin
        line.append(f"D={D:>3}:{psnr(ref, blend(order_b, u, v, a, b, cc, op, col, H, W)):5.1f}dB")
    print(f"  {tag:22s} N={u.size:>6} cover~{cover:4.1f}x | random={psnr(ref, rand):4.1f}dB  " + "  ".join(line))
    return cover


def project_cam(P, cam, LONG):
    Rv, tv, fx, fy, cx, cy, name = cam
    nativeW = 2 * float(cx); s = LONG / nativeW
    W = int(round(nativeW * s)); H = int(round(2 * float(cy) * s))
    with torch.no_grad():
        u, v, zc, (a, b, cc) = project_general(P, Rv, tv, fx * s, fy * s, cx * s, cy * s)
        cam_center = -Rv.T @ tv
        dirs = P["mean"] - cam_center; dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-9)
        col = sh_eval(P["sh"], dirs, P["deg"]).clamp(0, 1).numpy()
        op = torch.sigmoid(P["op"]).numpy()
    u, v, zc = u.numpy(), v.numpy(), zc.numpy(); a, b, cc = a.numpy(), b.numpy(), cc.numpy()
    vis = (zc > 1e-4) & (u > -W) & (u < 2 * W) & (v > -H) & (v < 2 * H)
    return name, H, W, [arr[vis] for arr in (u, v, a, b, cc, op, zc)] + [col[vis]]


def densify(arrs, target_cover, base_cover, H, W, rng):
    """Replicate Gaussians with positional + depth jitter to simulate the high-overlap millions regime
    (conservative: same opacity per replica = more semi-transparent layers = HARDER on ordering)."""
    u, v, a, b, cc, op, zc, col = arrs
    K = max(1, int(round(target_cover / base_cover)))
    rx = 3.0 * np.sqrt(np.clip(cc / (a * cc - b ** 2 + 1e-12), .25, None))
    ry = 3.0 * np.sqrt(np.clip(a / (a * cc - b ** 2 + 1e-12), .25, None))
    zspan = (zc.max() - zc.min() + 1e-9)
    rep = lambda x: np.tile(x, K)
    ru = rep(u) + rng.normal(0, 1, u.size * K) * rep(rx) * 0.5
    rv = rep(v) + rng.normal(0, 1, u.size * K) * rep(ry) * 0.5
    rz = rep(zc) + rng.normal(0, 1, u.size * K) * zspan * 0.04          # within-cluster depth spread
    return [ru, rv, rep(a), rep(b), rep(cc), rep(op), rz, np.tile(col, (K, 1))]


def main():
    P, n = load_ply(PLY)
    cams, _, _ = _load_colmap(DATASET)
    print(f"loaded trained ply: N={n} Gaussians, sh K={P['sh'].shape[1]}; {len(cams)} cameras")
    rng = np.random.default_rng(0)

    print("\n[baseline] real trained scene @96px (sparse, cover~1.7x):")
    base = None
    for ci in range(min(3, len(cams))):
        name, H, W, arrs = project_cam(P, cams[ci], 96)
        if arrs[0].size == 0:
            continue
        c = sweep(f"cam {name}", *arrs, H, W, rng)
        if base is None:
            base = (H, W, arrs, c)

    print("\n[STRESS] millions proxy — real Gaussians replicated w/ jitter to high cover (cam0 @96px):")
    H, W, arrs, base_cover = base
    for tc in (6, 15, 30):
        d = densify(arrs, tc, base_cover, H, W, rng)
        sweep(f"densified~{tc}x", *d, H, W, rng)
    print("\nGate: smallest D with >40dB at the millions-regime cover (15-30x) is the on-device bucket count.")


if __name__ == "__main__":
    main()
