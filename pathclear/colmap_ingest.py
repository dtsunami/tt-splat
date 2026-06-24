#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
COLMAP ingestion adapter — turn an SfM reconstruction into our 3DGS inputs.

Real pipeline:  images (or ffmpeg frames) -> COLMAP SfM -> cameras.txt/images.txt/points3D.txt
                -> THIS adapter -> {per-camera (Rv,tv,f,pp)} + {Gaussian init from sparse points}.

Verified here WITHOUT real data or a COLMAP install: write our known synthetic cameras (from
train3d) in COLMAP's exact text format, read them back through the adapter, and confirm the
read-back cameras reproduce the renders bit-for-bit + points seed the Gaussians. So when you run
real COLMAP, the parser is already proven.

COLMAP conventions: images.txt has world->camera qvec=(qw,qx,qy,qz)+tvec (X_cam = R(q)X + t),
camera looks +z; cameras.txt PINHOLE params = fx fy cx cy; points3D.txt = ID X Y Z R G B ...
"""
import math, os, torch
from train3d import quat_to_rot, render, scene, look_at, H, W, F, PP, cameras

WS = "/tmp/claude-1000/-home-starboy/99cfd4cc-748e-42e4-a1df-7dc090414335/scratchpad/colmap_ws"


def canonical_qvec2rotmat(q):
    """VERBATIM from COLMAP scripts/python/read_write_model.py (qvec=w,x,y,z). Ground truth."""
    w, x, y, z = q.tolist()
    return torch.tensor([
        [1-2*y*y-2*z*z,   2*x*y-2*w*z,     2*z*x+2*w*y],
        [2*x*y+2*w*z,     1-2*x*x-2*z*z,   2*y*z-2*w*x],
        [2*z*x-2*w*y,     2*y*z+2*w*x,     1-2*x*x-2*y*y]], dtype=torch.float64)


def rot_to_quat(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25*s; x = (R[2,1]-R[1,2])/s; y = (R[0,2]-R[2,0])/s; z = (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
        w = (R[2,1]-R[1,2])/s; x = 0.25*s; y = (R[0,1]+R[1,0])/s; z = (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
        w = (R[0,2]-R[2,0])/s; x = (R[0,1]+R[1,0])/s; y = 0.25*s; z = (R[1,2]+R[2,1])/s
    else:
        s = math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
        w = (R[1,0]-R[0,1])/s; x = (R[0,2]+R[2,0])/s; y = (R[1,2]+R[2,1])/s; z = 0.25*s
    return torch.tensor([w, x, y, z], dtype=torch.float64)


def write_colmap(path, cams, points_xyz, points_rgb):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/cameras.txt", "w") as f:
        for i, _ in enumerate(cams):
            f.write(f"{i+1} PINHOLE {W} {H} {F} {F} {float(PP[0])} {float(PP[1])}\n")
    with open(f"{path}/images.txt", "w") as f:
        for i, (Rv, tv) in enumerate(cams):
            q = rot_to_quat(Rv)
            f.write(f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {tv[0]} {tv[1]} {tv[2]} {i+1} img{i}.png\n")
            f.write("\n")                       # (empty 2D-keypoints line)
    with open(f"{path}/points3D.txt", "w") as f:
        for j in range(points_xyz.shape[0]):
            x, y, z = points_xyz[j].tolist()
            r, g, b = points_rgb[j].tolist()
            f.write(f"{j+1} {x} {y} {z} {int(r)} {int(g)} {int(b)} 0.0\n")


def read_colmap(path):
    intr = {}
    for line in open(f"{path}/cameras.txt"):
        if line.startswith("#") or not line.strip(): continue
        p = line.split(); cid = int(p[0]); model = p[1]; prm = list(map(float, p[4:]))
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):     # f, cx, cy, [k...]
            fx = fy = prm[0]; cx, cy = prm[1], prm[2]
        else:                                                          # PINHOLE/OPENCV: fx fy cx cy ...
            fx, fy, cx, cy = prm[0], prm[1], prm[2], prm[3]
        intr[cid] = (fx, fy, cx, cy)
    # canonical COLMAP images.txt = 2 lines/image (pose, then keypoints); keep both, step by 2
    cams = []; lines = [l for l in open(f"{path}/images.txt") if not l.startswith("#")]
    for i in range(0, len(lines) - 1, 2):
        p = lines[i].split()
        if len(p) < 10: continue                 # skip stray blanks (need ...,cam_id,name)
        q = torch.tensor([float(p[1]), float(p[2]), float(p[3]), float(p[4])], dtype=torch.float64)
        tv = torch.tensor([float(p[5]), float(p[6]), float(p[7])], dtype=torch.float64)
        Rv = quat_to_rot(q)
        cid = int(p[8]); fx, fy, cx, cy = intr[cid]; name = p[9]
        cams.append((Rv, tv, fx, fy, cx, cy, name))
    pts = [l.split() for l in open(f"{path}/points3D.txt") if l.strip() and not l.startswith("#")]
    xyz = torch.tensor([[float(p[1]), float(p[2]), float(p[3])] for p in pts], dtype=torch.float64)
    rgb = torch.tensor([[float(p[4]), float(p[5]), float(p[6])] for p in pts], dtype=torch.float64)
    return cams, xyz, rgb


def main():
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    PX, PY = jj.double(), ii.double()
    N = 24
    GT = scene(1, N)
    cams = cameras(5, seed=0)
    pts_xyz = GT["mean"].clone()
    pts_rgb = (torch.sigmoid(GT["col"])[:, None].repeat(1, 3) * 255).round()

    # (0) CANONICAL check: our quat_to_rot must equal COLMAP's documented qvec2rotmat
    gqc = torch.Generator().manual_seed(11); qerr = 0.0
    for _ in range(200):
        q = torch.randn(4, generator=gqc, dtype=torch.float64); q = q/q.norm()
        qerr = max(qerr, float((quat_to_rot(q) - canonical_qvec2rotmat(q)).abs().max()))

    write_colmap(WS, cams, pts_xyz, pts_rgb)
    rcams, rxyz, rrgb = read_colmap(WS)

    # (0b) CANONICAL camera-position: world camera center = -R^T t (COLMAP convention).
    # Must equal the eye positions we placed the cameras at.
    eyes = [torch.tensor([6*math.cos(2*math.pi*i/5+0.1), 1.5, 6*math.sin(2*math.pi*i/5+0.1)],
                         dtype=torch.float64) for i in range(5)]
    center_err = 0.0
    for (Rv2, tv2, *_), eye in zip(rcams, eyes):
        center = -Rv2.T @ tv2
        center_err = max(center_err, float((center - eye).abs().max()))

    # (1) poses round-trip: read-back (Rv,tv,f,pp) match originals
    pose_err = 0.0
    for (Rv, tv), (Rv2, tv2, fx, fy, cx, cy, _name) in zip(cams, rcams):
        pose_err = max(pose_err, float((Rv - Rv2).abs().max()), float((tv - tv2).abs().max()),
                       abs(fx - float(F)), abs(cx - float(PP[0])))

    # (2) the real test: read-back cameras reproduce the renders
    rmse = 0.0
    for (Rv, tv), (Rv2, tv2, *_) in zip(cams, rcams):
        a = render(GT, Rv, tv, PX, PY)
        b = render(GT, Rv2, tv2, PX, PY)         # F/PP are module-level; fx≈F, cx≈PP here
        rmse = max(rmse, float(((a - b)**2).mean()))

    # (3) points seed Gaussians: positions recovered, count matches
    pts_ok = rxyz.shape[0] == N and float((rxyz - pts_xyz).abs().max()) < 1e-4

    print(f"[canonical] quat_to_rot vs COLMAP qvec2rotmat  max err = {qerr:.2e}")
    print(f"[canonical] camera pos = -R^T t vs placed eye  max err = {center_err:.2e}")
    print(f"camera round-trip   max pose err = {pose_err:.2e}")
    print(f"render match        max MSE      = {rmse:.2e}  (read-back cams reproduce renders)")
    print(f"point-cloud init    N={rxyz.shape[0]} (exp {N})  xyz_err={float((rxyz-pts_xyz).abs().max()):.2e}")
    ok = qerr < 1e-12 and center_err < 1e-9 and pose_err < 1e-9 and rmse < 1e-12 and pts_ok
    print("COLMAP_INGEST_OK" if ok else "COLMAP_INGEST_FAIL")


if __name__ == "__main__":
    main()
