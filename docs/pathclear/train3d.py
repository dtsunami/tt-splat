#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Close the loop to 3DGS: the 2D pipeline + a 3D->2D EWA projection front-end.

3D Gaussian (mean xyz, scale, quaternion, opacity, color) --project--> per-camera
(2D mean, 2D conic, depth) --> our proven 2D raster (sort by depth, front->back blend).
Trained over MULTIPLE synthetic cameras (known poses), evaluated on a HELD-OUT novel view
— the real test that 3D structure (not per-view overfit) was learned.

Cameras here are synthetic (no COLMAP/ffmpeg needed). Real data path later:
  video --ffmpeg--> frames --COLMAP(SfM)--> intrinsics+poses+points --> 3DGS.

Backward via autograd (the hand-derived blend backward was verified vs autograd in train2d_verify;
here autograd carries it through the projection too). Verified: projection geometry + convergence
+ novel-view PSNR.
"""
import math, torch
torch.set_default_dtype(torch.float64)   # tight; perf phase uses fp32 on device

H = W = 64
F = 70.0
PP = torch.tensor([W/2, H/2])


def quat_to_rot(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y),
        2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def cov3d(scale_log, quat):
    Rq = quat_to_rot(quat)                       # (N,3,3)
    S2 = torch.exp(scale_log)**2                 # (N,3)
    return Rq @ (S2[..., None] * Rq.transpose(-1, -2))


def look_at(eye, center, up=torch.tensor([0., 1., 0.])):
    f = (center - eye); f = f / f.norm()
    r = torch.linalg.cross(f, up); r = r / r.norm()
    u = torch.linalg.cross(f, r)                 # proper rotation (det=+1); COLMAP-compatible
    Rv = torch.stack([r, u, f])                  # world->cam rows; cam looks +z
    tv = -Rv @ eye
    return Rv, tv


def project(P, Rv, tv):
    mean, sl, q = P["mean"], P["scale"], P["quat"]
    Sig3 = cov3d(sl, q)
    mc = mean @ Rv.T + tv                         # camera space (N,3)
    z = mc[:, 2].clamp(min=1e-4)
    u = F*mc[:, 0]/z + PP[0]; v = F*mc[:, 1]/z + PP[1]
    N = mean.shape[0]
    J = torch.zeros(N, 2, 3)
    J[:, 0, 0] = F/z; J[:, 0, 2] = -F*mc[:, 0]/z**2
    J[:, 1, 1] = F/z; J[:, 1, 2] = -F*mc[:, 1]/z**2
    Sig_cam = torch.einsum('ij,njk,lk->nil', Rv, Sig3, Rv)
    Sig2 = J @ Sig_cam @ J.transpose(-1, -2)
    Sig2 = Sig2 + 0.3*torch.eye(2)               # EWA low-pass
    a_, b_, c_ = Sig2[:, 0, 0], Sig2[:, 0, 1], Sig2[:, 1, 1]
    det = a_*c_ - b_*b_ + 1e-9
    return u, v, mc[:, 2], (c_/det, -b_/det, a_/det)


def render(P, Rv, tv, PX, PY):
    u, v, zc, (ca, cb, cc) = project(P, Rv, tv)
    order = torch.argsort(zc).tolist()           # near -> far
    C = torch.zeros(H, W); T = torch.ones(H, W)
    op, col = torch.sigmoid(P["op"]), torch.sigmoid(P["col"])
    for i in order:
        if zc[i] <= 0:           # behind camera
            continue
        dx, dy = PX - u[i], PY - v[i]
        al = (op[i] * torch.exp(-0.5*(ca[i]*dx*dx + 2*cb[i]*dx*dy + cc[i]*dy*dy))).clamp(max=0.99)
        C = C + T*al*col[i]; T = T*(1-al)
    return C


def scene(seed, n):
    g = torch.Generator().manual_seed(seed)
    return {"mean": (torch.rand(n, 3, generator=g)-0.5)*5,
            "scale": torch.log(0.25 + torch.rand(n, 3, generator=g)*0.35),
            "quat": torch.randn(n, 4, generator=g),
            "op": torch.logit(0.5 + torch.rand(n, generator=g)*0.3),
            "col": torch.logit(0.3 + torch.rand(n, generator=g)*0.5)}


def cameras(n, dist=8.0, radius=6.0, seed=0):
    g = torch.Generator().manual_seed(seed); cams = []
    for i in range(n):
        ang = 2*math.pi*i/n + 0.1
        eye = torch.tensor([radius*math.cos(ang), 2.0*math.sin(ang*0.7), radius*math.sin(ang)]) \
              * (dist/ (radius+ 1e-9)) if False else torch.tensor(
              [radius*math.cos(ang), 1.5, radius*math.sin(ang)])
        cams.append(look_at(eye, torch.zeros(3)))
    return cams


def main():
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    PX, PY = jj.double(), ii.double()
    N = 24
    GT = scene(1, N)
    train_cams = cameras(5, seed=0)
    novel_cam = look_at(torch.tensor([6*math.cos(0.8), -1.2, 6*math.sin(0.8)]), torch.zeros(3))  # unseen pose
    targets = [render(GT, R, t, PX, PY).detach() for (R, t) in train_cams]
    novel_target = render(GT, *novel_cam, PX, PY).detach()

    # ---- verify: projection geometry sanity ----
    pin = {k: GT[k][:1].clone() for k in GT}; pin["mean"] = torch.zeros(1, 3)  # Gaussian at origin
    R0, t0 = train_cams[0]
    u, v, zc, _ = project(pin, R0, t0)
    on_axis = abs(float(u[0]) - W/2) < 1.0 and abs(float(v[0]) - H/2) < 1.0  # origin -> principal pt
    print(f"projection sanity: origin->screen=({float(u[0]):.1f},{float(v[0]):.1f}) exp=({W/2},{H/2}) "
          f"depth={float(zc[0]):.2f} -> {'OK' if on_axis else 'FAIL'}")

    # ---- init = perturbed GT (isolate 3D-loop closure from densification) ----
    gp = torch.Generator().manual_seed(5)
    P = {k: GT[k].clone() for k in GT}
    P["mean"] = P["mean"] + torch.randn(N, 3, generator=gp)*0.25
    P["scale"] = P["scale"] + torch.randn(N, 3, generator=gp)*0.1
    P["quat"] = P["quat"] + torch.randn(N, 4, generator=gp)*0.1
    P["op"] = P["op"] + torch.randn(N, generator=gp)*0.1
    P["col"] = P["col"] + torch.randn(N, generator=gp)*0.1
    for k in P: P[k].requires_grad_()
    opt = torch.optim.Adam([
        {"params": [P["mean"]], "lr": 0.03}, {"params": [P["scale"]], "lr": 0.01},
        {"params": [P["quat"]], "lr": 0.01}, {"params": [P["op"]], "lr": 0.02},
        {"params": [P["col"]], "lr": 0.02}])

    def psnr(a, b):
        mse = float(((a-b)**2).mean()); return 10*math.log10(float(b.max())**2/max(mse, 1e-12))

    STEPS = 250
    for step in range(1, STEPS+1):
        opt.zero_grad()
        loss = sum(((render(P, R, t, PX, PY) - tg)**2).mean() for (R, t), tg in zip(train_cams, targets))
        loss.backward(); opt.step()
        if step == 1 or step % 50 == 0:
            with torch.no_grad():
                tr = sum(psnr(render(P, R, t, PX, PY), tg) for (R, t), tg in zip(train_cams, targets))/len(train_cams)
            print(f"  step {step:3d}  loss={float(loss.detach()):.5f}  train_PSNR={tr:.1f} dB")

    with torch.no_grad():
        tr = sum(psnr(render(P, R, t, PX, PY), tg) for (R, t), tg in zip(train_cams, targets))/len(train_cams)
        nv = psnr(render(P, *novel_cam, PX, PY), novel_target)
    print(f"final  train_PSNR={tr:.1f} dB   NOVEL-view PSNR={nv:.1f} dB")
    print("TRAIN3D_OK" if on_axis and tr > 35 and nv > 30 else "TRAIN3D_FAIL")


if __name__ == "__main__":
    main()
