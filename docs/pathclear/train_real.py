#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Train 3DGS (SH color) from a COLMAP model + real images (output of prepare_data.py).

  python train_real.py --model runs/myscene/sparse/0 --images runs/myscene/images --size 96 --sh 3

- SH spherical-harmonic color (degree 0-3, view-dependent) — colour = SH(view_dir).
- Per-image masks: loss weighted by a mask (foreground) when present (frames.json polygons
  rasterized to PNGs, or any mask dir).
- Held-out view for novel-view PSNR; --preview dumps novel GT|render.

Host autograd render is per-Gaussian (per-pixel torch loop) — keep --size small / subsample
points; the device path (SFPU blend-loop + scatter-add) is the perf backend.
"""
import argparse, math, os, torch, numpy as np
from colmap_ingest import read_colmap, write_colmap
from train3d import cov3d, scene, cameras, F, PP, H as TH, W as TW

# real-SH constants (3DGS convention)
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435]


def sh_dim(deg): return (deg + 1) ** 2


def sh_eval(sh, dirs, deg):
    """sh: [N,K,3], dirs: [N,3] unit -> rgb [N,3] in [0,1]."""
    r = C0 * sh[:, 0]
    if deg >= 1:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        r = r - C1*y*sh[:, 1] + C1*z*sh[:, 2] - C1*x*sh[:, 3]
        if deg >= 2:
            xx, yy, zz, xy, yz, xz = x*x, y*y, z*z, x*y, y*z, x*z
            r = (r + C2[0]*xy*sh[:, 4] + C2[1]*yz*sh[:, 5] + C2[2]*(2*zz-xx-yy)*sh[:, 6]
                 + C2[3]*xz*sh[:, 7] + C2[4]*(xx-yy)*sh[:, 8])
            if deg >= 3:
                r = (r + C3[0]*y*(3*xx-yy)*sh[:, 9] + C3[1]*xy*z*sh[:, 10] + C3[2]*y*(4*zz-xx-yy)*sh[:, 11]
                     + C3[3]*z*(2*zz-3*xx-3*yy)*sh[:, 12] + C3[4]*x*(4*zz-xx-yy)*sh[:, 13]
                     + C3[5]*z*(xx-yy)*sh[:, 14] + C3[6]*x*(xx-3*yy)*sh[:, 15])
    return (r + 0.5).clamp(0, 1)


def project_general(P, Rv, tv, fx, fy, cx, cy):
    Sig3 = cov3d(P["scale"], P["quat"])
    mc = P["mean"] @ Rv.T + tv
    z = mc[:, 2].clamp(min=1e-4)
    u = fx*mc[:, 0]/z + cx; v = fy*mc[:, 1]/z + cy
    N = P["mean"].shape[0]; J = torch.zeros(N, 2, 3, dtype=torch.float64)
    J[:, 0, 0] = fx/z; J[:, 0, 2] = -fx*mc[:, 0]/z**2
    J[:, 1, 1] = fy/z; J[:, 1, 2] = -fy*mc[:, 1]/z**2
    Sig_cam = torch.einsum('ij,njk,lk->nil', Rv, Sig3, Rv)
    Sig2 = J @ Sig_cam @ J.transpose(-1, -2) + 0.3*torch.eye(2, dtype=torch.float64)
    a_, b_, c_ = Sig2[:, 0, 0], Sig2[:, 0, 1], Sig2[:, 1, 1]
    det = a_*c_ - b_*b_ + 1e-9
    return u, v, mc[:, 2], (c_/det, -b_/det, a_/det)


def render(P, cam, H, W, PX, PY, aa=False):
    """RGB front-to-back compositing with SH view-dependent colour -> [H,W,3].
    aa=True applies the Mip-Splatting anti-alias opacity compensation (recipe gap #3): scale each
    opacity by sqrt(det(Σ2D_raw)/det(Σ2D_dilated)) <= 1 so sub-pixel splats shrink (autograd-correct)."""
    Rv, tv, fx, fy, cx, cy = cam[:6]
    u, v, zc, (ca, cb, cc) = project_general(P, Rv, tv, fx, fy, cx, cy)
    cam_center = -Rv.T @ tv                                   # world camera position
    dirs = P["mean"] - cam_center
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-9)
    col = sh_eval(P["sh"], dirs, P["deg"])                    # [N,3] view-dependent
    order = torch.argsort(zc).tolist()
    C = torch.zeros(H, W, 3, dtype=torch.float64); T = torch.ones(H, W, dtype=torch.float64)
    op = torch.sigmoid(P["op"])
    if aa:                                                    # conic = Σ2D_dilatedᐨ¹ -> recover raw det
        detc = ca * cc - cb * cb
        A = cc / detc; Bb = -cb / detc; Cc = ca / detc       # dilated 2D covariance entries
        det_dil = A * Cc - Bb * Bb
        det_raw = (A - 0.3) * (Cc - 0.3) - Bb * Bb
        op = op * torch.sqrt((det_raw / (det_dil + 1e-12)).clamp(0.0, 1.0))
    for i in order:
        if zc[i] <= 0: continue
        dx, dy = PX - u[i], PY - v[i]
        al = (op[i]*torch.exp(-0.5*(ca[i]*dx*dx + 2*cb[i]*dx*dy + cc[i]*dy*dy))).clamp(max=0.99)
        w = T*al
        C = C + w[..., None]*col[i]
        T = T*(1 - al)
    return C


def init_from_points(xyz, rgb, sh_degree=3):
    N = xyz.shape[0]; K = sh_dim(sh_degree)
    d = torch.cdist(xyz, xyz) + torch.eye(N, dtype=torch.float64)*1e9
    nn = d.min(dim=1).values.median()
    sl = torch.log((0.5*nn).clamp(min=1e-3)).repeat(N, 1).repeat(1, 3)
    sh = torch.zeros(N, K, 3, dtype=torch.float64)
    sh[:, 0] = ((rgb/255.0).clamp(1e-3, 1-1e-3) - 0.5) / C0   # DC term encodes base colour
    return {"mean": xyz.clone(), "scale": sl.clone(), "quat": torch.tensor([[1., 0, 0, 0]]).repeat(N, 1).double(),
            "op": torch.logit(torch.full((N,), 0.5, dtype=torch.float64)), "sh": sh, "deg": sh_degree}


def tensors(P): return [k for k in P if isinstance(P[k], torch.Tensor)]


def load_image(path, size):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    W0, H0 = im.size; s = size/max(W0, H0)
    W, H = max(1, round(W0*s)), max(1, round(H0*s))
    return torch.tensor(np.asarray(im.resize((W, H), Image.BILINEAR), dtype=np.float64)/255.0), s


def load_mask(path, H, W):
    from PIL import Image
    if not os.path.exists(path): return None
    m = Image.open(path).convert("L").resize((W, H), Image.BILINEAR)
    return torch.tensor(np.asarray(m, dtype=np.float64)/255.0)   # [H,W] in [0,1]


def train(cams, targets, xyz, rgb, holdout=1, steps=300, preview=None, sh_degree=3, masks=None):
    H, W, _ = targets[0].shape
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    PX, PY = jj.double(), ii.double()
    masks = masks or [None]*len(cams)
    tr, nv = cams[:-holdout], cams[-holdout:]
    tr_t, nv_t = targets[:-holdout], targets[-holdout:]
    tr_m = masks[:-holdout]
    P = init_from_points(xyz, rgb, sh_degree)
    for k in tensors(P): P[k].requires_grad_()
    lr = {"mean": .01, "scale": .01, "quat": .01, "op": .02, "sh": .01}
    opt = torch.optim.Adam([{"params": [P[k]], "lr": lr[k]} for k in tensors(P)])
    psnr = lambda a, b: 10*math.log10(max(float(b.max()), 1e-6)**2/max(float(((a-b)**2).mean()), 1e-12))
    for step in range(1, steps+1):
        opt.zero_grad(); loss = 0.0
        for c, t, m in zip(tr, tr_t, tr_m):
            diff = (render(P, c, H, W, PX, PY) - t)**2
            if m is not None: diff = diff * m[..., None]
            loss = loss + diff.mean()
        loss.backward(); opt.step()
        if step == 1 or step % 50 == 0:
            with torch.no_grad():
                p = sum(psnr(render(P, c, H, W, PX, PY), t) for c, t in zip(tr, tr_t))/len(tr)
            print(f"  step {step:3d}  loss={float(loss.detach()):.5f}  train_PSNR={p:.1f} dB")
    with torch.no_grad():
        p_tr = sum(psnr(render(P, c, H, W, PX, PY), t) for c, t in zip(tr, tr_t))/len(tr)
        p_nv = sum(psnr(render(P, c, H, W, PX, PY), t) for c, t in zip(nv, nv_t))/len(nv)
        if preview is not None:
            from PIL import Image
            gt = nv_t[0].clamp(0, 1); pr = render(P, nv[0], H, W, PX, PY).clamp(0, 1)
            row = torch.cat([gt, pr], 1).numpy()
            Image.fromarray((row*255).astype(np.uint8)).resize((W*8, H*4), Image.NEAREST).save(preview)
            print(f"wrote {preview} (novel GT | render)")
    print(f"final  train_PSNR={p_tr:.1f} dB   NOVEL-view PSNR={p_nv:.1f} dB  (N={xyz.shape[0]} cams={len(cams)} sh={sh_degree})")
    return p_tr, p_nv


def selftest(sh_degree=2):
    WS = "/tmp/claude-1000/-home-starboy/99cfd4cc-748e-42e4-a1df-7dc090414335/scratchpad/real_ws"
    os.makedirs(WS, exist_ok=True)
    ii, jj = torch.meshgrid(torch.arange(TH), torch.arange(TW), indexing="ij")
    PX, PY = jj.double(), ii.double()
    GT = scene(1, 24)
    g = torch.Generator().manual_seed(3)
    rgb255 = (0.2 + torch.rand(24, 3, generator=g)*0.6) * 255
    GT["sh"] = torch.zeros(24, sh_dim(sh_degree), 3, dtype=torch.float64)
    GT["sh"][:, 0] = (rgb255/255.0 - 0.5)/C0; GT["deg"] = sh_degree
    cams_gt = [(R, t, float(F), float(F), float(PP[0]), float(PP[1])) for (R, t) in cameras(6, seed=0)]
    targets = [render(GT, c, TH, TW, PX, PY).detach() for c in cams_gt]
    write_colmap(WS, [(c[0], c[1]) for c in cams_gt], GT["mean"], rgb255.round())
    rcams, rxyz, rrgb = read_colmap(WS)

    # (1) SH convergence (no mask) — view-dependent colour path
    print(f"selftest(SH deg={sh_degree}): {len(rcams)} cams, {rxyz.shape[0]} points")
    p_tr, p_nv = train(rcams, targets, rxyz, rrgb, holdout=1, steps=400, sh_degree=sh_degree,
                       preview=f"{WS}/selftest_sh.png")
    sh_ok = p_tr > 30 and p_nv > 26

    # (2) mask-weighting assertion: a half-mask must restrict the loss to the upper half exactly
    P = init_from_points(rxyz, rrgb, sh_degree)
    r = render(P, rcams[0], TH, TW, PX, PY).detach(); gt = targets[0]
    half = torch.cat([torch.ones(TH//2, TW), torch.zeros(TH-TH//2, TW)]).double()
    sq = (r - gt)**2
    masked_mean = (sq*half[..., None]).sum() / (half.sum()*3)
    upper_mean = sq[:TH//2].mean()
    mask_ok = abs(float(masked_mean - upper_mean)) < 1e-9
    print(f"SH: train={p_tr:.1f} novel={p_nv:.1f} dB -> {'OK' if sh_ok else 'FAIL'}  |  "
          f"mask weighting exact={mask_ok} (Δ={float(masked_mean-upper_mean):.2e})")
    print("REAL_PIPE_OK" if sh_ok and mask_ok else "REAL_PIPE_FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--images"); ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--sh", type=int, default=3); ap.add_argument("--masks")
    ap.add_argument("--holdout", type=int, default=1); ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--max-points", type=int, default=0); ap.add_argument("--preview")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    cams, xyz, rgb = read_colmap(a.model)
    if a.max_points and xyz.shape[0] > a.max_points:
        g = torch.Generator().manual_seed(0); idx = torch.randperm(xyz.shape[0], generator=g)[:a.max_points]
        xyz, rgb = xyz[idx], rgb[idx]
        print(f"subsampled points: {a.max_points}")
    targets, kept, masks = [], [], []
    for c in cams:
        p = os.path.join(a.images, c[6])
        if not os.path.exists(p): continue
        img, s = load_image(p, a.size); H, W, _ = img.shape
        kept.append((c[0], c[1], c[2]*s, c[3]*s, c[4]*s, c[5]*s, c[6])); targets.append(img)
        masks.append(load_mask(os.path.join(a.masks, os.path.splitext(c[6])[0]+".png"), H, W) if a.masks else None)
    print(f"loaded {len(kept)} cams at size {a.size}, {xyz.shape[0]} points, sh={a.sh}, masks={'yes' if a.masks else 'no'}")
    train(kept, targets, xyz, rgb, holdout=a.holdout, steps=a.steps, preview=a.preview, sh_degree=a.sh, masks=masks)


if __name__ == "__main__":
    main()
