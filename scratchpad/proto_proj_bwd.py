#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Stage D PROTOTYPE (host, pure torch, NO autograd): analytic backward of project_general + sh_eval +
sigmoid.  Given upstream dL/d{u,v,ca,cb,cc,op,col} (what Stage A's raster backward produces), compute
dL/d{mean,scale,quat,op_logit,sh} with explicit formulas, batched over N.

This nails the MATH before porting to ttnn (server/device_project_backward.py).  Golden = torch autograd
of train_real.project_general / sh_eval / sigmoid.  Grad-checked per parameter.
"""
import sys, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path.home() / "tt-splat" / "docs" / "pathclear"))
from train_real import project_general, sh_eval, C0, C1, C2, C3   # noqa: E402
from train3d import quat_to_rot                                    # noqa: E402

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------- SH basis + d/d dir
def sh_basis(dirs, deg):
    """Return (B [N,K], dB [N,K,3]) — basis_k(dir) and d basis_k / d dir.  basis_0 = C0 (const)."""
    N = dirs.shape[0]
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    K = (deg + 1) ** 2
    B = torch.zeros(N, K); dB = torch.zeros(N, K, 3)
    B[:, 0] = C0                                                   # dB[:,0]=0
    if deg >= 1:
        B[:, 1] = -C1 * y;  dB[:, 1, 1] = -C1
        B[:, 2] = C1 * z;   dB[:, 2, 2] = C1
        B[:, 3] = -C1 * x;  dB[:, 3, 0] = -C1
    if deg >= 2:
        xx, yy, zz = x * x, y * y, z * z
        B[:, 4] = C2[0] * x * y;            dB[:, 4, 0] = C2[0] * y;      dB[:, 4, 1] = C2[0] * x
        B[:, 5] = C2[1] * y * z;            dB[:, 5, 1] = C2[1] * z;      dB[:, 5, 2] = C2[1] * y
        B[:, 6] = C2[2] * (2 * zz - xx - yy)
        dB[:, 6, 0] = C2[2] * (-2 * x); dB[:, 6, 1] = C2[2] * (-2 * y); dB[:, 6, 2] = C2[2] * (4 * z)
        B[:, 7] = C2[3] * x * z;            dB[:, 7, 0] = C2[3] * z;      dB[:, 7, 2] = C2[3] * x
        B[:, 8] = C2[4] * (xx - yy);        dB[:, 8, 0] = C2[4] * 2 * x;  dB[:, 8, 1] = C2[4] * (-2 * y)
    if deg >= 3:
        xx, yy, zz = x * x, y * y, z * z
        B[:, 9] = C3[0] * y * (3 * xx - yy)
        dB[:, 9, 0] = C3[0] * y * 6 * x;          dB[:, 9, 1] = C3[0] * (3 * xx - 3 * yy)
        B[:, 10] = C3[1] * x * y * z
        dB[:, 10, 0] = C3[1] * y * z; dB[:, 10, 1] = C3[1] * x * z; dB[:, 10, 2] = C3[1] * x * y
        B[:, 11] = C3[2] * y * (4 * zz - xx - yy)
        dB[:, 11, 0] = C3[2] * y * (-2 * x); dB[:, 11, 1] = C3[2] * (4 * zz - xx - 3 * yy); dB[:, 11, 2] = C3[2] * y * 8 * z
        B[:, 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
        dB[:, 12, 0] = C3[3] * z * (-6 * x); dB[:, 12, 1] = C3[3] * z * (-6 * y); dB[:, 12, 2] = C3[3] * (6 * zz - 3 * xx - 3 * yy)
        B[:, 13] = C3[4] * x * (4 * zz - xx - yy)
        dB[:, 13, 0] = C3[4] * (4 * zz - 3 * xx - yy); dB[:, 13, 1] = C3[4] * x * (-2 * y); dB[:, 13, 2] = C3[4] * x * 8 * z
        B[:, 14] = C3[5] * z * (xx - yy)
        dB[:, 14, 0] = C3[5] * z * 2 * x; dB[:, 14, 1] = C3[5] * z * (-2 * y); dB[:, 14, 2] = C3[5] * (xx - yy)
        B[:, 15] = C3[6] * x * (xx - 3 * yy)
        dB[:, 15, 0] = C3[6] * (3 * xx - 3 * yy); dB[:, 15, 1] = C3[6] * x * (-6 * y)
    return B, dB


# ---------------------------------------------------------------------------- quat_to_rot derivative
def drot_dquat(qn):
    """qn normalized [N,4] (w,x,y,z). Return dR/dqn as [N,3,3,4] (dR[:,i,j,:] = d R_ij / d (w,x,y,z))."""
    N = qn.shape[0]
    w, x, y, z = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    J = torch.zeros(N, 3, 3, 4)
    # R00=1-2(yy+zz)
    J[:, 0, 0, 2] = -4 * y; J[:, 0, 0, 3] = -4 * z
    # R01=2(xy-wz)
    J[:, 0, 1, 1] = 2 * y; J[:, 0, 1, 2] = 2 * x; J[:, 0, 1, 0] = -2 * z; J[:, 0, 1, 3] = -2 * w
    # R02=2(xz+wy)
    J[:, 0, 2, 1] = 2 * z; J[:, 0, 2, 3] = 2 * x; J[:, 0, 2, 0] = 2 * y; J[:, 0, 2, 2] = 2 * w
    # R10=2(xy+wz)
    J[:, 1, 0, 1] = 2 * y; J[:, 1, 0, 2] = 2 * x; J[:, 1, 0, 0] = 2 * z; J[:, 1, 0, 3] = 2 * w
    # R11=1-2(xx+zz)
    J[:, 1, 1, 1] = -4 * x; J[:, 1, 1, 3] = -4 * z
    # R12=2(yz-wx)
    J[:, 1, 2, 2] = 2 * z; J[:, 1, 2, 3] = 2 * y; J[:, 1, 2, 0] = -2 * x; J[:, 1, 2, 1] = -2 * w
    # R20=2(xz-wy)
    J[:, 2, 0, 1] = 2 * z; J[:, 2, 0, 3] = 2 * x; J[:, 2, 0, 0] = -2 * y; J[:, 2, 0, 2] = -2 * w
    # R21=2(yz+wx)
    J[:, 2, 1, 2] = 2 * z; J[:, 2, 1, 3] = 2 * y; J[:, 2, 1, 0] = 2 * x; J[:, 2, 1, 1] = 2 * w
    # R22=1-2(xx+yy)
    J[:, 2, 2, 1] = -4 * x; J[:, 2, 2, 2] = -4 * y
    return J


# ---------------------------------------------------------------------------- analytic backward
def project_backward(P, cam, up):
    """up = dict of upstream grads: u,v,ca,cb,cc,op,colR,colG,colB  (each [N]).  Returns dL/dP dict."""
    Rv, tv, fx, fy, cx, cy = cam[:6]
    Rv = Rv.double(); tv = tv.double()
    mean, scale, quat = P["mean"], P["scale"], P["quat"]
    N = mean.shape[0]; deg = P["deg"]

    # ---- recompute forward intermediates (mirror project_general) ----
    qn = quat / quat.norm(dim=-1, keepdim=True)
    R = quat_to_rot(qn)                                   # [N,3,3]
    S2 = torch.exp(scale) ** 2                            # [N,3]
    Sig3 = R @ (S2[..., None] * R.transpose(-1, -2))      # [N,3,3]
    mc = mean @ Rv.T + tv                                 # [N,3]
    z = mc[:, 2].clamp(min=1e-4)
    zmask = (mc[:, 2] > 1e-4).double()
    mc0, mc1 = mc[:, 0], mc[:, 1]
    J = torch.zeros(N, 2, 3)
    J[:, 0, 0] = fx / z; J[:, 0, 2] = -fx * mc0 / z ** 2
    J[:, 1, 1] = fy / z; J[:, 1, 2] = -fy * mc1 / z ** 2
    Sig_cam = torch.einsum('ij,njk,lk->nil', Rv, Sig3, Rv)
    Sig2 = J @ Sig_cam @ J.transpose(-1, -2)
    a_ = Sig2[:, 0, 0] + 0.3; b_ = Sig2[:, 0, 1]; c_ = Sig2[:, 1, 1] + 0.3
    det = a_ * c_ - b_ * b_ + 1e-9

    # ---- color forward (for clamp mask + dirs) ----
    cam_center = -Rv.T @ tv
    d = mean - cam_center
    dn = d.norm(dim=-1, keepdim=True) + 1e-9
    dirs = d / dn
    Bk, dBk = sh_basis(dirs, deg)                         # [N,K], [N,K,3]
    sh = P["sh"]
    pre = (Bk[:, :, None] * sh).sum(dim=1) + 0.5          # [N,3]
    cmask = ((pre > 0) & (pre < 1)).double()              # clamp passthrough mask

    gP = {}

    # ===== (1) opacity: op = sigmoid(op_logit) =====
    op = torch.sigmoid(P["op"])
    gP["op"] = up["op"] * op * (1 - op)

    # ===== (2) SH color =====
    gcol = torch.stack([up["colR"], up["colG"], up["colB"]], dim=-1)  # [N,3]
    gpre = gcol * cmask                                   # [N,3]
    gP["sh"] = Bk[:, :, None] * gpre[:, None, :]          # [N,K,3]
    # color -> dirs -> mean
    gdirs_col = torch.einsum('nc,nkd,nkc->nd', gpre, dBk, sh)   # [N,3]

    # ===== (3) conic (ca,cb,cc) -> Sig2 entries (a_,b_,c_) =====
    D = det; D2 = D * D
    gca, gcb, gcc = up["ca"], up["cb"], up["cc"]
    ga = gca * (-c_ * c_ / D2) + gcb * (b_ * c_ / D2) + gcc * (1 / D - a_ * c_ / D2)
    gb = gca * (2 * b_ * c_ / D2) + gcb * (-1 / D - 2 * b_ * b_ / D2) + gcc * (2 * a_ * b_ / D2)
    gc = gca * (1 / D - a_ * c_ / D2) + gcb * (a_ * b_ / D2) + gcc * (-a_ * a_ / D2)

    # ===== (4) Sig2 = J SC J^T -> SC (6) and J (4 nonzero) =====
    J00, J02, J11, J12 = J[:, 0, 0], J[:, 0, 2], J[:, 1, 1], J[:, 1, 2]
    SC = Sig_cam
    SC00, SC01, SC02 = SC[:, 0, 0], SC[:, 0, 1], SC[:, 0, 2]
    SC11, SC12, SC22 = SC[:, 1, 1], SC[:, 1, 2], SC[:, 2, 2]
    gSC = torch.zeros(N, 3, 3)
    gSC[:, 0, 0] = ga * J00 * J00
    gSC[:, 1, 1] = gc * J11 * J11
    gSC[:, 2, 2] = ga * J02 * J02 + gb * J02 * J12 + gc * J12 * J12
    s01 = gb * J00 * J11
    s02 = ga * 2 * J00 * J02 + gb * J00 * J12
    s12 = gb * J02 * J11 + gc * 2 * J11 * J12
    gSC[:, 0, 1] = gSC[:, 1, 0] = s01 / 2                 # split symmetric off-diag (full-matrix convention)
    gSC[:, 0, 2] = gSC[:, 2, 0] = s02 / 2
    gSC[:, 1, 2] = gSC[:, 2, 1] = s12 / 2
    gJ00 = ga * (2 * J00 * SC00 + 2 * J02 * SC02) + gb * (J11 * SC01 + J12 * SC02)
    gJ02 = ga * (2 * J00 * SC02 + 2 * J02 * SC22) + gb * (J11 * SC12 + J12 * SC22)
    gJ11 = gc * (2 * J11 * SC11 + 2 * J12 * SC12) + gb * (J00 * SC01 + J02 * SC12)
    gJ12 = gc * (2 * J11 * SC12 + 2 * J12 * SC22) + gb * (J00 * SC02 + J02 * SC22)

    # ===== (5) J & uv -> mc -> mean =====
    gmc0 = up["u"] * (fx / z) + gJ02 * (-fx / z ** 2)
    gmc1 = up["v"] * (fy / z) + gJ12 * (-fy / z ** 2)
    gz = (up["u"] * (-fx * mc0 / z ** 2) + up["v"] * (-fy * mc1 / z ** 2)
          + gJ00 * (-fx / z ** 2) + gJ02 * (2 * fx * mc0 / z ** 3)
          + gJ11 * (-fy / z ** 2) + gJ12 * (2 * fy * mc1 / z ** 3))
    gmc2 = gz * zmask
    gmc = torch.stack([gmc0, gmc1, gmc2], dim=-1)         # [N,3]
    gmean_geom = gmc @ Rv                                 # dL/dmean = Rv^T gmc -> [N,3]

    # ===== (6) Sig_cam -> Sig3 -> {scale, quat} =====
    gSig3 = torch.einsum('ji,njk,kl->nil', Rv, gSC, Rv)   # Rv^T gSC Rv
    # Sig3 = R D R^T :  dL/dR = (G+G^T) R D ;  dL/dD_kk = (R^T G R)_kk
    G = gSig3
    GR = (G + G.transpose(-1, -2)) @ (R * S2[:, None, :])  # (G+G^T) R diag(S2)  -> dL/dR  [N,3,3]
    RtGR = R.transpose(-1, -2) @ G @ R                     # [N,3,3]
    gS2 = torch.diagonal(RtGR, dim1=-2, dim2=-1)           # [N,3]
    gP["scale"] = gS2 * 2 * S2                             # S2=exp(2 scale) -> dL/dscale

    # dL/dR -> dL/dqn -> dL/dq
    dRq = drot_dquat(qn)                                   # [N,3,3,4]
    gqn = torch.einsum('nij,nijk->nk', GR, dRq)            # [N,4]
    # normalization qn = q/|q| : dL/dq = (I - qn qn^T)/|q| @ gqn
    qnorm = quat.norm(dim=-1, keepdim=True)
    proj = gqn - (gqn * qn).sum(-1, keepdim=True) * qn
    gP["quat"] = proj / qnorm

    # ===== mean total = geom + color(dirs) =====
    # dirs -> mean :  ddir/dmean = (I - dirs dirs^T)/|d|
    gmean_col = (gdirs_col - (gdirs_col * dirs).sum(-1, keepdim=True) * dirs) / dn
    gP["mean"] = gmean_geom + gmean_col
    return gP


# ---------------------------------------------------------------------------- grad-check
def main():
    torch.manual_seed(0)
    N, deg = 40, 3
    K = (deg + 1) ** 2
    # random forward-facing scene
    P = {
        "mean": (torch.randn(N, 3) * 0.4 + torch.tensor([0., 0., 4.])).requires_grad_(True),
        "scale": (torch.randn(N, 3) * 0.3 - 1.5).requires_grad_(True),
        "quat": (torch.randn(N, 4)).requires_grad_(True),
        "op": (torch.randn(N) * 0.5).requires_grad_(True),
        "sh": (torch.randn(N, K, 3) * 0.2).requires_grad_(True),
        "deg": deg,
    }
    Rv = torch.eye(3); tv = torch.zeros(3); fx = fy = 100.; cx = cy = 48.
    cam = (Rv, tv, fx, fy, cx, cy)

    # forward + random upstream grads -> scalar loss for autograd golden
    u, v, zc, (ca, cb, cc) = project_general(P, Rv, tv, fx, fy, cx, cy)
    cam_center = -Rv.T @ tv
    dirs = P["mean"] - cam_center
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-9)
    col = sh_eval(P["sh"], dirs, deg)
    op = torch.sigmoid(P["op"])
    torch.manual_seed(1)
    g = {k: torch.randn(N) for k in ("u", "v", "ca", "cb", "cc", "op")}
    gcol = torch.randn(N, 3)
    L = (g["u"] * u + g["v"] * v + g["ca"] * ca + g["cb"] * cb + g["cc"] * cc + g["op"] * op
         + (gcol * col).sum(-1)).sum()
    L.backward()
    ref = {k: P[k].grad.clone() for k in ("mean", "scale", "quat", "op", "sh")}

    up = dict(u=g["u"], v=g["v"], ca=g["ca"], cb=g["cb"], cc=g["cc"], op=g["op"],
              colR=gcol[:, 0], colG=gcol[:, 1], colB=gcol[:, 2])
    Pd = {k: (P[k].detach() if torch.is_tensor(P[k]) else P[k]) for k in P}
    got = project_backward(Pd, cam, up)

    print(f"=== Stage D host analytic backward grad-check  (N={N}, deg={deg}) ===")
    allok = True
    for k in ("op", "sh", "mean", "scale", "quat"):
        r, gt = ref[k], got[k]
        rel = (gt - r).norm() / (r.norm() + 1e-12)
        ok = rel < 1e-9
        allok &= ok
        print(f"  {k:6s} rel={rel:.2e}  {'OK' if ok else 'FAIL'}")
    print("ALL OK" if allok else "SOME FAILED")


if __name__ == "__main__":
    main()
