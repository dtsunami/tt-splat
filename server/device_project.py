#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Device projection FORWARD (Stage B of the device-resident loop): mirror train_real.project_general in
batched ttnn ops over [N] component tensors, so projection runs on the Blackhole reading device-resident
params (no host readback). Per-Gaussian 3x3/2x2 matrix math is expanded into elementwise ttnn ops with
the (constant per frame) camera Rv/tv/fx.. folded in as host scalars. Golden = project_general + cov3d
(quat_to_rot conventions from train3d.py).

  u,v,zc,(a,b,c) = project_geom(dev, mean, scale, quat, cam)   # ttnn [N] tensors out
"""
from __future__ import annotations
import torch
import ttnn


def _dt(dev, t):
    return ttnn.from_torch(t.float().reshape(-1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)


def _is_tt(t):
    return isinstance(t, ttnn.Tensor)


def _col(dev, t, c):
    """Column c of a param as a [N] ttnn tensor. Accepts host torch [N]/[N,C] OR device-resident ttnn
    [N]/[N,C] (sliced ON DEVICE — no host readback, the Stage E device-resident path)."""
    if _is_tt(t):
        if len(t.shape) == 1:
            return t
        N = t.shape[0]
        return ttnn.reshape(ttnn.slice(t, [0, c], [N, c + 1]), (N,))
    return _dt(dev, t[:, c] if t.dim() > 1 else t)


def _shcol(dev, sh, k, c):
    """sh[:,k,c] as [N] ttnn. Host torch [N,K,3] or device-resident ttnn [N,K,3] (on-device slice)."""
    if _is_tt(sh):
        N = sh.shape[0]
        return ttnn.reshape(ttnn.slice(sh, [0, k, c], [N, k + 1, c + 1]), (N,))
    return _dt(dev, sh[:, k, c])


def project_geom(dev, mean, scale, quat, cam, aux=False):
    """mean,scale [N,3], quat [N,4] (host torch or already device-resident host slices). Returns ttnn
    [N] tensors: u, v, zc, (a, b, c) conic — matching project_general's geometry outputs.
    If aux=True also returns a dict of ttnn intermediates the analytic backward (Stage D) consumes."""
    Rv, tv, fx, fy, cx, cy = cam[:6]
    R = [[float(Rv[i][j]) for j in range(3)] for i in range(3)]
    tvs = [float(tv[i]) for i in range(3)]
    fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)

    mx, my, mz = _col(dev, mean, 0), _col(dev, mean, 1), _col(dev, mean, 2)
    # normalize quat on device
    qw, qx, qy, qz = (_col(dev, quat, i) for i in range(4))
    nrm = ttnn.sqrt(ttnn.add(ttnn.add(ttnn.square(qw), ttnn.square(qx)),
                             ttnn.add(ttnn.square(qy), ttnn.square(qz))))
    inrm = ttnn.reciprocal(nrm)
    qw, qx, qy, qz = (ttnn.mul(q, inrm) for q in (qw, qx, qy, qz))

    def sq(a): return ttnn.square(a)
    def m2(a, b): return ttnn.mul(a, b)
    def one_minus(t): return ttnn.add(ttnn.mul(t, -1.0), 1.0)   # ttnn.sub(scalar,tensor) unsupported
    two = 2.0
    # rotation matrix entries (R[i][j]) as [N] tensors
    Rij = {}
    Rij[0, 0] = one_minus(ttnn.mul(ttnn.add(sq(qy), sq(qz)), two))
    Rij[1, 1] = one_minus(ttnn.mul(ttnn.add(sq(qx), sq(qz)), two))
    Rij[2, 2] = one_minus(ttnn.mul(ttnn.add(sq(qx), sq(qy)), two))
    Rij[0, 1] = ttnn.mul(ttnn.sub(m2(qx, qy), m2(qw, qz)), two)
    Rij[1, 0] = ttnn.mul(ttnn.add(m2(qx, qy), m2(qw, qz)), two)
    Rij[0, 2] = ttnn.mul(ttnn.add(m2(qx, qz), m2(qw, qy)), two)
    Rij[2, 0] = ttnn.mul(ttnn.sub(m2(qx, qz), m2(qw, qy)), two)
    Rij[1, 2] = ttnn.mul(ttnn.sub(m2(qy, qz), m2(qw, qx)), two)
    Rij[2, 1] = ttnn.mul(ttnn.add(m2(qy, qz), m2(qw, qx)), two)

    s2 = [ttnn.exp(ttnn.mul(_col(dev, scale, k), two)) for k in range(3)]   # exp(scale)^2 = exp(2 scale)

    # Sig3 = R diag(S2) R^T  (symmetric); Sig3[i][j] = sum_k R[i][k]*S2[k]*R[j][k]
    def sig3(i, j):
        acc = None
        for k in range(3):
            term = ttnn.mul(ttnn.mul(Rij[i, k], s2[k]), Rij[j, k])
            acc = term if acc is None else ttnn.add(acc, term)
        return acc
    S3 = {(i, j): sig3(i, j) for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]}
    S3[1, 0], S3[2, 0], S3[2, 1] = S3[0, 1], S3[0, 2], S3[1, 2]               # symmetric

    # mc = mean @ Rv^T + tv
    comp = {0: mx, 1: my, 2: mz}
    def mc(c):
        acc = ttnn.add(ttnn.mul(comp[0], R[c][0]), ttnn.mul(comp[1], R[c][1]))
        return ttnn.add(ttnn.add(acc, ttnn.mul(comp[2], R[c][2])), tvs[c])
    mc0, mc1, mc2 = mc(0), mc(1), mc(2)
    z = ttnn.maximum(mc2, _dt(dev, torch.full((mean.shape[0],), 1e-4)))
    zi = ttnn.reciprocal(z); zi2 = ttnn.square(zi)
    u = ttnn.add(ttnn.mul(ttnn.mul(mc0, zi), fx), cx)
    v = ttnn.add(ttnn.mul(ttnn.mul(mc1, zi), fy), cy)
    J00 = ttnn.mul(zi, fx); J02 = ttnn.mul(ttnn.mul(mc0, zi2), -fx)
    J11 = ttnn.mul(zi, fy); J12 = ttnn.mul(ttnn.mul(mc1, zi2), -fy)

    # Sig_cam = Rv Sig3 Rv^T ; M = Rv Sig3 ; SC = M Rv^T
    def M(i, k):
        acc = ttnn.mul(S3[0, k], R[i][0])
        acc = ttnn.add(acc, ttnn.mul(S3[1, k], R[i][1]))
        return ttnn.add(acc, ttnn.mul(S3[2, k], R[i][2]))
    Mik = {(i, k): M(i, k) for i in range(3) for k in range(3)}
    def SC(i, l):
        acc = ttnn.mul(Mik[i, 0], R[l][0])
        acc = ttnn.add(acc, ttnn.mul(Mik[i, 1], R[l][1]))
        return ttnn.add(acc, ttnn.mul(Mik[i, 2], R[l][2]))
    SC00, SC01, SC02 = SC(0, 0), SC(0, 1), SC(0, 2)
    SC11, SC12, SC22 = SC(1, 1), SC(1, 2), SC(2, 2)

    # Sig2 = J Sig_cam J^T  (J row0=[J00,0,J02], row1=[0,J11,J12]) + 0.3 I
    a_ = ttnn.add(ttnn.add(ttnn.mul(SC00, ttnn.square(J00)),
                           ttnn.mul(ttnn.mul(ttnn.mul(J00, J02), SC02), 2.0)),
                  ttnn.mul(SC22, ttnn.square(J02)))
    c_ = ttnn.add(ttnn.add(ttnn.mul(SC11, ttnn.square(J11)),
                           ttnn.mul(ttnn.mul(ttnn.mul(J11, J12), SC12), 2.0)),
                  ttnn.mul(SC22, ttnn.square(J12)))
    b_ = ttnn.add(ttnn.add(ttnn.mul(ttnn.mul(J00, J11), SC01), ttnn.mul(ttnn.mul(J00, J12), SC02)),
                  ttnn.add(ttnn.mul(ttnn.mul(J02, J11), SC12), ttnn.mul(ttnn.mul(J02, J12), SC22)))
    a_ = ttnn.add(a_, 0.3); c_ = ttnn.add(c_, 0.3)
    det = ttnn.add(ttnn.sub(ttnn.mul(a_, c_), ttnn.square(b_)), 1e-9)
    di = ttnn.reciprocal(det)
    ca = ttnn.mul(c_, di); cb = ttnn.mul(ttnn.mul(b_, di), -1.0); cc = ttnn.mul(a_, di)
    if not aux:
        return u, v, mc2, (ca, cb, cc)
    A = dict(zi=zi, zi2=zi2, mc0=mc0, mc1=mc1, mc2=mc2, z=z,
             J00=J00, J02=J02, J11=J11, J12=J12,
             SC00=SC00, SC01=SC01, SC02=SC02, SC11=SC11, SC12=SC12, SC22=SC22,
             a_=a_, b_=b_, c_=c_, di=di,
             Rij=Rij, s2=s2, qn=(qw, qx, qy, qz), qnorm=nrm,
             R=R, tvs=tvs, fx=fx, fy=fy, N=mean.shape[0])
    return u, v, mc2, (ca, cb, cc), A


_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
       -0.4570457994644658, 1.445305721320277, -0.5900435899266435]


def project_color(dev, mean, sh, deg, cam, aux=False):
    """Device sh_eval: returns (colR,colG,colB) ttnn [N] in [0,1]. Mirrors train_real.sh_eval.
    If aux=True also returns a dict with dirs (x,y,z), |d|-recip, basis wb and pre-clamp values."""
    Rv, tv = cam[0], cam[1]
    cc = (-Rv.double().T @ tv.double()).tolist()                 # camera center (world)
    mx, my, mz = _col(dev, mean, 0), _col(dev, mean, 1), _col(dev, mean, 2)
    dx = ttnn.sub(mx, cc[0]); dy = ttnn.sub(my, cc[1]); dz = ttnn.sub(mz, cc[2])
    inv = ttnn.reciprocal(ttnn.add(ttnn.sqrt(ttnn.add(ttnn.add(ttnn.square(dx), ttnn.square(dy)),
                                                      ttnn.square(dz))), 1e-9))
    x, y, z = ttnn.mul(dx, inv), ttnn.mul(dy, inv), ttnn.mul(dz, inv)
    sh_c = lambda k, c: _shcol(dev, sh, k, c)
    wb = [None]                                                  # wb[k] = C-weighted spatial basis (k>=1)
    if deg >= 1:
        wb += [ttnn.mul(y, -_C1), ttnn.mul(z, _C1), ttnn.mul(x, -_C1)]
        if deg >= 2:
            xx, yy, zz = ttnn.square(x), ttnn.square(y), ttnn.square(z)
            xy, yz, xz = ttnn.mul(x, y), ttnn.mul(y, z), ttnn.mul(x, z)
            wb += [ttnn.mul(xy, _C2[0]), ttnn.mul(yz, _C2[1]),
                   ttnn.mul(ttnn.sub(ttnn.sub(ttnn.mul(zz, 2.0), xx), yy), _C2[2]),
                   ttnn.mul(xz, _C2[3]), ttnn.mul(ttnn.sub(xx, yy), _C2[4])]
            if deg >= 3:
                wb += [ttnn.mul(ttnn.mul(y, ttnn.sub(ttnn.mul(xx, 3.0), yy)), _C3[0]),
                       ttnn.mul(ttnn.mul(xy, z), _C3[1]),
                       ttnn.mul(ttnn.mul(y, ttnn.sub(ttnn.sub(ttnn.mul(zz, 4.0), xx), yy)), _C3[2]),
                       ttnn.mul(ttnn.mul(z, ttnn.sub(ttnn.sub(ttnn.mul(zz, 2.0), ttnn.mul(xx, 3.0)), ttnn.mul(yy, 3.0))), _C3[3]),
                       ttnn.mul(ttnn.mul(x, ttnn.sub(ttnn.sub(ttnn.mul(zz, 4.0), xx), yy)), _C3[4]),
                       ttnn.mul(ttnn.mul(z, ttnn.sub(xx, yy)), _C3[5]),
                       ttnn.mul(ttnn.mul(x, ttnn.sub(xx, ttnn.mul(yy, 3.0))), _C3[6])]
    out, pre = [], []
    for c in range(3):
        r = ttnn.add(ttnn.mul(sh_c(0, c), _C0), 0.5)             # DC + 0.5
        for k in range(1, len(wb)):
            r = ttnn.add(r, ttnn.mul(wb[k], sh_c(k, c)))
        pre.append(r)
        out.append(ttnn.clamp(r, 0.0, 1.0))
    if not aux:
        return out[0], out[1], out[2]
    A = dict(x=x, y=y, z=z, inv=inv, wb=wb, pre=pre, cc_world=cc)
    return out[0], out[1], out[2], A


def project_op(dev, op_logit):
    return ttnn.sigmoid(op_logit if _is_tt(op_logit) else _dt(dev, op_logit))
