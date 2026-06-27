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
from backend import DEFAULT


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


def project_geom(dev, mean, scale, quat, cam, aux=False, backend=None):
    """mean,scale [N,3], quat [N,4] (host torch or already device-resident host slices). Returns ttnn
    [N] tensors: u, v, zc, (a, b, c) conic — matching project_general's geometry outputs.
    If aux=True also returns a dict of ttnn intermediates the analytic backward (Stage D) consumes."""
    B = backend or DEFAULT
    Rv, tv, fx, fy, cx, cy = cam[:6]
    R = [[float(Rv[i][j]) for j in range(3)] for i in range(3)]
    tvs = [float(tv[i]) for i in range(3)]
    fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)

    mx, my, mz = _col(dev, mean, 0), _col(dev, mean, 1), _col(dev, mean, 2)
    # normalize quat on device
    qw, qx, qy, qz = (_col(dev, quat, i) for i in range(4))
    nrm = B.sqrt(B.add(B.add(B.square(qw), B.square(qx)),
                             B.add(B.square(qy), B.square(qz))))
    inrm = B.recip(nrm)
    qw, qx, qy, qz = (B.mul(q, inrm) for q in (qw, qx, qy, qz))

    def sq(a): return B.square(a)
    def m2(a, b): return B.mul(a, b)
    def one_minus(t): return B.add(B.mul(t, -1.0), 1.0)   # B.sub(scalar,tensor) unsupported
    two = 2.0
    # rotation matrix entries (R[i][j]) as [N] tensors
    Rij = {}
    Rij[0, 0] = one_minus(B.mul(B.add(sq(qy), sq(qz)), two))
    Rij[1, 1] = one_minus(B.mul(B.add(sq(qx), sq(qz)), two))
    Rij[2, 2] = one_minus(B.mul(B.add(sq(qx), sq(qy)), two))
    Rij[0, 1] = B.mul(B.sub(m2(qx, qy), m2(qw, qz)), two)
    Rij[1, 0] = B.mul(B.add(m2(qx, qy), m2(qw, qz)), two)
    Rij[0, 2] = B.mul(B.add(m2(qx, qz), m2(qw, qy)), two)
    Rij[2, 0] = B.mul(B.sub(m2(qx, qz), m2(qw, qy)), two)
    Rij[1, 2] = B.mul(B.sub(m2(qy, qz), m2(qw, qx)), two)
    Rij[2, 1] = B.mul(B.add(m2(qy, qz), m2(qw, qx)), two)

    s2 = [B.exp(B.mul(_col(dev, scale, k), two)) for k in range(3)]   # exp(scale)^2 = exp(2 scale)

    # Sig3 = R diag(S2) R^T  (symmetric); Sig3[i][j] = sum_k R[i][k]*S2[k]*R[j][k]
    def sig3(i, j):
        acc = None
        for k in range(3):
            term = B.mul(B.mul(Rij[i, k], s2[k]), Rij[j, k])
            acc = term if acc is None else B.add(acc, term)
        return acc
    S3 = {(i, j): sig3(i, j) for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]}
    S3[1, 0], S3[2, 0], S3[2, 1] = S3[0, 1], S3[0, 2], S3[1, 2]               # symmetric

    # mc = mean @ Rv^T + tv
    comp = {0: mx, 1: my, 2: mz}
    def mc(c):
        acc = B.add(B.mul(comp[0], R[c][0]), B.mul(comp[1], R[c][1]))
        return B.add(B.add(acc, B.mul(comp[2], R[c][2])), tvs[c])
    mc0, mc1, mc2 = mc(0), mc(1), mc(2)
    z = B.maximum(mc2, _dt(dev, torch.full((mean.shape[0],), 1e-4)))
    zi = B.recip(z); zi2 = B.square(zi)
    u = B.add(B.mul(B.mul(mc0, zi), fx), cx)
    v = B.add(B.mul(B.mul(mc1, zi), fy), cy)
    J00 = B.mul(zi, fx); J02 = B.mul(B.mul(mc0, zi2), -fx)
    J11 = B.mul(zi, fy); J12 = B.mul(B.mul(mc1, zi2), -fy)

    # Sig_cam = Rv Sig3 Rv^T ; M = Rv Sig3 ; SC = M Rv^T
    def M(i, k):
        acc = B.mul(S3[0, k], R[i][0])
        acc = B.add(acc, B.mul(S3[1, k], R[i][1]))
        return B.add(acc, B.mul(S3[2, k], R[i][2]))
    Mik = {(i, k): M(i, k) for i in range(3) for k in range(3)}
    def SC(i, l):
        acc = B.mul(Mik[i, 0], R[l][0])
        acc = B.add(acc, B.mul(Mik[i, 1], R[l][1]))
        return B.add(acc, B.mul(Mik[i, 2], R[l][2]))
    SC00, SC01, SC02 = SC(0, 0), SC(0, 1), SC(0, 2)
    SC11, SC12, SC22 = SC(1, 1), SC(1, 2), SC(2, 2)

    # Sig2 = J Sig_cam J^T  (J row0=[J00,0,J02], row1=[0,J11,J12]) + 0.3 I
    a_ = B.add(B.add(B.mul(SC00, B.square(J00)),
                           B.mul(B.mul(B.mul(J00, J02), SC02), 2.0)),
                  B.mul(SC22, B.square(J02)))
    c_ = B.add(B.add(B.mul(SC11, B.square(J11)),
                           B.mul(B.mul(B.mul(J11, J12), SC12), 2.0)),
                  B.mul(SC22, B.square(J12)))
    b_ = B.add(B.add(B.mul(B.mul(J00, J11), SC01), B.mul(B.mul(J00, J12), SC02)),
                  B.add(B.mul(B.mul(J02, J11), SC12), B.mul(B.mul(J02, J12), SC22)))
    a_ = B.add(a_, 0.3); c_ = B.add(c_, 0.3)
    det = B.add(B.sub(B.mul(a_, c_), B.square(b_)), 1e-9)
    di = B.recip(det)
    ca = B.mul(c_, di); cb = B.mul(B.mul(b_, di), -1.0); cc = B.mul(a_, di)
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


def project_color(dev, mean, sh, deg, cam, aux=False, backend=None):
    """Device sh_eval: returns (colR,colG,colB) ttnn [N] in [0,1]. Mirrors train_real.sh_eval.
    If aux=True also returns a dict with dirs (x,y,z), |d|-recip, basis wb and pre-clamp values."""
    B = backend or DEFAULT
    Rv, tv = cam[0], cam[1]
    cc = (-Rv.double().T @ tv.double()).tolist()                 # camera center (world)
    mx, my, mz = _col(dev, mean, 0), _col(dev, mean, 1), _col(dev, mean, 2)
    dx = B.sub(mx, cc[0]); dy = B.sub(my, cc[1]); dz = B.sub(mz, cc[2])
    inv = B.recip(B.add(B.sqrt(B.add(B.add(B.square(dx), B.square(dy)),
                                                      B.square(dz))), 1e-9))
    x, y, z = B.mul(dx, inv), B.mul(dy, inv), B.mul(dz, inv)
    sh_c = lambda k, c: _shcol(dev, sh, k, c)
    wb = [None]                                                  # wb[k] = C-weighted spatial basis (k>=1)
    if deg >= 1:
        wb += [B.mul(y, -_C1), B.mul(z, _C1), B.mul(x, -_C1)]
        if deg >= 2:
            xx, yy, zz = B.square(x), B.square(y), B.square(z)
            xy, yz, xz = B.mul(x, y), B.mul(y, z), B.mul(x, z)
            wb += [B.mul(xy, _C2[0]), B.mul(yz, _C2[1]),
                   B.mul(B.sub(B.sub(B.mul(zz, 2.0), xx), yy), _C2[2]),
                   B.mul(xz, _C2[3]), B.mul(B.sub(xx, yy), _C2[4])]
            if deg >= 3:
                wb += [B.mul(B.mul(y, B.sub(B.mul(xx, 3.0), yy)), _C3[0]),
                       B.mul(B.mul(xy, z), _C3[1]),
                       B.mul(B.mul(y, B.sub(B.sub(B.mul(zz, 4.0), xx), yy)), _C3[2]),
                       B.mul(B.mul(z, B.sub(B.sub(B.mul(zz, 2.0), B.mul(xx, 3.0)), B.mul(yy, 3.0))), _C3[3]),
                       B.mul(B.mul(x, B.sub(B.sub(B.mul(zz, 4.0), xx), yy)), _C3[4]),
                       B.mul(B.mul(z, B.sub(xx, yy)), _C3[5]),
                       B.mul(B.mul(x, B.sub(xx, B.mul(yy, 3.0))), _C3[6])]
    out, pre = [], []
    for c in range(3):
        r = B.add(B.mul(sh_c(0, c), _C0), 0.5)             # DC + 0.5
        for k in range(1, len(wb)):
            r = B.add(r, B.mul(wb[k], sh_c(k, c)))
        pre.append(r)
        out.append(B.clamp(r, 0.0, 1.0))
    if not aux:
        return out[0], out[1], out[2]
    A = dict(x=x, y=y, z=z, inv=inv, wb=wb, pre=pre, cc_world=cc)
    return out[0], out[1], out[2], A


def project_op(dev, op_logit, backend=None):
    B = backend or DEFAULT
    return B.sigmoid(op_logit if _is_tt(op_logit) else _dt(dev, op_logit))
