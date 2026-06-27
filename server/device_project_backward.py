#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Device projection BACKWARD (Stage D of the device-resident loop): the analytic backward of
project_general + sh_eval + sigmoid, in batched ttnn over [N], consuming the aux intermediates emitted by
device_project.project_geom/project_color (aux=True).  Given upstream dL/d{u,v,ca,cb,cc,op,col} (what the
Stage A raster backward produces per Gaussian), returns dL/d{mean,scale,quat,op_logit,sh}.

Math mirrors scratchpad/proto_proj_bwd.py, which is grad-checked vs torch autograd to ~1e-12.
Golden = torch autograd of train_real.project_general/sh_eval.  Device target (fp32): rel err < ~1e-2.
"""
from __future__ import annotations
import torch
import ttnn
from backend import DEFAULT
from device_project import project_geom, project_color, project_op, _dt, _shcol, _C0, _C1, _C2, _C3


def _t2t(dev, t):                       # torch/numpy [N] -> ttnn [N]
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    return ttnn.from_torch(t.float().reshape(-1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)


def _g(t):                              # ttnn [N] -> torch [N] (f64)
    return ttnn.to_torch(t).flatten().double()


# NOTE: the m()/ad()/neg()/s1() compute aliases are now BACKEND-LOCAL (bound to B inside each function that
# needs them), so the same code traces or dispatches. Structural _asm/_g/_t2t stay direct ttnn (marshalling).
def _aliases(B):
    """Backend-bound elementwise aliases: m=mul, ad=add, neg=*-1, s1=tensor*scalar."""
    return B.mul, B.add, B.neg, (lambda a, k: B.mul(a, float(k)))


def _asm(cols, shape=None):
    """Assemble a list of [N] ttnn tensors into [N, len(cols)] (ON DEVICE), optional reshape."""
    N = cols[0].shape[0]
    out = ttnn.concat([ttnn.reshape(c, (N, 1)) for c in cols], dim=1)
    return ttnn.reshape(out, shape) if shape else out


def _const_like(B, ref, v):
    return B.add(B.mul(ref, 0.0), float(v))


def _basis_deriv(B, x, y, z, deg):
    """k -> (dx,dy,dz) ttnn tensors for k>=1 (C-weighted, matches proto sh_basis.dB)."""
    m, ad, neg, s1 = _aliases(B)
    Z = B.mul(x, 0.0)
    K = lambda v: _const_like(B, Z, v)
    out = {}
    if deg >= 1:
        out[1] = (Z, K(-_C1), Z)
        out[2] = (Z, Z, K(_C1))
        out[3] = (K(-_C1), Z, Z)
    if deg >= 2:
        out[4] = (s1(y, _C2[0]), s1(x, _C2[0]), Z)
        out[5] = (Z, s1(z, _C2[1]), s1(y, _C2[1]))
        out[6] = (s1(x, -2 * _C2[2]), s1(y, -2 * _C2[2]), s1(z, 4 * _C2[2]))
        out[7] = (s1(z, _C2[3]), Z, s1(x, _C2[3]))
        out[8] = (s1(x, 2 * _C2[4]), s1(y, -2 * _C2[4]), Z)
    if deg >= 3:
        xx, yy, zz = B.square(x), B.square(y), B.square(z)
        xy, xz, yz = m(x, y), m(x, z), m(y, z)
        out[9] = (s1(xy, 6 * _C3[0]), s1(B.sub(xx, yy), 3 * _C3[0]), Z)
        out[10] = (s1(yz, _C3[1]), s1(xz, _C3[1]), s1(xy, _C3[1]))
        out[11] = (s1(xy, -2 * _C3[2]),
                   s1(B.sub(B.sub(s1(zz, 4), xx), s1(yy, 3)), _C3[2]),
                   s1(yz, 8 * _C3[2]))
        out[12] = (s1(xz, -6 * _C3[3]), s1(yz, -6 * _C3[3]),
                   s1(B.sub(B.sub(s1(zz, 6), s1(xx, 3)), s1(yy, 3)), _C3[3]))
        out[13] = (s1(B.sub(B.sub(s1(zz, 4), s1(xx, 3)), yy), _C3[4]),
                   s1(xy, -2 * _C3[4]), s1(xz, 8 * _C3[4]))
        out[14] = (s1(xz, 2 * _C3[5]), s1(yz, -2 * _C3[5]), s1(B.sub(xx, yy), _C3[5]))
        out[15] = (s1(B.sub(xx, yy), 3 * _C3[6]), s1(xy, -6 * _C3[6]), Z)
    return out


def _color_op_backward(B, shcol, up, AC, op_sig, deg):
    """Opacity + SH-color backward — the 'easy half' of Stage D (pure muls/adds + the >0/<1 cmask, NO
    transcendentals/reciprocals). One source for all three backends: ttnn (TtnnBackend), the fp64 oracle
    (NumpyBackend), or the DAG recorder (TraceBackend -> tile-compiler). Operands are backend values.
      shcol(k,c) -> sh[k,c];  up{op,colR,colG,colB};  AC{pre[3], wb[1..K-1], x,y,z, inv};
      op_sig = sigmoid(op_logit).   Returns (gop, gsh_t{(k,c)->[N]}, gmean_color[3])."""
    m, ad, neg, s1 = _aliases(B)
    K = (deg + 1) ** 2
    # (1) opacity:  gop = up_op * sigmoid * (1 - sigmoid)
    gop = m(m(up["op"], op_sig), ad(neg(op_sig), 1.0))
    # (2) SH:  gpre_c = up_col_c * (0 < pre_c < 1);  gsh[k,c] = gpre_c * wb_k  (wb_0 = C0)
    cmask = [m(B.gt(AC["pre"][c], 0.0), B.lt(AC["pre"][c], 1.0)) for c in range(3)]
    gpre = [m(up[f"col{ch}"], cmask[i]) for i, ch in enumerate("RGB")]
    gsh_t = {}
    for c in range(3):
        gsh_t[(0, c)] = s1(gpre[c], _C0)
        for k in range(1, K):
            gsh_t[(k, c)] = m(gpre[c], AC["wb"][k])
    # color -> dirs -> mean:  gdir_d = sum_c gpre_c * sum_k sh[k,c]*dB[k,d];  then project off the view dir
    db = _basis_deriv(B, AC["x"], AC["y"], AC["z"], deg)
    Z = B.mul(AC["x"], 0.0)
    gdir = [B.add(Z, 0.0) for _ in range(3)]
    for c in range(3):
        sc = [B.add(Z, 0.0) for _ in range(3)]            # sum_k sh[k,c]*dB[k,d]
        for k in range(1, K):
            shkc = shcol(k, c)
            for d in range(3):
                sc[d] = ad(sc[d], m(shkc, db[k][d]))
        for d in range(3):
            gdir[d] = ad(gdir[d], m(gpre[c], sc[d]))
    dirs = [AC["x"], AC["y"], AC["z"]]; inv = AC["inv"]
    gdot = None
    for d in range(3):
        t = m(gdir[d], dirs[d]); gdot = t if gdot is None else ad(gdot, t)
    gmean_color = [m(B.sub(gdir[d], m(gdot, dirs[d])), inv) for d in range(3)]
    return gop, gsh_t, gmean_color


def project_backward(dev, P, cam, up, aux=None, return_ttnn=False, backend=None):
    """P: dict with mean,scale,quat,op,sh (host torch OR device-resident ttnn), deg.
    cam: (Rv,tv,fx,fy,cx,cy).  up: dict of upstream grads u,v,ca,cb,cc,op,colR,colG,colB (host torch [N]).
    aux: optional (A_geom, A_color) from the Stage B forward (aux=True) — pass to SKIP the forward
    recompute (the Stage E device-resident path; the forward already produced them).
    return_ttnn: if True, grads stay DEVICE-RESIDENT (assembled ttnn [N,C]) to feed device Adam with NO
    host round-trip (8.6x faster Adam at scale).  Else returns host torch grads.
    Returns gP dict: mean[N,3], scale[N,3], quat[N,4], op[N], sh[N,K,3]."""
    B = backend or DEFAULT
    m, ad, neg, s1 = _aliases(B)
    mean, scale, quat, sh, deg = P["mean"], P["scale"], P["quat"], P["sh"], P["deg"]
    N = mean.shape[0]; K = sh.shape[1]
    Rv = cam[0]
    if aux is None:
        _, _, _, _, A = project_geom(dev, mean, scale, quat, cam, aux=True, backend=backend)
        _, _, _, AC = project_color(dev, mean, sh, deg, cam, aux=True, backend=backend)
    else:
        A, AC = aux
    U = {k: (up[k] if isinstance(up[k], ttnn.Tensor) else _t2t(dev, up[k])) for k in up}

    # ===== (1)+(2) opacity + SH-color backward (extracted -> one source for ttnn/numpy/trace; fusable) =====
    op_sig = project_op(dev, P["op"], backend=backend)
    gop, gsh_t, gmean_color = _color_op_backward(
        B, lambda k, c: _shcol(dev, sh, k, c), U, AC, op_sig, deg)

    # ===== (3) conic -> Sig2 (a_,b_,c_) =====
    a_, b_, c_, di = A["a_"], A["b_"], A["c_"], A["di"]
    di2 = m(di, di)
    gca, gcb, gcc = U["ca"], U["cb"], U["cc"]
    cc2 = m(c_, c_); aa2 = m(a_, a_); ac = m(a_, c_); bc = m(b_, c_); ab = m(a_, b_); bb = m(b_, b_)
    ga = ad(ad(m(gca, neg(m(cc2, di2))), m(gcb, m(bc, di2))), m(gcc, B.sub(di, m(ac, di2))))
    gb = ad(ad(m(gca, s1(m(bc, di2), 2.0)), m(gcb, ad(neg(di), neg(s1(m(bb, di2), 2.0))))),
            m(gcc, s1(m(ab, di2), 2.0)))
    gc = ad(ad(m(gca, B.sub(di, m(ac, di2))), m(gcb, m(ab, di2))), m(gcc, neg(m(aa2, di2))))

    # ===== (4) Sig2 = J SC J^T -> SC(3x3) and J(00,02,11,12) =====
    J00, J02, J11, J12 = A["J00"], A["J02"], A["J11"], A["J12"]
    SC = {(0, 0): A["SC00"], (0, 1): A["SC01"], (0, 2): A["SC02"],
          (1, 1): A["SC11"], (1, 2): A["SC12"], (2, 2): A["SC22"]}
    SC[1, 0], SC[2, 0], SC[2, 1] = SC[0, 1], SC[0, 2], SC[1, 2]
    g = {}
    g[0, 0] = m(ga, m(J00, J00))
    g[1, 1] = m(gc, m(J11, J11))
    g[2, 2] = ad(ad(m(ga, m(J02, J02)), m(gb, m(J02, J12))), m(gc, m(J12, J12)))
    s01 = m(gb, m(J00, J11))
    s02 = ad(m(ga, s1(m(J00, J02), 2.0)), m(gb, m(J00, J12)))
    s12 = ad(m(gb, m(J02, J11)), m(gc, s1(m(J11, J12), 2.0)))
    g[0, 1] = g[1, 0] = s1(s01, 0.5)
    g[0, 2] = g[2, 0] = s1(s02, 0.5)
    g[1, 2] = g[2, 1] = s1(s12, 0.5)
    gJ00 = ad(m(ga, ad(s1(m(J00, SC[0, 0]), 2), s1(m(J02, SC[0, 2]), 2))),
              m(gb, ad(m(J11, SC[0, 1]), m(J12, SC[0, 2]))))
    gJ02 = ad(m(ga, ad(s1(m(J00, SC[0, 2]), 2), s1(m(J02, SC[2, 2]), 2))),
              m(gb, ad(m(J11, SC[1, 2]), m(J12, SC[2, 2]))))
    gJ11 = ad(m(gc, ad(s1(m(J11, SC[1, 1]), 2), s1(m(J12, SC[1, 2]), 2))),
              m(gb, ad(m(J00, SC[0, 1]), m(J02, SC[1, 2]))))
    gJ12 = ad(m(gc, ad(s1(m(J11, SC[1, 2]), 2), s1(m(J12, SC[2, 2]), 2))),
              m(gb, ad(m(J00, SC[0, 2]), m(J02, SC[2, 2]))))

    # ===== (5) J & uv -> mc -> mean(geom) =====
    zi, zi2, mc0, mc1, mc2 = A["zi"], A["zi2"], A["mc0"], A["mc1"], A["mc2"]
    fx, fy = A["fx"], A["fy"]
    zi3 = m(zi2, zi)
    gmc0 = ad(m(U["u"], s1(zi, fx)), m(gJ02, s1(zi2, -fx)))
    gmc1 = ad(m(U["v"], s1(zi, fy)), m(gJ12, s1(zi2, -fy)))
    gz = ad(ad(ad(m(U["u"], s1(m(mc0, zi2), -fx)), m(U["v"], s1(m(mc1, zi2), -fy))),
               ad(m(gJ00, s1(zi2, -fx)), m(gJ02, s1(m(mc0, zi3), 2 * fx)))),
            ad(m(gJ11, s1(zi2, -fy)), m(gJ12, s1(m(mc1, zi3), 2 * fy))))
    zmask = B.gt(mc2, 1e-4)
    gmc2 = m(gz, zmask)
    gmc = [gmc0, gmc1, gmc2]
    Rvh = [[float(Rv[i][j]) for j in range(3)] for i in range(3)]
    gmean_geom = []
    for j in range(3):
        acc = s1(gmc[0], Rvh[0][j])
        acc = ad(acc, s1(gmc[1], Rvh[1][j]))
        acc = ad(acc, s1(gmc[2], Rvh[2][j]))
        gmean_geom.append(acc)

    # ===== (6) SC -> Sig3 -> {scale, quat} =====
    # gSig3 = Rv^T gSC Rv :  G[i][l] = sum_jk Rv[j][i] Rv[k][l] gSC[j][k]
    G = {}
    for i in range(3):
        for l in range(3):
            acc = None
            for j in range(3):
                for k in range(3):
                    w = Rvh[j][i] * Rvh[k][l]
                    if w == 0.0:
                        continue
                    term = s1(g[j, k], w)
                    acc = term if acc is None else ad(acc, term)
            G[i, l] = acc
    Rg = A["Rij"]; s2 = A["s2"]                          # Gaussian rotation (ttnn), S2 (ttnn)
    # dL/dRg = (G+G^T) Rg diag(S2) :  GR[i][k] = sum_j (G[i][j]+G[j][i]) Rg[j][k] s2[k]
    GR = {}
    for i in range(3):
        for k in range(3):
            acc = None
            for j in range(3):
                gij = ad(G[i, j], G[j, i])
                term = m(gij, Rg[j, k])
                acc = term if acc is None else ad(acc, term)
            GR[i, k] = m(acc, s2[k])
    # gS2_k = (Rg^T G Rg)[k][k] = sum_ij Rg[i][k] G[i][j] Rg[j][k]
    gscale_t = []
    for k in range(3):
        acc = None
        for i in range(3):
            for j in range(3):
                term = m(m(Rg[i, k], G[i, j]), Rg[j, k])
                acc = term if acc is None else ad(acc, term)
        gscale_t.append(m(m(acc, s2[k]), 2.0))           # *2*S2 (S2=exp(2 scale))

    # dL/dRg -> dL/dqn :  gqn_m = sum_ij GR[i][j] dR[i][j]/dqn_m
    qw, qx, qy, qz = A["qn"]
    dR = _drot_dquat_ttnn(B, qw, qx, qy, qz)             # dict (i,j,m) -> ttnn or None
    gqn = []
    for mq in range(4):
        acc = None
        for i in range(3):
            for j in range(3):
                d = dR.get((i, j, mq))
                if d is None:
                    continue
                term = m(GR[i, j], d)
                acc = term if acc is None else ad(acc, term)
        gqn.append(acc)
    # qn = q/|q| : gq = (gqn - (gqn.qn) qn)/|q|
    qnc = [qw, qx, qy, qz]
    dot = None
    for mq in range(4):
        t = m(gqn[mq], qnc[mq])
        dot = t if dot is None else ad(dot, t)
    qninv = B.recip(A["qnorm"])
    gquat_t = []
    for mq in range(4):
        proj = B.sub(gqn[mq], m(dot, qnc[mq]))
        gquat_t.append(m(proj, qninv))

    # ===== mean total = geometry grad + color(dirs) grad (color part from _color_op_backward) =====
    gmean_t = [ad(gmean_geom[dmn], gmean_color[dmn]) for dmn in range(3)]

    if return_ttnn:                                       # DEVICE-RESIDENT grads -> device Adam, no round-trip
        return dict(
            mean=_asm(gmean_t), scale=_asm(gscale_t), quat=_asm(gquat_t), op=gop,
            sh=_asm([gsh_t[(k, c)] for k in range(K) for c in range(3)], shape=(N, K, 3)))
    gsh = torch.zeros(N, K, 3)
    for k in range(K):
        for c in range(3):
            gsh[:, k, c] = _g(gsh_t[(k, c)])
    return dict(mean=_g_stack(gmean_t), scale=_g_stack(gscale_t), quat=_g_stack(gquat_t),
                op=_g(gop), sh=gsh)


def _g_stack(cols):
    return torch.stack([_g(c) for c in cols], dim=-1)


def _drot_dquat_ttnn(B, w, x, y, z):
    """dR_ij/dqn_m as ttnn tensors (qn already normalized). Mirrors proto drot_dquat. Key (i,j,m)."""
    m, ad, neg, s1 = _aliases(B)
    d = {}
    def put(i, j, mq, t): d[(i, j, mq)] = t
    # R00=1-2(yy+zz)
    put(0, 0, 2, s1(y, -4)); put(0, 0, 3, s1(z, -4))
    # R01=2(xy-wz)
    put(0, 1, 1, s1(y, 2)); put(0, 1, 2, s1(x, 2)); put(0, 1, 0, s1(z, -2)); put(0, 1, 3, s1(w, -2))
    # R02=2(xz+wy)
    put(0, 2, 1, s1(z, 2)); put(0, 2, 3, s1(x, 2)); put(0, 2, 0, s1(y, 2)); put(0, 2, 2, s1(w, 2))
    # R10=2(xy+wz)
    put(1, 0, 1, s1(y, 2)); put(1, 0, 2, s1(x, 2)); put(1, 0, 0, s1(z, 2)); put(1, 0, 3, s1(w, 2))
    # R11=1-2(xx+zz)
    put(1, 1, 1, s1(x, -4)); put(1, 1, 3, s1(z, -4))
    # R12=2(yz-wx)
    put(1, 2, 2, s1(z, 2)); put(1, 2, 3, s1(y, 2)); put(1, 2, 0, s1(x, -2)); put(1, 2, 1, s1(w, -2))
    # R20=2(xz-wy)
    put(2, 0, 1, s1(z, 2)); put(2, 0, 3, s1(x, 2)); put(2, 0, 0, s1(y, -2)); put(2, 0, 2, s1(w, -2))
    # R21=2(yz+wx)
    put(2, 1, 2, s1(z, 2)); put(2, 1, 3, s1(y, 2)); put(2, 1, 0, s1(x, 2)); put(2, 1, 1, s1(w, 2))
    # R22=1-2(xx+yy)
    put(2, 2, 1, s1(x, -4)); put(2, 2, 2, s1(y, -4))
    return d
