#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Densification pathclear: clone / split / prune (3DGS adaptive density control).

It's a dynamic, data-dependent reshaping of the Gaussian set -> a host/general-purpose
operation (same caveat as bin/sort), sitting on top of the device-proven render+backward (M4).
Here we iterate it in host torch-autograd (fast; device==host already verified) to prove the
operators AND that they help: start with TOO FEW Gaussians and show densification grows the set
to fit a target, beating a fixed-count baseline.

  prune : drop opacity < tau_op
  clone : high positional-grad + small  -> duplicate, nudged
  split : high positional-grad + large  -> 2 children, scale/1.6, jittered
"""
import math, torch

H = W = 64
torch.set_default_dtype(torch.float32)


def conic(sx, sy, th):
    ct, st = torch.cos(th), torch.sin(th)
    # Sigma = R diag(sx^2,sy^2) R^T ; conic = Sigma^-1 (closed form, batched)
    s2x, s2y = sx*sx, sy*sy
    cxx = ct*ct*s2x + st*st*s2y
    cyy = st*st*s2x + ct*ct*s2y
    cxy = ct*st*(s2x - s2y)
    det = cxx*cyy - cxy*cxy + 1e-9
    return cyy/det, -cxy/det, cxx/det          # a,b,c


def render(P, order, PX, PY):
    a, b, c = conic(P["sx"], P["sy"], P["th"])
    C = torch.zeros(H, W); T = torch.ones(H, W)
    for i in order:
        dx, dy = PX-P["cx"][i], PY-P["cy"][i]
        al = torch.sigmoid(P["op"][i]) * torch.exp(-0.5*(a[i]*dx*dx + 2*b[i]*dx*dy + c[i]*dy*dy))
        al = al.clamp(max=0.99)
        C = C + T*al*torch.sigmoid(P["col"][i]); T = T*(1-al)
    return C


def leaves(P):
    return [P[k].requires_grad_() for k in P]


def adam(P, lrs):
    groups = [{"params": [P[k]], "lr": lrs[k]} for k in P]
    return torch.optim.Adam(groups)


def densify(P, gpos, n_max):
    """gpos = accumulated positional-grad magnitude per Gaussian."""
    with torch.no_grad():
        op = torch.sigmoid(P["op"]); scale = torch.maximum(P["sx"], P["sy"])
        keep = op > 0.02                                   # PRUNE low opacity
        tau = gpos.mean() + gpos.std()                     # densify threshold
        big = scale > 6.0
        do_clone = keep & (gpos > tau) & (~big)
        do_split = keep & (gpos > tau) & big
        survive = keep & (~do_split)                       # clone keeps original; split removes it
        new = {k: [P[k][survive]] for k in P}
        if do_clone.any():                                 # CLONE: duplicate, nudged
            for k in P:
                v = P[k][do_clone].clone()
                if k in ("cx", "cy"): v = v + torch.randn_like(v)
                new[k].append(v)
        if do_split.any():                                 # SPLIT: 2 children, scale/1.6, jittered
            for _ in range(2):
                for k in P:
                    v = P[k][do_split].clone()
                    if k in ("sx", "sy"): v = v / 1.6
                    if k == "cx": v = v + torch.randn_like(v)*P["sx"][do_split]
                    if k == "cy": v = v + torch.randn_like(v)*P["sy"][do_split]
                    new[k].append(v)
        out = {k: torch.cat(new[k]) for k in P}
        if out["cx"].numel() > n_max:                      # cap
            idx = torch.randperm(out["cx"].numel())[:n_max]
            out = {k: out[k][idx] for k in P}
        return {k: out[k].clone() for k in P}, int(do_clone.sum()), int(do_split.sum())


def run(densify_on, seed_init, steps=600):
    torch.manual_seed(0)
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    PX, PY = jj.float(), ii.float()
    # ground truth: 8 Gaussians
    g = torch.Generator().manual_seed(1); NG = 8
    GT = {"cx": torch.rand(NG, generator=g)*W, "cy": torch.rand(NG, generator=g)*H,
          "sx": 3+torch.rand(NG, generator=g)*4, "sy": 3+torch.rand(NG, generator=g)*4,
          "th": torch.rand(NG, generator=g)*math.pi,
          "op": torch.logit(0.5+torch.rand(NG, generator=g)*0.3), "col": torch.logit(0.4+torch.rand(NG, generator=g)*0.4)}
    order_gt = list(range(NG))
    target = render(GT, order_gt, PX, PY).detach()

    gi = torch.Generator().manual_seed(seed_init); N0 = 3      # FEW, LARGE/blurry -> split into detail
    P = {"cx": torch.rand(N0, generator=gi)*W, "cy": torch.rand(N0, generator=gi)*H,
         "sx": 12+torch.rand(N0, generator=gi)*8, "sy": 12+torch.rand(N0, generator=gi)*8,
         "th": torch.rand(N0, generator=gi)*math.pi, "op": torch.zeros(N0), "col": torch.zeros(N0)}
    lrs = {"cx": .6, "cy": .6, "sx": .08, "sy": .08, "th": .03, "op": .08, "col": .08}
    leaves(P); opt = adam(P, lrs)
    gpos = torch.zeros(P["cx"].numel()); acc = 0

    for step in range(1, steps+1):
        C = render(P, list(range(P["cx"].numel())), PX, PY)
        loss = ((C - target)**2).mean()
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            gpos += (P["cx"].grad**2 + P["cy"].grad**2).sqrt(); acc += 1
        opt.step()
        if densify_on and step % 50 == 0 and step <= 400 and P["cx"].numel() < 48:
            P, nc, ns = densify({k: P[k].detach() for k in P}, gpos/max(acc,1), 48)
            leaves(P); opt = adam(P, lrs); gpos = torch.zeros(P["cx"].numel()); acc = 0
    with torch.no_grad():
        final = render(P, list(range(P["cx"].numel())), PX, PY)
        mse = float(((final-target)**2).mean()); psnr = 10*math.log10(float(target.max())**2/max(mse,1e-12))
    return psnr, P["cx"].numel()


def test_operators():
    # 8 Gaussians: g0 low-opacity (PRUNE), g1 high-grad small (CLONE), g2 high-grad big (SPLIT),
    # g3..g7 idle normal (SURVIVE). gpos outliers on g0..g2.
    n = 8
    op = torch.full((n,), float(torch.logit(torch.tensor(0.5))))
    op[0] = float(torch.logit(torch.tensor(0.001)))            # -> pruned
    sx = torch.full((n,), 3.0); sx[2] = 9.0                    # g2 big -> split
    P = {"cx": torch.arange(n).float()*5, "cy": torch.arange(n).float()*5,
         "sx": sx.clone(), "sy": sx.clone(), "th": torch.zeros(n), "op": op, "col": torch.zeros(n)}
    gpos = torch.zeros(n); gpos[0:3] = 10.0                    # g0,g1,g2 high (g0 pruned anyway)
    out, nc, ns = densify({k: P[k].clone() for k in P}, gpos, n_max=999)
    # survivors = keep(7: all but g0) minus split(g2) = 6 ; +1 clone +2 split = 9
    ok = (nc == 1 and ns == 1 and out["cx"].numel() == 9)
    n_children = int((torch.abs(out["sx"] - 9.0/1.6) < 1e-4).sum())  # 2 split children @ 9/1.6
    ok = ok and n_children == 2
    ok = ok and bool((torch.sigmoid(out["op"]) > 0.02).all())   # no pruned survivors
    print(f"operator test  clones={nc} splits={ns} N:8->{out['cx'].numel()} (exp 9) "
          f"split_children@{9/1.6:.3f}={n_children} (exp 2) -> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ops_ok = test_operators()
    base_psnr, base_n = run(False, seed_init=7)
    dens_psnr, dens_n = run(True, seed_init=7)
    print(f"baseline (fixed N=3):     PSNR={base_psnr:5.1f} dB  N={base_n}")
    print(f"densified (3 -> grow):    PSNR={dens_psnr:5.1f} dB  N={dens_n}")
    print("DENSIFY_OK" if ops_ok and dens_psnr > base_psnr + 3 and dens_n > base_n else "DENSIFY_FAIL")


if __name__ == "__main__":
    main()
