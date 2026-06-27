#!/usr/bin/env python3
# Decisive profile: where does Stage B (project_geom/color/op + readback) and Stage D (project_backward)
# time actually go at training scale? Synchronized, warmed, averaged. No production changes.
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.home()/"tt-splat"/"server")); sys.path.insert(0, str(Path.home()/"tt-splat"/"docs"/"pathclear"))
import numpy as np, torch
import ttnn
import device_project as DP
from device_project_backward import project_backward

N = int(os.environ.get("PN", "800")); deg = 3; K = (deg+1)**2
torch.manual_seed(0)
mean = (torch.randn(N,3)*0.4 + torch.tensor([0.,0.,4.]))
scale = (torch.randn(N,3)*0.3 - 1.5); quat = torch.randn(N,4)
op = torch.randn(N)*0.5; sh = torch.randn(N,K,3)*0.2
Rv, tv, fx, fy, cx, cy = torch.eye(3), torch.zeros(3), 100., 100., 48., 48.
cam = (Rv, tv, fx, fy, cx, cy, "t")
up = dict(u=torch.randn(N), v=torch.randn(N), ca=torch.randn(N), cb=torch.randn(N), cc=torch.randn(N),
          op=torch.randn(N), colR=torch.randn(N), colG=torch.randn(N), colB=torch.randn(N))
Pd = dict(mean=mean, scale=scale, quat=quat, op=op, sh=sh, deg=deg)

dev = ttnn.open_device(device_id=0)
def sync():
    try: ttnn.synchronize_device(dev)
    except Exception: pass
def timeit(fn, iters=10):
    fn(); sync()                                   # warm (JIT)
    t0 = time.perf_counter()
    for _ in range(iters): r = fn()
    sync(); return 1e3*(time.perf_counter()-t0)/iters, r
try:
    tg, (u_t,v_t,zc_t,(ca_t,cb_t,cc_t),Ageo) = timeit(lambda: DP.project_geom(dev, mean, scale, quat, cam, aux=True))
    tc, (cR,cG,cB,Acol) = timeit(lambda: DP.project_color(dev, mean, sh, deg, cam, aux=True))
    to, _ = timeit(lambda: DP.project_op(dev, op))
    def readback9():
        g = lambda t: ttnn.to_torch(t).flatten().numpy()
        return [g(x) for x in (u_t,v_t,zc_t,ca_t,cb_t,cc_t)]
    trb, _ = timeit(readback9)
    td, _ = timeit(lambda: project_backward(dev, Pd, cam, up, aux=(Ageo, Acol), return_ttnn=True))
    print(f"N={N} deg={deg}")
    print(f"  B.project_geom  = {tg:6.1f} ms")
    print(f"  B.project_color = {tc:6.1f} ms")
    print(f"  B.project_op    = {to:6.1f} ms")
    print(f"  B.readback(6x)  = {trb:6.1f} ms")
    print(f"  ---- B total    ~ {tg+tc+to+trb:6.1f} ms")
    print(f"  D.project_bwd   = {td:6.1f} ms  <-- the dominator")
finally:
    ttnn.close_device(dev)
