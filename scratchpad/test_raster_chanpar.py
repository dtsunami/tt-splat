#!/usr/bin/env python3
# Gate: M14 forward channel-parallel (_raster_rgb banded) vs serial (_raster_channel x3) -> pixels must match.
import os, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path.home()/"tt-splat"/"server")); sys.path.insert(0, str(Path.home()/"tt-splat"/"docs"/"pathclear"))
import ttnn
from bin_sort import bin_and_sort
import render_device as RD
import sfpu_raster_scaled as M14

np.random.seed(1); torch.manual_seed(1)
SZ = int(os.environ.get("SZ", "96")); N = int(os.environ.get("N", "400"))
Wp = Hp = SZ; TS = 32
cx = np.random.rand(N)*Wp; cy = np.random.rand(N)*Hp
sx = 4+np.random.rand(N)*4; sy = 4+np.random.rand(N)*4
a = 1/sx**2; c = 1/sy**2; b = np.zeros(N)
op = 0.4+np.random.rand(N)*0.4
col = [0.3+np.random.rand(N)*0.5 for _ in range(3)]   # [R,G,B] arrays
depth = np.random.rand(N)
s_gid,_,ranges,ntx,nty,_ = bin_and_sort(cx,cy,sx**2,sy**2,depth,Wp,Hp,ts=TS)
tl = [s_gid[ranges[t,0]:ranges[t,1]].tolist() for t in range(ntx*nty)]
maxc = max((len(l) for l in tl), default=0); nbatch = (maxc+M14.B-1)//M14.B

dev = ttnn.open_device(device_id=0)
try:
    res = RD._resources(dev, Wp, Hp)
    # serial reference (3 x _raster_channel)
    serR = [RD._raster_channel(dev, res, tl, ntx, nbatch, cx, cy, a, b, c, op, col[k], Wp, Hp,
                               want_T=(k == 0)) for k in range(3)]
    serT = serR[0][1]; serial = [serR[0][0]] + [serR[k] for k in range(1, 3)]
    # banded parallel
    os.environ["RAST_CHANPAR"] = "1"
    res2 = RD._resources(dev, Wp, Hp)             # fresh dict (cache may hold 'par')
    par_chans, parT = RD._raster_rgb(dev, res2, tl, ntx, nbatch, cx, cy, a, b, c, op, col, Wp, Hp, want_T=True)
    bands = res2.get("par") is not None
    worst = max(float(np.abs(par_chans[k]-serial[k]).max()) for k in range(3))
    tworst = float(np.abs(parT-serT).max())
    print(f"scene SZ={SZ} N={N} tiles={ntx}x{nty} nbatch={nbatch} banded={bands}")
    print(f"  RGB max|parallel-serial| = {worst:.3e}   T max diff = {tworst:.3e}")
    print(f"RASTER_CHANPAR {'OK' if (worst < 1e-5 and tworst < 1e-5) else 'FAIL'}")
finally:
    ttnn.close_device(dev)
