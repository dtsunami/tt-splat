#!/usr/bin/env python3
# Stage E perf scaling: projection stages (B fwd, D bwd, C Adam) vs N. Projection is elementwise over
# [N] -> light power (no raster). Shows dispatch-bound -> amortization crossover.
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path.home()/"tt-splat"/"server"))
sys.path.insert(0, str(Path.home()/"tt-splat"/"docs"/"pathclear"))
import ttnn
from device_project import project_geom, project_color, project_op
from device_project_backward import project_backward
from device_adam import DeviceAdam
torch.manual_seed(0)
deg,K=1,4
cam=(torch.eye(3,dtype=torch.float64),torch.zeros(3,dtype=torch.float64),80.,80.,32.,32.,"t")
dev=ttnn.open_device(device_id=0)
def mk(N):
    mean=torch.empty(N,3); mean[:,0]=torch.rand(N)*2-1; mean[:,1]=torch.rand(N)*2-1; mean[:,2]=2+torch.rand(N)*2
    return {"mean":mean.double(),"scale":torch.full((N,3),-1.6).double(),
            "quat":torch.tensor([[1.,0,0,0]]).repeat(N,1).double(),"op":torch.zeros(N).double(),
            "sh":(torch.randn(N,K,3)*0.3).double(),"deg":deg}
def sync():
    try: ttnn.synchronize_device(dev)
    except Exception: pass
try:
    print(f"{'N':>7} {'B(fwd)':>9} {'D(bwd)':>9} {'C(adam)':>9} {'proj/step':>10} {'us/Gauss':>9}")
    for N in (64,256,1024,4096,16384,65536):
        P=mk(N)
        ad=DeviceAdam(dev,{k:P[k] for k in ("mean","scale","quat","op","sh")},
                      {"mean":.01,"scale":.01,"quat":.01,"op":.02,"sh":.01})
        up={k:np.random.randn(N).astype(np.float64) for k in ("u","v","ca","cb","cc","op","colR","colG","colB")}
        def B():
            p=ad.p
            *_,Ag=project_geom(dev,p["mean"],p["scale"],p["quat"],cam,aux=True)
            *_,Ac=project_color(dev,p["mean"],p["sh"],deg,cam,aux=True)
            project_op(dev,p["op"]); return Ag,Ac
        # warmup
        for _ in range(2):
            Ag,Ac=B()
            Pd=dict(mean=ad.p["mean"],scale=ad.p["scale"],quat=ad.p["quat"],sh=ad.p["sh"],op=ad.p["op"],deg=deg)
            g3=project_backward(dev,Pd,cam,up,aux=(Ag,Ac),return_ttnn=True); ad.step({k:g3[k] for k in ("mean","scale","quat","op","sh")})
        sync()
        R=5; tb=td=tc=0.0
        for _ in range(R):
            t=time.perf_counter(); Ag,Ac=B(); sync(); tb+=time.perf_counter()-t
            Pd=dict(mean=ad.p["mean"],scale=ad.p["scale"],quat=ad.p["quat"],sh=ad.p["sh"],op=ad.p["op"],deg=deg)
            t=time.perf_counter(); g3=project_backward(dev,Pd,cam,up,aux=(Ag,Ac),return_ttnn=True); sync(); td+=time.perf_counter()-t
            t=time.perf_counter(); ad.step({k:g3[k] for k in ("mean","scale","quat","op","sh")}); sync(); tc+=time.perf_counter()-t
        b,d,c=1e3*tb/R,1e3*td/R,1e3*tc/R; proj=b+d+c
        print(f"{N:>7} {b:>8.1f} {d:>8.1f} {c:>8.1f} {proj:>9.1f} {1e3*proj/N:>8.2f}",flush=True)
finally:
    ttnn.close_device(dev)
