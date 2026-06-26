#!/usr/bin/env python3
# Profile the device-resident loop OVER TIME to catch "ramps up then stalls": per-step wall time,
# per-stage breakdown (first vs last window), device memory/program-cache growth (leak detection).
import sys, os, time, gc
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path.home()/"tt-splat"/"server"))
sys.path.insert(0, str(Path.home()/"tt-splat"/"docs"/"pathclear"))
import ttnn
from render_device import _device
from device_resident import DeviceResidentTrainer
torch.manual_seed(0)
N   = int(os.environ.get("PN", "1024"))
SZ  = int(os.environ.get("PSZ", "96"))
STEPS = int(os.environ.get("PSTEPS", "80"))
deg, K = 1, 4
H = W = SZ
mean = torch.empty(N,3); mean[:,0]=torch.rand(N)*2-1; mean[:,1]=torch.rand(N)*2-1; mean[:,2]=2+torch.rand(N)*2
P0={"mean":mean.double(),"scale":torch.full((N,3),-1.6).double(),"quat":torch.tensor([[1.,0,0,0]]).repeat(N,1).double(),
    "op":torch.zeros(N).double(),"sh":(torch.randn(N,K,3)*0.3).double(),"deg":deg}
cam=(torch.eye(3,dtype=torch.float64),torch.zeros(3,dtype=torch.float64),float(SZ*1.2),float(SZ*1.2),SZ/2,SZ/2,"t")
gt=torch.rand(H,W,3,dtype=torch.float64)

def mem(dev):
    for fn in ("num_program_cache_entries",):
        try: pc = getattr(dev, fn)()
        except Exception: pc = -1
    dram = -1
    try: dram = ttnn._ttnn.reports.get_memory_view(dev, ttnn.BufferType.DRAM)  # may not exist
    except Exception: pass
    return pc

dev=_device()
try:
    tr=DeviceResidentTrainer(dev, P0, deg=deg)
    print(f"N={N} SZ={SZ} STEPS={STEPS}", flush=True)
    print(f"{'step':>4} {'ms':>7} {'B':>6} {'raster':>7} {'A':>6} {'D':>6} {'C':>6} {'progcache':>9} {'loss':>8}", flush=True)
    rss0 = None
    for s in range(STEPS):
        l, im = tr.step(cam, gt)
        e = tr.step_log[-1]
        pc = -1
        try: pc = dev.num_program_cache_entries()
        except Exception: pass
        # host RSS (catch host-side leak)
        rss = int(open(f"/proc/self/statm").read().split()[1]) * 4096 // (1024*1024)
        if rss0 is None: rss0 = rss
        if s < 5 or s % 5 == 0 or s == STEPS-1:
            print(f"{s:>4} {e['step']:>7.1f} {e['B']:>6.1f} {e['raster']:>7.1f} {e['A']:>6.1f} {e['D']:>6.1f} {e['C']:>6.1f} {pc:>9} {l:>8.5f}  rss={rss}MB(+{rss-rss0})", flush=True)
    # window summary: mean of steps 2-6 vs last 5
    def win(a,b):
        xs=tr.step_log[a:b]; return {k:sum(x[k] for x in xs)/len(xs) for k in ('step','B','raster','A','D','C')}
    w0=win(2,7); w1=win(STEPS-5,STEPS)
    print(f"WINDOW early(2-6) step={w0['step']:.1f}  late(last5) step={w1['step']:.1f}  ratio={w1['step']/w0['step']:.2f}x", flush=True)
    print(f"  early B={w0['B']:.1f} raster={w0['raster']:.1f} A={w0['A']:.1f} D={w0['D']:.1f} C={w0['C']:.1f}", flush=True)
    print(f"  late  B={w1['B']:.1f} raster={w1['raster']:.1f} A={w1['A']:.1f} D={w1['D']:.1f} C={w1['C']:.1f}", flush=True)
finally:
    ttnn.close_device(dev)
