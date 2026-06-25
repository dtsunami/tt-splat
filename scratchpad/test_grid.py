import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path.home()/"tt-splat"/"server")); sys.path.insert(0, str(Path.home()/"tt-splat"/"docs"/"pathclear"))
import ttnn
from bin_sort import bin_and_sort
import fused_backward as FB
np.random.seed(0); torch.manual_seed(0)
Wp=Hp=64; ntx=nty=2; N=40
cx=np.random.rand(N)*Wp; cy=np.random.rand(N)*Hp
sx=4+np.random.rand(N)*4; sy=4+np.random.rand(N)*4
# conic from variance
a=1/sx**2; c=1/sy**2; b=np.zeros(N)
op=0.4+np.random.rand(N)*0.4; col=[0.3+np.random.rand(N)*0.5 for _ in range(3)]
depth=np.random.rand(N)
s_gid,_,ranges,nx,ny,_=bin_and_sort(cx,cy,sx**2,sy**2,depth,Wp,Hp,ts=32)
tl=[s_gid[ranges[t,0]:ranges[t,1]].tolist() for t in range(nx*ny)]
gi=np.random.rand(Hp,Wp,3); Tfin=np.ones((Hp,Wp))
dev=ttnn.open_device(device_id=0)
try:
    # reference: host-tile-loop chunked (mirror DeviceRaster.backward)
    geomR={k:np.zeros(N) for k in ("cx","cy","a","b","c","op")}; colR=[np.zeros(N) for _ in range(3)]
    for t,lst in enumerate(tl):
        if not lst: continue
        ty,tx=divmod(t,ntx); oy,ox=ty*32,tx*32; rev=lst[::-1]
        Tt=Tfin[oy:oy+32,ox:ox+32]
        for k in range(3):
            dl=torch.from_numpy(gi[oy:oy+32,ox:ox+32,k].copy()); S=None; T=torch.from_numpy(Tt.copy())
            for cs in range(0,len(rev),FB.FUSED_K):
                ch=rev[cs:cs+FB.FUSED_K]
                pr=[{"cx":cx[i],"cy":cy[i],"a":a[i],"b":b[i],"c":c[i],"op":op[i],"col":col[k][i]} for i in ch]
                g,S,T=FB.fused_backward(dev,pr,dl,T,ox=ox,oy=oy,S_init=S,return_state=True)
                for j,i in enumerate(ch):
                    for key in ("cx","cy","a","b","c","op"): geomR[key][i]+=float(g[key][j])
                    colR[k][i]+=float(g["col"][j])
    # grid-sharded
    geomG,colG=FB.fused_backward_grid(dev,cx,cy,a,b,c,op,col,tl,ntx,nty,Wp,Hp,gi,Tfin)
    scale=max(abs(geomR[k]).max() for k in geomR)+1e-9
    worst=0.0
    for k in geomR:
        e=abs(geomG[k]-geomR[k]).max()/scale; worst=max(worst,e); print(f"  {k:3} err/scale={e:.2e}")
    for k in range(3):
        e=abs(colG[k]-colR[k]).max()/scale; worst=max(worst,e)
    print(f"GRID_BACKWARD worst={worst:.2e} -> {'OK' if worst<1e-2 else 'FAIL'}")
finally: ttnn.close_device(dev)
