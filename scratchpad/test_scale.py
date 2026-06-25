import sys, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path.home()/"tt-splat"/"server")); sys.path.insert(0, str(Path.home()/"tt-splat"/"docs"/"pathclear"))
from train_real import render
import device_raster as DR
torch.manual_seed(0)
H=W=96; N=200; deg=1; K=4
mean=torch.empty(N,3,dtype=torch.float64)
mean[:,0]=(torch.rand(N)*2-1).double(); mean[:,1]=(torch.rand(N)*2-1).double(); mean[:,2]=(2+torch.rand(N)*2).double()
P={"mean":mean,"scale":torch.full((N,3),math.log(0.15),dtype=torch.float64),
   "quat":torch.tensor([[1.,0,0,0]]).repeat(N,1).double(),"op":torch.zeros(N,dtype=torch.float64),
   "sh":torch.randn(N,K,3,dtype=torch.float64)*0.3,"deg":deg}
OPT=["mean","scale","quat","op","sh"]
cam=(torch.eye(3,dtype=torch.float64),torch.zeros(3,dtype=torch.float64),120.,120.,48.,48.,"t")
ii,jj=torch.meshgrid(torch.arange(H),torch.arange(W),indexing="ij"); PX,PY=jj.double(),ii.double()
gt=torch.rand(H,W,3,dtype=torch.float64)
def grads(img,g):
    for k in OPT:
        if P[k].grad is not None: P[k].grad=None
    (((img-g)**2).mean()).backward()
    return {k:P[k].grad.detach().clone().double() for k in OPT}
for k in OPT: P[k].requires_grad_(True)
gh=grads(render(P,cam,H,W,PX,PY),gt)
gd=grads(DR.render_train(P,cam,H,W),gt.float())
worst=max((gd[k]-gh[k]).norm().item()/(gh[k].norm().item()+1e-12) for k in OPT)
for k in OPT:
    print(f"  grad[{k:5}] rel={ (gd[k]-gh[k]).norm().item()/(gh[k].norm().item()+1e-12):.2e}")
print(f"SCALE_GRAD N={N} {H}px(3x3 tiles) worst={worst:.2e} -> {'OK' if worst<5e-2 else 'FAIL'}")
