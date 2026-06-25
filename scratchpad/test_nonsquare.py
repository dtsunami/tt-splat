import sys, math, torch
from pathlib import Path
sys.path.insert(0, str(Path.home()/"tt-splat"/"server")); sys.path.insert(0, str(Path.home()/"tt-splat"/"docs"/"pathclear"))
import device_raster as DR
torch.manual_seed(0)
H, W = 72, 96; N=60; deg=1; K=4         # non-square, non-32-multiple H (corgi aspect)
m=torch.empty(N,3,dtype=torch.float64)
m[:,0]=(torch.rand(N)*1.6-0.8).double(); m[:,1]=(torch.rand(N)*1.2-0.6).double(); m[:,2]=(2.5+torch.rand(N)).double()
P={"mean":m,"scale":torch.full((N,3),math.log(0.15),dtype=torch.float64),
   "quat":torch.tensor([[1.,0,0,0]]).repeat(N,1).double(),"op":torch.zeros(N,dtype=torch.float64),
   "sh":torch.randn(N,K,3,dtype=torch.float64)*0.4,"deg":deg}
for k in ["mean","scale","quat","op","sh"]: P[k].requires_grad_(True)
cam=(torch.eye(3,dtype=torch.float64),torch.zeros(3,dtype=torch.float64),110.,110.,48.,36.,"t")
gt=torch.rand(H,W,3,dtype=torch.float64)
img=DR.render_train(P,cam,H,W)
print("img shape", tuple(img.shape))
loss=((img-gt.float())**2).mean(); loss.backward()
ok = all(P[k].grad is not None and torch.isfinite(P[k].grad).all() for k in ["mean","scale","op","sh"])
print(f"NONSQUARE_OK loss={float(loss):.4f} grads_finite={ok} -> {'OK' if ok else 'FAIL'}")
