import sys, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path.home()/"tt-splat"/"server")); sys.path.insert(0, str(Path.home()/"tt-splat"/"docs"/"pathclear"))
import device_raster as DR
torch.manual_seed(1)
H=W=96; N=200; deg=1; K=4
def mk():
    m=torch.empty(N,3,dtype=torch.float64)
    m[:,0]=(torch.rand(N)*1.6-0.8).double(); m[:,1]=(torch.rand(N)*1.6-0.8).double(); m[:,2]=(2.5+torch.rand(N)).double()
    return {"mean":m,"scale":torch.full((N,3),math.log(0.13),dtype=torch.float64),
            "quat":torch.tensor([[1.,0,0,0]]).repeat(N,1).double(),"op":torch.zeros(N,dtype=torch.float64),
            "sh":torch.randn(N,K,3,dtype=torch.float64)*0.4,"deg":deg}
cam=(torch.eye(3,dtype=torch.float64),torch.zeros(3,dtype=torch.float64),120.,120.,48.,48.,"t")
gtP=mk()
target=DR.render_train(gtP,cam,H,W).detach()        # device-culled target (self-consistent)
P={k:(v.clone() if torch.is_tensor(v) else v) for k,v in gtP.items()}
P["mean"]=P["mean"]+torch.randn(N,3).double()*0.12; P["sh"]=P["sh"]+torch.randn(N,K,3).double()*0.2
P["op"]=P["op"]+torch.randn(N).double()*0.3
OPT=["mean","scale","quat","op","sh"]
for k in OPT: P[k].requires_grad_(True)
lr={"mean":.01,"scale":.01,"quat":.01,"op":.02,"sh":.01}
opt=torch.optim.Adam([{"params":[P[k]],"lr":lr[k]} for k in OPT])
psnr=lambda a,b:10*math.log10(1.0/max(float(((a-b)**2).mean()),1e-12))
print(f"scale convergence  {H}px N={N} (3x3 tiles, M14 fwd + culled chunked bwd)")
for step in range(1,26):
    opt.zero_grad(); img=DR.render_train(P,cam,H,W); loss=((img-target)**2).mean(); loss.backward(); opt.step()
    if step==1 or step%5==0: print(f"  step {step:3d} loss={float(loss):.6f} PSNR={psnr(img.detach(),target):.1f} dB")
f=psnr(DR.render_train(P,cam,H,W).detach(),target)
print(f"SCALE_CONV final PSNR={f:.1f} dB -> {'OK' if f>30 else 'FAIL'}")
