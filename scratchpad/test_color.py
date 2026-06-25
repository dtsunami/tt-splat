import sys, math, torch
from pathlib import Path
sys.path.insert(0, str(Path.home()/"tt-splat"/"server")); sys.path.insert(0, str(Path.home()/"tt-splat"/"docs"/"pathclear"))
import ttnn
from train_real import sh_eval
import device_project as DP
torch.manual_seed(0)
N=200; deg=3; K=(deg+1)**2
mean=torch.randn(N,3,dtype=torch.float64); mean[:,2]=2+torch.rand(N).double()
sh=torch.randn(N,K,3,dtype=torch.float64)*0.3
op=torch.randn(N,dtype=torch.float64)
Rv=torch.eye(3,dtype=torch.float64); tv=torch.zeros(3,dtype=torch.float64)
cam=(Rv,tv,120.,120.,48.,48.,"t")
# host
cc=(-Rv.T@tv); dirs=mean-cc; dirs=dirs/(dirs.norm(dim=-1,keepdim=True)+1e-9)
col_h=sh_eval(sh,dirs,deg); op_h=torch.sigmoid(op)
dev=ttnn.open_device(device_id=0)
try:
    r,gC,b=DP.project_color(dev,mean,sh,deg,cam); opd=DP.project_op(dev,op)
    g=lambda t: ttnn.to_torch(t).reshape(-1)[:N].double()
    rel=lambda dv,hv:((g(dv)-hv).norm()/(hv.norm()+1e-12)).item()
    er=rel(r,col_h[:,0]); eg=rel(gC,col_h[:,1]); eb=rel(b,col_h[:,2]); eo=rel(opd,op_h)
    print(f"  colR={er:.2e} colG={eg:.2e} colB={eb:.2e} op={eo:.2e}")
    print(f"DEVICE_COLOR worst={max(er,eg,eb,eo):.2e} -> {'OK' if max(er,eg,eb,eo)<5e-3 else 'FAIL'}")
finally: ttnn.close_device(dev)
