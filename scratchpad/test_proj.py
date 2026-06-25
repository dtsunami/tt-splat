import sys, math, torch
from pathlib import Path
sys.path.insert(0, str(Path.home()/"tt-splat"/"server")); sys.path.insert(0, str(Path.home()/"tt-splat"/"docs"/"pathclear"))
import ttnn
from train_real import project_general
import device_project as DP
torch.manual_seed(0)
N=200
mean=torch.empty(N,3,dtype=torch.float64); mean[:,0]=torch.rand(N).double()*2-1; mean[:,1]=torch.rand(N).double()*2-1; mean[:,2]=2+torch.rand(N).double()*2
scale=torch.full((N,3),math.log(0.15),dtype=torch.float64)+torch.randn(N,3).double()*0.1
quat=torch.randn(N,4,dtype=torch.float64)
P={"mean":mean,"scale":scale,"quat":quat,"op":torch.zeros(N),"sh":torch.zeros(N,1,3),"deg":0}
Rv=torch.eye(3,dtype=torch.float64); tv=torch.zeros(3,dtype=torch.float64); fx=fy=120.; cx=cy=48.
u_h,v_h,zc_h,(a_h,b_h,c_h)=project_general(P,Rv,tv,fx,fy,cx,cy)
dev=ttnn.open_device(device_id=0)
try:
    cam=(Rv,tv,fx,fy,cx,cy,"t")
    u_d,v_d,zc_d,(a_d,b_d,c_d)=DP.project_geom(dev,mean,scale,quat,cam)
    g=lambda t: ttnn.to_torch(t).reshape(-1)[:N].double()
    def rel(dv,hv):
        return ((g(dv)-hv).norm()/(hv.norm()+1e-12)).item()
    for nm,dv,hv in [("u",u_d,u_h),("v",v_d,v_h),("zc",zc_d,zc_h),("a",a_d,a_h),("b",b_d,b_h),("c",c_d,c_h)]:
        print(f"  {nm:2} rel={rel(dv,hv):.2e}")
    worst=max(rel(dv,hv) for dv,hv in [(u_d,u_h),(v_d,v_h),(zc_d,zc_h),(a_d,a_h),(b_d,b_h),(c_d,c_h)])
    print(f"DEVICE_PROJ worst={worst:.2e} -> {'OK' if worst<5e-3 else 'FAIL'}")
finally: ttnn.close_device(dev)
