import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path.home()/"tt-splat"/"server"))
import ttnn
from device_adam import DeviceAdam
dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    init = torch.randn(200, 3)
    da = DeviceAdam(dev, {"x": init.clone()}, {"x": 0.01})
    # manual torch Adam fed the SAME grad sequence (isolates Adam math)
    p = init.clone(); m = torch.zeros_like(p); v = torch.zeros_like(p)
    torch.manual_seed(7)
    for step in range(1, 51):
        g = torch.randn(200, 3) * 0.1
        da.step({"x": g})
        m = 0.9*m + 0.1*g; v = 0.999*v + 0.001*g*g
        mh = m/(1-0.9**step); vh = v/(1-0.999**step)
        p = p - 0.01*mh/(vh.sqrt()+1e-8)
    dp = da.params()["x"][:200, :3]
    err = (dp - p).abs().max().item() / (p.abs().max().item() + 1e-12)
    print(f"DEVICE_ADAM 50 steps  rel_err={err:.2e} -> {'OK' if err < 1e-3 else 'FAIL'}")
finally:
    ttnn.close_device(dev)
