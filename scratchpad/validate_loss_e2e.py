import os
os.environ["TT_DEVICE_RESIDENT"]="1"; os.environ["TT_SIZE"]="64"; os.environ["TT_MAX_POINTS"]="500"; os.environ["TT_DENSIFY"]="0"
os.environ.setdefault("TT_METAL_HOME","/home/starboy/tt-metal")
import sys
from pathlib import Path
ROOT=Path.home()/"tt-splat"; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"server"))
SP=Path("/tmp/claude-1000/-home-starboy/e585199c-0ca1-4e6e-96ae-274e4ac91dee/scratchpad/loss_e2e")
from ttgs.config import TrainConfig
from ttgs.viewer.dashboard import TrainingController
import train_tt
cfg=TrainConfig(); cfg.iterations=24; cfg.dashboard_every=2; cfg.sh_degree=1; cfg.lambda_dssim=0.2
ctrl=TrainingController(output_dir=SP)
train_tt.run(ROOT/"work"/"scene", SP, cfg, None, dashboard=ctrl)
hist=ctrl.get_history()
ps=[h.get("psnr") for h in hist if h.get("psnr") is not None]
ls=[h.get("loss") for h in hist if h.get("loss") is not None]
print("RESULT psnr:", [round(p,1) for p in ps])
print("RESULT loss:", [round(l,4) for l in ls])
print("RESULT", "LOSS_E2E_OK" if len(ps)>=2 and ps[-1]>ps[0] and ls[-1]<ls[0] else "FAIL")
