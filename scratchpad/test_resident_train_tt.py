#!/usr/bin/env python3
# Drive train_tt.run in device-resident mode through the dashboard contract; verify stage_timings populate.
import os, sys
from pathlib import Path
os.environ["TT_DEVICE_RESIDENT"] = "1"
os.environ.setdefault("TT_MAX_POINTS", "800")
os.environ.setdefault("TT_SIZE", "96")
sys.path.insert(0, str(Path.home()/"tt-splat"))
sys.path.insert(0, str(Path.home()/"tt-splat"/"server"))
import train_tt
from ttgs.config import TrainConfig
from ttgs.viewer.dashboard import TrainingController

cfg = TrainConfig(); cfg.iterations = 8; cfg.dashboard_every = 2
ctrl = TrainingController(output_dir=Path("work/tt_out"))
out = train_tt.run(Path("work/scene"), Path("work/tt_out"), cfg, backend=None, dashboard=ctrl)
hist = ctrl.get_history()
latest = ctrl.get_latest()
st = (latest or {}).get("stage_timings")
print(f"ply={out.name} updates={len(hist)} last_step={hist[-1]['step'] if hist else None}")
print(f"stage_timings={ {k: round(v,1) for k,v in st.items()} if st else None}")
ok = bool(st) and all(k in st for k in ("B", "raster", "A", "D", "C", "step")) and st["step"] > 0
# every history entry from this resident run carries the breakdown
ok = ok and all("stage_timings" in h for h in hist)
print("RESIDENT_TRAIN_TT_OK" if ok else "RESIDENT_TRAIN_TT_FAIL")
