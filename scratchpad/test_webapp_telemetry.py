#!/usr/bin/env python3
# Smoke: stage_timings flows build_update -> /state + history; /docs cartoons serve. No device needed.
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home()/"tt-splat"))
sys.path.insert(0, str(Path.home()/"tt-splat"/"server"))
import numpy as np
from ttgs.viewer.dashboard import build_update, TrainingController, build_app
try:
    from fastapi.testclient import TestClient
except Exception:
    from starlette.testclient import TestClient

class Pipe:
    def __init__(self): self.training = TrainingController()

st = dict(B=14.8, bin=2.1, raster=20.3, loss=1.0, A=34.0, D=25.1, C=3.0, step=97.9)
u = build_update(step=10, total_steps=100, loss=0.08, n_gaussians=1024, camera_name="cam0.jpg",
                 render=np.zeros((8, 8, 3), np.float32), gt=np.zeros((8, 8, 3), np.float32),
                 l1=0.1, ssim=12.0, mse=0.08, stage_timings=st)
assert "stage_timings" in u and u["stage_timings"]["A"] == 34.0, u.get("stage_timings")
# host path (no stage_timings) must NOT add the key
u2 = build_update(step=1, total_steps=100, loss=0.1, n_gaussians=10, camera_name="c.jpg",
                  render=np.zeros((8, 8, 3), np.float32), gt=np.zeros((8, 8, 3), np.float32))
assert "stage_timings" not in u2, "host path leaked stage_timings"

p = Pipe(); p.training.push_update(u)
app = build_app(p); c = TestClient(app)
r = c.get("/state"); assert r.status_code == 200, r.status_code
assert r.json()["stage_timings"]["raster"] == 20.3, r.json().get("stage_timings")
h = c.get("/state/history").json(); assert h[-1]["stage_timings"]["C"] == 3.0
pc = c.get("/docs/pipeline_cartoon.html"); assert pc.status_code == 200 and "trains" in pc.text
ac = c.get("/docs/adam_cartoon.html"); assert ac.status_code == 200 and "Adam" in ac.text
assert c.get("/docs/nope.html").status_code == 404
assert c.get("/docs/secrets.txt").status_code == 404
print("WEBAPP_TELEMETRY_OK")
