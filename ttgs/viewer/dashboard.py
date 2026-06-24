"""ttgs dashboard — FastAPI server that owns the full pipeline.

Architecture
────────────
  uvicorn (calling thread)      ← always responsive
  PipelineController            ← owns stage threads, exclusions, masks
    └── TrainingController      ← pause/resume/commands for training loop

Routes
──────
  GET  /                         pipeline hub + training monitor
  GET  /images                   image gallery
  GET  /images/list              JSON metadata for all frames
  GET  /images/{name}/thumb      thumbnail JPEG (200 px wide)
  GET  /images/{name}/full       full-resolution image
  GET  /images/{name}/edit       per-image mask editor
  GET  /images/{name}/mask.png   current mask (404 if none)
  POST /images/{name}/mask       save mask  { mask_b64: str }
  DELETE /images/{name}/mask     clear mask
  POST /images/{name}/exclude    mark excluded
  DELETE /images/{name}/exclude  clear exclusion

  GET  /pipeline/status          stage status JSON
  POST /pipeline/run             { from_stage: str }
  POST /pipeline/interrupt

  GET  /state                    latest training snapshot (poll ~500 ms)
  POST /pause                    toggle pause / resume
  POST /command                  training command (see _CommandPayload)
  POST /mcp                      MCP JSON-RPC 2.0 tool interface
"""

from __future__ import annotations

import base64
import io
import json
import queue
import threading
import time as _time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import Request as _Request  # Must be at module level — PEP 563 breaks local resolution
from PIL import Image
from pydantic import BaseModel
from rich.console import Console

console = Console()


# ─── TrainingController ───────────────────────────────────────────────────────

class TrainingController:
    """Thread-safe bridge for the training loop.

    FastAPI handlers write from the server thread; training loop reads each step.
    """

    def __init__(self, output_dir: Path | None = None, snapshot_every: int = 0) -> None:
        self._lock        = threading.Lock()
        self._pause_event = threading.Event()
        self._stop_event  = threading.Event()
        self._global_mask: np.ndarray | None = None
        self._commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self._latest: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []     # rolling loss history (no images)
        self._output_dir  = output_dir                # for saving latest render to disk
        self._snapshot_every = snapshot_every          # save per-camera snapshots every N steps
        self._live_config: dict[str, Any] = {}        # current training config (set by training loop)
        self._command_log: list[dict[str, Any]] = []  # recent executed commands with stats

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def set_global_mask(self, mask: np.ndarray | None) -> None:
        with self._lock:
            self._global_mask = mask

    def queue_command(self, cmd: dict[str, Any]) -> None:
        self._commands.put_nowait(cmd)

    def stop(self) -> None:
        """Signal the training loop to save and exit cleanly."""
        self._stop_event.set()
        # Also un-pause so the loop isn't stuck in wait_if_paused
        self._pause_event.clear()

    @property
    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def wait_if_paused(self) -> None:
        while self._pause_event.is_set() and not self._stop_event.is_set():
            self._pause_event.wait(0.1)

    def get_mask(self) -> np.ndarray | None:
        with self._lock:
            return self._global_mask

    def drain_commands(self) -> list[dict[str, Any]]:
        cmds: list[dict[str, Any]] = []
        while True:
            try:
                cmds.append(self._commands.get_nowait())
            except queue.Empty:
                break
        return cmds

    def push_update(self, update: dict[str, Any]) -> None:
        with self._lock:
            self._latest = dict(update)  # snapshot — prevents mutation during serialize
            # Append scalar fields to rolling history (no images)
            entry = {k: v for k, v in update.items() if not k.endswith("_b64")}
            self._history.append(entry)
        # Save render/gt/diff per camera to disk (non-blocking, best-effort)
        if self._output_dir is not None:
            try:
                cam = update.get("camera_name", "")
                if cam:
                    stem = Path(cam).stem  # P4120179.JPG → P4120179
                    for key, subdir in (("render_b64", "renders"),
                                        ("gt_b64", "gt"),
                                        ("diff_b64", "diffs")):
                        if key in update:
                            d = self._output_dir / subdir
                            d.mkdir(parents=True, exist_ok=True)
                            (d / f"{stem}.png").write_bytes(
                                base64.b64decode(update[key])
                            )
            except Exception:
                pass
        # Save timestamped snapshots for training-progression videos
        if self._output_dir is not None and self._snapshot_every > 0:
            step = update.get("step", -1)
            if step >= 0 and step % self._snapshot_every == 0:
                try:
                    cam = update.get("camera_name", "")
                    if cam:
                        stem = Path(cam).stem
                        snap_dir = self._output_dir / "snapshots" / stem
                        snap_dir.mkdir(parents=True, exist_ok=True)
                        # Render frame (changes each step)
                        if "render_b64" in update:
                            (snap_dir / f"step_{step:06d}.png").write_bytes(
                                base64.b64decode(update["render_b64"])
                            )
                        # GT frame (constant — save once as reference)
                        gt_path = snap_dir / "gt.png"
                        if "gt_b64" in update and not gt_path.exists():
                            gt_path.write_bytes(
                                base64.b64decode(update["gt_b64"])
                            )
                except Exception:
                    pass

    def get_history(self, last_n: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            if last_n > 0:
                return list(self._history[-last_n:])
            return list(self._history)

    def get_latest(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._latest) if self._latest is not None else None

    def set_config(self, cfg_dict: dict[str, Any]) -> None:
        with self._lock:
            self._live_config = dict(cfg_dict)

    def get_config(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._live_config)

    def log_command(self, cmd_type: str, step: int,
                    detail: str = "", stats: dict | None = None) -> None:
        with self._lock:
            entry: dict[str, Any] = {
                "type": cmd_type,
                "step": step,
                "time": _time.time(),
                "detail": detail,
            }
            if stats:
                entry["stats"] = stats
            self._command_log.append(entry)
            if len(self._command_log) > 50:
                self._command_log = self._command_log[-50:]

    def get_command_log(self, last_n: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            if last_n > 0:
                return list(self._command_log[-last_n:])
            return list(self._command_log)


# ─── Image helpers ────────────────────────────────────────────────────────────

def _to_b64(arr: np.ndarray) -> str:
    img = Image.fromarray((arr.clip(0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode()


def _lum_diff_b64(render: np.ndarray, gt: np.ndarray) -> str:
    lum = lambda x: 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
    diff = np.abs(lum(render) - lum(gt))
    r = np.clip(diff * 3, 0, 1)
    g = np.clip(diff * 3 - 1, 0, 1)
    b = np.clip(diff * 3 - 2, 0, 1)
    return _to_b64(np.stack([r, g, b], axis=-1).astype(np.float32))


def build_update(
    step: int,
    total_steps: int,
    loss: float,
    n_gaussians: int,
    camera_name: str,
    render: np.ndarray,
    gt: np.ndarray,
    is_paused: bool = False,
    focus_camera: str | None = None,
    l1: float = 0.0,
    ssim: float = 0.0,
    mse: float = 0.0,
) -> dict[str, Any]:
    return {
        "step":         step,
        "total_steps":  total_steps,
        "loss":         round(float(loss), 6),
        "l1":           round(float(l1), 6),
        "ssim":         round(float(ssim), 6),
        "mse":          round(float(mse), 6),
        "n_gaussians":  int(n_gaussians),
        "camera_name":  camera_name,
        "is_paused":    is_paused,
        "focus_camera": focus_camera,
        "render_b64":   _to_b64(render),
        "gt_b64":       _to_b64(gt),
        "diff_b64":     _lum_diff_b64(render, gt),
    }


# ─── Pydantic models ──────────────────────────────────────────────────────────

class _MaskPayload(BaseModel):
    mask_b64: str

class _CommandPayload(BaseModel):
    type: str
    threshold:     float        = 0.005
    max_log_scale: float        = 2.5
    lr_factor:     float        = 1.0
    camera_name:   str | None   = None

class _PipelineRunPayload(BaseModel):
    from_stage: str


# ─── MCP tool list ────────────────────────────────────────────────────────────

_MCP_TOOLS = [
    {"name": "get_state",
     "description": "Current training state + render/diff images. Call first to assess quality.",
     "inputSchema": {"type": "object", "properties": {
         "include_images": {"type": "boolean"}}}},
    {"name": "set_pause",
     "description": "Pause or resume training.",
     "inputSchema": {"type": "object", "properties": {
         "paused": {"type": "boolean"}}, "required": ["paused"]}},
    {"name": "prune_gaussians",
     "description": "Remove low-opacity Gaussians (floaters). threshold: 0–1, default 0.005.",
     "inputSchema": {"type": "object", "properties": {
         "threshold": {"type": "number"}}}},
    {"name": "clamp_scale",
     "description": "Kill elongated needle Gaussians by clamping log-scale. Typical: 1.5–3.0.",
     "inputSchema": {"type": "object", "properties": {
         "max_log_scale": {"type": "number"}}}},
    {"name": "set_lr_scale",
     "description": "Multiply all LRs by factor. >1 speeds up, <1 stabilises.",
     "inputSchema": {"type": "object", "properties": {
         "factor": {"type": "number"}}, "required": ["factor"]}},
    {"name": "focus_camera",
     "description": "Lock training to one camera (null to clear).",
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": ["string", "null"]}}, "required": ["name"]}},
    {"name": "densify_now",
     "description": "Force a densification pass on the next step.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "reset_opacities",
     "description": "Reset all opacities to near-zero.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "save_checkpoint",
     "description": "Save checkpoint immediately.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "get_pipeline_status",
     "description": "Get current pipeline stage statuses (extract/sfm/train/export).",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "run_pipeline",
     "description": "Start or restart pipeline from a stage: extract | sfm | train | export.",
     "inputSchema": {"type": "object", "properties": {
         "from_stage": {"type": "string"}}, "required": ["from_stage"]}},
    {"name": "interrupt_pipeline",
     "description": "Interrupt the currently running pipeline stage.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "list_frames",
     "description": "List all frames with excluded/masked status.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "exclude_image",
     "description": "Exclude or include an image from SfM and training.",
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string"},
         "excluded": {"type": "boolean"}}, "required": ["name", "excluded"]}},
    {"name": "get_config",
     "description": "Get current live training config (iterations, save_every, snapshot_every, etc.).",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "update_config",
     "description": "Update training config on-the-fly. Pass any TrainConfig fields to change.",
     "inputSchema": {"type": "object", "properties": {
         "iterations":        {"type": "integer", "description": "Max training steps"},
         "save_every":        {"type": "integer", "description": "Checkpoint interval (0=end only)"},
         "snapshot_every":    {"type": "integer", "description": "Per-camera snapshot interval (0=off)"},
         "dashboard_every":   {"type": "integer", "description": "Dashboard push interval"},
         "lambda_dssim":      {"type": "number",  "description": "SSIM loss weight (0-1)"},
         "densify_from":      {"type": "integer", "description": "Start densifying at step N"},
         "densify_until":     {"type": "integer", "description": "Stop densifying at step N"},
         "densify_every":     {"type": "integer", "description": "Densify interval"},
         "densify_grad_threshold": {"type": "number", "description": "Gradient threshold for densify"},
         "opacity_reset_every":    {"type": "integer", "description": "Opacity reset interval (0=off)"},
     }}},
]


# ─── FastAPI app ──────────────────────────────────────────────────────────────

def build_app(pipeline):
    """Build the FastAPI app.

    *pipeline* is a PipelineController (or any object with a .training attribute
    that is a TrainingController).  For training-only use pass an object with
    training=<TrainingController>.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    app = FastAPI(title="ttgs dashboard")
    # NOTE: always use pipeline.training (not a cached reference) — the
    # PipelineController creates a fresh TrainingController for each run,
    # so a snapshot taken here would go stale.

    # GPU hardware monitor (optional — degrades gracefully)
    _gpu_mon = None
    try:
        from ttgs.backend.monitor import GpuMonitor
        _gpu_mon = GpuMonitor()
        if _gpu_mon.available:
            _gpu_mon.start()
        else:
            _gpu_mon = None
    except Exception:
        pass

    _tpl    = Path(__file__).parent / "templates"
    _static = Path(__file__).parent / "static"

    # ── HTML pages ─────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return HTMLResponse(_tpl.joinpath("dashboard.html").read_text(encoding="utf-8"))

    @app.get("/training", response_class=HTMLResponse)
    async def training_page():
        return HTMLResponse(_tpl.joinpath("training.html").read_text(encoding="utf-8"))

    @app.get("/images", response_class=HTMLResponse)
    async def gallery_page():
        return HTMLResponse(_tpl.joinpath("gallery.html").read_text(encoding="utf-8"))

    @app.get("/images/{name}/edit", response_class=HTMLResponse)
    async def image_edit_page(name: str):
        html = _tpl.joinpath("image_edit.html").read_text(encoding="utf-8")
        return HTMLResponse(html.replace("__IMAGE_NAME__", name))

    @app.get("/static/{filename}")
    async def static_file(filename: str):
        path = _static / filename
        if not path.exists() or not path.is_file():
            return Response(status_code=404)
        ct = "text/css" if filename.endswith(".css") else \
             "application/javascript" if filename.endswith(".js") else \
             "application/octet-stream"
        return Response(content=path.read_bytes(), media_type=ct)

    @app.get("/favicon.ico")
    async def favicon():
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
               "<text y='1.05em' font-size='28'>🫧</text></svg>")
        return Response(content=svg, media_type="image/svg+xml")

    # ── Image data routes ───────────────────────────────────────────────────

    @app.get("/images/list")
    async def images_list():
        return pipeline.list_frames()

    @app.get("/frames.json")
    async def frames_json():
        """Unified per-frame edit state (exclusions + masks + filters).

        Polled by the editor for real-time updates.  Written automatically
        whenever any edit changes.
        """
        return pipeline.read_frames_json()

    @app.get("/images/{name}/thumb")
    async def image_thumb(name: str):
        path = pipeline.frames_dir / name
        if not path.exists():
            return JSONResponse(status_code=404, content={"error": "not found"})
        img = Image.open(path).convert("RGB")
        img.thumbnail((240, 180), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=72)
        return Response(buf.getvalue(), media_type="image/jpeg")

    @app.get("/images/{name}/full")
    async def image_full(name: str):
        from fastapi.responses import FileResponse
        path = pipeline.frames_dir / name
        if not path.exists():
            return JSONResponse(status_code=404, content={"error": "not found"})
        return FileResponse(str(path))

    @app.get("/images/{name}/processed")
    async def image_processed(name: str):
        """Serve the filter-processed frame, or fall back to the raw frame."""
        from fastapi.responses import FileResponse
        processed = pipeline.output_dir / "processed_frames" / name
        if processed.exists():
            return FileResponse(str(processed))
        raw = pipeline.frames_dir / name
        if raw.exists():
            return FileResponse(str(raw))
        return JSONResponse(status_code=404, content={"error": "not found"})

    @app.get("/images/{name}/mask.png")
    async def get_image_mask(name: str):
        p = pipeline.mask_path(name)
        if not p.exists():
            return JSONResponse(status_code=404, content={"error": "no mask"})
        return Response(p.read_bytes(), media_type="image/png")

    @app.post("/images/{name}/mask")
    async def save_image_mask(name: str, payload: _MaskPayload):
        data = base64.b64decode(payload.mask_b64)
        img  = Image.open(io.BytesIO(data)).convert("L")
        mask = np.array(img, dtype=np.float32) / 255.0
        pipeline.save_mask(name, mask)
        return {"ok": True}

    @app.delete("/images/{name}/mask")
    async def delete_image_mask(name: str):
        pipeline.delete_mask(name)
        return {"ok": True}

    @app.get("/images/{name}/colmap")
    async def colmap_features(name: str):
        data = pipeline.get_colmap_features(name)
        if data is None:
            return JSONResponse(status_code=404, content={"error": "no SfM data"})
        return data

    @app.get("/images/{name}/mask-data")
    async def get_mask_data(name: str):
        return pipeline.get_mask_data(name)

    @app.post("/images/{name}/mask-data")
    async def save_mask_data(name: str, request: _Request):
        data = await request.json()
        pipeline.save_mask_data(name, data)
        # Tell the training loop to reload this mask from disk
        pipeline.training.queue_command({
            "type": "reload_masks", "image_name": name,
        })
        return {"ok": True}

    @app.post("/images/{name}/exclude")
    async def exclude_image(name: str):
        pipeline.set_exclusion(name, True)
        return {"ok": True, "excluded": True}

    @app.delete("/images/{name}/exclude")
    async def include_image(name: str):
        pipeline.set_exclusion(name, False)
        return {"ok": True, "excluded": False}

    # ── Pipeline control ────────────────────────────────────────────────────

    @app.get("/pipeline/status")
    async def pipeline_status():
        return pipeline.get_status()

    @app.post("/pipeline/run")
    async def pipeline_run(payload: _PipelineRunPayload):
        err = pipeline.run_from(payload.from_stage)
        if err:
            return JSONResponse(status_code=400, content={"error": err})
        return {"ok": True, "from_stage": payload.from_stage}

    @app.post("/pipeline/interrupt")
    async def pipeline_interrupt():
        pipeline.interrupt()
        return {"ok": True}

    # ── GPU hardware metrics ──────────────────────────────────────────────

    @app.get("/gpu")
    async def gpu_metrics():
        if _gpu_mon is None:
            return JSONResponse({"available": False})
        data = _gpu_mon.snapshot()
        data["available"] = bool(data)
        return data

    # ── Command log ──────────────────────────────────────────────────────

    @app.get("/state/commands")
    async def state_commands(last: int = 20):
        return pipeline.training.get_command_log(last_n=last)

    # ── Live training config ──────────────────────────────────────────────

    @app.get("/config")
    async def get_config():
        return pipeline.training.get_config()

    @app.post("/config")
    async def update_config(request: _Request):
        data = await request.json()
        pipeline.training.queue_command({"type": "update_config", **data})
        return {"ok": True, "queued": data}

    # ── Training state (polled by dashboard) ────────────────────────────────

    @app.get("/state/history")
    async def state_history(last: int = 0, camera: str = ""):
        """Return loss history. Optional ?last=N or ?camera=name filters."""
        history = pipeline.training.get_history(last_n=last)
        if camera:
            history = [h for h in history if h.get("camera_name") == camera]
        return history

    @app.get("/state")
    async def state():
        latest = pipeline.training.get_latest()
        if latest is None:
            return Response(status_code=204)
        # Serialize once to avoid Content-Length mismatch when the dict
        # is mutated between Starlette's two serialization passes.
        body = json.dumps(latest).encode()
        return Response(content=body, media_type="application/json")

    @app.post("/pause")
    async def toggle_pause():
        if pipeline.training.is_paused:
            pipeline.training.resume()
            return {"paused": False}
        else:
            pipeline.training.pause()
            return {"paused": True}

    @app.post("/mask")
    async def set_global_mask(payload: _MaskPayload):
        data = base64.b64decode(payload.mask_b64)
        img  = Image.open(io.BytesIO(data)).convert("L")
        mask = np.array(img, dtype=np.float32) / 255.0
        pipeline.training.set_global_mask(mask)
        return {"ok": True, "shape": list(mask.shape)}

    @app.delete("/mask")
    async def clear_global_mask():
        pipeline.training.set_global_mask(None)
        return {"ok": True}

    @app.post("/command")
    async def command(payload: _CommandPayload):
        pipeline.training.queue_command(payload.model_dump())
        return {"ok": True}

    # ── MCP JSON-RPC 2.0 ───────────────────────────────────────────────────

    @app.post("/mcp")
    async def mcp(request: _Request):
        body   = await request.json()
        rpc_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params", {})

        def ok(result):
            return {"jsonrpc": "2.0", "id": rpc_id, "result": result}
        def err(code, msg):
            return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": msg}}
        def tok(*blocks):
            return ok({"content": list(blocks)})
        def terr(msg):
            return ok({"content": [{"type": "text", "text": f"Error: {msg}"}], "isError": True})
        def txt(s):
            return {"type": "text", "text": str(s)}
        def img_block(b64):
            return {"type": "image", "data": b64, "mimeType": "image/png"}

        if method == "initialize":
            return ok({"protocolVersion": "2024-11-05",
                       "capabilities": {"tools": {}},
                       "serverInfo": {"name": "ttgs", "version": "1.0"}})
        if method == "notifications/initialized":
            return ok({})
        if method == "tools/list":
            return ok({"tools": _MCP_TOOLS})
        if method != "tools/call":
            return err(-32601, f"Method not found: {method!r}")

        tool = params.get("name", "")
        args = params.get("arguments", {})

        # ── training tools ─────────────────────────────────────────────
        if tool == "get_state":
            latest = pipeline.training.get_latest()
            if latest is None:
                return tok(txt("Training has not started yet."))
            scalar = {k: v for k, v in latest.items() if not k.endswith("_b64")}
            blocks = [txt(json.dumps(scalar, indent=2))]
            if args.get("include_images", True):
                blocks += [img_block(latest["render_b64"]),
                           img_block(latest["diff_b64"])]
            return tok(*blocks)

        if tool == "set_pause":
            if args.get("paused"):
                pipeline.training.pause();  return tok(txt("Paused."))
            else:
                pipeline.training.resume(); return tok(txt("Resumed."))

        if tool == "prune_gaussians":
            t = float(args.get("threshold", 0.005))
            pipeline.training.queue_command({"type": "prune", "threshold": t})
            return tok(txt(f"prune queued (threshold={t:.3f})"))

        if tool == "clamp_scale":
            v = float(args.get("max_log_scale", 2.5))
            pipeline.training.queue_command({"type": "clamp_scale", "max_log_scale": v})
            return tok(txt(f"clamp_scale queued (max={v})"))

        if tool == "set_lr_scale":
            f = args.get("factor")
            if f is None: return terr("'factor' required")
            pipeline.training.queue_command({"type": "set_lr", "lr_factor": float(f)})
            return tok(txt(f"set_lr queued (×{f})"))

        if tool == "focus_camera":
            n = args.get("name")
            pipeline.training.queue_command({"type": "focus_camera", "camera_name": n})
            return tok(txt(f"focus_camera → {n!r}"))

        if tool == "densify_now":
            pipeline.training.queue_command({"type": "densify_now"})
            return tok(txt("densify_now queued"))

        if tool == "reset_opacities":
            pipeline.training.queue_command({"type": "reset_opacities"})
            return tok(txt("reset_opacities queued"))

        if tool == "save_checkpoint":
            pipeline.training.queue_command({"type": "save"})
            return tok(txt("save queued"))

        # ── pipeline tools ─────────────────────────────────────────────
        if tool == "get_pipeline_status":
            return tok(txt(json.dumps(pipeline.get_status(), indent=2)))

        if tool == "run_pipeline":
            s = args.get("from_stage")
            if not s: return terr("'from_stage' required")
            e = pipeline.run_from(s)
            return tok(txt(f"Error: {e}" if e else f"Pipeline running from '{s}'"))

        if tool == "interrupt_pipeline":
            pipeline.interrupt()
            return tok(txt("Interrupted."))

        if tool == "list_frames":
            frames = pipeline.list_frames()
            summary = f"{len(frames)} frames, " \
                      f"{sum(1 for f in frames if f['excluded'])} excluded, " \
                      f"{sum(1 for f in frames if f['masked'])} masked"
            return tok(txt(summary), txt(json.dumps(frames[:50], indent=2)))

        if tool == "exclude_image":
            n = args.get("name"); ex = args.get("excluded")
            if n is None or ex is None: return terr("'name' and 'excluded' required")
            pipeline.set_exclusion(n, ex)
            return tok(txt(f"{'Excluded' if ex else 'Included'}: {n}"))

        if tool == "get_config":
            cfg = pipeline.training.get_config()
            return tok(txt(json.dumps(cfg, indent=2) if cfg else "No config — training not started."))

        if tool == "update_config":
            pipeline.training.queue_command({"type": "update_config", **args})
            return tok(txt(f"Config update queued: {json.dumps(args)}"))

        return terr(f"Unknown tool: {tool!r}")

    return app


# ─── Server lifecycle ─────────────────────────────────────────────────────────

class DashboardServer:
    """Owns the uvicorn server; pipeline runs in a background thread.

    Usage:
        server = DashboardServer(pipeline_controller, port=7860)
        server.run()           # blocks — uvicorn in calling thread
        server.stop()          # signal shutdown from another thread
    """

    def __init__(self, pipeline, port: int = 7860) -> None:
        self._pipeline = pipeline
        self._port     = port
        self._server   = None

    @property
    def controller(self):
        """Back-compat: expose the training controller."""
        return self._pipeline.training

    def run(self) -> None:
        import uvicorn
        app    = build_app(self._pipeline)
        config = uvicorn.Config(app, host="0.0.0.0", port=self._port, log_level="error")
        self._server = uvicorn.Server(config)
        url = f"http://localhost:{self._port}"
        console.print(f"[bold cyan]dashboard[/] [link={url}]{url}[/link]  "
                      f"[dim]MCP → {url}/mcp[/dim]  Ctrl-C to stop")
        try:
            self._server.run()
        except KeyboardInterrupt:
            console.print("\n[yellow]dashboard[/] stopped.")
            self._pipeline.training.stop()
            self._server.should_exit = True

    def run_training(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Run *fn* in a background thread; uvicorn in calling thread.

        Ctrl-C triggers a graceful shutdown: training is interrupted via
        a 'save' command so it can write a checkpoint before exiting.
        """
        import uvicorn

        result_box: list = []
        exc_box:    list = []

        app    = build_app(self._pipeline)
        config = uvicorn.Config(app, host="0.0.0.0", port=self._port,
                                log_level="error")
        server = uvicorn.Server(config)
        self._server = server

        def _train():
            try:
                result_box.append(fn(*args, **kwargs))
            except Exception as exc:
                exc_box.append(exc)
            finally:
                server.should_exit = True

        t = threading.Thread(target=_train, daemon=False, name="ttgs-training")
        t.start()

        url = f"http://localhost:{self._port}"
        console.print(f"[bold cyan]dashboard[/] [link={url}]{url}[/link]  "
                      f"[dim]MCP → {url}/mcp[/dim]")
        try:
            server.run()
        except KeyboardInterrupt:
            console.print("\n[yellow]dashboard[/] shutting down…")
            # Signal the training loop to save + exit cleanly
            self._pipeline.training.stop()
            server.should_exit = True

        t.join(timeout=60)

        if exc_box:
            raise exc_box[0]
        return result_box[0] if result_box else None

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
