"""PipelineController — owns the full extract → sfm → train → export pipeline.

Each stage runs in a managed background thread.  The controller exposes
thread-safe methods so FastAPI handlers can read status, interrupt, and
restart from any stage without touching the training thread directly.
"""

from __future__ import annotations

import json
import sys
import threading
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

from ttgs.viewer.dashboard import TrainingController

console = Console()

STAGES = ["extract", "sfm", "train", "export"]


@dataclass
class StageState:
    status: str = "idle"     # idle | running | done | error
    message: str = ""        # last status line or error
    start_time: float | None = None
    end_time: float | None = None


class PipelineController:
    """Thread-safe pipeline owner.

    Attributes:
        training:   Embedded TrainingController; used by training loop and
                    the /state /pause /command /mcp routes.
        frames_dir: Where extracted frames live.  Set after extract or by
                    caller when source is already an image directory.
    """

    def __init__(
        self,
        output_dir: Path,
        source: Path | None = None,
        frames_dir: Path | None = None,
        cfg=None,           # ttgs.config.Config
        backend=None,       # ttgs.backend.detect.BackendInfo
        colmap_bin: str | None = None,
    ) -> None:
        self.output_dir  = output_dir
        self.source      = source
        self.cfg         = cfg
        self.backend     = backend
        self.colmap_bin  = colmap_bin

        # Derived directories
        self.frames_dir  = frames_dir or (output_dir / "frames")
        self.sfm_dir     = output_dir / "sfm"
        self.train_dir   = output_dir / "train"
        self.export_dir  = output_dir / "export"
        self.masks_dir   = output_dir / "masks"

        # Stage tracking
        self._lock   = threading.Lock()
        self._stages: dict[str, StageState] = {s: StageState() for s in STAGES}
        self._current: str | None = None
        self._interrupt = threading.Event()
        self._thread: threading.Thread | None = None

        # Embedded training controller
        snapshot_every = cfg.train.snapshot_every if cfg else 0
        self.training = TrainingController(output_dir=self.output_dir,
                                           snapshot_every=snapshot_every)

        # Single source of truth for per-frame edit state.
        # Migrate from legacy masks.json + excluded.json if frames.json
        # doesn't exist yet, then sync derived mask PNGs.
        self._migrate_legacy_json()
        self._sync_mask_pngs()

    # ── Stage status ──────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            now = _time.time()
            stages = {}
            for s, st in self._stages.items():
                d: dict[str, Any] = {"status": st.status, "message": st.message}
                if st.start_time is not None:
                    if st.end_time is not None:
                        d["duration"] = round(st.end_time - st.start_time, 1)
                    elif st.status == "running":
                        d["duration"] = round(now - st.start_time, 1)
                stages[s] = d
            return {
                "stages": stages,
                "current": self._current,
                "interrupting": self._interrupt.is_set(),
                "frames_dir": str(self.frames_dir) if self.frames_dir.exists() else None,
            }

    def _set(self, stage: str, status: str, message: str = "") -> None:
        with self._lock:
            st = self._stages[stage]
            st.status = status
            st.message = message
            if status == "running":
                st.start_time = _time.time()
                st.end_time = None
            elif status in ("done", "error"):
                st.end_time = _time.time()
            self._current = stage if status == "running" else None

    # ── frames.json — single source of truth ────────────────────────────────
    #
    # All per-frame edit state lives in frames_dir/frames.json:
    #   { "image.jpg": { "excluded": false, "polygons": [...], ... }, ... }
    #
    # Rasterised mask PNGs in masks/ are *derived* from the JSON and
    # consumed by COLMAP + the training loop.

    @property
    def _frames_json_path(self) -> Path:
        return self.frames_dir / "frames.json"

    def _load_frames_json(self) -> dict:
        if self._frames_json_path.exists():
            try:
                return json.loads(self._frames_json_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_frames_json(self, data: dict) -> None:
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._frames_json_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def _migrate_legacy_json(self) -> None:
        """One-time migration from legacy masks.json + excluded.json."""
        if self._frames_json_path.exists():
            return  # already migrated
        legacy_excl = self.output_dir / "excluded.json"
        legacy_masks = self.output_dir / "masks.json"
        if not legacy_excl.exists() and not legacy_masks.exists():
            return  # nothing to migrate
        excluded: set[str] = set()
        masks: dict = {}
        if legacy_excl.exists():
            try:
                excluded = set(json.loads(legacy_excl.read_text()))
            except Exception:
                pass
        if legacy_masks.exists():
            try:
                masks = json.loads(legacy_masks.read_text())
            except Exception:
                pass
        # Build frames.json from legacy data
        if not self.frames_dir.exists():
            return
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        frames: dict[str, Any] = {}
        for p in sorted(self.frames_dir.iterdir()):
            if p.suffix not in exts:
                continue
            name = p.name
            entry: dict[str, Any] = {"excluded": name in excluded}
            md = masks.get(name, {})
            for k in ("polygons", "weights", "filters"):
                if md.get(k):
                    entry[k] = md[k]
            frames[name] = entry
        self._save_frames_json(frames)
        console.print(f"[dim]migrated legacy excluded.json + masks.json → frames.json[/]")

    def _sync_mask_pngs(self) -> None:
        """Render mask PNGs for all entries in frames.json that have polygons.

        This ensures derived PNGs exist even when frames.json was copied
        from another output directory (e.g. starting a new training run
        with pre-existing masks).
        """
        data = self._load_frames_json()
        if not data:
            return
        count = 0
        for name, entry in data.items():
            polys = entry.get("polygons", [])
            weights = entry.get("weights", [])
            if polys or weights:
                self._render_mask_png(name, polys, weights)
                count += 1
            filters = entry.get("filters", [])
            if filters:
                self._apply_filters(name, filters)
        if count:
            console.print(f"[dim]synced {count} mask PNGs from frames.json[/]")

    # ── Exclusions ───────────────────────────────────────────────────────────

    def get_exclusions(self) -> set[str]:
        data = self._load_frames_json()
        return {name for name, entry in data.items() if entry.get("excluded")}

    def set_exclusion(self, name: str, excluded: bool) -> None:
        data = self._load_frames_json()
        entry = data.setdefault(name, {})
        entry["excluded"] = excluded
        self._save_frames_json(data)

    # ── Per-image masks ──────────────────────────────────────────────────────

    def get_mask_data(self, image_name: str) -> dict:
        """Return mask primitives for one image.

        Returns ``{"polygons": [...], ...}`` or ``{}`` if none.
        """
        entry = self._load_frames_json().get(image_name, {})
        return {k: entry[k] for k in ("polygons", "weights", "filters") if k in entry}

    def save_mask_data(self, image_name: str, data: dict) -> None:
        """Save mask primitives for one image, regenerate mask PNG, and
        apply any filter primitives to produce a processed frame.

        Expected data format::

            {
              "polygons": [[[x,y], ...], ...],
              "weights":  [{"polygon": [[x,y],...], "weight": 0.5}, ...],
              "filters":  [{"polygon": [[x,y],...], "type": "gaussian_blur", "radius": 15}, ...]
            }
        """
        all_data = self._load_frames_json()
        entry = all_data.setdefault(image_name, {})
        # Update mask fields, remove empty ones
        for k in ("polygons", "weights", "filters"):
            if data.get(k):
                entry[k] = data[k]
            else:
                entry.pop(k, None)
        self._save_frames_json(all_data)
        self._render_mask_png(image_name, data.get("polygons", []),
                              data.get("weights", []))
        self._apply_filters(image_name, data.get("filters", []))

    def _render_mask_png(
        self, image_name: str, polygons: list, weights: list | None = None,
    ) -> None:
        """Render primitives to a grayscale weight-map PNG.

        Pixel values are loss weights:  0 = excluded, 255 = full weight.
        Intermediate values = proportional training attention.
        """
        import numpy as np
        from PIL import Image as PILImage, ImageDraw

        png_path = self.mask_path(image_name)

        if not polygons and not weights:
            if png_path.exists():
                png_path.unlink()
            return

        img_path = self.frames_dir / image_name
        if not img_path.exists():
            return
        with PILImage.open(img_path) as src:
            w, h = src.size

        # Start with all-zero (excluded)
        arr = np.zeros((h, w), dtype=np.float32)

        # Layer 1: foreground polygons → weight 1.0
        if polygons:
            fg = PILImage.new("L", (w, h), 0)
            draw = ImageDraw.Draw(fg)
            for poly in polygons:
                coords = [(int(round(p[0])), int(round(p[1]))) for p in poly]
                if len(coords) >= 3:
                    draw.polygon(coords, fill=255)
            arr = np.array(fg, dtype=np.float32) / 255.0

        # Layer 2: weight regions — overwrite their area with the specified weight
        if weights:
            for wr in weights:
                poly = wr.get("polygon", [])
                wval = float(wr.get("weight", 0.5))
                if len(poly) < 3:
                    continue
                region = PILImage.new("L", (w, h), 0)
                ImageDraw.Draw(region).polygon(
                    [(int(round(p[0])), int(round(p[1]))) for p in poly],
                    fill=255,
                )
                mask_bool = np.array(region) > 0
                arr[mask_bool] = wval

        out = PILImage.fromarray((arr.clip(0, 1) * 255).astype(np.uint8))
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        out.save(png_path)

    def _apply_filters(self, image_name: str, filters: list) -> None:
        """Apply spatial filter primitives to produce a processed frame.

        Writes to output_dir/processed_frames/{image_name}.  If no filters,
        removes the processed file so the pipeline falls back to the raw frame.
        """
        import numpy as np
        from PIL import Image as PILImage, ImageDraw, ImageFilter, ImageEnhance

        processed_dir = self.output_dir / "processed_frames"
        out_path = processed_dir / image_name
        src_path = self.frames_dir / image_name

        if not filters or not src_path.exists():
            if out_path.exists():
                out_path.unlink()
            return

        processed_dir.mkdir(parents=True, exist_ok=True)
        img = PILImage.open(src_path).convert("RGB")
        w, h = img.size
        arr = np.array(img)

        for flt in filters:
            poly = flt.get("polygon", [])
            ftype = flt.get("type", "gaussian_blur")
            if len(poly) < 3:
                continue

            # Build boolean mask for this polygon region
            region_mask = PILImage.new("L", (w, h), 0)
            coords = [(int(round(p[0])), int(round(p[1]))) for p in poly]
            ImageDraw.Draw(region_mask).polygon(coords, fill=255)
            rmask = np.array(region_mask) > 0

            if ftype == "gaussian_blur":
                radius = int(flt.get("radius", 15))
                blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=radius)))
                arr[rmask] = blurred[rmask]

            elif ftype == "median":
                radius = int(flt.get("radius", 5))
                # PIL MedianFilter needs odd kernel size
                ks = max(3, radius * 2 + 1)
                if ks % 2 == 0:
                    ks += 1
                filtered = np.array(img.filter(ImageFilter.MedianFilter(size=ks)))
                arr[rmask] = filtered[rmask]

            elif ftype == "sharpen":
                sharpened = np.array(ImageEnhance.Sharpness(img).enhance(2.0))
                arr[rmask] = sharpened[rmask]

            elif ftype == "brightness":
                factor = float(flt.get("factor", 1.4))
                bright = np.array(ImageEnhance.Brightness(img).enhance(factor))
                arr[rmask] = bright[rmask]

        PILImage.fromarray(arr).save(out_path, quality=95)

    def mask_path(self, image_name: str) -> Path:
        return self.masks_dir / (Path(image_name).stem + ".png")

    def has_mask(self, image_name: str) -> bool:
        return self.mask_path(image_name).exists()

    def save_mask(self, image_name: str, arr) -> None:
        """Save a raw rasterised mask (from brush paint). No JSON update."""
        import numpy as np
        from PIL import Image as PILImage
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray((arr.clip(0, 1) * 255).astype(np.uint8)).save(self.mask_path(image_name))

    def load_mask(self, image_name: str):
        import numpy as np
        from PIL import Image as PILImage
        p = self.mask_path(image_name)
        if not p.exists():
            return None
        return np.array(PILImage.open(p).convert("L"), dtype=np.float32) / 255.0

    def delete_mask(self, image_name: str) -> None:
        p = self.mask_path(image_name)
        if p.exists():
            p.unlink()
        # Clear mask fields from frames.json
        all_data = self._load_frames_json()
        entry = all_data.get(image_name, {})
        for k in ("polygons", "weights", "filters"):
            entry.pop(k, None)
        if entry:
            all_data[image_name] = entry
        self._save_frames_json(all_data)

    # ── COLMAP overlay data ──────────────────────────────────────────────────

    def get_colmap_features(self, image_name: str) -> dict[str, Any] | None:
        """Load 2D observations for *image_name* from COLMAP images.bin.

        Returns ``{keypoints: [[x,y,triangulated], ...]}`` where triangulated
        is 1 if the observation maps to a 3D point, 0 otherwise.
        Returns None if the SfM data doesn't exist.
        """
        import struct

        # Find images.bin
        sparse = self.sfm_dir / "undistorted" / "sparse"
        if not sparse.is_dir():
            sparse = self.sfm_dir / "sparse" / "0"
        if not sparse.is_dir():
            sparse = self.sfm_dir / "sparse"
        images_bin = sparse / "images.bin"
        if not images_bin.exists():
            return None

        try:
            with open(images_bin, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                for _ in range(n):
                    _image_id = struct.unpack("<I", f.read(4))[0]
                    f.read(32)  # qvec (4 doubles)
                    f.read(24)  # tvec (3 doubles)
                    _camera_id = struct.unpack("<I", f.read(4))[0]
                    # Null-terminated name
                    name_bytes = b""
                    while True:
                        c = f.read(1)
                        if c == b"\x00":
                            break
                        name_bytes += c
                    name = name_bytes.decode("utf-8", errors="replace")
                    # 2D observations
                    (num_pts2d,) = struct.unpack("<Q", f.read(8))
                    if name == image_name or name.endswith("/" + image_name):
                        pts = []
                        for _ in range(num_pts2d):
                            x, y = struct.unpack("<2d", f.read(16))
                            pt3d_id = struct.unpack("<q", f.read(8))[0]
                            pts.append([round(x, 1), round(y, 1),
                                        1 if pt3d_id >= 0 else 0])
                        return {"keypoints": pts}
                    else:
                        f.read(num_pts2d * 24)  # skip
        except Exception:
            return None
        return None

    # ── Frame list ────────────────────────────────────────────────────────────

    def list_frames(self) -> list[dict[str, Any]]:
        """Return metadata for every frame in frames_dir."""
        if not self.frames_dir.exists():
            return []
        data = self._load_frames_json()
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        frames = []
        for p in sorted(self.frames_dir.iterdir()):
            if p.suffix not in exts:
                continue
            entry = data.get(p.name, {})
            frames.append({
                "name":     p.name,
                "excluded": entry.get("excluded", False),
                "masked":   self.has_mask(p.name),
            })
        return frames

    def read_frames_json(self) -> dict:
        """Public accessor for the /frames.json API route."""
        return self._load_frames_json()

    # ── Pipeline execution ────────────────────────────────────────────────────

    def interrupt(self) -> None:
        """Signal the current stage to stop."""
        self._interrupt.set()
        self.training.stop()      # tell the training loop to save + exit

    def run_from(self, from_stage: str) -> str:
        """Start (or restart) the pipeline from *from_stage* in a background thread.

        Returns an error string if the stage name is invalid or a run is already
        in progress; otherwise returns "".
        """
        if from_stage not in STAGES:
            return f"unknown stage: {from_stage!r}"
        if self._thread is not None and self._thread.is_alive():
            return "a stage is already running — interrupt first"
        self._interrupt.clear()
        self.training.resume()

        self._thread = threading.Thread(
            target=self._run_stages,
            args=(from_stage,),
            daemon=True,
            name=f"ttgs-pipeline-{from_stage}",
        )
        self._thread.start()
        return ""

    def _run_stages(self, from_stage: str) -> None:
        idx = STAGES.index(from_stage)
        for stage in STAGES[idx:]:
            if self._interrupt.is_set():
                self._set(stage, "idle", "interrupted")
                break
            self._set(stage, "running")
            try:
                self._run_stage(stage)
                self._set(stage, "done")
            except Exception as exc:
                self._set(stage, "error", str(exc))
                console.print(f"[red]{stage} failed:[/] {exc}")
                break

    def _run_stage(self, stage: str) -> None:
        if stage == "extract":
            self._stage_extract()
        elif stage == "sfm":
            self._stage_sfm()
        elif stage == "train":
            self._stage_train()
        elif stage == "export":
            self._stage_export()

    # ── Individual stages ─────────────────────────────────────────────────────

    def _stage_extract(self) -> None:
        if self.source is None:
            raise ValueError("source is required for the extract stage")
        from ttgs.stages.extract import run as extract_run
        result = extract_run(self.source, self.frames_dir, self.cfg.extract)
        self.frames_dir = result

    def _stage_sfm(self) -> None:
        from ttgs.stages.sfm import run as sfm_run
        images_path = self._prepare_sfm_images()
        masks_arg = self.masks_dir if (self.masks_dir.exists()
                                       and any(self.masks_dir.glob("*.png"))) else None
        sfm_run(images_path, self.sfm_dir, self.cfg.sfm, self.colmap_bin,
                masks_dir=masks_arg)

    def _stage_train(self) -> None:
        from ttgs.stages.train import run as train_run

        # Resolve backend lazily
        if self.backend is None:
            from ttgs.backend.detect import best as best_backend
            self.backend = best_backend()

        # Fresh training controller for each run
        snapshot_every = self.cfg.train.snapshot_every if self.cfg else 0
        self.training = TrainingController(output_dir=self.output_dir,
                                           snapshot_every=snapshot_every)

        dataset_dir = self.sfm_dir / "undistorted"
        if not dataset_dir.exists():
            dataset_dir = self.sfm_dir

        train_run(
            dataset_dir,
            self.train_dir,
            self.cfg.train,
            self.backend,
            resume=False,
            dashboard=self.training,
            masks_dir=self.masks_dir,  # always pass — may be created mid-training
            excluded=self.get_exclusions(),
        )

    def _stage_export(self) -> None:
        from ttgs.stages.export import run as export_run
        ply = self.train_dir / "splat.ply"
        if not ply.exists():
            raise FileNotFoundError(f"splat.ply not found at {ply}")
        export_run(ply, self.export_dir, self.cfg.export)

    # ── SfM image preparation ─────────────────────────────────────────────────

    def _prepare_sfm_images(self) -> Path:
        """Return a directory of images for COLMAP.

        If some images are excluded, copies/symlinks the included subset into
        sfm_dir/filtered_images/ so COLMAP only sees them.
        """
        import shutil
        excluded = self.get_exclusions()
        if not excluded:
            return self.frames_dir

        filtered = self.sfm_dir / "filtered_images"
        filtered.mkdir(parents=True, exist_ok=True)
        # Remove stale entries
        for f in filtered.iterdir():
            f.unlink(missing_ok=True)

        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        for p in sorted(self.frames_dir.iterdir()):
            if p.suffix not in exts or p.name in excluded:
                continue
            dest = filtered / p.name
            if sys.platform == "win32":
                shutil.copy2(p, dest)
            else:
                dest.symlink_to(p.resolve())

        return filtered
