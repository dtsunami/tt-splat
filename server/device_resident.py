#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Stage E — the DEVICE-RESIDENT 3DGS training loop.  Params + Adam m/v live on the Blackhole as ttnn
tensors and NEVER leave the card in the inner loop (only the projected 2D params + grads round-trip to
host for the M6 bin/sort, which is explicitly deferred — GPU radix sort is the >100k follow-up).

Per step (NO host autograd, NO 3D-param readback):
  B  device projection FORWARD   (device_project: project_geom/color/op, reading resident params)
  -> raster FORWARD              (M14 sfpu_raster_scaled, culled, multi-tile; image + final-T)
  -> loss / dL-dimage           (host elementwise — trivial)
  A  device raster BACKWARD      (fused_backward_grid: one grid dispatch/chunk, all tiles parallel)
  D  device projection BACKWARD  (device_project_backward, consuming B's aux — no forward recompute)
  C  device Adam                 (device_adam, batched update of resident params)

Correctness gate: convergence on real data + per-step 3D grads match the host-autograd render_train path.
Perf: `.timings` accumulates per-stage wall time (synchronized) for bottleneck analysis.
"""
from __future__ import annotations
import sys, time, os
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "pathclear"))
import ttnn                                                # noqa: E402
from bin_sort import bin_and_sort                          # noqa: E402  M6 per-tile cull + depth sort
from device_binsort import device_binsort                  # noqa: E402  S2 on-device counting bin/sort
from raster_blocked import needs_blocking, raster_rgb_blocked, fused_backward_blocked  # noqa: E402  >352px tile-block raster
import sfpu_raster_scaled as M14                            # noqa: E402  TS, B
from render_device import _device, _resources, _raster_channel, _raster_rgb   # noqa: E402  M14 forward
from device_project import project_geom, project_color, project_op   # noqa: E402  Stage B
from device_project_backward import project_backward        # noqa: E402  Stage D
from device_adam import DeviceAdam                          # noqa: E402  Stage C
from fused_backward import fused_backward_grid              # noqa: E402  Stage A

_GEOM = ("cx", "cy", "a", "b", "c", "op")
_DBG = __import__("os").environ.get("RESIDENT_DBG", "0") == "1"


def _dbg(msg):
    if _DBG:
        print(f"      [resident] {msg}", flush=True)


def _sync(dev):
    try:
        ttnn.synchronize_device(dev)
    except Exception:
        pass


class DeviceResidentTrainer:
    def __init__(self, dev, P, lr=None, deg=None, lambda_dssim=0.2, sh_interval=0, aa=False):
        self.dev = dev
        self.deg = int(P["deg"] if deg is None else deg)
        self.lambda_dssim = float(lambda_dssim)    # L1 + D-SSIM loss (0 = pure L1; 3DGS default 0.2)
        self.sh_interval = int(sh_interval)        # progressive-SH: steps per band (0 = full deg, warmup off)
        self.deg_eff = self.deg                    # effective SH degree this step (ramps under warmup)
        self.aa = bool(aa)                         # Mip-Splatting anti-alias opacity compensation
        self.near_frac = float(os.environ.get("TT_NEAR_FRAC", "0.05"))   # near-plane cull = frac × median scene depth
        self.debug = int(os.environ.get("TT_DEBUG", "0"))   # --debug: per-step finite-stats triage (see step())
        self.grad_clip = float(os.environ.get("TT_GRAD_CLIP", "10.0"))   # clip 2D grads pre-Stage-D (anti-divergence; 0=off)
        self.proj_sanitize = os.environ.get("TT_PROJ_SANITIZE", "1") == "1"  # #1: source-side floors in project_backward (default ON)
        self.skip_on_nan = os.environ.get("TT_SKIP_STEP", "1") == "1"        # #3: skip a step on non-finite loss, don't abort (default ON)
        self._skips = 0                            # consecutive skipped (non-finite) steps — patience counter
        self.skip_patience = int(os.environ.get("TT_SKIP_PATIENCE", "16"))   # give up after this many in a row
        # Device D-SSIM loss — runs the L1+D-SSIM loss & dL/dimage on the Blackhole (matmul-blur), killing the
        # ~300-420ms/step host SSIM at high res. On unless TT_DSSIM_DEVICE=0; masked frames / lambda_dssim<=0
        # fall back to the host autograd path transparently. See dssim_device.py.
        self.dssim_dev = None
        if self.lambda_dssim > 0 and int(os.environ.get("TT_DSSIM_DEVICE", "1")):
            try:
                from dssim_device import DeviceDSSIM
                self.dssim_dev = DeviceDSSIM(dev)
                print("  [resident] device D-SSIM loss ON (matmul-blur; TT_DSSIM_DEVICE=0 to use host)")
            except Exception as exc:
                print(f"  [resident] device D-SSIM unavailable ({type(exc).__name__}: {exc}); host loss path")
        self.last_screen = None                    # last step's per-Gaussian (u,v,zc,du,dv,valid) for pose-opt
        init = {k: P[k].detach().clone() for k in ("mean", "scale", "quat", "op", "sh")}
        self.K = init["sh"].shape[1]
        lr = lr or {"mean": .01, "scale": .01, "quat": .01, "op": .02, "sh": .01}
        self.adam = DeviceAdam(dev, init, lr)              # params + m/v resident on device
        self.timings = {k: 0.0 for k in ("B", "bin", "raster", "loss", "A", "D", "C", "step")}
        self.nstep = 0
        self.N = int(init["mean"].shape[0])   # live Gaussian count (mutated by prune)
        # SCALE BAND (anti-degenerate): the device raster/backward use the RAW conic with no screen-space
        # low-pass, so a sub-pixel Gaussian (e.g. a split child) gives conic ~1/scale^2 -> exploding position
        # gradients -> mean blows to ±inf -> NaN loss. Until a proper 2D low-pass lands, keep every scale in a
        # scene-relative band [med-band, med+band] (clamped each step + used as the split floor). 0 = disabled.
        _smed = float(init["scale"].float().median())
        _band = float(os.environ.get("TT_SCALE_BAND", "4.0"))     # e^4 ≈ 55x each way (~3000x total range)
        self.scale_lo = (_smed - _band) if _band > 0 else None
        self.scale_hi = (_smed + _band) if _band > 0 else None
        self.step_log = []          # per-step {stage: ms} — for profiling the ramp/stall curve
        self._gpos = np.zeros(self.N, dtype=np.float64)   # accumulated 2D positional-grad magnitude (densify signal)
        self._gacc = 0
        self.profiler = None        # --profile / TT_PROFILE=1: live per-step device compute-util
        if os.environ.get("TT_PROFILE", "0") == "1":
            try:
                from profiler import DeviceProfiler
                self.profiler = DeviceProfiler()
                print("  [resident] --profile ON — live per-step device compute-util (tt-metal device profiler)")
            except Exception as exc:
                print(f"  [resident] profiler unavailable ({type(exc).__name__}: {exc}); continuing without util")

    # ---- resident params (ttnn) ----
    @property
    def _p(self):
        return self.adam.p

    def step(self, cam, gt, mask=None):
        """cam: (Rv,tv,fx,fy,cx,cy[,name]).  gt: [H,W,3] torch/np.  mask: [H,W] or None.
        Returns (loss float, image np[H,W,3]).  All param math on device; params stay resident."""
        dev = self.dev
        gt = gt.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(gt) else np.asarray(gt, np.float64)
        H, W = gt.shape[:2]
        t0 = time.perf_counter()

        # progressive-SH warmup (recipe gap #2): ramp the EFFECTIVE degree 0->deg over the run so high-freq
        # view-dependent colour isn't fit before geometry settles. sh is full-size; only the first
        # (deg_eff+1)^2 bands contribute fwd, and the rest get ZERO grad in Stage D (padded there).
        self.deg_eff = self.deg if self.sh_interval <= 0 else min(self.deg, self.nstep // self.sh_interval)

        # ===== B: device projection forward (reads resident params) =====
        _dbg("B: project_geom...")
        p = self._p
        u_t, v_t, zc_t, (ca_t, cb_t, cc_t), Ageo = project_geom(
            dev, p["mean"], p["scale"], p["quat"], cam, aux=True)
        _dbg("B: project_color...")
        cR_t, cG_t, cB_t, Acol = project_color(dev, p["mean"], p["sh"], self.deg_eff, cam, aux=True)
        _dbg("B: project_op + 2D readback...")
        op_t = project_op(dev, p["op"])
        g = lambda t: ttnn.to_torch(t).flatten().numpy().astype(np.float64)   # 2D readback (deferred bin)
        u, v, zc = g(u_t), g(v_t), g(zc_t)
        ca, cb, cc = g(ca_t), g(cb_t), g(cc_t)
        opn = g(op_t); col = np.stack([g(cR_t), g(cG_t), g(cB_t)], axis=-1)
        N = u.shape[0]
        # Mip-Splatting anti-aliasing (recipe gap #3): scale opacity by sqrt(det ratio) (<=1) so sub-pixel
        # splats shrink. Computed from the read-back conic (no extra device op). The op-grad is scaled to
        # match in Stage D below (the factor's coupling to scale/quat is a 2nd-order term, left out).
        aaf = None
        if self.aa:
            from device_project import aa_factor
            aaf = aa_factor(ca, cb, cc)
            opn = opn * aaf
        _sync(dev); t1 = time.perf_counter(); self.timings["B"] += t1 - t0
        # ===== bin/sort (host, deferred) + cull =====
        TS = M14.TS
        Wp, Hp = ((W + TS - 1) // TS) * TS, ((H + TS - 1) // TS) * TS
        # NEAR-PLANE CULL (the real anti-NaN fix): a Gaussian with zc just above 0 projects to u,v ∝ 1/zc with
        # position gradients ∝ 1/zc² — enormous. Densify-flung children land near the camera, Adam then blows
        # their mean to ±inf → inf cx → NaN loss (the divergence we chased). zc>1e-4 admitted them; instead cull
        # anything closer than near_frac × the typical (median) scene depth. Frozen near splats are harmless.
        _zpos = zc[zc > 1e-4]
        _near = max(1e-3, self.near_frac * float(np.median(_zpos))) if _zpos.size else 1e-3
        valid = zc > _near
        vidx = np.nonzero(valid)[0]
        img = np.zeros((H, W, 3), np.float64)
        grads2d = {k: np.zeros(N, np.float64) for k in _GEOM}
        col_g = [np.zeros(N, np.float64) for _ in range(3)]
        if vidx.size:
            cxv, cyv, av, bv, ccv, opv, zcv = (a[valid] for a in (u, v, ca, cb, cc, opn, zc))
            colv = [col[valid, k] for k in range(3)]
            blocked = needs_blocking(dev, Wp // TS, Hp // TS)   # >352px: tile-block raster (raster_blocked.py)
            res = None if blocked else _resources(dev, Wp, Hp)
            if os.environ.get("TT_DEVICE_BINSORT", "0") == "1":   # S5: on-device counting bin/sort (device_binsort.py)
                s_gid, _st, ranges, ntx, nty, _tot = device_binsort(dev, cxv, cyv, av, bv, ccv, zcv, Wp, Hp, ts=TS)
            else:
                detc = av * ccv - bv * bv; detc = np.where(np.abs(detc) < 1e-12, 1e-12, detc)
                var_x = np.clip(ccv / detc, 0.25, None); var_y = np.clip(av / detc, 0.25, None)
                s_gid, _st, ranges, ntx, nty, _tot = bin_and_sort(cxv, cyv, var_x, var_y, zcv, Wp, Hp, ts=TS)
            tile_lists = [s_gid[ranges[t, 0]:ranges[t, 1]].tolist() for t in range(ntx * nty)]
            maxc = max((len(l) for l in tile_lists), default=0)
            nbatch = (maxc + M14.B - 1) // M14.B
            t2 = time.perf_counter(); self.timings["bin"] += t2 - t1
            _dbg(f"raster fwd (N={N} valid={vidx.size} nbatch={nbatch} ntx={ntx} nty={nty} blocked={blocked})...")

            # ===== raster forward (M14; tile-block loop when >352px) =====
            if blocked:
                chans, Tfin = raster_rgb_blocked(dev, tile_lists, ntx, nty, cxv, cyv, av, bv, ccv, opv,
                                                 colv, Wp, Hp, want_T=True)
            else:
                chans, Tfin = _raster_rgb(dev, res, tile_lists, ntx, nbatch, cxv, cyv, av, bv, ccv, opv,
                                          colv, Wp, Hp, want_T=True)
            img = np.clip(np.stack(chans, axis=-1)[:H, :W, :], 0.0, 1.0)
            _sync(dev); t3 = time.perf_counter(); self.timings["raster"] += t3 - t2
            # ===== loss (L1 + D-SSIM) + dL/dimage — ON DEVICE (matmul-blur SSIM) when enabled & unmasked =====
            mk = (mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)) \
                if mask is not None else None
            if self.dssim_dev is not None and mk is None:        # device SSIM (no host autograd; ~2.6x faster @1600px)
                loss, gi = self.dssim_dev.loss_grad(img, gt, self.lambda_dssim)
            else:                                                # host autograd path (always; + masked frames)
                from loss import dloss_dimage
                loss, gi = dloss_dimage(img, gt, mk, self.lambda_dssim)
            gp = np.zeros((Hp, Wp, 3), np.float64); gp[:H, :W, :] = gi
            t4 = time.perf_counter(); self.timings["loss"] += t4 - t3

            # ===== #3 SKIP-STEP: a non-finite loss means this step's FORWARD is poisoned. Skip the whole
            # backward+Adam so no garbage update lands; params stay as-is and the next camera usually recovers.
            # Patience-bounded so a genuinely stuck run still terminates (vs spinning forever). =====
            if self.skip_on_nan and not np.isfinite(loss):
                self._skips += 1
                self.last_screen = None                 # don't feed a poisoned pose/densify signal
                self.nstep += 1
                self.step_log.append(dict(B=1e3 * (t1 - t0), bin=1e3 * (t2 - t1), raster=1e3 * (t3 - t2),
                                          loss=1e3 * (t4 - t3), A=0.0, D=0.0, C=0.0,
                                          step=1e3 * (t4 - t0), skipped=1))
                print(f"  [skip-step @{self.nstep}] non-finite loss — backward/Adam skipped "
                      f"({self._skips}/{self.skip_patience} in a row)", flush=True)
                return loss, img.astype(np.float32)     # params UNCHANGED this step
            self._skips = 0                             # finite step → reset the patience counter

            # ===== A: raster backward (grid-sharded; tile-block loop when >352px) =====
            if blocked:
                gv, cgv = fused_backward_blocked(dev, cxv, cyv, av, bv, ccv, opv, colv,
                                                 tile_lists, ntx, nty, Wp, Hp, gp, Tfin)
            else:
                gv, cgv = fused_backward_grid(dev, cxv, cyv, av, bv, ccv, opv, colv,
                                              tile_lists, ntx, Hp // 32, Wp, Hp, gp, Tfin,
                                              stage=os.environ.get("TT_FB_STAGE", "s4"))   # Stage 4 default
            for key in _GEOM:
                grads2d[key][vidx] = gv[key]
            for k in range(3):
                col_g[k][vidx] = cgv[k]
            if aaf is not None:                # AA: dL/d(sigmoid_op) = dL/d(eff_op)*aa (eff_op = sigmoid*aa)
                grads2d["op"] = grads2d["op"] * aaf
            _dbg("A done")
            _sync(dev); t5 = time.perf_counter(); self.timings["A"] += t5 - t4
        else:
            loss = float((img - gt).__pow__(2).mean())
            t5 = time.perf_counter()

        if self.grad_clip > 0:                             # GRADIENT CLIP — the fix the --debug telemetry pinpointed: a
            for _k in grads2d:                             # split child's 2D conic-grad explodes (~1e4 vs ~1e-3) and Stage D
                grads2d[_k] = np.clip(np.nan_to_num(grads2d[_k]), -self.grad_clip, self.grad_clip)   # project_backward → NaN. Bound it finite.
        if self._gpos.shape[0] != N:                       # defensive: never let a stale densify-signal size hard-crash a run
            self._gpos = np.zeros(N, dtype=np.float64); self._gacc = 0
        self._gpos += np.sqrt(grads2d["cx"] ** 2 + grads2d["cy"] ** 2)   # densify signal: screen-space ‖∂L/∂(u,v)‖
        self._gacc += 1
        # cache per-Gaussian screen state for host-side camera pose-opt (recipe gap #1): dL/du=cx, dL/dv=cy
        self.last_screen = dict(u=u, v=v, zc=zc, du=grads2d["cx"], dv=grads2d["cy"], valid=valid)

        # ===== D: device projection backward (reuse B aux) =====
        up = dict(u=grads2d["cx"], v=grads2d["cy"], ca=grads2d["a"], cb=grads2d["b"], cc=grads2d["c"],
                  op=grads2d["op"], colR=col_g[0], colG=col_g[1], colB=col_g[2])
        Pd = dict(mean=p["mean"], scale=p["scale"], quat=p["quat"], sh=p["sh"], op=p["op"], deg=self.deg_eff)
        _dbg("D: project_backward...")
        g3 = project_backward(dev, Pd, cam, up, aux=(Ageo, Acol), return_ttnn=True,
                              sanitize=self.proj_sanitize)  # #1 source-side floors when enabled; grads stay resident
        self.last_g3 = g3                                  # ttnn; grad-equivalence gate reads via to_torch
        _sync(dev); t6 = time.perf_counter(); self.timings["D"] += t6 - t5
        # ===== C: device Adam (resident grads -> NO host round-trip) =====
        _dbg("C: adam.step...")
        _gin = {k: g3[k] for k in ("mean", "scale", "quat", "op", "sh")}
        if self.grad_clip > 0 or self.proj_sanitize:       # SANITIZE 3D grads — THE net: project_backward can emit NaN/inf
            _C = self.grad_clip if self.grad_clip > 0 else 1e30   # from degenerate split-child geometry. #1's on-device floors
            _gin = {k: torch.nan_to_num(ttnn.to_torch(v), nan=0.0, posinf=_C, neginf=-_C).clamp(-_C, _C)   # bound inf; this host
                    for k, v in _gin.items()}              # nan_to_num kills any residual NaN → params stay finite. (TT_GRAD_CLIP=0 off)
        self.adam.step(_gin)
        if self.scale_lo is not None:                      # anti-degenerate scale band — bounds the conic so a
            self.adam.clamp_param("scale", min_val=self.scale_lo, max_val=self.scale_hi)   # tiny splat can't explode grads
        _sync(dev); t7 = time.perf_counter(); self.timings["C"] += t7 - t6
        self.timings["step"] += t7 - t0
        self.nstep += 1
        _log = dict(
            B=1e3 * (t1 - t0), bin=1e3 * (t2 - t1) if vidx.size else 0.0,
            raster=1e3 * (t3 - t2) if vidx.size else 0.0,
            loss=1e3 * (t4 - t3) if vidx.size else 0.0, A=1e3 * (t5 - t4) if vidx.size else 0.0,
            D=1e3 * (t6 - t5), C=1e3 * (t7 - t6), step=1e3 * (t7 - t0))
        if self.profiler is not None:        # adds util/dev_us/cores (device-side; --profile only)
            try:                              # one read after the step (cheap; util is a lower bound under heavy marker load)
                _log.update(self.profiler.step(dev, _log["step"]))
            except Exception:
                pass
        self.step_log.append(_log)
        if self.debug:                                     # --debug numerical-stability triage: flag the FIRST
            p = self._p                                    # non-finite quantity in pipeline order so divergence is sourced
            pl, pb = self._finite({"mean": p["mean"], "scale": p["scale"], "op": p["op"]})
            g3l, g3b = self._finite({k: g3[k] for k in ("mean", "scale", "op")})
            if vidx.size:
                vl, vb = self._finite({"u": u[valid], "zc": zc[valid], "conic": np.stack([ca[valid], cb[valid], cc[valid]])})
                zf = zc[valid]; zf = zf[np.isfinite(zf)]
                proj = (f"zc_min={zf.min():.1e} " if zf.size else "") + vl
                g2l, g2b = self._finite({"d_uv": np.stack([grads2d['cx'], grads2d['cy']]),
                                         "d_conic": np.stack([grads2d['a'], grads2d['b'], grads2d['c']])})
            else:
                proj, vb, g2l, g2b = "(nothing visible)", None, "-", None
            first = vb or g2b or g3b or pb                 # proj→2D grad→3D grad→param(after Adam); earliest = the source
            print(f"  [dbg @{self.nstep}] vis={vidx.size}/{N} | proj {proj} | grad2d {g2l} | grad3d {g3l} | param {pl}"
                  + (f"  ⚠FIRST-NONFINITE={first}" if first else "  ✓"), flush=True)
        return loss, img.astype(np.float32)

    def _finite(self, arrs):
        """--debug telemetry: arrs = {name: np.ndarray | ttnn.Tensor}. Returns ('name=|max| …', first-non-finite name|None)."""
        out, bad = [], None
        for k, v in arrs.items():
            a = ttnn.to_torch(v).float().numpy() if isinstance(v, ttnn.Tensor) else np.asarray(v)
            fin = np.isfinite(a)
            if fin.all():
                out.append(f"{k}={np.abs(a).max():.1e}")
            else:
                out.append(f"{k}=NaN/inf×{int((~fin).sum())}"); bad = bad or k
        return " ".join(out), bad

    def params_host(self):
        """Read resident params back to host torch (for PLY export / checkpoint — NOT per inner step)."""
        out = {k: ttnn.to_torch(self._p[k]).double() for k in ("mean", "scale", "quat", "op", "sh")}
        out["deg"] = self.deg
        return out

    def report(self):
        n = max(1, self.nstep)
        ms = {k: 1e3 * v / n for k, v in self.timings.items()}
        order = ["B", "bin", "raster", "loss", "A", "D", "C", "step"]
        line = "  ".join(f"{k}={ms[k]:.1f}" for k in order)
        return f"per-step ms (avg over {n}):  {line}"

    # ---- dashboard commands on the RESIDENT params (no host autograd; params stay on device) ----
    def prune(self, threshold):
        """Drop Gaussians with sigmoid(op) <= threshold, preserving survivors' Adam state.
        Returns the new Gaussian count."""
        op = ttnn.to_torch(self._p["op"]).reshape(-1)
        keep = torch.sigmoid(op) > float(threshold)
        nkeep = int(keep.sum().item())
        if nkeep == 0 or nkeep == keep.numel():
            return self.N                       # refuse a prune that would empty or change nothing
        self.adam.prune(keep)
        self.N = nkeep
        self._gpos = np.zeros(self.N, dtype=np.float64); self._gacc = 0   # densify signal must track N (like cull/densify/spawn)
        return nkeep

    def reset_opacities(self, prob=0.01):
        import math
        self.adam.fill_param("op", math.log(prob / (1.0 - prob)))   # op is a logit

    def clamp_scale(self, max_log_scale):
        self.adam.clamp_param("scale", max_val=max_log_scale)

    def set_lr(self, factor):
        self.adam.scale_lr(factor)

    def densify(self, grad_threshold=0.0, n_max=100000):
        """Clone/split/prune the resident Gaussians from the accumulated screen-space positional grad,
        then rebuild the resident DeviceAdam at the new N (fresh m/v — the proven M7 reshape). Returns stats."""
        from densify import densify_3d
        Ph = self.params_host()                                   # torch dict mean/scale/quat/op/sh + deg
        g = torch.from_numpy(self._gpos / max(self._gacc, 1)).float()
        _msl = {"min_log_scale": self.scale_lo} if self.scale_lo is not None else {}   # split floor = scene band
        _guard = os.environ.get("TT_DENSIFY_GUARD", "1") == "1"   # #2: prune collapsed splats + unit-quat children (default ON)
        newP, st = densify_3d(Ph, g, grad_threshold=grad_threshold, n_max=n_max, guard=_guard, **_msl)
        # NO-OP SHORT-CIRCUIT: when densify changes nothing (no grad signal, or the DIVERGENCE bail), do NOT
        # rebuild the DeviceAdam. A rebuild wipes m/v momentum and (t kept large, m/v=0) jolts EVERY param by
        # ~lr·sign(g) on the next step — so a no-op densify every `densify_every` steps was silently thrashing
        # training. Keep Adam state intact; just refresh the grad-signal window and report why it no-op'd.
        if st["n_after"] == st["n_before"] and not (st["clone"] or st["split"] or st["prune"]):
            st["reason"] = "diverged(params non-finite)" if st.get("diverged") else \
                           f"no grad signal (gmax={float(g.max()) if g.numel() else 0.0:.2e}, gacc={self._gacc})"
            self._gpos = np.zeros(self.N, dtype=np.float64); self._gacc = 0
            return st
        init = {k: newP[k].detach() for k in ("mean", "scale", "quat", "op", "sh")}
        t_keep = self.adam.t
        self.adam = DeviceAdam(self.dev, init, self.adam.lr, self.adam.b1, self.adam.b2, self.adam.eps)
        self.adam.t = t_keep                                      # keep Adam bias-correction continuity
        self.N = int(init["mean"].shape[0])
        self._gpos = np.zeros(self.N, dtype=np.float64); self._gacc = 0
        return st

    def cull(self, keep_mask):
        """Interactive eraser: delete the Gaussians where keep_mask (torch/np bool [N]) is False
        (e.g. floaters/fog the human brushed over). Returns the new count."""
        keep = torch.as_tensor(np.asarray(keep_mask)).to(torch.bool)
        nkeep = int(keep.sum().item())
        if nkeep == 0 or nkeep == self.N:
            return self.N                        # refuse to empty the scene or no-op
        self.adam.prune(keep)
        self.N = nkeep
        self._gpos = np.zeros(self.N, dtype=np.float64); self._gacc = 0
        return self.N

    def spawn(self, means, colors):
        """Bubble gun: append new Gaussians at world `means` [M,3] with RGB `colors` [M,3], then
        rebuild the resident DeviceAdam at the new N. Returns the new count."""
        from densify import spawn_gaussians
        Ph = self.params_host()
        newP = spawn_gaussians(Ph, torch.as_tensor(np.asarray(means)), torch.as_tensor(np.asarray(colors)))
        init = {k: newP[k].detach() for k in ("mean", "scale", "quat", "op", "sh")}
        t_keep = self.adam.t
        self.adam = DeviceAdam(self.dev, init, self.adam.lr, self.adam.b1, self.adam.b2, self.adam.eps)
        self.adam.t = t_keep
        self.N = int(init["mean"].shape[0])
        self._gpos = np.zeros(self.N, dtype=np.float64); self._gacc = 0
        return self.N
