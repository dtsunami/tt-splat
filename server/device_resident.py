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
import sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "pathclear"))
import ttnn                                                # noqa: E402
from bin_sort import bin_and_sort                          # noqa: E402  M6 per-tile cull + depth sort
import sfpu_raster_scaled as M14                            # noqa: E402  TS, B
from render_device import _device, _resources, _raster_channel   # noqa: E402  M14 forward
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
    def __init__(self, dev, P, lr=None, deg=None):
        self.dev = dev
        self.deg = int(P["deg"] if deg is None else deg)
        init = {k: P[k].detach().clone() for k in ("mean", "scale", "quat", "op", "sh")}
        self.K = init["sh"].shape[1]
        lr = lr or {"mean": .01, "scale": .01, "quat": .01, "op": .02, "sh": .01}
        self.adam = DeviceAdam(dev, init, lr)              # params + m/v resident on device
        self.timings = {k: 0.0 for k in ("B", "bin", "raster", "loss", "A", "D", "C", "step")}
        self.nstep = 0

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

        # ===== B: device projection forward (reads resident params) =====
        _dbg("B: project_geom...")
        p = self._p
        u_t, v_t, zc_t, (ca_t, cb_t, cc_t), Ageo = project_geom(
            dev, p["mean"], p["scale"], p["quat"], cam, aux=True)
        _dbg("B: project_color...")
        cR_t, cG_t, cB_t, Acol = project_color(dev, p["mean"], p["sh"], self.deg, cam, aux=True)
        _dbg("B: project_op + 2D readback...")
        op_t = project_op(dev, p["op"])
        g = lambda t: ttnn.to_torch(t).flatten().numpy().astype(np.float64)   # 2D readback (deferred bin)
        u, v, zc = g(u_t), g(v_t), g(zc_t)
        ca, cb, cc = g(ca_t), g(cb_t), g(cc_t)
        opn = g(op_t); col = np.stack([g(cR_t), g(cG_t), g(cB_t)], axis=-1)
        N = u.shape[0]
        _sync(dev); t1 = time.perf_counter(); self.timings["B"] += t1 - t0

        # ===== bin/sort (host, deferred) + cull =====
        TS = M14.TS
        Wp, Hp = ((W + TS - 1) // TS) * TS, ((H + TS - 1) // TS) * TS
        valid = zc > 1e-4
        vidx = np.nonzero(valid)[0]
        img = np.zeros((H, W, 3), np.float64)
        grads2d = {k: np.zeros(N, np.float64) for k in _GEOM}
        col_g = [np.zeros(N, np.float64) for _ in range(3)]
        if vidx.size:
            cxv, cyv, av, bv, ccv, opv, zcv = (a[valid] for a in (u, v, ca, cb, cc, opn, zc))
            colv = [col[valid, k] for k in range(3)]
            detc = av * ccv - bv * bv; detc = np.where(np.abs(detc) < 1e-12, 1e-12, detc)
            var_x = np.clip(ccv / detc, 0.25, None); var_y = np.clip(av / detc, 0.25, None)
            res = _resources(dev, Wp, Hp)
            s_gid, _st, ranges, ntx, nty, _tot = bin_and_sort(cxv, cyv, var_x, var_y, zcv, Wp, Hp, ts=TS)
            tile_lists = [s_gid[ranges[t, 0]:ranges[t, 1]].tolist() for t in range(ntx * nty)]
            maxc = max((len(l) for l in tile_lists), default=0)
            nbatch = (maxc + M14.B - 1) // M14.B
            t2 = time.perf_counter(); self.timings["bin"] += t2 - t1
            _dbg(f"raster fwd (N={N} valid={vidx.size} nbatch={nbatch} ntx={ntx} nty={nty})...")

            # ===== raster forward (M14) =====
            chans, Tfin = [], None
            for k in range(3):
                r = _raster_channel(dev, res, tile_lists, ntx, nbatch, cxv, cyv, av, bv, ccv, opv,
                                    colv[k], Wp, Hp, want_T=(k == 0))
                Ck, Tfin = r if k == 0 else (r, Tfin)
                chans.append(Ck)
            img = np.clip(np.stack(chans, axis=-1)[:H, :W, :], 0.0, 1.0)
            _sync(dev); t3 = time.perf_counter(); self.timings["raster"] += t3 - t2

            # ===== loss + dL/dimage (host) =====
            diff = img - gt
            if mask is not None:
                mk = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
                diff = diff * mk[..., None]
            loss = float((diff ** 2).mean())
            gi = (2.0 * diff / diff.size)
            gp = np.zeros((Hp, Wp, 3), np.float64); gp[:H, :W, :] = gi
            t4 = time.perf_counter(); self.timings["loss"] += t4 - t3

            # ===== A: raster backward (grid-sharded) =====
            gv, cgv = fused_backward_grid(dev, cxv, cyv, av, bv, ccv, opv, colv,
                                          tile_lists, ntx, Hp // 32, Wp, Hp, gp, Tfin)
            for key in _GEOM:
                grads2d[key][vidx] = gv[key]
            for k in range(3):
                col_g[k][vidx] = cgv[k]
            _dbg("A done")
            _sync(dev); t5 = time.perf_counter(); self.timings["A"] += t5 - t4
        else:
            loss = float((img - gt).__pow__(2).mean())
            t5 = time.perf_counter()

        # ===== D: device projection backward (reuse B aux) =====
        up = dict(u=grads2d["cx"], v=grads2d["cy"], ca=grads2d["a"], cb=grads2d["b"], cc=grads2d["c"],
                  op=grads2d["op"], colR=col_g[0], colG=col_g[1], colB=col_g[2])
        Pd = dict(mean=p["mean"], scale=p["scale"], quat=p["quat"], sh=p["sh"], op=p["op"], deg=self.deg)
        _dbg("D: project_backward...")
        g3 = project_backward(dev, Pd, cam, up, aux=(Ageo, Acol))
        self.last_g3 = g3                                  # for the grad-equivalence gate
        _sync(dev); t6 = time.perf_counter(); self.timings["D"] += t6 - t5

        # ===== C: device Adam (updates resident params) =====
        _dbg("C: adam.step...")
        self.adam.step({k: g3[k] for k in ("mean", "scale", "quat", "op", "sh")})
        _sync(dev); t7 = time.perf_counter(); self.timings["C"] += t7 - t6
        self.timings["step"] += t7 - t0
        self.nstep += 1
        return loss, img.astype(np.float32)

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
