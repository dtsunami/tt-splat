# The Splatting Game — Engine Plan 🫧🔫

> A real, async, crowdsourced game engine for Gaussian-splat reconstruction.
> **Thesis: decouple two kinds of compute.** A human can spot a floater, a missing
> wall, or a patch of fog in a single photo in *milliseconds* — the irregular
> **spatial reasoning** half of 3DGS. The Blackhole grinds the **dense optimization**
> half (Adam against photometric loss) without ever stalling on a human. Wire them
> together so neither waits on the other: the human *steers* in 3D at 60fps, the
> device *settles* at its own train rate, and an async edit queue is the only thing
> between them.

This is the engineering plan behind the **Moonshot — the Splatting Game (GWAP)**
section of [`ROADMAP.md`](./ROADMAP.md) (the "what it would actually take" backlog
at `ROADMAP.md:136`). The headline finding: **the async edit queue, the
authoritative trainer, and the param-streaming primitive already exist in the
codebase.** The game is a *re-wiring of three existing pieces* plus exactly one
net-new client renderer — not new infrastructure.

The single biggest lift — and the one piece that turns "edit-and-wait" into "a
game" — is the **live in-browser splat renderer fed by a param stream**. Prototype
it first.

---

## Architecture (render-rate and train-rate are decoupled)

```
                          edits (async, lossless, ordered)
   ┌───────────────────────────┐  POST /command        ┌──────────────┐
   │  PLAYER BROWSER            │ ────────────────────► │   FastAPI    │
   │  • WebGL/WebGPU splat      │  splat_spawn /        │ dashboard.py │
   │    renderer  (60fps)       │  cull_region          └──────┬───────┘
   │  • free orbit + aim        │                              │ queue_command
   │  • optimistic local edit   │                       ┌──────▼───────┐
   │                            │                       │  EDIT QUEUE  │
   │                            │                       │ queue.Queue  │
   └───────────▲────────────────┘                       └──────┬───────┘
               │                                                │ drain_commands
               │  param stream                                  │ (once per step)
               │  (means,scales,quats,                   ┌──────▼────────────────┐
               │   opacities,sh_dc) +                    │  BLACKHOLE DEVICE-     │
               │  deltas/keyframes                       │  RESIDENT TRAINER      │
               │  ~2–4 Hz, stale-tolerant                │  = AUTHORITATIVE PARAMS│
               └─────────────────────────────────────────┤  Adam vs photo loss    │
                  WS /ws/splat  (or GET /splat.bin)       │  never leaves the card │
                                                          └────────────────────────┘

   RENDER CLOCK  ≠  TRAIN CLOCK
   client GPU @ 60fps (orbit/aim, never touches device)
   device @ its own train rate; snapshots gated ~2–4 Hz (params_host() is heavy)
```

Two intentionally-unsynchronized clocks. The client renders and aims off its **local**
splat buffer; the device trains and reconciles off its **authoritative** copy; the
edit queue and the param stream are the only coupling, both async and lossless.

---

## Track A — Async Architecture (it's already async; extend it)

**Verdict: it can be async, and most of it already is.** The tt-splat dashboard was
built around a thread-safe command queue that the training loop drains once per step.
That queue is exactly the async edit channel a game needs. The work is not *adding*
concurrency — it is **changing what flows back to the client** (params, not PNGs) and
**adding a client-side predictor** on top of the existing reconcile-by-training loop.

### Roles: authoritative trainer, async queue, optimistic client

- **Authoritative state = the trainer.** In the device-resident path the Gaussian
  params + Adam m/v live on the Blackhole and never leave the card in the inner loop
  (`server/device_resident.py:54-79`); the only readback is the explicit, out-of-loop
  `params_host()` (`server/device_resident.py:196-200`). There is a single writable
  copy of the scene, so there is nothing to desync against. Every edit mutates *that*
  copy: `spawn` (`device_resident.py:259-271`), `cull` (`device_resident.py:247-257`),
  `prune` (`device_resident.py:210-220`), each rebuilding the resident `DeviceAdam`.
- **Async channel = `TrainingController` command queue.** `queue_command`
  (`ttgs/viewer/dashboard.py:88-89`) wraps a `queue.Queue`. FastAPI `POST /command`
  (`dashboard.py:613-616`) writes it from the uvicorn server thread; the training loop
  drains it from its own thread via `drain_commands` (`dashboard.py:109-116`). Producer
  and consumer never block each other — this is already a decoupled, concurrent,
  lossless edit log.
- **Client = optimistic predictor (the new piece).** Today the client is a thin
  viewer: it POSTs an edit and waits for the next server frame. For a game it must
  apply the edit to its *local* splat set immediately, then let the authoritative
  param stream correct it.

### How a human edit and a training step interleave without conflict

The interleaving is already correct and needs no locking. Per step the loop does:

```python
for step in range(...):                       # server/train_tt.py:294
    for cmd in dashboard.drain_commands():     # :299  — ALL edits applied first
        ... splat_spawn / cull_region / prune / densify_now / set_lr ...   # :325-341
    loss, img = trainer.step(cam, gt, mask)    # :352  — THEN one optimize step
```

An edit is **atomic with respect to a step**: it lands entirely in the gap between
step *N* and step *N+1*, never mid-gradient. The `queue.Queue` serializes concurrent
editors into one ordered stream, so even simultaneous bubble + cull from two players
apply in a defined order.

**Training "out-votes" bad edits over time.** Right after an edit, the next
`trainer.step` runs Adam against the photometric loss (`device_resident.py:179`,
`loss` at `:143`). A human who sprays splats into empty space or erases real geometry
creates a transient error that the gradient immediately starts correcting — the
optimizer pulls the scene back toward the captured photos. The human perturbs; the
trainer reconciles. **This is the core game loop: the human *steers*, the Blackhole
*settles*.** (It is also the anti-griefing backstop named in `ROADMAP.md:144`.)

### What must change vs. today

**Today:** the server renders the frame and ships a PNG. `build_update` base64-encodes
`render_b64` / `gt_b64` / `diff_b64` (`dashboard.py:249-251`, `_to_b64` at `:207-211`);
the browser just blits `data:image/png` (`base.js:376`, `training.html:376`). The frame
rate is bolted to the train rate (`dashboard_every`, `train_tt.py:295`) and the camera
is fixed to whatever view the trainer happened to pick
(`vi = (step-1) % len(views)`, `train_tt.py:344`).

**For a game, stream params, not pixels.** Push the 5 raw arrays
`(means, scales, quats, opacities, sh_dc)` to a **client-side GPU renderer**
(WebGL/WASM gsplat) that draws at its own framerate from a **free, client-owned
camera**. This decouples render rate from train rate and gives the player a smooth
fly-around even while the trainer grinds.

This primitive **already exists in the repo**: `LiveViewer.update`
(`ttgs/viewer/__init__.py:56-150`) accepts exactly those 5 arrays and streams them
over viser's WebSocket, already sanitizing NaNs and capping by opacity to bound the
message (`__init__.py:90-101`). The game just needs:

- a periodic param push from the loop (where `push_update` is called today:
  `train_tt.py:364` / `:484`) — replace/augment the PNG `build_update` with a params
  snapshot from `trainer.params_host()`;
- delta encoding for steady state (most steps change few splats) with occasional full
  keyframes, since the full set grows toward the scale-campaign millions;
- edits stay events on the existing `/command` path — `splat_spawn` / `cull_region`
  already carry screen-space points + brush (`_CommandPayload`, `dashboard.py:263-272`;
  emitted by `_splatFire`, `base.js:482-489`).

### Latency / consistency model

**Eventual consistency, trainer-authoritative.** Two async clocks: the train clock
(each `trainer.step`) and the render clock (client GPU). They are intentionally
unsynchronized.

- **Client predicts.** On a brush stroke the client mutates its local splats now (add
  billboards / hide erased ones) for zero-perceived-latency — instead of today's
  wait-for-next-`/state`-PNG round trip (`base.js:482-489` → 500ms
  `setInterval(tick, 500)` at `base.js:416`).
- **Server reconciles.** The same stroke is POSTed as a command, drained at the next
  step boundary, applied to authoritative params, and reflected in the next param push.
  The client snaps its local prediction to the authoritative stream when it arrives.
- **Convergence.** Because the trainer is the only writer and every step nudges toward
  the photos, divergence is bounded and self-healing: a dropped command just means that
  edit didn't happen (queue is lossless, so this is rare); a mis-predicted client splat
  gets overwritten by the next stream frame. No global lock, no transaction — just
  last-writer-wins where the trainer is always the last writer.

### Multiplayer: N async editors, one authoritative trainer

This falls out almost for free. `queue.Queue` is multi-producer-safe, so N browsers
POSTing `/command` (`dashboard.py:613`) already serialize into one ordered edit stream
that the single trainer drains — **N async editors, one authoritative trainer, zero
added locking** on the edit path. The pieces to add are above the core model:

- **Fan-out:** broadcast the param stream to all N clients (the viser/WebSocket layer
  in `ttgs/viewer/__init__.py` is single-room today; promote to a broadcast hub).
- **Identity/echo:** tag commands with an editor id so each client can match its
  optimistic prediction to its own authoritative echo (and ignore/merge others').
- **Presence (optional):** stream each editor's free camera so players see each other's
  viewpoints.

The authoritative-trainer + serialized-edit-queue design means there is no consensus
problem to solve: conflicts are resolved by the single trainer applying edits in queue
order and the optimizer reconciling the result.

### Summary schema

| Concern | Today | Game extension | Already exists? |
|---|---|---|---|
| Edit channel | `queue_command`/`drain_commands` | unchanged | ✅ `dashboard.py:88,109` |
| Authoritative state | resident trainer | unchanged | ✅ `device_resident.py:54` |
| Edit→step interleave | drain-then-step | unchanged | ✅ `train_tt.py:299,352` |
| Reconcile bad edits | Adam vs. photo loss | unchanged | ✅ `device_resident.py:179` |
| Client display | server PNG push (`render_b64`) | **param stream → client renderer** | partial: `LiveViewer.update` `__init__.py:56` |
| Render vs. train rate | coupled (`dashboard_every`) | **decoupled (client clock)** | ⬜ new client renderer |
| Client edits | round-trip, wait for PNG | **optimistic predict + reconcile** | ⬜ new client logic |
| Multiplayer | single viewer | **fan-out + editor id** | partial: queue is multi-writer-safe |

---

## Track B — Real Game Engine (live WebGL/WebGPU splat renderer)

**Verdict: feasible and mostly additive.** Keep the resident TT trainer untouched, add
a `/splat.bin` streaming endpoint + a thin client renderer. Recommend **PlayCanvas
Engine** (WebGPU compute radix sort, MIT) for the scale target, with
**antimatter15/splat** as the 1-day MVP.

### Where we are today (grounded)

The "viewer" is a polled PNG. `training.html` runs `setInterval(tick, 500)` (line 416),
calls `pollState()` (`base.js:6` → `GET /state`), and assigns the server-rendered frame
into an `<img>` via `setImg('render-img', ..., d.render_b64)` (`training.html:376`). That
`render_b64` is a host-side PIL PNG produced by `_to_b64` (`dashboard.py:207`). **There is
no client-side 3D and no endpoint that hands splat params to the browser** — `splat.ply`
is only written to disk by `_write_ply` (`train_tt.py:94`) for checkpoint/export
(`pipeline_controller.py:592`). So the game engine is net-new client rendering plus
exactly one new streaming endpoint; the FastAPI app in `dashboard.py` and the resident
trainer stay otherwise untouched.

The params are already canonical 3DGS. `_write_ply` emits `x,y,z`, zeroed `nx,ny,nz`,
`f_dc_0..2`, channel-major `f_rest_*`, `opacity` (logit), `scale_0..2` (log), `rot_0..3`
(wxyz, normalized). On device they live as `P{mean,scale,quat,op,sh[N,K,3]}` and read
back through `trainer.params_host()` (`device_resident.py:196`), explicitly flagged "NOT
per inner step" — a full device→host copy. **That single fact drives the whole streaming
cadence.**

### Recommended renderer

**Primary target: PlayCanvas Engine (MIT).** Its 2.19+ build ships a compute-based
**WebGPU** renderer that culls, projects, and **sorts splats with a GPU radix sort every
frame**, keeps selection over 5M Gaussians real-time, streams a LoD/SOG format past 10M,
and auto-falls back to WebGL2 (~85–98% reach). For the scale campaign (see
[`SCALE_CAMPAIGN_PLAN.md`](./SCALE_CAMPAIGN_PLAN.md) — full-res + millions) the on-GPU
sort is decisive: a live param swap becomes a GPU upload, not a CPU stall.

**Alternative if we want three.js: Spark (`sparkjsdev/spark`, World Labs).** three.js-
native, 98% WebGL2, multi-splat scenes, real-time editing/relighting, a shader-graph for
dynamic effects, and Spark 2.0 LoD streaming (`.RAD`, virtual splat paging, fixed GPU
budget) — a near-perfect fit if mesh+splat fusion or effects are wanted.

**MVP floor: antimatter15/splat (MIT, WebGL1, zero deps).** CPU web-worker counting-sort
(~150ms @ 1M), progressive load, no SH. `gsplat.js` (MIT, built on antimatter15) adds
light editing. `mkkellogg/GaussianSplats3D` is CPU-sort and **no longer actively
developed** — avoid as a base.

| Renderer | Backend | Sort | Streaming/partial | Editing | N=100k–1M | License |
|---|---|---|---|---|---|---|
| antimatter15/splat | WebGL1 | CPU worker | progressive load, full-buffer swap | none (we add) | ~150ms sort @1M | MIT |
| gsplat.js | WebGL | CPU worker | progressive | basic | similar | MIT |
| mkkellogg GaussianSplats3D | WebGL/three | CPU | progressive (.ksplat) | none | CPU-bound; **inactive** | MIT |
| **PlayCanvas Engine** | **WebGPU**(+WebGL2 fb) | **GPU radix** | **SOG/LoD >10M** | **GPU brush/lasso/sphere** | real-time | **MIT** |
| Spark (three.js) | WebGL2 | GPU-assisted | **.RAD LoD paging** | **real-time + relight** | real-time | open source |

### Param streaming (resident → client)

Serve a **packed binary, never PNG**. Build it from `trainer.params_host()`.

- **MVP tier — `GET /splat.bin`**: the antimatter15 **32-byte/splat** layout: position
  `f32×3`, scale `f32×3` (`exp(scale_log)`), `RGBA u8×4` (R/G/B = `0.5 + C0·f_dc`,
  A = `sigmoid(op)`), rotation as `u8×4` quantized wxyz quaternion. SH band-0 only on the
  wire (matches every WebGL viewer). Add an `ETag`/`?step=` so the client skips unchanged
  snapshots.
- **Scale tier — `WS /ws/splat`**: push **deltas**, not full N. The trainer already
  produces structured edits — `spawn` appends, `cull`/`prune` remove
  (`device_resident.py`) — so a frame is `{appended:[...32B], removed:[idx...]}`. A
  5k-splat bubble-gun spray is a few KB, not an N re-upload. Quantize means to f16 /
  adopt SOG for the millions tier; add `f_rest` SH as an optional second buffer for
  view-dependent color.

**Sort-on-GPU is the central concern.** WebGL bases re-sort on CPU on every buffer change
(100–150ms @1M), which both re-uploads N and stalls — fine at MVP N (current
`HOST_MAX_POINTS` / a few 100k) but a hard cap on live-update rate. PlayCanvas/Spark sort
on-GPU per frame, so a param swap is just an upload — the reason they are the millions-
tier target.

### Edit → server channel (reuse, don't rebuild)

The channel is already camera-driven and un-projects server-side. The client POSTs
`/command` (`base.js:39`) with
`{type:'splat_spawn'|'cull_region', camera_name, points:[[px,py]…], brush, n_per}`
(`training.html:482`). Server: `splat_spawn` (`train_tt.py:325`) projects splats to that
camera (`_project_uvz`), `_unproject_spawn` sets depth = median camera-depth of nearby
splats (else scene median) and color = GT pixel; `cull_region` masks via `_select_region`.
With a real client camera there are two send modes — **(i)** screen points + the **nearest
dataset `camera_name`** (works today, zero server change — the un-projection stays
server-side, the safe path), or **(ii)** a true 3D world point/ray via a tiny new
`world_spawn`/`world_cull` command that skips un-projection. **Ship (i) first.**

### Render-rate vs train-rate decoupling

The client renders its local splat buffer at **60fps**; orbit/aim never touch the device.
The server pushes a fresh snapshot only every K steps / min-interval (reuse
`dashboard_every=5` and the existing `drain_commands` loop). Because `params_host()` is a
heavy readback, **gate snapshots to ~2–4 Hz** and treat them as stale-tolerant. Edits ride
the existing `queue.Queue` (`dashboard.py:66`), apply on the next train step, and surface
in the next snapshot; the client shows an **optimistic local preview** (spawn the brush
splats client-side immediately) so the game feels instant.

---

## Track C — Interactive COLMAP / SfM (human-in-the-loop)

**Verdict: feasible, and partly already shipping.** Masks + exclusion already ARE human
SfM inputs and the bubble gun already seeds structure; **2D-2D / 2D-3D correspondence-
marking is the next tractable piece** — a thin pycolmap wrapper, not a big lift. The same
per-frame edit surface that drives training (`image_edit.html` + `pipeline_controller.py`)
already feeds COLMAP the two canonical human-SfM inputs (feature masks, image exclusion),
and the bubble gun already lets a human hand-place 3D structure.

### What already exists (these ARE human SfM)

- **Dynamic-object masking** (mask out the corgi's tongue): the human paints polygons in
  `image_edit.html`; `pipeline_controller.save_mask_data` (`pipeline_controller.py:231`)
  rasterises a weight PNG, and `_stage_sfm` passes `masks_dir` to `sfm.run`
  (`pipeline_controller.py:546-552`), which becomes `--ImageReader.mask_path`
  (`sfm.py:138-139`) / `reader.mask_path` (`sfm.py:258-259`). This is exactly COLMAP's
  "black = no features in dynamic regions" workflow.
- **Excluding blurry frames**: `set_exclusion` (`pipeline_controller.py:215`) +
  `_prepare_sfm_images` (`pipeline_controller.py:599-626`) physically removes excluded
  frames so COLMAP never sees them — the human deciding which views enter SfM.
- **The keypoint overlay (read-only correspondence view)**: `get_colmap_features` parses
  `images.bin` and flags each 2D observation triangulated (1) or not (0)
  (`pipeline_controller.py:408-457`), served at `/images/{name}/colmap`
  (`dashboard.py:494-499`) and drawn as green/red dots on `colmap-canvas`
  (`image_edit.html:365-406`). The human already *sees* where SfM succeeded vs failed per
  pixel.

### The bubble gun is already SfM-seeding

`_unproject_spawn` (`train_tt.py:180-203`) converts a screen click into a **new world-
space Gaussian**: depth = median camera-depth of nearby splats (local surface) or
scene-median depth — its own comment (line 183) names this *"empty region -> SfM-seeding"*
— and colour is sampled from the GT photo at that pixel. A human clicking empty space to
add 3D structure with photo-derived colour is structurally `triangulate_point` +
`add_point3D`, only without the multi-view ray constraint. The eraser (`_select_region` +
`cull`, `train_tt.py:206-214, 334-341`) is the inverse: `delete_point3D` / `deregister`.

### New tasks and their pycolmap feasibility

All verified present in pycolmap 4.0.4 in `~/tt-metal/python_env` (the venv `ttgs sfm`
already uses):

| Game task | pycolmap primitive | Lift |
|---|---|---|
| Mask movers / exclude blurry | `mask_path` / `_prepare_sfm_images` | **Done** |
| Anchor a failed pose (mark 2D↔3D) | `estimate_and_refine_absolute_pose(points2D, points3D, camera)` | **Thin** — one call |
| Register a dropped frame (mark 2D↔2D) | `Database.write_two_view_geometry` → `IncrementalMapper.register_next_image` → `triangulate_image` → `adjust_local_bundle` | **Medium** — the documented `custom_incremental_pipeline.py` idiom |
| Hand-place a 3D point that persists | `Reconstruction.add_point3D` / `add_observation` (+ `triangulate_point`) | **Thin** |
| Flag / fix drift | `deregister_frame` + re-`register_next_image`, or scoped `bundle_adjustment` | **Medium** |

The Python interface deliberately exposes the `IncrementalMapper` core directly
(bypassing the C++ `IncrementalPipeline` controller), which is precisely what a human-
driven, one-image-at-a-time loop needs.

### SfM and densification are one verb

Both reduce to *"human places or fixes 3D structure by clicking on an image."*
`image_edit.html` is already that canvas (mask + COLMAP overlay layers). The mapping is
one-to-one: bubble-gun spawn ↔ `add_point3D`; eraser ↔
`delete_point3D`/`deregister_frame`; the mask weight-PNG is consumed by **both** COLMAP
(feature suppression) and the training loop (loss weighting) — one artifact, two stages.
Promoting the bubble gun to also write into `Reconstruction.points3D` (not just the live
Gaussian tensor) would make hand-placed structure survive re-train/re-export and literally
unify the two stores.

### Camera-pose optimization (the ghost-buster) — the fourth structure type

The human doesn't only place *gaussians* and *points* — they can fix the **cameras**. A
mis-registered COLMAP pose shows up in the render as **ghosting / double-images** (the same
geometry painted twice, slightly offset), and a human reads that in milliseconds. So camera
pose is a first-class human target, and it has two halves:

- **Automatic (port gsplat's `pose_opt`).** gsplat ships joint camera-extrinsic optimization
  (`pose_opt`, `pose_opt_lr≈1e-5`, reg `≈1e-6`) precisely to correct COLMAP pose error —
  and tt-splat is in the **"absent"** column for it today (poses are fixed from COLMAP at
  `train_tt._load_colmap`). But tt-splat is **unusually well-positioned** to add it: the
  analytic **projection backward already exists** (`server/device_project_backward.py`,
  `project_backward`) — that Jacobian `∂(u,v)/∂(R,t)` is exactly what you differentiate to
  flow gradients into a 6-DoF pose. So pose-opt is mostly *"make each camera's R,t trainable
  params and route the projection grads to them + a small-LR Adam group,"* not a new kernel.
  Medium lift, real quality win on noisy captures.
- **Interactive (grab the camera).** In the live splat view the player **grabs a ghosting
  camera's frustum and nudges it** until the doubles snap into focus, or flags a frame
  *"this pose is wrong — re-solve it."* pycolmap closes the loop: `bundle_adjustment` /
  `estimate_and_refine_absolute_pose` (already in the feasibility table) refine or re-anchor
  the pose, and the next training steps lock it in. Same gesture as the bubble gun — *aim,
  then fix structure* — except the structure is a **camera** instead of a gaussian.

This couples to the free-view renderer (Track B): once cameras are draggable objects in the
3D scene, pose editing is just another thing you point at.

### Sources
- [PyCOLMAP — Python Interface (DeepWiki)](https://deepwiki.com/colmap/colmap/5-python-interface-(pycolmap))
- [Incremental Mapping Process (DeepWiki)](https://deepwiki.com/colmap/colmap/8.1-incremental-mapping-process)
- [COLMAP FAQ — masking dynamic objects](https://colmap.github.io/faq.html)
- [COLMAP GUI — database management / manual correspondences](https://colmap.github.io/gui.html)
- Live `dir()` probe of `pycolmap` 4.0.4 in `/home/starboy/tt-metal/python_env`

---

## The one interaction: the human places/fixes 3D structure, the trainer reconciles

Step back and every feature in all three tracks collapses into **a single interaction
model**: *aim somewhere in 3D, then add / remove / anchor structure.* The trainer
reconciles whatever the human did against the photos.

| Human verb (in-game) | What it is, mechanically | Existing handle |
|---|---|---|
| **Densify** (shoot a bubble) | add a Gaussian at an aimed point | `splat_spawn` → `spawn` (`device_resident.py:259`) |
| **Cull** (vacuum the fog) | remove Gaussians in a region | `cull_region` → `cull` (`device_resident.py:247`) |
| **SfM-seed** (fill empty space) | add world-space structure w/ photo color | `_unproject_spawn` (`train_tt.py:180`) |
| **Correspondence-mark** (anchor a failed frame) | tie a 2D pixel to a 3D point/another view | pycolmap `IncrementalMapper` (Track C) |
| **Nudge a camera** (bust a ghost) | fix a 6-DoF pose (interactive or auto pose-opt) | gsplat `pose_opt` port + `project_backward` Jacobian; pycolmap `bundle_adjustment` (Track C) |
| **Mask / exclude** (ignore movers/blur) | suppress features / drop a view | `save_mask_data`, `set_exclusion` |

All six are the same gesture — *click on an image / in the 3D view to place, erase, anchor,
or align* — and all are reconciled by the **same authoritative loop**: edits drain into
the device-resident trainer, Adam runs against photometric loss, and the result streams
back. Densification and SfM are not two systems; they are **one verb against two stores**
(the live Gaussian tensor and `Reconstruction.points3D`), which the unification in Track C
proposes to merge. The human supplies irregular spatial judgment; the Blackhole supplies
dense optimization; the queue keeps them decoupled. **That is the whole engine.**

---

## MVP path / phased build order

Build the renderer first — it is the piece that turns "edit-and-wait" into "a game"
(`ROADMAP.md:139`), and everything else already exists.

**Phase 0 — Live WebGL viewer fed by the param stream (the prototype).** *Zero changes
to the TT training loop.*
1. Add `GET /splat.bin` to `dashboard.py` (`params_host()` → 32B antimatter15 layout,
   `ETag` on step).
2. Embed antimatter15/splat's WebGL canvas in `training.html` next to the PNG; point it
   at `/splat.bin`, poll ~0.5–1s.
3. Wire its orbit camera (mouse-drag) — render at 60fps off the local buffer.
4. Reuse the **existing** splat-tool overlay: on click, raycast client-side, pick nearest
   dataset `camera_name` from `/frames.json` / `/images/{name}/colmap`, and POST the
   **unchanged** `splat_spawn` / `cull_region`. → *live, orbitable, shoot-into scene.*

**Phase 1 — Optimistic edits + decoupled clocks.** Apply brush edits to the local splat
buffer immediately (Track A §latency); snap to the authoritative `/splat.bin` snapshot
when it arrives. Gate snapshots to ~2–4 Hz; never block render on train.

**Phase 2 — Scale renderer + deltas.** Swap to PlayCanvas Engine (WebGPU GPU radix sort)
and `WS /ws/splat` delta frames (`{appended, removed}`) for the millions tier
([`SCALE_CAMPAIGN_PLAN.md`](./SCALE_CAMPAIGN_PLAN.md)). Add f16 means / SOG; optional
`f_rest` SH buffer.

**Phase 3 — Multiplayer.** Promote the viser/WebSocket layer
(`ttgs/viewer/__init__.py:56`) to a broadcast hub; tag commands with editor id for
prediction/echo matching; optional camera-presence stream. Then scoring (Δloss/ΔPSNR
attribution — already streamed per stage), leaderboards, and anti-griefing (edit budgets,
undo, N-player consensus before destructive culls; `ROADMAP.md:144`).

**Phase 4 — Interactive SfM.** Promote the bubble gun to also write
`Reconstruction.points3D`; add correspondence-marking via the pycolmap `IncrementalMapper`
surface (Track C) so failed frames can be registered by hand.

### Open questions
- **Snapshot cadence vs. `params_host()` cost.** Is a full device→host copy at 2–4 Hz
  cheap enough at the millions tier, or do we need a device-side delta export so the
  stream never pays for the full N?
- **Send mode for edits.** Ship server-side un-projection (mode i, nearest
  `camera_name`) — but when do we need true world-space `world_spawn`/`world_cull` (mode
  ii), and how do we pick depth without a reference camera?
- **Delta indexing under churn.** `cull`/`prune` renumber the resident array; how do we
  keep client-side splat indices stable across removals for delta `{removed:[idx...]}`
  frames (generation ids? stable handles?).
- **Anti-griefing threshold.** "Training out-votes a bad edit" is the backstop, but how
  many consensus pulls before a *destructive* cull lands, and does the trainer's
  reconciliation rate set that budget?
- **One store or two.** Do we merge the live Gaussian tensor and `Reconstruction.points3D`
  now (Track C unification) or keep hand-placed SfM structure separate until export?
