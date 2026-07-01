/* pose3d.js — three.js scene for /pose: every camera as an SO(3) "flag" (x=red, y=green, z=blue triad +
 * frustum) against the gaussian point cloud. For the focused camera: the ORIGINAL COLMAP flag (dim), the
 * CURRENT correction (bright), and the full HISTORY trajectory between them — scrubbable by SCROLL WHEEL.
 * A TransformControls gizmo on the focused camera emits pose_nudge (camera-local ω / t increments).
 *
 * Globals exposed to pose.html: initPose3D(), pose3dPoll(), scrubTo(i), setGizmo(mode).
 * Depends on: THREE + OrbitControls + TransformControls (vendor), base.js poseNudgeRaw/postConfig.
 * Backend: GET /cameras {cameras:[{name,R[9],t[3],fx,fy,cx,cy}], pose:{deltas,focus,step}}, /pointcloud, /pose/history.
 */
(function () {
  const THREE = window.THREE;
  if (!THREE) { console.warn('pose3d: three.js not loaded'); return; }

  let scene, cam, renderer, controls, tctl, gizmoObj;
  let camGroup, focusGroup, cloud = null, cloudStep = -1;
  let baseCams = [], byName = {}, deltas = {}, focusName = null;
  let traj = [], scrubIdx = -1, sceneScale = 1, fitted = false, gizmoMode = 'rotate', gizmoBusy = false;

  // ── flat-9 (row-major) 3×3 + vec helpers ───────────────────────────────────
  const mm = (A, B) => { const C = new Array(9); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { let s = 0; for (let k = 0; k < 3; k++) s += A[i * 3 + k] * B[k * 3 + j]; C[i * 3 + j] = s; } return C; };
  const tpose = A => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
  const mv = (A, v) => [A[0] * v[0] + A[1] * v[1] + A[2] * v[2], A[3] * v[0] + A[4] * v[1] + A[5] * v[2], A[6] * v[0] + A[7] * v[1] + A[8] * v[2]];
  function so3exp(w) {                              // Rodrigues -> 3×3 row-major
    const th = Math.hypot(w[0], w[1], w[2]); if (th < 1e-9) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const x = w[0] / th, y = w[1] / th, z = w[2] / th, c = Math.cos(th), s = Math.sin(th), C = 1 - c;
    return [c + x * x * C, x * y * C - z * s, x * z * C + y * s,
            y * x * C + z * s, c + y * y * C, y * z * C - x * s,
            z * x * C - y * s, z * y * C + x * s, c + z * z * C];
  }
  // corrected extrinsic (mirror pose_opt.corrected_cam): R'=Exp(ω)R0, t'=Exp(ω)t0 + tcorr
  function corrected(R0, t0, d) { const Rd = so3exp(d.slice(0, 3)); return { R: mm(Rd, R0), t: mv(Rd, t0).map((v, i) => v + d[3 + i]) }; }
  // camera center in world = -Rᵀt ; world axes = columns of Rᵀ
  function world(R, t) { const Rt = tpose(R), c = mv(Rt, t).map(v => -v); return { c, ax: [[Rt[0], Rt[3], Rt[6]], [Rt[1], Rt[4], Rt[7]], [Rt[2], Rt[5], Rt[8]]] }; }

  const AXC = [0xff5555, 0x55ff55, 0x5599ff];      // x=red y=green z=blue (the SO(3) flag)

  function _line(a, b, color, op) {
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]);
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: op < 1, opacity: op }));
  }
  // a "flag" = triad (3 colored axes) + frustum, at a corrected pose
  function buildFlag(R, t, intr, opt) {
    const op = opt.op == null ? 1 : opt.op, L = opt.len || sceneScale * 0.12, grp = new THREE.Group();
    const { c, ax } = world(R, t);
    for (let i = 0; i < 3; i++) grp.add(_line(c, [c[0] + ax[i][0] * L, c[1] + ax[i][1] * L, c[2] + ax[i][2] * L], opt.mono != null ? opt.mono : AXC[i], op));
    // frustum (approx W=2cx,H=2cy): 4 corner rays + far rectangle
    const D = L * 1.4, Rt = tpose(R), W = 2 * intr.cx, H = 2 * intr.cy, corn = [];
    for (const [px, py] of [[0, 0], [W, 0], [W, H], [0, H]]) {
      const dc = [(px - intr.cx) / intr.fx * D, (py - intr.cy) / intr.fy * D, D];
      const wc = mv(Rt, dc); corn.push([c[0] + wc[0], c[1] + wc[1], c[2] + wc[2]]);
    }
    const fc = opt.mono != null ? opt.mono : (opt.frust || 0x66788a);
    for (let i = 0; i < 4; i++) { grp.add(_line(c, corn[i], fc, op * 0.7)); grp.add(_line(corn[i], corn[(i + 1) % 4], fc, op * 0.7)); }
    return grp;
  }

  function initPose3D() {
    const canvas = document.getElementById('scene-canvas'); if (!canvas) return;
    scene = new THREE.Scene(); scene.background = new THREE.Color(0x07090d);
    cam = new THREE.PerspectiveCamera(50, 1, 0.001, 5000); cam.position.set(0, 0, 5);
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    controls = new THREE.OrbitControls(cam, renderer.domElement);
    controls.enableZoom = false;                   // scroll is reserved for the flag-history scrubber
    controls.enableDamping = true; controls.dampingFactor = 0.1;
    tctl = new THREE.TransformControls(cam, renderer.domElement);
    tctl.setMode('rotate'); tctl.addEventListener('dragging-changed', e => { controls.enabled = !e.value; });
    tctl.addEventListener('objectChange', _onGizmo);
    scene.add(tctl);
    camGroup = new THREE.Group(); focusGroup = new THREE.Group(); scene.add(camGroup); scene.add(focusGroup);
    scene.add(new THREE.AxesHelper(0.5));           // world origin reference
    canvas.addEventListener('wheel', e => {         // SCROLL = scrub the flag history (up = forward in time)
      if (!traj.length) return; e.preventDefault();
      const cur = scrubIdx < 0 ? traj.length - 1 : scrubIdx;
      scrubTo(cur + (e.deltaY > 0 ? -1 : 1));
    }, { passive: false });
    _resize(); window.addEventListener('resize', _resize);
    (function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, cam); })();
  }

  function _resize() {
    const w = renderer.domElement.clientWidth, h = renderer.domElement.clientHeight; if (!w || !h) return;
    renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix();
  }

  async function pose3dPoll() {
    try {
      const cr = await fetch('/cameras'); if (cr.ok) _applyCameras(await cr.json());
      if (cloudStep < 0 || (Math.random() < 0.15)) { const pr = await fetch('/pointcloud'); if (pr.ok) _applyCloud(await pr.json()); }
    } catch (e) { /* training not up yet */ }
  }

  function _applyCameras(data) {
    const cams = data.cameras || [], pose = data.pose || {};
    if (cams.length && !baseCams.length) {           // build static base flags once
      baseCams = cams; byName = {}; cams.forEach((c, i) => byName[c.name] = i);
      const cs = cams.map(c => world(c.R, c.t).c);
      sceneScale = Math.max(0.5, _spread(cs));
      cams.forEach(c => { const f = buildFlag(c.R, c.t, c, { op: 0.32, mono: 0x44505c }); f.userData.name = c.name; camGroup.add(f); });
      if (!fitted) { _fit(cs); fitted = true; }
    }
    (pose.deltas || []).forEach((d, i) => { if (baseCams[i]) deltas[baseCams[i].name] = d; });
    const nf = pose.focus || (cams[0] && cams[0].name);
    if (nf && nf !== focusName) { focusName = nf; _loadTraj(focusName); }
    _rebuildFocus();
  }

  function _applyCloud(pc) {
    if (!pc || !pc.means || pc.step === cloudStep) return; cloudStep = pc.step;
    const n = pc.means.length, pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { pos[i * 3] = pc.means[i][0]; pos[i * 3 + 1] = pc.means[i][1]; pos[i * 3 + 2] = pc.means[i][2]; const c = pc.colors[i] || [1, 1, 1]; col[i * 3] = c[0]; col[i * 3 + 1] = c[1]; col[i * 3 + 2] = c[2]; }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    if (cloud) { cloud.geometry.dispose(); cloud.geometry = g; }
    else { cloud = new THREE.Points(g, new THREE.PointsMaterial({ size: Math.max(0.004, sceneScale * 0.004), vertexColors: true, sizeAttenuation: true })); scene.add(cloud); }
  }

  async function _loadTraj(name) {
    traj = []; scrubIdx = -1;
    try { const r = await fetch('/pose/history?camera=' + encodeURIComponent(name)); if (r.ok) traj = await r.json(); } catch (e) { }
    const sl = document.getElementById('scrub'); if (sl) { sl.max = Math.max(0, traj.length - 1); sl.value = sl.max; }
    _updateScrubLbl();
  }

  // (re)draw the focused camera's original / current / history flags + the gizmo
  function _rebuildFocus() {
    while (focusGroup.children.length) focusGroup.remove(focusGroup.children[0]);
    if (!focusName || byName[focusName] == null) return;
    const base = baseCams[byName[focusName]], d = deltas[focusName] || [0, 0, 0, 0, 0, 0];
    // dim original (COLMAP) flag
    focusGroup.add(buildFlag(base.R, base.t, base, { op: 0.45, mono: 0x8a8a8a, len: sceneScale * 0.16 }));
    // history trajectory: faint ghost triads from original -> current
    for (let i = 0; i < traj.length; i += Math.max(1, Math.floor(traj.length / 24))) {
      const cc = corrected(base.R, base.t, traj[i].d); focusGroup.add(buildFlag(cc.R, cc.t, base, { op: 0.16, len: sceneScale * 0.13 }));
    }
    // the scrubbed / current flag (bright)
    const showD = (scrubIdx >= 0 && traj[scrubIdx]) ? traj[scrubIdx].d : d;
    const cur = corrected(base.R, base.t, showD);
    focusGroup.add(buildFlag(cur.R, cur.t, base, { op: 1, len: sceneScale * 0.2 }));
    _placeGizmo(cur);
  }

  function _placeGizmo(cur) {
    if (gizmoMode === 'off') { if (gizmoObj) tctl.detach(); return; }
    const { c } = world(cur.R, cur.t);
    if (!gizmoObj) { gizmoObj = new THREE.Object3D(); scene.add(gizmoObj); }
    gizmoObj.position.set(c[0], c[1], c[2]);
    const Rt = tpose(cur.R), m = new THREE.Matrix4().set(Rt[0], Rt[1], Rt[2], 0, Rt[3], Rt[4], Rt[5], 0, Rt[6], Rt[7], Rt[8], 0, 0, 0, 0, 1);
    gizmoObj.quaternion.setFromRotationMatrix(m); gizmoObj.updateMatrixWorld();
    gizmoObj.userData.q0 = gizmoObj.quaternion.clone(); gizmoObj.userData.p0 = gizmoObj.position.clone();
    tctl.attach(gizmoObj); tctl.setMode(gizmoMode);
  }

  // gizmo drag -> camera-local ω / t increment -> pose_nudge (approx; pad/drive remain authoritative)
  function _onGizmo() {
    if (!gizmoObj || gizmoBusy) return; gizmoBusy = true;
    const q0 = gizmoObj.userData.q0, p0 = gizmoObj.userData.p0;
    if (gizmoMode === 'rotate' && q0) {
      const dq = gizmoObj.quaternion.clone().multiply(q0.clone().invert());
      const ang = 2 * Math.acos(Math.min(1, Math.abs(dq.w))), s = Math.sqrt(1 - dq.w * dq.w);
      if (ang > 1e-4 && s > 1e-6) { const ax = [dq.x / s, dq.y / s, dq.z / s]; if (typeof poseNudgeRaw === 'function') poseNudgeRaw(ax.map(v => v * ang), [0, 0, 0]); }
      gizmoObj.userData.q0 = gizmoObj.quaternion.clone();
    } else if (gizmoMode === 'translate' && p0) {
      const dp = gizmoObj.position.clone().sub(p0); if (dp.length() > 1e-5 && typeof poseNudgeRaw === 'function') poseNudgeRaw([0, 0, 0], [dp.x, dp.y, dp.z]);
      gizmoObj.userData.p0 = gizmoObj.position.clone();
    }
    gizmoBusy = false;
  }

  window.scrubTo = function (i) {
    if (!traj.length) return;
    scrubIdx = (i >= traj.length - 1) ? -1 : Math.max(0, i);   // last = "live"
    const sl = document.getElementById('scrub'); if (sl) sl.value = scrubIdx < 0 ? traj.length - 1 : scrubIdx;
    _updateScrubLbl(); _rebuildFocus();
  };
  function _updateScrubLbl() {
    const el = document.getElementById('scrub-lbl'); if (!el) return;
    if (!traj.length) { el.textContent = 'no history'; return; }
    if (scrubIdx < 0) { const d = (deltas[focusName] || [0, 0, 0]); el.textContent = `live · ω${_deg(d)}°`; }
    else { const t = traj[scrubIdx]; el.textContent = `@${t.step} · ω${_deg(t.d)}°`; }
  }
  const _deg = d => (Math.hypot(d[0], d[1], d[2]) * 180 / Math.PI).toFixed(2);

  window.setGizmo = function (mode) {
    gizmoMode = mode;
    ['rot', 'trn', 'off'].forEach(k => { const e = document.getElementById('giz-' + k); if (e) e.classList.toggle('sel', mode === ({ rot: 'rotate', trn: 'translate', off: 'off' })[k]); });
    if (mode === 'off') { tctl.detach(); } else if (typeof postConfig === 'function') { postConfig({ pose_opt: 1 }); _rebuildFocus(); }
  };

  function _spread(pts) { if (pts.length < 2) return 1; let mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9]; pts.forEach(p => { for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], p[k]); mx[k] = Math.max(mx[k], p[k]); } }); return Math.hypot(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) || 1; }
  function _fit(pts) {
    if (!pts.length) return; const ctr = [0, 0, 0]; pts.forEach(p => { ctr[0] += p[0]; ctr[1] += p[1]; ctr[2] += p[2]; }); ctr.forEach((_, k) => ctr[k] /= pts.length);
    controls.target.set(ctr[0], ctr[1], ctr[2]); const r = sceneScale * 1.6; cam.position.set(ctr[0] + r, ctr[1] + r * 0.5, ctr[2] + r); cam.updateProjectionMatrix();
  }

  window.initPose3D = initPose3D; window.pose3dPoll = pose3dPoll;
})();
