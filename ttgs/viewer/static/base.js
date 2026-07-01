/* ttgs shared JS — polling, commands, charts */

// ── Training state ───────────────────────────────────────────────────────────
let _trainState = null;

async function pollState() {
  try {
    const r = await fetch('/state');
    if (r.status === 204) { _trainState = null; return null; }
    if (!r.ok) throw new Error(r.status);
    _trainState = await r.json();
    return _trainState;
  } catch { _trainState = null; return null; }
}

function getTrainState() { return _trainState; }

// ── Pipeline status ──────────────────────────────────────────────────────────
let _pipelineStatus = null;

async function pollPipeline() {
  try {
    const r = await fetch('/pipeline/status');
    if (!r.ok) return null;
    _pipelineStatus = await r.json();
    return _pipelineStatus;
  } catch { return null; }
}

function updatePills() {
  if (!_pipelineStatus) return;
  for (const s of ['extract','sfm','train','export']) {
    const pill = document.querySelector(`.pill[data-stage="${s}"]`);
    if (pill) pill.className = `pill ${_pipelineStatus.stages[s].status}`;
  }
}

// ── Commands ─────────────────────────────────────────────────────────────────
async function postCommand(type, extra = {}) {
  await fetch('/command', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({type, ...extra}),
  });
}

async function togglePause() { await fetch('/pause', {method:'POST'}); }

async function runFrom(stage) {
  await fetch('/pipeline/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({from_stage: stage}),
  });
  pollPipeline().then(updatePills);
}

async function interruptPipeline() {
  await fetch('/pipeline/interrupt', {method: 'POST'});
  pollPipeline().then(updatePills);
}

async function postConfig(data) {
  try {
    const r = await fetch('/config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data),
    });
    return r.ok;
  } catch { return false; }
}

function pruneGaussians() {
  const el = document.getElementById('prune-thresh');
  const t = el ? parseFloat(el.value) : 0.005;
  postCommand('prune', {threshold: t});
}

// Interactive "grab the camera" — nudge the CURRENT camera's 6-DoF extrinsics (requires pose-opt on).
// Low-level: post an arbitrary (omega, trans) correction for the current camera. Returns false if no cam.
// omega = so(3) rotation [rx,ry,rz] (rad); trans = [tx,ty,tz] (world units). Used by the buttons AND the
// WASD/mouse drive (which sends fractional, combined nudges).
function poseNudgeRaw(omega, trans) {
  const st = (typeof getTrainState === 'function') ? getTrainState() : null;
  const cam = st && st.camera_name;
  if (!cam) return false;
  postCommand('pose_nudge', {camera_name: cam, omega: omega || [0, 0, 0], trans: trans || [0, 0, 0]});
  return true;
}

// Button helper: one axis by the pose-step magnitude. kind: 'omega'|'trans'; axis: 0|1|2; sign: ±1.
function poseNudge(kind, axis, sign) {
  const el = document.getElementById('pose-step');
  let step = el ? parseFloat(el.value) : NaN;
  if (!isFinite(step) || step <= 0) step = (kind === 'omega') ? 0.005 : 0.01;
  const omega = [0, 0, 0], trans = [0, 0, 0];
  (kind === 'omega' ? omega : trans)[axis] = sign * step;
  if (!poseNudgeRaw(omega, trans)) alert('No active camera — start training first.');
}

// ── Image helpers ────────────────────────────────────────────────────────────
function setImg(imgId, phId, b64) {
  const img = document.getElementById(imgId);
  const ph  = document.getElementById(phId);
  if (!img || !ph) return;
  img.src = 'data:image/png;base64,' + b64;
  img.style.display = 'block';
  ph.style.display  = 'none';
}

function renderTrainStats(containerId, d) {
  const el = document.getElementById(containerId);
  if (!el || !d) return;
  el.innerHTML =
    `step <b>${d.step.toLocaleString()}</b>/${d.total_steps.toLocaleString()} ` +
    `loss <b>${d.loss.toFixed(4)}</b> ` +
    `L1 ${d.l1.toFixed(4)} PSNR ${(d.psnr || 0).toFixed(2)}dB ` +
    `<b>${d.n_gaussians.toLocaleString()}</b>G ` +
    `cam: ${d.camera_name}`;
}

// ── Loss chart ───────────────────────────────────────────────────────────────
//
// drawLossChart(canvasId, history, {camera})
//
// history: [{step, loss, l1, psnr, mse, camera_name}, ...]
// Draws multi-line loss curves on a <canvas>.

const CHART_COLORS = {
  loss: '#4af',
  l1:   '#f84',
  psnr: '#a6f',
};

async function fetchHistory(camera) {
  const url = camera ? `/state/history?camera=${encodeURIComponent(camera)}` : '/state/history';
  try {
    const r = await fetch(url);
    if (!r.ok) return [];
    return await r.json();
  } catch { return []; }
}

// Stable color per camera: hash the name → hue. The same camera is always the
// same color across redraws, so a recurring loss-spike offender shows up as the
// same hue marching across the chart (the phase-lock made visible).
function _camHue(name) {
  let h = 0;
  const s = String(name || '');
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h % 360;
}
function camColor(name, a = 1) { return `hsla(${_camHue(name)},72%,62%,${a})`; }

// Exponential moving average (trend line + spike-prominence baseline).
function _ema(vals, alpha) {
  let e = null; const out = [];
  for (const v of vals) {
    if (v === undefined || v === null || !isFinite(v)) { out.push(e); continue; }
    e = (e == null) ? v : alpha * v + (1 - alpha) * e;
    out.push(e);
  }
  return out;
}

function drawLossChart(canvasId, history, opts = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !history.length) return;
  const ctx = canvas.getContext('2d');
  // Keep the backing store matched to the displayed size so the chart stays crisp
  // when its container is resized (the loss panel is user-expandable).
  if (canvas.clientWidth && canvas.clientHeight &&
      (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight)) {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
  }
  const W = canvas.width, H = canvas.height;
  const pad = {l: 52, r: 10, t: 20, b: 24};
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;

  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#0e0e0e';
  ctx.fillRect(0, 0, W, H);

  // Determine which series to show
  const series = opts.series || ['loss', 'l1'];   // psnr is dB (different scale) — shown in the stats bar, not on this axis
  const highlighted = opts.highlight || null;  // camera name to emphasize (white ring)
  const logY = opts.logY !== false;            // log scale by default — unsquashes the late-training crawl

  // Compute Y range across all visible series (positive values only for log)
  let yMin = Infinity, yMax = -Infinity;
  for (const key of series) {
    for (const h of history) {
      const v = h[key];
      if (v !== undefined && v !== null && isFinite(v) && (!logY || v > 0)) {
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
      }
    }
  }
  if (!isFinite(yMin)) return;

  let toY, gridVals;
  if (logY) {
    const lo = Math.log10(yMin), hi = Math.log10(yMax);
    const lr = Math.max(hi - lo, 1e-6);
    const lMin = lo - lr * 0.05, lMax = hi + lr * 0.05;
    const floor = Math.pow(10, lMin);
    toY = v => pad.t + (1 - (Math.log10(Math.max(v, floor)) - lMin) / (lMax - lMin)) * ch;
    gridVals = [];
    const nGrid = 4;
    for (let i = 0; i <= nGrid; i++) gridVals.push(Math.pow(10, lMin + (lMax - lMin) * (i / nGrid)));
  } else {
    const yRange = Math.max(yMax - yMin, 1e-6);
    const a = yMin - yRange * 0.05, b = yMax + yRange * 0.05;
    toY = v => pad.t + (1 - (v - a) / (b - a)) * ch;
    gridVals = [];
    const nGrid = 4;
    for (let i = 0; i <= nGrid; i++) gridVals.push(a + (b - a) * (i / nGrid));
  }

  const xMin = history[0].step;
  const xMax = history[history.length - 1].step;
  const xRange = Math.max(xMax - xMin, 1);
  const toX = s => pad.l + ((s - xMin) / xRange) * cw;

  // Grid lines + y labels (in the left gutter, off the trace)
  ctx.strokeStyle = '#222'; ctx.lineWidth = 1;
  ctx.font = '10px monospace'; ctx.fillStyle = '#666'; ctx.textAlign = 'right';
  for (const v of gridVals) {
    const y = toY(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillText(v < 0.01 ? v.toExponential(1) : v.toFixed(4), pad.l - 4, y + 3);
  }

  // X axis labels
  ctx.textAlign = 'center'; ctx.fillStyle = '#555';
  const nXLabels = 5;
  for (let i = 0; i <= nXLabels; i++) {
    const s = xMin + xRange * (i / nXLabels);
    ctx.fillText(Math.round(s).toLocaleString(), toX(s), H - 4);
  }

  // Draw each series
  for (const key of series) {
    ctx.strokeStyle = CHART_COLORS[key] || '#888';
    ctx.lineWidth = key === 'loss' ? 2 : 1;
    ctx.globalAlpha = key === 'loss' ? 1 : 0.7;
    ctx.beginPath();
    let started = false;
    for (const h of history) {
      const v = h[key];
      if (v === undefined || v === null || !isFinite(v) || (logY && v <= 0)) continue;
      const x = toX(h.step), y = toY(v);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // EMA trend overlay for loss — the smoothed signal under the spikes.
  const lossVals = history.map(h => h.loss);
  const ema = _ema(lossVals, 0.15);
  ctx.strokeStyle = '#fff'; ctx.globalAlpha = 0.28; ctx.lineWidth = 1.5;
  ctx.beginPath();
  let es = false;
  for (let i = 0; i < history.length; i++) {
    const v = ema[i];
    if (v == null || !isFinite(v) || (logY && v <= 0)) continue;
    const x = toX(history[i].step), y = toY(v);
    if (!es) { ctx.moveTo(x, y); es = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke(); ctx.globalAlpha = 1;

  // Per-camera markers: color = camera, radius = how far the point juts above its
  // local EMA (spike prominence). Spikes become big colored dots; the recurring
  // offender marches across in one hue. Highlighted camera gets a white ring.
  for (let i = 0; i < history.length; i++) {
    const h = history[i];
    if (h.loss == null || !isFinite(h.loss) || (logY && h.loss <= 0)) continue;
    const base = ema[i] || h.loss;
    const prom = base > 0 ? Math.max(0, (h.loss - base) / base) : 0;   // relative jut above trend
    const isHi = highlighted && h.camera_name === highlighted;
    const r = isHi ? 3.2 : Math.min(1.4 + prom * 9, 5);
    if (!isHi && prom < 0.02 && r < 1.6) continue;   // skip the calm baseline to cut clutter
    const x = toX(h.step), y = toY(h.loss);
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = camColor(h.camera_name, 0.95); ctx.fill();
    if (isHi) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.4; ctx.stroke(); }
  }

  // Legend
  ctx.font = '10px monospace'; ctx.textAlign = 'left';
  let lx = pad.l + 6;
  for (const key of series) {
    ctx.fillStyle = CHART_COLORS[key] || '#888';
    ctx.fillText(key, lx, pad.t + 12);
    lx += ctx.measureText(key).width + 12;
  }
  ctx.fillStyle = '#fff'; ctx.globalAlpha = 0.28;
  ctx.fillText('ema', lx, pad.t + 12); lx += ctx.measureText('ema').width + 14;
  ctx.globalAlpha = 1; ctx.fillStyle = '#567';
  ctx.fillText('· dot = camera (click → focus)', lx, pad.t + 12);

  // Interaction state + handlers (attach once)
  canvas._history = history;
  canvas._opts = {pad, cw, ch, xMin, xRange, series, toX, toY, onPick: opts.onPick};
  canvas._drawArgs = {id: canvasId, opts};
  canvas.style.cursor = 'pointer';
  if (!canvas._hasHover) {
    canvas._hasHover = true;
    canvas.addEventListener('mousemove', _chartHover);
    canvas.addEventListener('click', _chartClick);
    // Redraw at full resolution when the (resizable) container is dragged.
    if (window.ResizeObserver) {
      const ro = new ResizeObserver(() => {
        const a = canvas._drawArgs;
        if (a && canvas._history && canvas._history.length) drawLossChart(a.id, canvas._history, a.opts);
      });
      ro.observe(canvas);
    }
  }
}

// Find the history point nearest the mouse x (shared by hover + click).
function _chartNearest(canvas, e) {
  const {pad, cw, xMin, xRange} = canvas._opts;
  const history = canvas._history;
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
  const step = xMin + ((mx - pad.l) / cw) * xRange;
  let closest = null, minDist = Infinity;
  for (const h of history) {
    const d = Math.abs(h.step - step);
    if (d < minDist) { minDist = d; closest = h; }
  }
  return closest;
}

// Click a marker → focus training on that camera (render jumps to that view).
// Pages may pass opts.onPick(camera_name, point) to customize (e.g. navigate).
function _chartClick(e) {
  const canvas = e.target;
  if (!canvas._opts || !canvas._history) return;
  const p = _chartNearest(canvas, e);
  if (!p || !p.camera_name) return;
  if (canvas._opts.onPick) { canvas._opts.onPick(p.camera_name, p); return; }
  try { postCommand('focus_camera', {camera_name: p.camera_name}); } catch {}
  const tip = document.getElementById('chart-tooltip');
  if (tip) { tip.textContent = `focus → ${p.camera_name}`; tip.style.borderColor = camColor(p.camera_name); }
}

function _chartHover(e) {
  const canvas = e.target;
  const {pad, cw, xMin, xRange, series} = canvas._opts;
  const history = canvas._history;
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (canvas.width / rect.width);

  // Find nearest data point
  const step = xMin + ((mx - pad.l) / cw) * xRange;
  let closest = null, minDist = Infinity;
  for (const h of history) {
    const d = Math.abs(h.step - step);
    if (d < minDist) { minDist = d; closest = h; }
  }
  if (!closest) return;

  // Show tooltip
  let tip = document.getElementById('chart-tooltip');
  if (!tip) {
    tip = document.createElement('div');
    tip.id = 'chart-tooltip';
    tip.style.cssText = 'position:fixed;background:#222;color:#ccc;padding:4px 8px;border:1px solid #444;border-radius:3px;font:11px monospace;pointer-events:none;z-index:100;white-space:pre';
    document.body.appendChild(tip);
  }
  const lines = [`step ${closest.step.toLocaleString()}  ●${closest.camera_name || ''}`];
  for (const k of series) {
    const v = closest[k];
    if (v !== undefined) lines.push(`${k}: ${v.toFixed(5)}`);
  }
  lines.push('click → focus this camera');
  tip.textContent = lines.join('\n');
  tip.style.borderColor = camColor(closest.camera_name);
  tip.style.left = (e.clientX + 12) + 'px';
  tip.style.top  = (e.clientY - 10) + 'px';
  tip.style.display = 'block';
}

// Hide tooltip on mouseout
document.addEventListener('mouseover', e => {
  if (e.target.tagName !== 'CANVAS') {
    const tip = document.getElementById('chart-tooltip');
    if (tip) tip.style.display = 'none';
  }
});


// ── Clean tooltips (data-tip) ────────────────────────────────────────────────
//
// Any element with data-tip="..." shows a floating, body-anchored tooltip on
// hover or keyboard focus.  Works for dynamically-inserted controls (delegated).
//   data-tip        body text
//   data-tip-title  optional bold heading
//   data-tip-badge  optional status chip: live | startup | off | both | host
//
// Badge → human label.  These describe whether a control is actually honored by
// the tt-splat (Tenstorrent) training backend — see /docs/controls.html.
const _TT_BADGE = {
  live:    {cls: 'live',    label: 'live-editable'},
  both:    {cls: 'both',    label: 'works · both modes'},
  startup: {cls: 'startup', label: 'startup only'},
  host:    {cls: 'host',    label: 'host mode only'},
  off:     {cls: 'off',     label: 'not wired into TT loop'},
};

let _ttipEl = null;
function _ttip() {
  if (!_ttipEl) {
    _ttipEl = document.createElement('div');
    _ttipEl.id = 'ttgs-tip';
    document.body.appendChild(_ttipEl);
  }
  return _ttipEl;
}

function _showTip(el) {
  const body = el.getAttribute('data-tip');
  if (!body) return;
  const title = el.getAttribute('data-tip-title');
  const badge = el.getAttribute('data-tip-badge');
  const tip = _ttip();

  let h = '';
  if (title || badge) {
    h += '<div class="tt-head">';
    if (title) h += '<span class="tt-title">' + title + '</span>';
    if (badge && _TT_BADGE[badge]) {
      const b = _TT_BADGE[badge];
      h += '<span class="tt-badge ' + b.cls + '">' + b.label + '</span>';
    }
    h += '</div>';
  }
  h += '<div class="tt-body">' + body + '</div>';
  if (el.hasAttribute('data-tip-more'))
    h += '<div class="tt-more">' + el.getAttribute('data-tip-more') + '</div>';
  else
    h += '<div class="tt-more">ⓘ full guide → /docs/controls.html</div>';
  tip.innerHTML = h;

  // Measure, then position: prefer below the element, flip above if no room.
  tip.style.left = '0px'; tip.style.top = '0px';
  tip.classList.add('show');
  const r = el.getBoundingClientRect();
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  const margin = 8, vw = window.innerWidth, vh = window.innerHeight;
  let left = r.left + r.width / 2 - tw / 2;
  left = Math.max(margin, Math.min(left, vw - tw - margin));
  let top = r.bottom + margin;
  if (top + th > vh - margin) top = r.top - th - margin;   // flip above
  if (top < margin) top = margin;
  tip.style.left = Math.round(left) + 'px';
  tip.style.top = Math.round(top) + 'px';
}

function _hideTip() {
  if (_ttipEl) _ttipEl.classList.remove('show');
}

function initTooltips() {
  // Delegated — covers controls injected later by components.js.
  document.addEventListener('mouseover', e => {
    const el = e.target.closest && e.target.closest('[data-tip]');
    if (el) _showTip(el);
  });
  document.addEventListener('mouseout', e => {
    const el = e.target.closest && e.target.closest('[data-tip]');
    if (el && !el.contains(e.relatedTarget)) _hideTip();
  });
  document.addEventListener('focusin', e => {
    const el = e.target.closest && e.target.closest('[data-tip]');
    if (el) _showTip(el);
  });
  document.addEventListener('focusout', _hideTip);
  window.addEventListener('scroll', _hideTip, true);
}

if (document.readyState === 'loading')
  document.addEventListener('DOMContentLoaded', initTooltips);
else
  initTooltips();
