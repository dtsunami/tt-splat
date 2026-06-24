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
    `L1 ${d.l1.toFixed(4)} SSIM ${d.ssim.toFixed(4)} MSE ${d.mse.toFixed(5)} ` +
    `<b>${d.n_gaussians.toLocaleString()}</b>G ` +
    `cam: ${d.camera_name}`;
}

// ── Loss chart ───────────────────────────────────────────────────────────────
//
// drawLossChart(canvasId, history, {camera})
//
// history: [{step, loss, l1, ssim, mse, camera_name}, ...]
// Draws multi-line loss curves on a <canvas>.

const CHART_COLORS = {
  loss: '#4af',
  l1:   '#f84',
  ssim: '#a6f',
  mse:  '#f44',
};

async function fetchHistory(camera) {
  const url = camera ? `/state/history?camera=${encodeURIComponent(camera)}` : '/state/history';
  try {
    const r = await fetch(url);
    if (!r.ok) return [];
    return await r.json();
  } catch { return []; }
}

function drawLossChart(canvasId, history, opts = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !history.length) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = {l: 50, r: 10, t: 20, b: 24};
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;

  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#0e0e0e';
  ctx.fillRect(0, 0, W, H);

  // Determine which series to show
  const series = opts.series || ['loss', 'l1', 'ssim', 'mse'];
  const highlighted = opts.highlight || null;  // camera name to highlight

  // Compute Y range across all visible series
  let yMin = Infinity, yMax = -Infinity;
  for (const key of series) {
    for (const h of history) {
      const v = h[key];
      if (v !== undefined && v !== null && isFinite(v)) {
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
      }
    }
  }
  if (!isFinite(yMin)) return;
  const yRange = Math.max(yMax - yMin, 1e-6);
  // Add 10% padding
  yMin -= yRange * 0.05;
  yMax += yRange * 0.05;

  const xMin = history[0].step;
  const xMax = history[history.length - 1].step;
  const xRange = Math.max(xMax - xMin, 1);

  const toX = s => pad.l + ((s - xMin) / xRange) * cw;
  const toY = v => pad.t + (1 - (v - yMin) / (yMax - yMin)) * ch;

  // Grid lines
  ctx.strokeStyle = '#222'; ctx.lineWidth = 1;
  const nGrid = 4;
  ctx.font = '10px monospace'; ctx.fillStyle = '#555'; ctx.textAlign = 'right';
  for (let i = 0; i <= nGrid; i++) {
    const v = yMin + (yMax - yMin) * (i / nGrid);
    const y = toY(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillText(v.toFixed(4), pad.l - 4, y + 3);
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
      if (v === undefined || v === null || !isFinite(v)) continue;
      const x = toX(h.step), y = toY(v);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // Highlighted camera dots (if filtering per-camera)
  if (highlighted) {
    ctx.fillStyle = '#ff0';
    for (const h of history) {
      if (h.camera_name === highlighted) {
        ctx.beginPath(); ctx.arc(toX(h.step), toY(h.loss), 3, 0, Math.PI * 2); ctx.fill();
      }
    }
  }

  // Legend
  ctx.font = '10px monospace'; ctx.textAlign = 'left';
  let lx = pad.l + 6;
  for (const key of series) {
    ctx.fillStyle = CHART_COLORS[key] || '#888';
    ctx.fillText(key, lx, pad.t + 12);
    lx += ctx.measureText(key).width + 12;
  }

  // Hover handler (attach once)
  if (!canvas._hasHover) {
    canvas._hasHover = true;
    canvas._history = history;
    canvas._opts = {pad, cw, ch, xMin, xRange, yMin, yMax, series};
    canvas.addEventListener('mousemove', _chartHover);
  } else {
    canvas._history = history;
    canvas._opts = {pad, cw, ch, xMin, xRange, yMin, yMax, series};
  }
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
  const lines = [`step ${closest.step.toLocaleString()}  ${closest.camera_name || ''}`];
  for (const k of series) {
    const v = closest[k];
    if (v !== undefined) lines.push(`${k}: ${v.toFixed(5)}`);
  }
  tip.textContent = lines.join('\n');
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
