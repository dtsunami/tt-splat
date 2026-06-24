/* ttgs shared UI components v2 — run bar + pipeline controls + frame list
 *
 * Depends on base.js (pollState, pollPipeline, postCommand, postConfig, etc.)
 *
 * Usage:
 *   initRunBar('run-bar');
 *   initPipelineControls('pipeline-controls', { stages: true });
 *   initFrameList('frame-list');
 *   initTrainingPreview('train-preview');
 *
 *   // in your tick():
 *   updateRunBar(data);
 *   updatePipelineControls(data);
 *   updateTrainingPreview(data);
 */


// ═══════════════════════════════════════════════════════════════════════
// Run Bar — shared header: live metrics + editable training config
//
// The status row shows real-time metrics (read-only).
// The params row shows grouped, inline-editable config chips:
//   • Scroll wheel on a chip = increment / decrement by step
//   • Click / tab into a chip = type a value
//   • Enter / blur = apply immediately
//   • Colored left border per group = orientation marker
//     Blue = training  |  Purple = quality  |  Green = densification
// ═══════════════════════════════════════════════════════════════════════

const _RB_GROUPS = [
  { key: 'training', groupLabel: 'Training', color: '#4af', params: [
    { key: 'iterations',      label: 'iterations',  fullLabel: 'Iterations',          step: 1000, min: 1,
      desc: 'Total training steps. More iterations refine detail, but too many causes overtraining \u2014 floaters and artifacts. If the render looked better earlier, reduce this.' },
    { key: 'save_every',      label: 'checkpoint',  fullLabel: 'Checkpoint Interval', step: 500,  min: 0,
      desc: 'Save model state every N steps. Lower is safer against crashes (~3 GB each). 0 = save only at the end.' },
    { key: 'snapshot_every',  label: 'snapshot',     fullLabel: 'Snapshot Interval',   step: 100,  min: 0,
      desc: 'Save a render PNG per camera every N steps for training progression videos. Stitch with ffmpeg. 0 = off.' },
    { key: 'dashboard_every', label: 'refresh',      fullLabel: 'Dashboard Refresh',   step: 5,    min: 1,
      desc: 'Push a live preview every N steps. Lower = smoother monitoring but slightly slows training.' },
  ]},
  { key: 'quality', groupLabel: 'Quality', color: '#a6f', params: [
    { key: 'lambda_dssim',        label: '\u03bb dssim',  fullLabel: 'DSSIM Weight (\u03bb)', step: 0.05, min: 0, max: 1,
      desc: 'Balance between L1 pixel loss and SSIM structural similarity. 0 = pure L1 (sharp, noisy), 1 = pure SSIM (smooth, blurry). Default 0.2 is good for most scenes.' },
    { key: 'opacity_reset_every', label: 'opacity reset', fullLabel: 'Opacity Reset Interval', step: 500, min: 0,
      desc: 'Reset all opacities to near-zero every N steps, forcing Gaussians to re-earn visibility. Eliminates floaters. 0 = off.' },
  ]},
  { key: 'densify', groupLabel: 'Densification', color: '#4f4', params: [
    { key: 'densify_from',           label: 'from',        fullLabel: 'Start Step',         step: 100,     min: 0,
      desc: 'Step when densification begins. Before this, the model only moves and recolors existing Gaussians without adding new ones.' },
    { key: 'densify_until',          label: 'until',       fullLabel: 'Stop Step',          step: 1000,    min: 0,
      desc: 'Step when densification stops. After this, only refinement \u2014 no new Gaussians. Lower if you have too many.' },
    { key: 'densify_every',          label: 'every',       fullLabel: 'Interval',           step: 10,      min: 1,
      desc: 'Densification frequency within the start\u2013stop window. Lower = more frequent = more Gaussians.' },
    { key: 'densify_grad_threshold', label: 'grad thresh', fullLabel: 'Gradient Threshold', step: 0.00005, min: 0,
      desc: 'Gradient threshold for splitting Gaussians. Lower = more aggressive \u2014 more Gaussians to fill gaps. Raise if model creates too many.' },
  ]},
];

let _rbConfigLoaded = false;
let _rbConfigTick = 0;
let _rbDebounce = null;
let _rbPending = {};
let _trainStartTime = null;
let _trainElapsedStr = '';

function _fmtElapsed(ms) {
  var s = Math.floor(ms / 1000), m = Math.floor(s / 60), h = Math.floor(m / 60);
  if (h > 0) return h + 'h ' + (m % 60) + 'm';
  if (m > 0) return m + 'm ' + (s % 60) + 's';
  return s + 's';
}

function initRunBar(id) {
  const el = document.getElementById(id);
  if (!el) return;

  let h = '';

  // ── Queue strip (pipeline stages + recent commands) ──
  h += '<div class="rb-queue" id="rb-queue"></div>';

  // ── Status row ──
  h += '<div class="rb-status">';
  h += '<span class="rb-dot" id="rb-dot"></span>';

  h += '<div class="rb-metric">';
  h += '  <span class="rb-label">Step</span>';
  h += '  <span class="rb-val" id="rb-step">-- / --</span>';
  h += '</div>';

  h += '<div class="rb-progress"><div class="rb-progress-fill" id="rb-fill"></div></div>';
  h += '<span class="rb-pct" id="rb-pct">--%</span>';

  h += '<div class="rb-divider"></div>';

  h += '<div class="rb-metric">';
  h += '  <span class="rb-label">Loss</span>';
  h += '  <span class="rb-val" id="rb-loss">--</span>';
  h += '</div>';

  h += '<div class="rb-metric">';
  h += '  <span class="rb-label">Gaussians</span>';
  h += '  <span class="rb-val" id="rb-n">--</span>';
  h += '</div>';

  h += '<div class="rb-metric">';
  h += '  <span class="rb-label">Camera</span>';
  h += '  <span class="rb-val cam" id="rb-cam">--</span>';
  h += '</div>';

  h += '<span class="spacer"></span>';

  // Live preview thumbnails
  h += '<div class="rb-previews">';
  h += '  <div class="rb-thumb"><img id="rb-render" /><span>Render</span></div>';
  h += '  <div class="rb-thumb"><img id="rb-gt" /><span>GT</span></div>';
  h += '  <div class="rb-thumb"><img id="rb-diff" /><span>Diff</span></div>';
  h += '</div>';

  h += '<button class="rb-pause" id="rb-pause" onclick="togglePause()">Pause</button>';
  h += '<button class="rb-edit" id="rb-edit">Edit</button>';
  h += '</div>';

  // ── Params row ──
  h += '<div class="rb-params">';
  for (const group of _RB_GROUPS) {
    h += '<div class="rb-group" data-group="' + group.key + '">';
    for (const p of group.params) {
      const maxAttr = p.max !== undefined ? ' max="' + p.max + '"' : '';
      h += '<div class="rb-chip" data-key="' + p.key + '" data-step="' + p.step + '">';
      h += '  <label>' + p.label + '</label>';
      h += '  <input type="number" id="rb-' + p.key + '"'
         + ' min="' + p.min + '" step="' + p.step + '"' + maxAttr + '>';
      h += '</div>';
    }
    h += '</div>';
  }
  h += '</div>';

  el.innerHTML = h;

  // ── Wire up interaction ──
  for (const chip of el.querySelectorAll('.rb-chip')) {
    const input = chip.querySelector('input');
    const key = chip.dataset.key;
    const step = parseFloat(chip.dataset.step);

    // Scroll wheel = increment / decrement
    chip.addEventListener('wheel', function(e) {
      e.preventDefault();
      const delta = e.deltaY < 0 ? step : -step;
      let val = parseFloat(input.value) || 0;
      val += delta;
      // Shift key = 10x step
      if (e.shiftKey) val += delta * 9;
      val = Math.max(parseFloat(input.min) || 0, val);
      if (input.max && val > parseFloat(input.max)) val = parseFloat(input.max);
      if (step < 1) val = parseFloat(val.toFixed(6));
      input.value = val;
      _scheduleParamUpdate(key, val, chip);
    }, { passive: false });

    // Enter = apply + blur
    input.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
      if (e.key === 'Escape') { input.blur(); }
    });

    // Blur / change = apply immediately
    input.addEventListener('change', function() {
      const val = parseFloat(input.value);
      if (isNaN(val)) return;
      _flushParamUpdate();
      _scheduleParamUpdate(key, val, chip);
      _flushParamUpdate();
    });
  }

  // Wire up Edit button (click = modal, right-click = context menu)
  var editBtn = document.getElementById('rb-edit');
  if (editBtn) {
    editBtn.addEventListener('click', function() { _showEditModal(); });
    editBtn.addEventListener('contextmenu', function(e) {
      e.preventDefault();
      _showContextMenu(e.clientX, e.clientY);
    });
  }

  // Global: Escape closes modal/menu, click outside closes menu
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') { _hideEditModal(); _hideContextMenu(); }
  });
  document.addEventListener('click', function(e) {
    var menu = document.getElementById('ttgs-ctx');
    if (menu && !menu.contains(e.target) && e.target.id !== 'rb-edit') _hideContextMenu();
  });

  // Load server config into inputs
  _loadRunBarConfig();
}


function updateRunBar(d) {
  const bar = document.getElementById('run-bar');
  const dot = document.getElementById('rb-dot');

  if (!d) {
    if (bar) bar.classList.add('idle');
    if (dot) dot.className = 'rb-dot';
    const btn = document.getElementById('rb-pause');
    if (btn) { btn.textContent = 'Pause'; btn.className = 'rb-pause'; }
    _trainStartTime = null; _trainElapsedStr = '';
    return;
  }

  if (bar) bar.classList.remove('idle');
  if (dot) dot.className = d.is_paused ? 'rb-dot paused' : 'rb-dot live';

  const $ = function(id) { return document.getElementById(id); };

  // Step
  if ($('rb-step')) $('rb-step').textContent =
    d.step.toLocaleString() + ' / ' + d.total_steps.toLocaleString();

  // Progress
  const pct = d.total_steps > 0 ? (d.step / d.total_steps * 100) : 0;
  if ($('rb-fill')) $('rb-fill').style.width = pct.toFixed(1) + '%';
  if ($('rb-pct'))  $('rb-pct').textContent  = pct.toFixed(1) + '%';

  // Metrics
  if ($('rb-loss')) $('rb-loss').textContent = d.loss.toFixed(4);
  if ($('rb-n'))    $('rb-n').textContent    = d.n_gaussians.toLocaleString();
  if ($('rb-cam'))  $('rb-cam').textContent  = d.camera_name || '--';

  // Preview thumbnails
  if ($('rb-render') && d.render_b64) { $('rb-render').src = 'data:image/png;base64,' + d.render_b64; $('rb-render').style.display = 'block'; }
  if ($('rb-gt') && d.gt_b64)         { $('rb-gt').src = 'data:image/png;base64,' + d.gt_b64;         $('rb-gt').style.display = 'block'; }
  if ($('rb-diff') && d.diff_b64)     { $('rb-diff').src = 'data:image/png;base64,' + d.diff_b64;     $('rb-diff').style.display = 'block'; }

  // Pause button
  const btn = $('rb-pause');
  if (btn) {
    btn.textContent = d.is_paused ? 'Resume' : 'Pause';
    btn.className = d.is_paused ? 'rb-pause paused' : 'rb-pause';
  }

  // Track elapsed time
  if (!_trainStartTime) _trainStartTime = Date.now();
  _trainElapsedStr = _fmtElapsed(Date.now() - _trainStartTime);

  // Refresh config periodically (every ~10 s at 500 ms poll)
  if (!_rbConfigLoaded || ++_rbConfigTick % 20 === 0) {
    _loadRunBarConfig();
  }

  // Update modal pipeline status if open
  _updateModalPipeline();

  // Update queue strip
  _updateQueueStrip();
}


async function _loadRunBarConfig() {
  try {
    const r = await fetch('/config');
    if (!r.ok) return;
    const cfg = await r.json();
    if (!cfg || Object.keys(cfg).length === 0) return;
    _rbConfigLoaded = true;
    for (const group of _RB_GROUPS) {
      for (const p of group.params) {
        // Skip if user is editing or has pending changes
        if (_rbPending[p.key]) continue;
        const input = document.getElementById('rb-' + p.key);
        if (!input) continue;
        if (document.activeElement === input) continue;
        if (cfg[p.key] !== undefined) input.value = cfg[p.key];
      }
    }
  } catch(e) {}
}


function _scheduleParamUpdate(key, value, chip) {
  _rbPending[key] = { value: value, chip: chip };
  if (_rbDebounce) clearTimeout(_rbDebounce);
  _rbDebounce = setTimeout(_flushParamUpdate, 300);
}


async function _flushParamUpdate() {
  if (_rbDebounce) { clearTimeout(_rbDebounce); _rbDebounce = null; }
  const pending = {};
  const chips = [];
  for (var k in _rbPending) {
    pending[k] = _rbPending[k].value;
    chips.push(_rbPending[k].chip);
  }
  _rbPending = {};
  if (Object.keys(pending).length === 0) return;

  try {
    const ok = await postConfig(pending);
    for (var i = 0; i < chips.length; i++) {
      chips[i].classList.add(ok ? 'flash-ok' : 'flash-err');
      (function(c) {
        setTimeout(function() { c.classList.remove('flash-ok', 'flash-err'); }, 600);
      })(chips[i]);
    }
  } catch(e) {
    for (var j = 0; j < chips.length; j++) {
      chips[j].classList.add('flash-err');
      (function(c) {
        setTimeout(function() { c.classList.remove('flash-err'); }, 600);
      })(chips[j]);
    }
  }
}


// ── Queue strip: pipeline stages + recent commands ──
var _queueTick = 0;
var _cmdLog = [];

async function _updateQueueStrip() {
  var el = document.getElementById('rb-queue');
  if (!el) return;

  // Poll pipeline status (already polled by main tick, reuse global)
  var ps = (typeof _pipelineStatus !== 'undefined') ? _pipelineStatus : null;

  // Poll command log every ~4s
  if (++_queueTick % 8 === 0) {
    try {
      var r = await fetch('/state/commands?last=8');
      if (r.ok) _cmdLog = await r.json();
    } catch(e) {}
  }

  var h = '';

  // Pipeline stage pills
  if (ps && ps.stages) {
    h += '<div class="rq-stages">';
    var stageNames = ['extract', 'sfm', 'train', 'export'];
    for (var i = 0; i < stageNames.length; i++) {
      var sn = stageNames[i];
      var st = ps.stages[sn];
      if (!st) continue;
      if (i > 0) h += '<span class="rq-arrow">\u2192</span>';

      var cls = 'rq-pill ' + st.status;
      var dur = '';
      if (st.duration !== undefined) {
        dur = ' <span class="rq-dur">' + _fmtElapsed(st.duration * 1000) + '</span>';
      }
      var label = sn;
      if (st.status === 'running') label = sn.toUpperCase();

      // Hover tooltip
      var tip = '<div class="rq-tip"><b>' + sn + '</b> \u2014 ' + st.status;
      if (st.duration !== undefined) tip += '<br>Duration: ' + _fmtElapsed(st.duration * 1000);
      if (st.message) tip += '<br>' + st.message;
      tip += '</div>';

      h += '<span class="' + cls + '">' + label + dur + tip + '</span>';
    }
    h += '</div>';
  }

  // Recent command pills
  if (_cmdLog.length > 0) {
    h += '<span class="rq-sep">\u2502</span>';
    h += '<div class="rq-cmds">';
    for (var j = 0; j < _cmdLog.length; j++) {
      var c = _cmdLog[j];
      var ago = _fmtElapsed(Date.now() - c.time * 1000);
      var statsHtml = '';
      if (c.stats) {
        statsHtml += '<br>Loss: ' + (c.stats.loss !== undefined ? c.stats.loss.toFixed(4) : '--');
        statsHtml += '<br>Gaussians: ' + (c.stats.n_gaussians !== undefined ? c.stats.n_gaussians.toLocaleString() : '--');
      }
      var tip = '<div class="rq-tip">';
      tip += '<b>' + c.type + '</b> at step ' + (c.step || 0).toLocaleString();
      tip += '<br>' + ago + ' ago';
      if (c.detail) tip += '<br>' + c.detail;
      tip += statsHtml;
      tip += '</div>';
      h += '<span class="rq-cmd">' + c.type + tip + '</span>';
    }
    h += '</div>';
  }

  // Training elapsed
  if (_trainElapsedStr) {
    h += '<span class="rq-elapsed">train ' + _trainElapsedStr + '</span>';
  }

  el.innerHTML = h;
}

// ── Backward compat: old training header maps to run bar ──
function initTrainingHeader(id) { initRunBar(id); }
function updateTrainingHeader(d) { updateRunBar(d); }


// ═══════════════════════════════════════════════════════════════════════
// Pipeline Controls (sidebar — operations only, config moved to run bar)
// ═══════════════════════════════════════════════════════════════════════

function initPipelineControls(id, opts) {
  opts = opts || {};
  var el = document.getElementById(id);
  if (!el) return;
  var showStages = opts.stages !== false;

  var h = '';

  if (showStages) {
    h += '<div class="ctrl-section"><h3>Pipeline <span class="ctrl-tag" id="pl-stage">idle</span></h3>';
    h += '<button class="ctrl-lg primary" onclick="runFrom(\'train\')">Run Training</button>';
    h += '<button class="ctrl-lg" onclick="runFrom(\'sfm\')">Re-run SfM + Train</button>';
    h += '<button class="ctrl-lg" onclick="runFrom(\'extract\')">Re-run from Extract</button>';
    h += '<button class="ctrl-lg danger" onclick="interruptPipeline()">Interrupt</button>';
    h += '</div>';
  }

  h += '<div class="ctrl-section"><h3>Operations</h3>';
  h += '<button class="ctrl-lg danger" onclick="pruneGaussians()">Prune</button>';
  h += '<div class="ctrl-field">';
  h += '  <label>opacity threshold <span class="ctrl-value" id="prune-val">0.005</span></label>';
  h += '  <input type="range" id="prune-thresh" min="0.001" max="0.1" step="0.001" value="0.005"';
  h += '         oninput="document.getElementById(\'prune-val\').textContent=parseFloat(this.value).toFixed(3)">';
  h += '</div>';
  h += '<button class="ctrl-lg" onclick="postCommand(\'densify_now\')">Densify</button>';
  h += '<button class="ctrl-lg" onclick="postCommand(\'clamp_scale\',{max_log_scale:2.5})">Clamp Scale</button>';
  h += '<button class="ctrl-lg" onclick="postCommand(\'reset_opacities\')">Reset Opacities</button>';
  h += '<button class="ctrl-lg" onclick="postCommand(\'save\')">Save Checkpoint</button>';
  h += '</div>';

  if (opts.image) {
    h += '<div class="ctrl-section"><h3>This Image</h3>';
    h += '<button class="ctrl-lg" id="pl-excl" onclick="toggleExclude()">Exclude [X]</button>';
    h += '<button class="ctrl-lg" onclick="focusCamera()">Focus here [F]</button>';
    h += '<button class="ctrl-lg" onclick="focusClear()">Clear focus</button>';
    h += '<button class="ctrl-lg danger" onclick="clearMask()">Clear all masks</button>';
    h += '</div>';
  }

  el.innerHTML = h;
}

function updatePipelineControls(trainData) {
  // Pause button
  var btn = document.getElementById('pl-pause');
  if (btn && trainData) {
    btn.textContent = trainData.is_paused ? 'Resume' : 'Pause';
    btn.className = trainData.is_paused ? 'ctrl-lg active' : 'ctrl-lg';
  }

  // Pipeline stage tag
  if (typeof _pipelineStatus !== 'undefined' && _pipelineStatus) {
    var tag = document.getElementById('pl-stage');
    if (tag) {
      var cur = _pipelineStatus.current;
      if (cur) {
        tag.textContent = cur;
        tag.className = 'ctrl-tag running';
      } else {
        tag.textContent = 'idle';
        tag.className = 'ctrl-tag';
      }
    }
  }

  // Exclude button (if per-image)
  if (typeof isExcluded !== 'undefined') {
    var excl = document.getElementById('pl-excl');
    if (excl) {
      excl.textContent = isExcluded ? 'Include [X]' : 'Exclude [X]';
      excl.className = isExcluded ? 'ctrl-lg active' : 'ctrl-lg';
    }
  }
}


// ═══════════════════════════════════════════════════════════════════════
// Training Preview (render / GT / diff thumbnails)
// ═══════════════════════════════════════════════════════════════════════

function initTrainingPreview(id) {
  var el = document.getElementById(id);
  if (!el) return;
  el.innerHTML =
    '<div class="ctrl-section">' +
    '<h3>Training Preview</h3>' +
    '<div class="tp-grid">' +
    '  <div class="tp-cell"><div class="tp-label">Render</div><img id="tp-render" class="tp-img" /></div>' +
    '  <div class="tp-cell"><div class="tp-label">GT</div><img id="tp-gt" class="tp-img" /></div>' +
    '  <div class="tp-cell"><div class="tp-label">Diff</div><img id="tp-diff" class="tp-img" /></div>' +
    '</div>' +
    '<div class="tp-stats" id="tp-stats">--</div>' +
    '</div>';
}

function updateTrainingPreview(d) {
  if (!d) return;
  var r = document.getElementById('tp-render');
  var g = document.getElementById('tp-gt');
  var df = document.getElementById('tp-diff');
  var st = document.getElementById('tp-stats');
  if (r && d.render_b64) { r.src = 'data:image/png;base64,' + d.render_b64; r.style.display = 'block'; }
  if (g && d.gt_b64)     { g.src = 'data:image/png;base64,' + d.gt_b64;     g.style.display = 'block'; }
  if (df && d.diff_b64)  { df.src = 'data:image/png;base64,' + d.diff_b64;  df.style.display = 'block'; }
  if (st) st.textContent = 'L1 ' + d.l1.toFixed(4) + '  SSIM ' + d.ssim.toFixed(4) + '  cam: ' + d.camera_name;
}


// ═══════════════════════════════════════════════════════════════════════
// Frame List (toggle-enabled, real-time)
// ═══════════════════════════════════════════════════════════════════════

var _frameListData = [];
var _frameListEl = null;
var _frameListHighlight = null;
var _frameListHash = '';

function initFrameList(id, highlightName) {
  _frameListEl = document.getElementById(id);
  _frameListHighlight = highlightName || null;
  if (_frameListEl) pollFrameList();
}

async function pollFrameList() {
  try {
    var r = await fetch('/images/list');
    if (!r.ok) return;
    var data = await r.json();
    var hash = JSON.stringify(data);
    if (hash !== _frameListHash) {
      _frameListHash = hash;
      _frameListData = data;
      renderFrameList();
    }
  } catch(e) {}
}

function renderFrameList() {
  if (!_frameListEl) return;
  if (!_frameListData.length) {
    _frameListEl.innerHTML = '<div class="fl-empty">No frames</div>';
    return;
  }
  // Update frame count badge if present
  var fc = document.getElementById('frame-count');
  if (fc) fc.textContent = _frameListData.length;

  var h = '';
  for (var i = 0; i < _frameListData.length; i++) {
    var f = _frameListData[i];
    var active = _frameListHighlight && f.name === _frameListHighlight;
    var cls = 'fl-entry' + (f.excluded ? ' excluded' : '') + (active ? ' active' : '');
    h += '<div class="' + cls + '" data-name="' + f.name + '">';
    h += '  <label class="fl-toggle" onclick="event.stopPropagation()">';
    h += '    <input type="checkbox"' + (f.excluded ? '' : ' checked') + ' onchange="toggleFrameEntry(\'' + f.name + '\', !this.checked)">';
    h += '    <span class="fl-switch"></span>';
    h += '  </label>';
    h += '  <img class="fl-thumb" src="/images/' + f.name + '/thumb" loading="lazy">';
    h += '  <span class="fl-name">' + f.name + '</span>';
    if (f.masked) h += '<span class="badge mask">M</span>';
    h += '</div>';
  }
  _frameListEl.innerHTML = h;

  // Click on entry navigates to edit page
  var entries = _frameListEl.querySelectorAll('.fl-entry');
  for (var j = 0; j < entries.length; j++) {
    entries[j].addEventListener('click', function() {
      location.href = '/images/' + this.dataset.name + '/edit';
    });
  }
}

async function toggleFrameEntry(name, excluded) {
  await fetch('/images/' + name + '/exclude', { method: excluded ? 'POST' : 'DELETE' });
  var f = _frameListData.find(function(x) { return x.name === name; });
  if (f) { f.excluded = excluded; _frameListHash = ''; }
  renderFrameList();
}


// ═══════════════════════════════════════════════════════════════════════
// Edit Modal — full config editor with descriptions + pipeline commands
// ═══════════════════════════════════════════════════════════════════════

var _modalEl = null;

function _ensureModal() {
  if (_modalEl) return;
  var ov = document.createElement('div');
  ov.id = 'ttgs-modal';
  ov.className = 'em-overlay em-hidden';
  ov.addEventListener('click', function(e) { if (e.target === ov) _hideEditModal(); });

  var h = '<div class="em-dialog">';

  // Header
  h += '<div class="em-header">';
  h += '  <h2>Training Configuration</h2>';
  h += '  <button class="em-close" onclick="_hideEditModal()">\u00d7</button>';
  h += '</div>';

  h += '<div class="em-body">';

  // Pipeline status
  h += '<div class="em-pipeline" id="em-pipeline">';
  h += '  <div class="em-pl-status" id="em-pl-status">Checking pipeline...</div>';
  h += '  <div class="em-pl-actions">';
  h += '    <button onclick="togglePause();_updateModalPipeline()">Pause / Resume</button>';
  h += '    <button onclick="postCommand(\'save\')">Save Checkpoint</button>';
  h += '    <button onclick="pruneGaussians()">Prune</button>';
  h += '    <button onclick="postCommand(\'densify_now\')">Densify Now</button>';
  h += '    <button onclick="postCommand(\'clamp_scale\',{max_log_scale:2.5})">Clamp Scale</button>';
  h += '    <button onclick="postCommand(\'reset_opacities\')">Reset Opacities</button>';
  h += '  </div>';
  h += '  <div class="em-pl-actions" style="margin-top:6px">';
  h += '    <button class="primary" onclick="runFrom(\'train\')">Run Training</button>';
  h += '    <button onclick="runFrom(\'sfm\')">Run SfM + Train</button>';
  h += '    <button onclick="runFrom(\'extract\')">Run from Extract</button>';
  h += '    <button onclick="runFrom(\'export\')">Export</button>';
  h += '    <button class="danger" onclick="interruptPipeline()">Interrupt</button>';
  h += '  </div>';
  h += '</div>';

  // Param groups
  for (var gi = 0; gi < _RB_GROUPS.length; gi++) {
    var group = _RB_GROUPS[gi];
    h += '<div class="em-group">';
    h += '<div class="em-group-hdr">';
    h += '  <span class="em-group-mark" style="background:' + group.color + '"></span>';
    h += '  <h3>' + group.groupLabel + '</h3>';
    h += '</div>';

    for (var pi = 0; pi < group.params.length; pi++) {
      var p = group.params[pi];
      var maxAttr = p.max !== undefined ? ' max="' + p.max + '"' : '';
      h += '<div class="em-param">';
      h += '  <div class="em-param-info">';
      h += '    <div class="em-param-label">' + (p.fullLabel || p.label) + '</div>';
      h += '    <div class="em-param-desc">' + (p.desc || '') + '</div>';
      h += '  </div>';
      h += '  <div class="em-param-input">';
      h += '    <input type="number" id="em-' + p.key + '"'
         + ' min="' + p.min + '" step="' + p.step + '"' + maxAttr
         + ' data-key="' + p.key + '">';
      h += '  </div>';
      h += '</div>';
    }
    h += '</div>';
  }

  h += '</div>'; // em-body
  h += '</div>'; // em-dialog

  ov.innerHTML = h;
  document.body.appendChild(ov);
  _modalEl = ov;

  // Wire up modal inputs — apply on change
  var inputs = ov.querySelectorAll('.em-param-input input');
  for (var i = 0; i < inputs.length; i++) {
    (function(inp) {
      inp.addEventListener('change', function() {
        var val = parseFloat(inp.value);
        if (isNaN(val)) return;
        var payload = {};
        payload[inp.dataset.key] = val;
        postConfig(payload).then(function(ok) {
          inp.parentElement.parentElement.classList.add(ok ? 'em-flash-ok' : 'em-flash-err');
          setTimeout(function() {
            inp.parentElement.parentElement.classList.remove('em-flash-ok', 'em-flash-err');
          }, 700);
          // Sync run bar chip
          var rbInput = document.getElementById('rb-' + inp.dataset.key);
          if (rbInput && ok) rbInput.value = inp.value;
        });
      });
      inp.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') { e.preventDefault(); inp.blur(); }
      });
    })(inputs[i]);
  }
}


function _showEditModal() {
  _ensureModal();
  // Load current values
  _loadModalConfig();
  _updateModalPipeline();
  _modalEl.classList.remove('em-hidden');
}

function _hideEditModal() {
  if (_modalEl) _modalEl.classList.add('em-hidden');
}

async function _loadModalConfig() {
  try {
    var r = await fetch('/config');
    if (!r.ok) return;
    var cfg = await r.json();
    for (var gi = 0; gi < _RB_GROUPS.length; gi++) {
      for (var pi = 0; pi < _RB_GROUPS[gi].params.length; pi++) {
        var p = _RB_GROUPS[gi].params[pi];
        var inp = document.getElementById('em-' + p.key);
        if (inp && cfg[p.key] !== undefined) inp.value = cfg[p.key];
      }
    }
  } catch(e) {}
}

function _updateModalPipeline() {
  var el = document.getElementById('em-pl-status');
  if (!el) return;
  var d = getTrainState();
  var lines = [];
  if (d) {
    var state = d.is_paused ? 'PAUSED' : 'TRAINING';
    lines.push('<b>' + state + '</b>');
    lines.push('step <b>' + d.step.toLocaleString() + ' / ' + d.total_steps.toLocaleString() + '</b>');
    lines.push('loss <b>' + d.loss.toFixed(4) + '</b>');
    lines.push('<b>' + d.n_gaussians.toLocaleString() + '</b> gaussians');
    if (_trainElapsedStr) lines.push('elapsed <span class="em-time">' + _trainElapsedStr + '</span>');
  } else {
    lines.push('Idle \u2014 no training running');
  }
  // Pipeline stages
  if (typeof _pipelineStatus !== 'undefined' && _pipelineStatus && _pipelineStatus.stages) {
    var stages = _pipelineStatus.stages;
    var parts = [];
    for (var s in stages) {
      var st = stages[s].status;
      if (st === 'running') parts.push('<span class="em-stage-run">' + s + '</span>');
      else if (st === 'done') parts.push('<span class="em-stage-done">' + s + '</span>');
      else if (st === 'error') parts.push('<span class="em-stage-err">' + s + '</span>');
      else parts.push('<span class="em-stage-idle">' + s + '</span>');
    }
    lines.push(parts.join(' \u2192 '));
  }
  el.innerHTML = lines.join('<span class="em-sep">\u00a0\u00a0\u2502\u00a0\u00a0</span>');
}


// ═══════════════════════════════════════════════════════════════════════
// Context Menu — right-click on Edit button for quick pipeline commands
// ═══════════════════════════════════════════════════════════════════════

var _ctxEl = null;

function _ensureContextMenu() {
  if (_ctxEl) return;
  var el = document.createElement('div');
  el.id = 'ttgs-ctx';
  el.className = 'ctx-menu ctx-hidden';
  document.body.appendChild(el);
  _ctxEl = el;
}

function _showContextMenu(x, y) {
  _ensureContextMenu();
  var d = getTrainState();

  var h = '<div class="ctx-status">';
  if (d) {
    h += '<b>' + (d.is_paused ? 'PAUSED' : 'TRAINING') + '</b>';
    h += ' \u00a0 step ' + d.step.toLocaleString();
    if (_trainElapsedStr) h += ' \u00a0 <span class="ctx-time">' + _trainElapsedStr + '</span>';
  } else {
    h += 'Idle';
  }
  h += '</div>';

  h += '<div class="ctx-sep"></div>';
  h += _ctxItem('Pause / Resume', 'togglePause()');
  h += _ctxItem('Save Checkpoint', "postCommand('save')");
  h += '<div class="ctx-sep"></div>';
  h += _ctxItem('Prune Gaussians', 'pruneGaussians()');
  h += _ctxItem('Densify Now', "postCommand('densify_now')");
  h += _ctxItem('Clamp Scale', "postCommand('clamp_scale',{max_log_scale:2.5})");
  h += _ctxItem('Reset Opacities', "postCommand('reset_opacities')");
  h += '<div class="ctx-sep"></div>';
  h += _ctxItem('Run Training', "runFrom('train')", 'primary');
  h += _ctxItem('Run SfM + Train', "runFrom('sfm')");
  h += _ctxItem('Run from Extract', "runFrom('extract')");
  h += _ctxItem('Export', "runFrom('export')");
  h += '<div class="ctx-sep"></div>';
  h += _ctxItem('Interrupt', 'interruptPipeline()', 'danger');

  _ctxEl.innerHTML = h;
  _ctxEl.classList.remove('ctx-hidden');

  // Position: keep within viewport
  var vw = window.innerWidth, vh = window.innerHeight;
  var mx = Math.min(x, vw - 250);
  var my = Math.min(y, vh - _ctxEl.offsetHeight - 10);
  _ctxEl.style.left = mx + 'px';
  _ctxEl.style.top = my + 'px';
}

function _ctxItem(label, action, cls) {
  return '<div class="ctx-item' + (cls ? ' ' + cls : '') + '" '
    + 'onclick="' + action + ';_hideContextMenu()">' + label + '</div>';
}

function _hideContextMenu() {
  if (_ctxEl) _ctxEl.classList.add('ctx-hidden');
}
