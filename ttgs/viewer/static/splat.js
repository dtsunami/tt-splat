/* ttgs bubble-gun "splat tool" — aim on the live render, click-drag to fire. Shared by /training and
 * /densify. ADD = spray new Gaussians (near a surface = densify, in empty space = SfM seed) via splat_spawn;
 * ERASE = vacuum the Gaussians under the brush (floaters/fog) via cull_region. Think Slime Rancher: a
 * crosshair you sweep over the error surface to fill in missing detail.
 *
 * Depends on base.js: postCommand, and a page-global `_currentCam` (the camera the render belongs to).
 * Optional: drive (drive.js) — mutually exclusive with driving. DOM it needs (all on the render cell):
 *   #render-img  #render-cell  #splat-overlay  #splat-toggle  #splat-bar  #splat-add  #splat-remove  #splat-brush
 */
const splatTool = { active:false, mode:'add', brush:8, nPer:3, drawing:false, points:[] };

function splatToggle() {
  splatTool.active = !splatTool.active;
  if (splatTool.active && typeof drive !== 'undefined' && drive.active && typeof driveToggle === 'function') driveToggle();
  const tg=document.getElementById('splat-toggle'), bar=document.getElementById('splat-bar'),
        ov=document.getElementById('splat-overlay');
  if (tg) tg.classList.toggle('on', splatTool.active);
  if (bar) bar.classList.toggle('active', splatTool.active);
  if (ov) ov.classList.toggle('active', splatTool.active);
  _splatClear();
}
function splatSetMode(m) {
  splatTool.mode = m;
  const a=document.getElementById('splat-add'), r=document.getElementById('splat-remove');
  if (a) a.classList.toggle('sel', m === 'add');
  if (r) r.classList.toggle('sel', m === 'remove');
}
function _imgPixel(cx, cy) {                 // client coords -> render pixel coords (object-fit:contain aware)
  const img = document.getElementById('render-img');
  if (!img) return null;
  const r = img.getBoundingClientRect(), nw = img.naturalWidth, nh = img.naturalHeight;
  if (!nw || !nh || r.width === 0) return null;
  const s = Math.min(r.width / nw, r.height / nh);
  const ox = r.left + (r.width - nw * s) / 2, oy = r.top + (r.height - nh * s) / 2;
  const px = (cx - ox) / s, py = (cy - oy) / s;
  if (px < 0 || py < 0 || px >= nw || py >= nh) return null;
  return [px, py, s];
}
function _splatCanvas() {
  const cv = document.getElementById('splat-overlay');
  const cell = document.getElementById('render-cell').getBoundingClientRect();
  if (cv.width !== Math.round(cell.width) || cv.height !== Math.round(cell.height)) {
    cv.width = Math.round(cell.width); cv.height = Math.round(cell.height);
  }
  return cv;
}
function _splatClear() {
  const cv = document.getElementById('splat-overlay'); if (!cv) return;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height); splatTool.points = []; splatTool.drawing = false;
}
function _splatDraw(ev) {
  const cv = _splatCanvas(), ctx = cv.getContext('2d');
  const cell = document.getElementById('render-cell').getBoundingClientRect();
  ctx.clearRect(0, 0, cv.width, cv.height);
  const add = splatTool.mode === 'add';
  const fill = add ? 'rgba(120,224,138,0.22)' : 'rgba(255,90,90,0.20)', line = add ? '#7de08a' : '#ff6a6a';
  for (const p of splatTool.points) {        // p = [imgpx, imgpy, clientX, clientY, dispR]
    ctx.beginPath(); ctx.arc(p[2] - cell.left, p[3] - cell.top, p[4], 0, 7);
    ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.stroke();
  }
  if (ev) {                                  // dashed brush cursor
    const pp = _imgPixel(ev.clientX, ev.clientY);
    if (pp) {
      ctx.beginPath(); ctx.arc(ev.clientX - cell.left, ev.clientY - cell.top, splatTool.brush * pp[2], 0, 7);
      ctx.strokeStyle = line; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
    }
  }
}
function _splatPush(ev) {
  const pp = _imgPixel(ev.clientX, ev.clientY);
  if (!pp) return;
  const last = splatTool.points[splatTool.points.length - 1];
  if (last && Math.hypot(pp[0] - last[0], pp[1] - last[1]) < splatTool.brush * 0.6) return;   // throttle (image px)
  splatTool.points.push([pp[0], pp[1], ev.clientX, ev.clientY, splatTool.brush * pp[2]]);
}
async function _splatFire() {
  const pts = splatTool.points.map(p => [Math.round(p[0]), Math.round(p[1])]);
  _splatClear();
  const cam = (typeof _currentCam !== 'undefined') ? _currentCam : null;
  if (!pts.length || !cam) return;
  const type = splatTool.mode === 'remove' ? 'cull_region' : 'splat_spawn';
  try { await postCommand(type, { camera_name: cam, points: pts, brush: splatTool.brush, n_per: splatTool.nPer }); }
  catch (e) { console.warn('splat:', e); }
}
(function _splatInit() {
  const cv = document.getElementById('splat-overlay');
  if (!cv) return;
  cv.addEventListener('mousedown',  e => { if (!splatTool.active) return; splatTool.drawing = true; splatTool.points = []; _splatPush(e); _splatDraw(e); });
  cv.addEventListener('mousemove',  e => { if (!splatTool.active) return; if (splatTool.drawing) _splatPush(e); _splatDraw(e); });
  cv.addEventListener('mouseup',    () => { if (!splatTool.active) return; splatTool.drawing = false; _splatFire(); });
  cv.addEventListener('mouseleave', () => { if (splatTool.drawing) { splatTool.drawing = false; _splatFire(); } });
  const br = document.getElementById('splat-brush');
  if (br) br.addEventListener('input', () => { splatTool.brush = parseFloat(br.value); });
})();
