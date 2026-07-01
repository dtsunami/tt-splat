/* ttgs camera drive — WASD + mouse-look pose nudging (game-style), shared by /training and /pose.
 *
 * Drives the CURRENT camera's 6-DoF extrinsics — the interactive "grab the ghosting camera and nudge"
 * handle. Reuses the render overlay (#splat-overlay) for mouse capture + a global keydown for WASD/QE/ZC.
 * Inputs are COALESCED and sent at ~10 Hz as one combined pose_nudge(omega,trans), so the command queue
 * (drained once per training step) isn't flooded. Turning Drive on enables pose-opt (driving IS pose-opt).
 * Note: with a ~1 s/step loop the render catches up a step behind the input — expected.
 *
 * Depends on base.js: poseNudgeRaw, postConfig, getTrainState. Optional: splatTool/splatToggle (training).
 * DOM it touches (all optional/guarded): #drive-toggle, #splat-overlay, #pose-hud.
 */
const drive = { active:false, keys:{}, lookDX:0, lookDY:0, dragging:false, lx:0, ly:0, timer:null,
                mov:0.02, rot:0.012, look:0.0018 };   // per-tick step sizes (world units / rad)

function _driveTick() {
  if (!drive.active) return;
  const o=[0,0,0], t=[0,0,0], k=drive.keys, m=drive.mov, r=drive.rot;
  if (k['w']) t[2]-=m; if (k['s']) t[2]+=m;      // forward / back
  if (k['a']) t[0]-=m; if (k['d']) t[0]+=m;      // strafe left / right
  if (k['q']) t[1]-=m; if (k['e']) t[1]+=m;      // down / up
  if (k['z']) o[2]-=r; if (k['c']) o[2]+=r;      // roll
  o[1]+=drive.lookDX*drive.look;                 // mouse X -> yaw
  o[0]+=drive.lookDY*drive.look;                 // mouse Y -> pitch
  drive.lookDX=0; drive.lookDY=0;
  if (o.some(v=>v) || t.some(v=>v)) { if (typeof poseNudgeRaw==='function') poseNudgeRaw(o, t); }
}

function driveToggle() {
  drive.active = !drive.active;
  const btn=document.getElementById('drive-toggle'), cv=document.getElementById('splat-overlay');
  const hud=document.getElementById('pose-hud');
  if (btn) btn.classList.toggle('on', drive.active);
  if (hud) hud.classList.toggle('show', drive.active);
  if (drive.active) {
    if (typeof splatTool!=='undefined' && splatTool.active && typeof splatToggle==='function') splatToggle();  // exclusive w/ bubble gun
    if (cv) { cv.classList.add('active'); cv.classList.add('drive'); }
    if (typeof postConfig==='function') postConfig({pose_opt: 1});           // driving IS pose-opt → ensure on
    if (!drive.timer) drive.timer=setInterval(_driveTick, 100);             // 10 Hz coalesced send
  } else {
    if (cv) { cv.classList.remove('active'); cv.classList.remove('drive'); }
    drive.keys={}; drive.lookDX=drive.lookDY=0; drive.dragging=false;
    if (drive.timer) { clearInterval(drive.timer); drive.timer=null; }
  }
}

function _updatePoseHud(d) {
  const hud=document.getElementById('pose-hud'); if(!hud) return;
  const pc = d && d.pose_corr;
  const corr = pc ? `Δrot ${pc.dw_deg.toFixed(2)}°  Δpos ${pc.dt.toFixed(3)}`
                  : 'pose-opt off (enable in Camera Pose-Opt, or hit Drive)';
  hud.innerHTML = drive.active
    ? `<b>🎮 Drive</b> WASD move · QE down/up · ZC roll · drag = look<br><span class="hud-corr">${corr}</span>`
    : `<span class="hud-corr">${corr}</span>`;
  hud.classList.toggle('show', drive.active || !!pc);   // show whenever there's a correction to see
}

// keyboard (only while Drive is on, and not while typing in a field)
document.addEventListener('keydown', e => {
  if (!drive.active) return;
  const tag=(e.target&&e.target.tagName)||'';
  if (tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT') return;
  const k=(e.key||'').toLowerCase();
  if ('wasdqezc'.indexOf(k)>=0 && k.length===1) { drive.keys[k]=true; e.preventDefault(); }
});
document.addEventListener('keyup', e => { const k=(e.key||'').toLowerCase(); if (drive.keys[k]!==undefined) drive.keys[k]=false; });

// mouse-look: drag on the render overlay (drive handlers coexist with the splat ones; each guards its tool)
(function _driveMouseInit(){
  const cv=document.getElementById('splat-overlay'); if(!cv) return;
  cv.addEventListener('mousedown', e => { if(!drive.active) return; drive.dragging=true; drive.lx=e.clientX; drive.ly=e.clientY; e.preventDefault(); });
  cv.addEventListener('mousemove', e => { if(!drive.active||!drive.dragging) return; drive.lookDX+=e.clientX-drive.lx; drive.lookDY+=e.clientY-drive.ly; drive.lx=e.clientX; drive.ly=e.clientY; });
  window.addEventListener('mouseup', () => { drive.dragging=false; });
})();
