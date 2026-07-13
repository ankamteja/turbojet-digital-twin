// scene.js — Three.js procedural turbojet, ported unchanged in spirit from the
// teammate's reference HUD. Geometry (bladeDiskMesh, ringMesh, nacelle lathe,
// shaft) is identical so the look/feel is preserved.
//
// What changed: the reference recoloured stages from a fake "degradation"
// timer. Here we expose two hooks the dashboard drives from REAL data:
//   - setStageHealth(stage, value): colour a stage green->yellow->red by health
//   - selectStage(stage): highlight the clicked stage (green) + dim the rest
// Purely-cosmetic motion (shaft spin, auto-rotate) has no real counterpart and
// is kept.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---- palette (matches the CSS custom properties) ----
const STAGE_COLOR_GREEN = 0x50ff78;
const GREEN_EMISSIVE = 0x40cc60;
const NEUTRAL = 0xcfd6dc;
const NEUTRAL_DIM = 0x50565c;
const HEALTH_YELLOW = 0xffcc00;
const HEALTH_RED = 0xff3b3b;

let scene, camera, renderer, canvas, controls;
let proceduralGroup, shaftSpin;
let stages = { fan: [], compressor: [], combustor: [], turbine: [] };

// per-stage health colour applied by setStageHealth (0..1 -> hex).
// Kept separate from selection so a selected stage can still restore to its
// health tint when deselected.
let stageHealthHex = { fan: NEUTRAL, combustor: NEUTRAL, compressor: NEUTRAL, turbine: NEUTRAL };
let activeStage = null;

// Callback invoked when a stage mesh region is clicked (wired by main.js).
// The reference used HTML buttons only; we keep that path and simply notify.
let onStageSelect = null;

/**
 * Boot the 3D scene into #engine-canvas.
 * @param {(stage:string)=>void} stageSelectCb notified on programmatic select.
 */
export function init(stageSelectCb) {
  onStageSelect = stageSelectCb || null;
  // Defer one frame so the container has a measured size before we size the GL
  // buffer (identical guard to the reference).
  requestAnimationFrame(() => {
    try {
      initScene();
      buildProceduralEngine();
      animate();
      console.log('[JET] 3D engine initialized');
    } catch (err) {
      console.error('[JET] 3D init failed:', err);
    }
  });
}

function initScene() {
  canvas = document.getElementById('engine-canvas');
  const container = document.getElementById('viewport-frame');
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(42, 1, 0.1, 2000);
  camera.position.set(8, 4, 10);
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const rect = container.getBoundingClientRect();
  const w = Math.max(rect.width, 300);
  const h = Math.max(rect.height, 200);
  renderer.setSize(w, h, false);
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  camera.aspect = w / h;
  camera.updateProjectionMatrix();

  if (window.ResizeObserver) {
    new ResizeObserver(() => resize()).observe(container);
  }
  window.addEventListener('resize', resize);

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.0); key.position.set(6, 9, 8); scene.add(key);
  const rim = new THREE.DirectionalLight(0x88aaff, 0.5); rim.position.set(-8, -4, -6); scene.add(rim);
  const fill = new THREE.DirectionalLight(0xffffff, 0.3); fill.position.set(0, -6, 4); scene.add(fill);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;          // cosmetic — no real counterpart
  controls.autoRotateSpeed = 0.8;
  controls.minDistance = 5;
  controls.maxDistance = 30;
  controls.target.set(-1, 0, 0);
  controls.update();

  canvas.addEventListener('pointerdown', () => { controls.autoRotate = false; canvas.classList.add('dragging'); });
  canvas.addEventListener('pointerup', () => {
    canvas.classList.remove('dragging');
    setTimeout(() => { controls.autoRotate = true; }, 2500);
  });

  proceduralGroup = new THREE.Group();
  scene.add(proceduralGroup);
  shaftSpin = new THREE.Group();
  proceduralGroup.add(shaftSpin);
}

function resize() {
  const container = document.getElementById('viewport-frame');
  const rect = container.getBoundingClientRect();
  const w = Math.max(rect.width, 100);
  const h = Math.max(rect.height, 100);
  if (!renderer) return;
  renderer.setSize(w, h, false);
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

// ---- geometry helpers (verbatim from reference) ----
function bladeDiskMesh({ x, hubR, bladeCount, bladeLen, bladeW, colorHex }) {
  const group = new THREE.Group();
  group.position.x = x;

  const hubGeo = new THREE.CylinderGeometry(hubR, hubR, 0.34, 20, 1, false);
  hubGeo.rotateZ(Math.PI / 2);
  const hubMat = new THREE.MeshStandardMaterial({ color: 0x8b9198, metalness: 0.75, roughness: 0.35 });
  group.add(new THREE.Mesh(hubGeo, hubMat));

  const bladeMat = new THREE.MeshStandardMaterial({
    color: colorHex, metalness: 0.4, roughness: 0.45, emissive: 0x010101, emissiveIntensity: 0, side: THREE.DoubleSide
  });
  const meshes = [];
  for (let i = 0; i < bladeCount; i++) {
    const a = (i / bladeCount) * Math.PI * 2;
    const bGeo = new THREE.BoxGeometry(0.035, bladeLen, bladeW);
    const mat = bladeMat.clone();
    const pivot = new THREE.Group();
    pivot.rotation.x = a;
    const inner = new THREE.Mesh(bGeo, mat);
    inner.position.y = hubR;
    inner.rotation.z = 0.3;
    pivot.add(inner);
    group.add(pivot);
    meshes.push(mat);
  }
  return { group, mats: meshes, hubMat };
}

function ringMesh(x, r1, r2, colorHex, opacity) {
  const geo = new THREE.RingGeometry(r1, r2, 40);
  const mat = new THREE.MeshBasicMaterial({ color: colorHex, transparent: true, opacity: opacity, side: THREE.DoubleSide });
  const m = new THREE.Mesh(geo, mat);
  m.rotation.y = Math.PI / 2;
  m.position.x = x;
  return { mesh: m, mat };
}

function buildProceduralEngine() {
  // central shaft
  const shaftGeo = new THREE.CylinderGeometry(0.16, 0.16, 8.6, 16);
  shaftGeo.rotateZ(Math.PI / 2);
  const shaftMat = new THREE.MeshStandardMaterial({ color: 0x555b61, metalness: 0.7, roughness: 0.4 });
  const shaft = new THREE.Mesh(shaftGeo, shaftMat);
  shaft.position.x = -0.7;
  shaftSpin.add(shaft);

  // spinner nose cone
  const noseGeo = new THREE.ConeGeometry(0.42, 0.9, 20);
  noseGeo.rotateZ(-Math.PI / 2);
  const noseMat = new THREE.MeshStandardMaterial({ color: 0x9aa0a6, metalness: 0.6, roughness: 0.35 });
  const nose = new THREE.Mesh(noseGeo, noseMat);
  nose.position.x = -5.0;
  shaftSpin.add(nose);

  // FAN
  const fan = bladeDiskMesh({ x: -4.45, hubR: 0.42, bladeCount: 20, bladeLen: 1.55, bladeW: 0.62, colorHex: NEUTRAL });
  shaftSpin.add(fan.group);
  stages.fan.push(...fan.mats, fan.hubMat);

  // COMPRESSOR (3 disks)
  const comp1 = bladeDiskMesh({ x: -3.15, hubR: 0.44, bladeCount: 26, bladeLen: 1.02, bladeW: 0.34, colorHex: NEUTRAL });
  const comp2 = bladeDiskMesh({ x: -2.45, hubR: 0.46, bladeCount: 30, bladeLen: 0.84, bladeW: 0.28, colorHex: NEUTRAL });
  const comp3 = bladeDiskMesh({ x: -1.85, hubR: 0.47, bladeCount: 32, bladeLen: 0.66, bladeW: 0.24, colorHex: NEUTRAL });
  [comp1, comp2, comp3].forEach(c => { shaftSpin.add(c.group); stages.compressor.push(...c.mats, c.hubMat); });

  // COMBUSTOR (liner + glow rings)
  const combGroup = new THREE.Group();
  combGroup.position.x = -1.05;
  const combGeo = new THREE.CylinderGeometry(0.58, 0.5, 1.05, 24, 1, true);
  combGeo.rotateZ(Math.PI / 2);
  const combMat = new THREE.MeshStandardMaterial({
    color: 0x707680, metalness: 0.3, roughness: 0.6, transparent: true, opacity: 0.55,
    side: THREE.DoubleSide, emissive: 0x010101, emissiveIntensity: 0
  });
  const combMesh = new THREE.Mesh(combGeo, combMat);
  combGroup.add(combMesh);
  const glowRings = [];
  for (let i = 0; i < 3; i++) {
    const rg = ringMesh(-0.32 + i * 0.32, 0.1, 0.5, NEUTRAL, 0.0);
    combGroup.add(rg.mesh);
    glowRings.push(rg.mat);
  }
  proceduralGroup.add(combGroup);
  // combMat is first — used as the opacity-controlled liner in selection logic
  stages.combustor.push(combMat, ...glowRings);

  // TURBINE (2 disks)
  const turb1 = bladeDiskMesh({ x: -0.15, hubR: 0.42, bladeCount: 24, bladeLen: 0.6, bladeW: 0.3, colorHex: NEUTRAL });
  const turb2 = bladeDiskMesh({ x: 0.42, hubR: 0.4, bladeCount: 22, bladeLen: 0.72, bladeW: 0.32, colorHex: NEUTRAL });
  [turb1, turb2].forEach(t => { shaftSpin.add(t.group); stages.turbine.push(...t.mats, t.hubMat); });

  // exhaust nozzle
  const nozGeo = new THREE.CylinderGeometry(0.55, 0.32, 1.4, 20, 1, true);
  nozGeo.rotateZ(Math.PI / 2);
  const nozMat = new THREE.MeshStandardMaterial({ color: 0x74797f, metalness: 0.55, roughness: 0.4, transparent: true, opacity: 0.6, side: THREE.DoubleSide });
  const nozzle = new THREE.Mesh(nozGeo, nozMat);
  nozzle.position.x = 1.55;
  proceduralGroup.add(nozzle);

  // outer nacelle (lathe profile — verbatim)
  const nacelleProfile = [
    new THREE.Vector2(0.38, -2.50), new THREE.Vector2(0.45, -2.40), new THREE.Vector2(0.58, -2.25),
    new THREE.Vector2(0.72, -2.05), new THREE.Vector2(0.88, -1.80), new THREE.Vector2(1.05, -1.50),
    new THREE.Vector2(1.22, -1.20), new THREE.Vector2(1.38, -0.80), new THREE.Vector2(1.50, -0.40),
    new THREE.Vector2(1.62, 0.00), new THREE.Vector2(1.72, 0.50), new THREE.Vector2(1.80, 1.00),
    new THREE.Vector2(1.86, 1.50), new THREE.Vector2(1.90, 2.00), new THREE.Vector2(1.94, 2.50),
    new THREE.Vector2(1.98, 3.00), new THREE.Vector2(2.02, 3.40), new THREE.Vector2(2.05, 3.80),
    new THREE.Vector2(2.08, 4.20), new THREE.Vector2(2.10, 4.55), new THREE.Vector2(2.15, 4.80),
    new THREE.Vector2(2.05, 5.00), new THREE.Vector2(1.60, 5.20),
  ];
  const nacGeo = new THREE.LatheGeometry(nacelleProfile, 32);
  nacGeo.rotateZ(Math.PI / 2);
  const nacMat = new THREE.MeshBasicMaterial({ color: 0xaabbcc, wireframe: true, transparent: true, opacity: 0.06 });
  proceduralGroup.add(new THREE.Mesh(nacGeo, nacMat));

  // secondary inner cowl wireframe
  const innerProfile = [
    new THREE.Vector2(0.50, -0.90), new THREE.Vector2(0.58, -0.40), new THREE.Vector2(0.66, 0.20),
    new THREE.Vector2(0.72, 0.80), new THREE.Vector2(0.76, 1.40), new THREE.Vector2(0.78, 2.00),
    new THREE.Vector2(0.76, 2.60), new THREE.Vector2(0.72, 3.20), new THREE.Vector2(0.65, 3.60),
  ];
  const innerGeo = new THREE.LatheGeometry(innerProfile, 20);
  innerGeo.rotateZ(Math.PI / 2);
  const innerMat = new THREE.MeshBasicMaterial({ color: 0x8899aa, wireframe: true, transparent: true, opacity: 0.04 });
  proceduralGroup.add(new THREE.Mesh(innerGeo, innerMat));

  // axis guideline
  const axisGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-6, 0, 0), new THREE.Vector3(4, 0, 0)]);
  const axisMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.06 });
  proceduralGroup.add(new THREE.Line(axisGeo, axisMat));
}

// Map a 0..1 health value to a HUD colour using the same bands as the panels:
// >=0.90 green, 0.85-0.90 yellow, <0.85 red.
function healthToHex(value) {
  if (value >= 0.90) return STAGE_COLOR_GREEN;
  if (value >= 0.85) return HEALTH_YELLOW;
  return HEALTH_RED;
}

/**
 * Colour a stage by its REAL health value. This is the base tint; a selected
 * stage temporarily overrides it with the bright green highlight but restores
 * to this tint on deselect.
 * @param {'fan'|'compressor'|'combustor'|'turbine'} stage
 * @param {number} value 0..1 health
 */
export function setStageHealth(stage, value) {
  if (!stages[stage]) return;
  stageHealthHex[stage] = healthToHex(value);
  // Repaint only if this stage is not the currently-selected (highlighted) one.
  if (activeStage !== stage) paintStage(stage);
}

// Apply the resolved colour/emissive for one stage based on selection state.
function paintStage(stage) {
  const isSelected = activeStage === stage;
  const isNone = activeStage === null;
  const baseHex = stageHealthHex[stage];

  // Selection semantics preserved from the reference: selected = bright green,
  // others dimmed when something is selected, otherwise show their own tint.
  const color = isSelected ? STAGE_COLOR_GREEN : (isNone ? baseHex : NEUTRAL_DIM);
  const emissive = isSelected ? GREEN_EMISSIVE : 0x010101;
  const emissiveIntensity = isSelected ? 0.4 : 0;

  stages[stage].forEach(mat => {
    if (mat.color) mat.color.setHex(color);
    if ('emissive' in mat) { mat.emissive.setHex(emissive); mat.emissiveIntensity = emissiveIntensity; }
    // Combustor liner is the first material — vary its opacity like the reference.
    if (stage === 'combustor' && mat === stages.combustor[0] && mat.transparent) {
      mat.opacity = isSelected ? 0.85 : (isNone ? 0.55 : 0.2);
    }
  });
}

/**
 * Highlight one stage (toggle). Pass null to clear. Repaints all stages so the
 * others dim/restore correctly. Returns the resolved active stage.
 */
export function selectStage(stage) {
  activeStage = (activeStage === stage) ? null : stage;
  Object.keys(stages).forEach(paintStage);
  if (onStageSelect) onStageSelect(activeStage);
  return activeStage;
}

export function getActiveStage() {
  return activeStage;
}

// ---- animation loop (cosmetic motion only) ----
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  controls.update();
  shaftSpin.rotation.x += dt * 2.4;   // cosmetic engine spin
  renderer.render(scene, camera);
}
