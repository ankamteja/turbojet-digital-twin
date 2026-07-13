// main.js — bootstrap + orchestration.
// Wires the API, the 3D scene, the dashboard renderer and the controls together:
//   1. load engine list
//   2. init the 3D scene (with stage-click callback)
//   3. init controls (dropdown / slider / play)
//   4. on every engine/cycle change -> fetch state -> render dashboard
//   5. wire the left-column stage buttons to the 3D highlight + stage detail

import { getEngines, getState } from './api.js';
import * as scene from './scene.js';
import * as dashboard from './dashboard.js';
import { initControls } from './controls.js';

// Fetch one state snapshot and paint it. Errors are surfaced in the system log
// so a dead backend is visible rather than silent.
async function loadState(engineId, cycle) {
  try {
    const state = await getState(engineId, cycle);
    dashboard.render(state);
  } catch (err) {
    console.error('[JET] state fetch failed:', err);
    const box = document.getElementById('sysLog');
    if (box) box.innerHTML = `<div class="row flag"><span>BACKEND UNREACHABLE — start uvicorn :8000</span><span>ERR</span></div>`;
  }
}

// Sync the left-column stage buttons' .active class to the scene's selection.
function syncStageButtons(active) {
  document.querySelectorAll('.stage-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.stage === active);
  });
}

// Called when a stage is (de)selected — from a button click or the 3D scene.
function onStageSelected(active) {
  syncStageButtons(active);
  if (active) dashboard.renderStageDetail(active);
  else dashboard.clearStageDetail();
}

async function boot() {
  // 1. engines
  let engines;
  try {
    engines = await getEngines();
  } catch (err) {
    console.error('[JET] cannot reach backend:', err);
    const box = document.getElementById('sysLog');
    if (box) box.innerHTML = `<div class="row flag"><span>BACKEND UNREACHABLE — start uvicorn :8000</span><span>ERR</span></div>`;
    return;
  }
  if (!engines.length) return;

  // 2. 3D scene — notify us on programmatic stage selection.
  scene.init(onStageSelected);

  // 3. controls own current engine/cycle; they call back on change.
  initControls({
    engines,
    onEngineChange: () => { /* cycle change fires alongside; nothing extra */ },
    onCycleChange: (engineId, cycle) => loadState(engineId, cycle),
  });

  // 4. left-column stage buttons drive the same selection path as the 3D model.
  document.getElementById('stageSelect').addEventListener('click', (e) => {
    const btn = e.target.closest('.stage-btn');
    if (btn) scene.selectStage(btn.dataset.stage);
  });
}

boot();
