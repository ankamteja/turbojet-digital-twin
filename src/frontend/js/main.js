// main.js — bootstrap + orchestration.
// Wires the API, the 3D scene, the dashboard renderer and the controls together:
//   1. load engine list
//   2. init the 3D scene (with stage-click callback)
//   3. init controls (dropdown / slider / play)
//   4. on every engine/cycle change -> fetch state -> render dashboard
//   5. wire the left-column stage buttons to the 3D highlight + stage detail

import { getEngines, getState, LOCAL } from './api.js';
import * as scene from './scene.js';
import * as dashboard from './dashboard.js';
import { initControls } from './controls.js';

// The active controls handle — lets the stage-selection path know which engine
// to request a projection for.
let controls = null;

// "start uvicorn :8000" is a dev instruction — meaningless, and slightly
// alarming, to a visitor on the public dashboard who has no backend to
// start. Same failure, different message depending on where we're running.
const BACKEND_DOWN_MSG = LOCAL
  ? 'BACKEND UNREACHABLE — start uvicorn :8000'
  : 'Backend unreachable — it may be waking from a cold start; try reloading in a minute';

// Fetch one state snapshot and paint it. Errors are surfaced in the system log
// so a dead backend is visible rather than silent.
async function loadState(engineId, cycle) {
  try {
    const state = await getState(engineId, cycle);
    dashboard.render(state);
  } catch (err) {
    console.error('[JET] state fetch failed:', err);
    const box = document.getElementById('sysLog');
    if (box) box.innerHTML = `<div class="row flag"><span>${BACKEND_DOWN_MSG}</span><span>ERR</span></div>`;
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
  if (active) {
    dashboard.renderStageDetail(active);
    // Feature 1: fetch + draw the projected future degradation drill-down for
    // the selected stage on the Health Trend chart.
    if (controls) dashboard.projectStage(controls.getEngineId(), active);
  } else {
    dashboard.clearStageDetail();
    dashboard.clearProjection();
  }
}

async function boot() {
  // 1. engines
  let engines;
  try {
    engines = await getEngines({
      // The backend may be cold (Render free tier sleeps after ~15 min idle);
      // getEngines retries for up to ~30-60s. Without this, that whole window
      // looks blank to a visitor instead of "loading, hang on."
      onRetry: (attempt, total) => {
        const box = document.getElementById('sysLog');
        if (box) box.innerHTML = `<div class="row"><span>waking backend (${attempt}/${total}) — cold start on free hosting, ~30-60s</span><span>...</span></div>`;
      },
    });
  } catch (err) {
    console.error('[JET] cannot reach backend:', err);
    const box = document.getElementById('sysLog');
    if (box) box.innerHTML = `<div class="row flag"><span>${BACKEND_DOWN_MSG}</span><span>ERR</span></div>`;
    return;
  }
  if (!engines.length) return;

  // 2. 3D scene — notify us on programmatic stage selection.
  scene.init(onStageSelected);

  // 3. controls own current engine/cycle; they call back on change.
  controls = initControls({
    engines,
    onEngineChange: (engineId) => {
      // If a stage is selected, re-project against the newly selected engine.
      const active = scene.getActiveStage();
      if (active) dashboard.projectStage(engineId, active);
    },
    onCycleChange: (engineId, cycle) => loadState(engineId, cycle),
  });

  // 4. left-column stage buttons drive the same selection path as the 3D model.
  document.getElementById('stageSelect').addEventListener('click', (e) => {
    const btn = e.target.closest('.stage-btn');
    if (btn) scene.selectStage(btn.dataset.stage);
  });
}

boot();
