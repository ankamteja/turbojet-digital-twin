// controls.js — the interactive controls added on top of the reference HUD:
//   - engine dropdown  (choose which engine's telemetry to view)
//   - cycle slider     (scrub through the engine's life, cycle 1..max_cycle)
//   - play/pause       (auto-advance ~1s per cycle, looping the engine's life)
//
// The controls are pure UI plumbing: they own the "current engine / current
// cycle" state and call back into main.js whenever either changes. main.js does
// the fetch + render. This keeps data-loading out of the widget layer.

export function initControls({ engines, onEngineChange, onCycleChange }) {
  const engineSel = document.getElementById('engineSelect');
  const slider = document.getElementById('cycleSlider');
  const cycleLabel = document.getElementById('cycleLabel');
  const playBtn = document.getElementById('playBtn');

  let engineId = engines[0].engine_id;
  let maxCycle = engines[0].max_cycle;
  // Open on the engine's latest cycle — a digital twin should show the engine's
  // *current* (most-aged) state, not a brand-new one that always reads ~100%.
  // Scrub back or hit Play to replay its life from the start.
  let cycle = maxCycle;
  let playing = false;
  let timer = null;

  // Populate the engine dropdown from /api/engines.
  engineSel.innerHTML = engines
    .map(e => `<option value="${e.engine_id}">ENGINE ${String(e.engine_id).padStart(2, '0')}</option>`)
    .join('');

  function setCycleBounds() {
    slider.min = 1;
    slider.max = maxCycle;
  }

  function updateCycleLabel() {
    cycleLabel.textContent = `${cycle} / ${maxCycle}`;
  }

  // Apply the current cycle to the slider + label and notify main.js.
  function emitCycle() {
    slider.value = cycle;
    updateCycleLabel();
    onCycleChange(engineId, cycle);
  }

  // ---- engine selection ----
  engineSel.addEventListener('change', () => {
    engineId = parseInt(engineSel.value, 10);
    const info = engines.find(e => e.engine_id === engineId);
    maxCycle = info.max_cycle;
    cycle = maxCycle;              // show the new engine at its current (latest) cycle
    setCycleBounds();
    emitCycle();
    onEngineChange(engineId);
  });

  // ---- cycle scrubbing ----
  slider.addEventListener('input', () => {
    cycle = parseInt(slider.value, 10);
    // Manual scrub pauses auto-advance so the user keeps control.
    if (playing) stop();
    emitCycle();
  });

  // ---- play / pause ----
  function step() {
    // Loop the engine's life: wrap back to cycle 1 after the last cycle.
    cycle = cycle >= maxCycle ? 1 : cycle + 1;
    emitCycle();
  }

  function start() {
    playing = true;
    playBtn.textContent = '❚❚ PAUSE';
    playBtn.classList.add('playing');
    timer = setInterval(step, 1000);   // ~1s per cycle
  }

  function stop() {
    playing = false;
    playBtn.textContent = '▶ PLAY';
    playBtn.classList.remove('playing');
    if (timer) { clearInterval(timer); timer = null; }
  }

  playBtn.addEventListener('click', () => (playing ? stop() : start()));

  // ---- initial state ----
  setCycleBounds();
  emitCycle();

  return {
    getEngineId: () => engineId,
    getCycle: () => cycle,
  };
}
