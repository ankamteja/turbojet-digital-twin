// test_controls.mjs — behaviour of the engine/cycle/play controls widget.
//
// Run: node --test tests/frontend/
//
// controls.js owns "current engine / current cycle" state and is pure event
// plumbing (see its header comment) — no fetch, no rendering — which is what
// makes it testable against a hand-rolled DOM stub instead of a real browser.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { makeControlsDom } from './helpers/dom_stub.mjs';

const ENGINES = [
  { engine_id: 1, max_cycle: 30 },
  { engine_id: 2, max_cycle: 20 },
];

async function loadControls() {
  // Fresh module instance per test: controls.js keeps its state in module-
  // level closures, and importing it twice from the same URL would hand back
  // the cached instance instead of a clean one.
  const mod = await import(`../../src/frontend/js/controls.js?t=${Date.now()}-${Math.random()}`);
  return mod.initControls;
}

test('opens on the engine\'s latest cycle, not cycle 1', async () => {
  const dom = makeControlsDom();
  const initControls = await loadControls();
  const calls = [];
  const api = initControls({
    engines: ENGINES,
    onEngineChange: () => {},
    onCycleChange: (id, cycle) => calls.push([id, cycle]),
  });

  assert.equal(api.getCycle(), 30, 'a fresh digital twin should open at the aged end of the data');
  assert.equal(dom.cycleSlider.value, '30');
  assert.equal(dom.cycleLabel.textContent, '30 / 30');
  assert.deepEqual(calls.at(-1), [1, 30]);
});

test('switching engines resets to the new engine\'s latest cycle', async () => {
  const dom = makeControlsDom();
  const initControls = await loadControls();
  const api = initControls({ engines: ENGINES, onEngineChange: () => {}, onCycleChange: () => {} });

  dom.engineSelect.value = '2';
  dom.engineSelect.dispatch('change');

  assert.equal(api.getEngineId(), 2);
  assert.equal(api.getCycle(), 20, 'engine 2 tops out at cycle 20, not engine 1\'s 30');
  assert.equal(dom.cycleSlider.max, 20);
});

test('scrubbing the slider pauses playback', async () => {
  const dom = makeControlsDom();
  const initControls = await loadControls();
  initControls({ engines: ENGINES, onEngineChange: () => {}, onCycleChange: () => {} });

  dom.playBtn.dispatch('click'); // start playing
  assert.equal(dom.playBtn.textContent, '❚❚ PAUSE');

  dom.cycleSlider.value = '5';
  dom.cycleSlider.dispatch('input');

  assert.equal(dom.playBtn.textContent, '▶ PLAY', 'manual scrub should hand control back to the user');
});

test('play from the end replays the engine\'s life from cycle 1', async () => {
  const dom = makeControlsDom();
  const initControls = await loadControls();
  const api = initControls({ engines: ENGINES, onEngineChange: () => {}, onCycleChange: () => {} });

  assert.equal(api.getCycle(), 30); // opened at the end
  dom.playBtn.dispatch('click');
  assert.equal(api.getCycle(), 1, 'pressing play at the end should restart, not stay stuck at max');
  assert.equal(dom.playBtn.textContent, '❚❚ PAUSE');

  // Playing started a setInterval; clear it or the process hangs at exit.
  dom.playBtn.dispatch('click');
});
