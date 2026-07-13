// dashboard.js — render one EngineState into every HUD panel.
// Everything here is driven by REAL backend data; there are no Math.random
// generators feeding real fields. The only cosmetic motion (airflow wave,
// shaft spin, auto-rotate) lives in main.js / scene.js.

import * as scene from './scene.js';
import { getSimulate } from './api.js';

// Health bands shared across the whole HUD (matches backend _alert_for cuts).
//   >=0.90 green / NOMINAL, 0.85–0.90 yellow / CAUTION, <0.85 red / WARNING
function band(value) {
  if (value >= 0.90) return 'green';
  if (value >= 0.85) return 'yellow';
  return 'red';
}

// Pretty stage metadata reused from the reference for the Stage Detail panel.
// Only the health numbers are real; blade/material specs are static reference.
const STAGE_META = {
  fan:        { name: 'FAN' },
  compressor: { name: 'COMPRESSOR' },
  combustor:  { name: 'COMBUSTOR' },
  turbine:    { name: 'TURBINE' },
};

// Cache the latest state so stage-detail clicks can read component data without
// re-fetching.
let lastState = null;
export function currentState() { return lastState; }

/**
 * Top-level entry: push a full EngineState into the DOM.
 * @param {object} state EngineState JSON from the backend.
 */
export function render(state) {
  lastState = state;
  renderPerformance(state);
  renderHealthOverall(state);
  render3DStages(state);
  renderTrend(state);
  renderRadar(state);
  renderThermal(state);
  renderCycleLog(state);
  renderSystemLog(state);
  renderViewportHud(state);
  // Refresh the stage-detail panel for whatever stage is currently selected.
  const active = scene.getActiveStage();
  if (active) renderStageDetail(active);
}

// ---- Performance metrics (right column) ----
function renderPerformance(state) {
  const s = state.sensors;
  const perf = state.performance;
  const cond = state.conditions;

  // Mapping from spec:
  //   Thrust  = thrust_n / 1000            (kN)
  //   Fuel    = fuel_flow_kg_s * 3600      (kg/h)
  //   EGT     = t4_k - 273.15              (°C)
  //   N1      = n_pct
  //   N2      = n_pct * 0.97               (single-spool proxy)
  //   Oil     = cosmetic — no real source, small static jitter
  setMetric('m-thrust', (perf.thrust_n.value / 1000).toFixed(1));
  setMetric('m-fuel', (cond.fuel_flow_kg_s * 3600).toFixed(0));
  setMetric('m-egt', (s.t4_k - 273.15).toFixed(0));
  setMetric('m-n1', s.n_pct.toFixed(1));
  setMetric('m-n2', (s.n_pct * 0.97).toFixed(1));
  // Oil pressure has no telemetry source; keep a small deterministic wobble so
  // the gauge looks alive without pretending to be real data.
  setMetric('m-oil', (52 + Math.round(Math.sin(state.cycle) * 2)).toFixed(0));

  // EGT alert styling from turbine health band (turbine drives exhaust temp).
  const egtEl = document.getElementById('m-egt');
  egtEl.classList.remove('alert-yellow', 'alert-red');
  const tb = band(state.health.turbine.value);
  if (tb === 'yellow') egtEl.classList.add('alert-yellow');
  else if (tb === 'red') egtEl.classList.add('alert-red');

  // Top-bar throttle chip mirrors spool speed (real).
  const thrVal = document.getElementById('thrVal');
  if (thrVal) thrVal.textContent = `THR ${s.n_pct.toFixed(0)}%`;
}

function setMetric(id, text) {
  const el = document.getElementById(id);
  if (el) el.querySelector('.mv').textContent = text;
}

// ---- Overall health: dots, status text, alert ring ----
function renderHealthOverall(state) {
  const overall = state.health.overall.value;
  const b = band(overall);

  const dots = ['hDot1', 'hDot2', 'hDot3'].map(id => document.getElementById(id));
  dots.forEach(d => { if (d) d.className = 'health-dot'; });
  if (b === 'yellow') { if (dots[0]) dots[0].className = 'health-dot yellow'; }
  else if (b === 'red') dots.forEach(d => { if (d) d.className = 'health-dot red'; });

  const status = document.getElementById('healthStatus');
  const ring = document.getElementById('alertTriangle');
  if (b === 'green') {
    status.textContent = 'HEALTH: NOMINAL';
    status.style.color = '';
    ring.setAttribute('stroke', 'var(--ink)');
  } else if (b === 'yellow') {
    status.textContent = 'HEALTH: CAUTION';
    status.style.color = 'var(--yellow)';
    ring.setAttribute('stroke', 'var(--yellow)');
  } else {
    status.textContent = 'HEALTH: WARNING';
    status.style.color = 'var(--red)';
    ring.setAttribute('stroke', 'var(--red)');
  }
}

// ---- 3D stage colours from real per-component health ----
function render3DStages(state) {
  const h = state.health;
  // fan has no dedicated health signal -> use overall health for its tint.
  scene.setStageHealth('fan', h.overall.value);
  scene.setStageHealth('compressor', h.compressor.value);
  scene.setStageHealth('combustor', h.combustor.value);
  scene.setStageHealth('turbine', h.turbine.value);
}

// ---- Stage Detail panel (fills on stage select) ----
// Reads component data (value, confidence, RUL, recommendation, contributing)
// straight from the last state. 'fan' maps to overall since the model has no
// fan-specific health head.
export function renderStageDetail(stage) {
  if (!lastState) return;
  const key = stage === 'fan' ? 'overall' : stage;
  const comp = lastState.health[key];
  const label = document.getElementById('stageLabel');
  label.textContent = STAGE_META[stage].name + ' — ACTIVE';
  label.className = 'active';

  // Build contributing-sensor rows (real engineered ratios from the backend).
  const contribRows = Object.entries(comp.contributing).map(([k, v]) =>
    `<div class="drow"><span class="k">${k}</span><span class="v">${fmt(v)}</span></div>`
  ).join('');

  const rul = comp.rul_cycles == null ? '∞' : `${comp.rul_cycles} cyc`;
  document.getElementById('stageDetail').innerHTML = `
    <div class="drow"><span class="k">Selected</span><span class="v">${STAGE_META[stage].name}</span></div>
    <div class="drow"><span class="k">Health</span><span class="v">${(comp.value * 100).toFixed(1)} %</span></div>
    <div class="drow"><span class="k">Confidence</span><span class="v">${(comp.confidence * 100).toFixed(0)} %</span></div>
    <div class="drow"><span class="k">RUL</span><span class="v">${rul}</span></div>
    <div class="drow"><span class="k">Action</span><span class="v">${comp.recommendation}</span></div>
    ${contribRows}`;
}

// Reset the stage-detail panel to its empty state (no stage selected).
export function clearStageDetail() {
  const label = document.getElementById('stageLabel');
  label.textContent = 'NONE SELECTED';
  label.className = 'none';
  document.getElementById('stageDetail').innerHTML = `
    <div class="drow"><span class="k">Selected</span><span class="v">—</span></div>
    <div class="drow"><span class="k">Health</span><span class="v">-- %</span></div>
    <div class="drow"><span class="k">Confidence</span><span class="v">-- %</span></div>
    <div class="drow"><span class="k">RUL</span><span class="v">--</span></div>
    <div class="drow"><span class="k">Action</span><span class="v">--</span></div>`;
}

// Compact number formatting for the contributing ratios (they span 1e-7..1e3).
function fmt(v) {
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.01 || a >= 10000)) return v.toExponential(2);
  return v.toFixed(a < 10 ? 3 : 1);
}

// ---- Health Trend (was "Vibration Trend"): overall health polyline ----
// The SVG viewBox is 220x90 with the plot area x∈[20,220], y∈[6,82] and health
// scaled 0.70..1.0 across the vertical span so degradation is visible.
//
// Both the observed trend and the projected drill-down (Feature 1) share one
// cycle→x mapping so the dashed projection continues seamlessly from the solid
// observed line. The x-domain spans [minCycle, maxCycle], where maxCycle is
// stretched to the projection horizon while a stage is selected.
const PLOT = { x0: 20, x1: 220, yTop: 6, yBot: 82, lo: 0.70, hi: 1.0 };

// Current cycle domain for the trend chart, updated by renderTrend and reused
// by the projection so the two lines align on the same axis.
let trendDomain = { min: 1, max: 30 };

function healthToY(v) {
  const norm = Math.max(0, Math.min(1, (v - PLOT.lo) / (PLOT.hi - PLOT.lo)));
  return PLOT.yBot - norm * (PLOT.yBot - PLOT.yTop);
}

function cycleToX(cycle) {
  const { min, max } = trendDomain;
  if (max === min) return PLOT.x0;
  const t = (cycle - min) / (max - min);
  return PLOT.x0 + Math.max(0, Math.min(1, t)) * (PLOT.x1 - PLOT.x0);
}

function renderTrend(state) {
  const hist = state.history;
  if (!hist.length) return;

  // Domain: observed cycles, extended to the projection horizon if a stage is
  // selected (so the dashed future line has room). Recompute on every render.
  trendDomain.min = hist[0].cycle;
  const observedMax = hist[hist.length - 1].cycle;
  trendDomain.max = Math.max(observedMax, projHorizonCycle || observedMax);

  const pts = hist.map(p => `${cycleToX(p.cycle).toFixed(1)},${healthToY(p.overall).toFixed(1)}`);
  const line = document.getElementById('trendLine');
  line.setAttribute('points', pts.join(' '));
  const b = band(observedMax != null ? hist[hist.length - 1].overall : 1);
  line.setAttribute('stroke', b === 'red' ? 'var(--red)' : b === 'yellow' ? 'var(--yellow)' : 'var(--green-solid)');

  renderConfidenceBand(state, hist);
  // Keep any active stage projection aligned to the (possibly new) domain.
  if (projStage) drawProjection();
}

// ---- Feature 2: confidence band around the overall trend ----
// A faint shaded area whose half-width scales with uncertainty = 1 - confidence.
// Lower confidence => wider band. Rendered as a filled polygon (upper edge then
// lower edge reversed) behind the trend line.
function renderConfidenceBand(state, hist) {
  const poly = document.getElementById('trendBand');
  if (!poly) return;
  const conf = state.health.overall.confidence;
  const uncertainty = Math.max(0, Math.min(1, 1 - conf));
  // Constant half-width per point, in viewBox units, scaled by uncertainty.
  // Max ~14 units (~18% of the 76-unit plot height) at zero confidence.
  const halfW = uncertainty * 14;
  if (halfW < 0.01) { poly.setAttribute('points', ''); return; }

  const clampY = y => Math.max(PLOT.yTop, Math.min(PLOT.yBot, y));
  const upper = hist.map(p => {
    const x = cycleToX(p.cycle);
    return `${x.toFixed(1)},${clampY(healthToY(p.overall) - halfW).toFixed(1)}`;
  });
  const lower = hist.slice().reverse().map(p => {
    const x = cycleToX(p.cycle);
    return `${x.toFixed(1)},${clampY(healthToY(p.overall) + halfW).toFixed(1)}`;
  });
  poly.setAttribute('points', upper.concat(lower).join(' '));
  const b = band(hist[hist.length - 1].overall);
  poly.setAttribute('fill', b === 'red' ? 'rgba(255,59,59,0.10)' : b === 'yellow' ? 'rgba(255,204,0,0.10)' : 'rgba(80,255,120,0.10)');
}

// ---- Feature 1: projected future degradation drill-down ----
// When a stage is selected we fetch getSimulate(engine, maxCycle+20) and plot
// that component's projected health[] over cycles[]. The observed portion
// (cycle <= projected_from_cycle) reuses the solid trend styling; the future
// portion is drawn here as a DASHED line beyond the current data.
let projStage = null;            // selected stage key we are projecting, or null
let projData = null;             // last getSimulate component payload {health, cycles, projected_from_cycle}
let projHorizonCycle = null;     // furthest projected cycle (extends trend domain)

// Fetch + cache a projection for the selected stage, then draw it.
export async function projectStage(engineId, stage) {
  projStage = stage === 'fan' ? 'overall' : stage;
  try {
    const maxCycle = trendDomain.max || (lastState ? lastState.cycle : 1);
    const sim = await getSimulate(engineId, maxCycle + 20);
    const comp = sim.components[projStage];
    if (!comp) { clearProjection(); return; }
    projData = {
      cycles: sim.cycles,
      health: comp.health,
      projectedFrom: comp.projected_from_cycle,
    };
    projHorizonCycle = sim.cycles[sim.cycles.length - 1];
    // Re-run trend so the domain stretches to the horizon, then draw projection.
    if (lastState) renderTrend(lastState);
    else drawProjection();
  } catch (err) {
    console.error('[JET] projection fetch failed:', err);
    clearProjection();
  }
}

// Draw only the FUTURE (cycle > projected_from_cycle) portion as a dashed line.
function drawProjection() {
  const line = document.getElementById('projLine');
  if (!line || !projData) return;
  const { cycles, health, projectedFrom } = projData;
  const pts = [];
  for (let i = 0; i < cycles.length; i++) {
    if (cycles[i] <= projectedFrom) continue;   // observed part stays solid
    pts.push(`${cycleToX(cycles[i]).toFixed(1)},${healthToY(health[i]).toFixed(1)}`);
  }
  // Anchor the dashed line to the last observed projected point so it connects.
  const anchorIdx = cycles.findIndex(c => c > projectedFrom) - 1;
  if (anchorIdx >= 0) {
    pts.unshift(`${cycleToX(cycles[anchorIdx]).toFixed(1)},${healthToY(health[anchorIdx]).toFixed(1)}`);
  }
  line.setAttribute('points', pts.join(' '));
}

// Clear the projection (stage deselected) and collapse the trend domain back.
export function clearProjection() {
  projStage = null;
  projData = null;
  projHorizonCycle = null;
  const line = document.getElementById('projLine');
  if (line) line.setAttribute('points', '');
  if (lastState) renderTrend(lastState);
}

// ---- Structural radar: 4 axes = compressor, combustor, turbine, overall ----
function renderRadar(state) {
  const h = state.health;
  const vals = [h.compressor.value, h.combustor.value, h.turbine.value, h.overall.value];
  const cx = 60, cy = 60, maxR = 42;
  const pts = vals.map((v, i) => {
    const a = (i / vals.length) * Math.PI * 2 - Math.PI / 2;
    const r = Math.max(0, Math.min(1, v)) * maxR;   // 0..1 health -> radius
    return `${(cx + Math.cos(a) * r).toFixed(1)},${(cy + Math.sin(a) * r).toFixed(1)}`;
  });
  const poly = document.getElementById('radarPoly');
  poly.setAttribute('points', pts.join(' '));
  const b = band(h.overall.value);
  poly.setAttribute('fill', b === 'red' ? 'rgba(255,59,59,0.15)' : b === 'yellow' ? 'rgba(255,204,0,0.12)' : 'rgba(80,255,120,0.12)');
  poly.setAttribute('stroke', b === 'red' ? 'var(--red)' : b === 'yellow' ? 'var(--yellow)' : 'var(--green-solid)');
}

// ---- Thermal Map bars: driven by the three real station temps t2,t3,t4 ----
// The reference had 24 cosmetic bars; we render 3 groups (one per station),
// each bar height = temperature normalised into a plausible gas-path range.
function renderThermal(state) {
  const bars = document.querySelectorAll('#thermalBars i');
  if (!bars.length) return;
  const s = state.sensors;
  const temps = [s.t2_k, s.t3_k, s.t4_k];     // inlet, compressor exit, turbine
  // Normalise each temperature to a 0..1 fraction over a wide gas-path window
  // (ambient ~250 K up to ~2500 K hot section).
  const norm = t => Math.max(0.08, Math.min(1, (t - 250) / (2500 - 250)));
  const n = bars.length;
  const perGroup = Math.ceil(n / 3);
  bars.forEach((b, i) => {
    const group = Math.min(2, Math.floor(i / perGroup));
    const frac = norm(temps[group]);
    b.style.height = (frac * 100).toFixed(0) + '%';
    // Hotter station -> warmer colour. Turbine group (t4) reads hottest.
    const hot = frac > 0.6;
    b.style.background = hot
      ? 'linear-gradient(to top, rgba(255,120,60,0.25), rgba(255,90,40,0.65))'
      : 'linear-gradient(to top, rgba(255,255,255,0.05), rgba(255,255,255,0.16))';
  });
}

// ---- Cycle Log (left): real rows from history (cycle -> health, thrust) ----
// Newest cycle at the top; flag rows whose overall health drops below caution.
function renderCycleLog(state) {
  const box = document.getElementById('cycleLog');
  const rows = state.history.slice(-11).reverse();
  box.innerHTML = rows.map(p => {
    const flag = p.overall < 0.90 ? ' flag' : '';
    const kN = (p.thrust_n / 1000).toFixed(1);
    return `<div class="row${flag}"><span>CYC-${String(p.cycle).padStart(2, '0')}</span><span>${(p.overall * 100).toFixed(0)}% · ${kN}kN</span></div>`;
  }).join('');
}

// ---- System Log (bottom): real alerts, or recommendations when none ----
function renderSystemLog(state) {
  const box = document.getElementById('sysLog');
  let rows;
  if (state.alerts.length) {
    // Real alerts; style critical/warn as flagged rows.
    rows = state.alerts.map(a => {
      const flag = (a.level === 'critical' || a.level === 'warn') ? ' flag' : '';
      return `<div class="row${flag}"><span>${a.msg}</span><span>${a.level.toUpperCase()}</span></div>`;
    });
  } else {
    // No alerts -> surface each component's recommendation (still real data).
    rows = Object.entries(state.health).map(([key, c]) =>
      `<div class="row"><span>${key.toUpperCase()}: ${c.recommendation}</span><span>${(c.value * 100).toFixed(0)}%</span></div>`
    );
  }
  box.innerHTML = rows.join('');
}

// ---- Viewport HUD signal readout = overall confidence * 100 (%) ----
function renderViewportHud(state) {
  const sig = document.getElementById('sigVal');
  if (sig) sig.textContent = `${(state.health.overall.confidence * 100).toFixed(0)}%`;
}
