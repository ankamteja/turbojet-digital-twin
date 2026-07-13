# 8. The Dashboard

File 7 built the backend API — the menu of requests that hands out `EngineState`
objects. This file covers the **dashboard**: the web page that asks for those
objects and paints them onto a live 3D engine.

The code is in `src/frontend/`: `index.html`, `js/api.js`, `js/controls.js`,
`js/scene.js`, `js/dashboard.js`, and `js/main.js`.

## What is a frontend, in plain language?

The **frontend** is everything that runs in your web browser: the layout you see,
the 3D graphics, the buttons. It's written in **HTML** (the page structure),
**CSS** (the styling), and **JavaScript** (the behavior). It holds no model and
no data of its own — it *fetches* data from the backend and displays it.

Analogy: the frontend is the cockpit instrument panel. The gauges don't generate
the readings; they receive signals and show them. Our "signals" are `EngineState`
objects arriving from the API.

Every number on the panel comes from real backend data. The only motion that is
purely cosmetic — a decorative airflow wave, the slowly spinning 3D shaft,
auto-rotate — is clearly marked as such in the code and drives no gauge
(`src/frontend/js/dashboard.js:2`, `src/frontend/js/scene.js:8`).

## The layout (index.html)

`index.html` defines the panels of the HUD (heads-up display). Reading the file
top to bottom, the screen is divided into columns:

- **Left column** — stage selector, the **Telemetry Feed** controls (engine
  dropdown, cycle slider, play button), an alert ring, and a cycle log
  (`src/frontend/index.html:44`).
- **Center** — the 3D engine canvas (`src/frontend/index.html:96`).
- **Right column** — a Performance grid, a Stage Detail panel, and a Health Trend
  chart (`src/frontend/index.html:112`).
- **Bottom row** — thermal map, airflow, structural radar, and a system log
  (`src/frontend/index.html:154`).

The 3D library, **Three.js**, is pulled in from a CDN via an import map
(`src/frontend/index.html:16`).

## Fetching data (api.js)

`api.js` is a thin set of wrappers around the four backend endpoints. It just does
the HTTP round-trip and hands back the parsed JSON (`src/frontend/js/api.js:19`):

```javascript
export function getEngines()          { return getJSON('/api/engines'); }
export function getState(id, cycle)   { return getJSON(`/api/engine/${id}/state?cycle=${cycle}`); }
export function getHistory(id)        { return getJSON(`/api/engine/${id}/history`); }
export function getSimulate(id, toCycle){ return getJSON(`/api/engine/${id}/simulate?to_cycle=${toCycle}`); }
```

The backend runs on port 8000; the frontend is served separately, so the origin
is hard-coded (`src/frontend/js/api.js:8`). If a request fails, the error is
surfaced in the system log so a dead backend is visible rather than silent
(`src/frontend/js/main.js:20`).

## Orchestration (main.js)

`main.js` is the conductor. On boot it: loads the engine list, initializes the 3D
scene, initializes the controls, and wires the stage buttons
(`src/frontend/js/main.js:52`). The crucial loop is: whenever the engine or cycle
changes, fetch that state and render it (`src/frontend/js/main.js:20`):

```javascript
async function loadState(engineId, cycle) {
  const state = await getState(engineId, cycle);
  dashboard.render(state);
}
```

That single line is the whole data flow of the UI: **ask the API for one
`EngineState`, hand it to the renderer.** Selecting a stage additionally triggers
a projection fetch, described at the end of this file
(`src/frontend/js/main.js:39`).

## The controls: engine selector and cycle play (controls.js)

`controls.js` owns "which engine, which cycle" and calls back into `main.js` when
either changes. It's pure UI plumbing — no data loading of its own.

- The **engine dropdown** is filled from `/api/engines`; picking a new engine
  resets the view to cycle 1 (`src/frontend/js/controls.js:44`).
- The **cycle slider** scrubs through the engine's life from cycle 1 to its max
  (`src/frontend/js/controls.js:55`).
- **Play** auto-advances one cycle per second, looping back to cycle 1 at the end
  — so you can watch an engine age in real time
  (`src/frontend/js/controls.js:63`):

```javascript
function step() {
  cycle = cycle >= maxCycle ? 1 : cycle + 1;
  emitCycle();               // -> triggers a fresh /state fetch + render
}
```

Every "step" fires another `/state?cycle=n` request, so pressing Play makes the
whole HUD animate through the engine's degradation using real predictions at each
cycle.

## The 3D engine (scene.js)

`scene.js` builds a procedural turbojet with **Three.js** — a fan, three
compressor disks, a combustor liner, two turbine disks, a shaft and nacelle. The
important part for us is that each stage's *color* is driven by real health.

`dashboard.js` pushes per-component health into the scene
(`src/frontend/js/dashboard.js:118`):

```javascript
scene.setStageHealth('compressor', h.compressor.value);
scene.setStageHealth('combustor',  h.combustor.value);
scene.setStageHealth('turbine',    h.turbine.value);
scene.setStageHealth('fan',        h.overall.value);  // no fan head → use overall
```

`scene.js` maps that 0–1 value to a color using the same bands as every other
panel (`src/frontend/js/scene.js:256`):

```javascript
function healthToHex(value) {
  if (value >= 0.90) return STAGE_COLOR_GREEN;
  if (value >= 0.85) return HEALTH_YELLOW;
  return HEALTH_RED;
}
```

So a degrading turbine literally turns yellow, then red, on the 3D model. Clicking
a stage (or its left-column button) highlights it and fills the Stage Detail
panel (`src/frontend/js/scene.js:302`).

## Mapping EngineState to the HUD (dashboard.js)

`dashboard.js` is where one `EngineState` becomes every panel on screen. Its
top-level `render` fans the state out to a dozen small painters
(`src/frontend/js/dashboard.js:35`). A few concrete mappings, all from real
fields:

- **Performance grid** (`src/frontend/js/dashboard.js:52`) — thrust shown in kN
  (`thrust_n / 1000`), fuel in kg/h (`fuel_flow_kg_s * 3600`), EGT in °C
  (`t4_k − 273.15`), spool speed from `n_pct`.
- **Overall health** sets the status text and alert ring color — NOMINAL / CAUTION
  / WARNING by the same 0.90 / 0.85 bands (`src/frontend/js/dashboard.js:91`).
- **Health Trend** draws the `history` array of overall-health points as a
  polyline (`src/frontend/js/dashboard.js:200`).
- **Structural radar** plots the four healths on four axes
  (`src/frontend/js/dashboard.js:310`).
- **Thermal map** heights come from the three real station temperatures
  (`src/frontend/js/dashboard.js:329`).
- **System log** shows real `alerts`, or each component's recommendation when
  there are none (`src/frontend/js/dashboard.js:364`).

### The Stage Detail drill-down

Selecting a stage shows the full model story for that component — health,
confidence, RUL and recommendation, plus the **contributing** engineered ratios
the backend surfaced (`src/frontend/js/dashboard.js:131`):

```javascript
const comp = lastState.health[key];       // one ComponentHealth
...
const rul = comp.rul_cycles == null ? '∞' : `${comp.rul_cycles} cyc`;
// renders value, confidence, RUL, recommendation, and comp.contributing rows
```

This is the direct payoff of files 4–6: the confidence (ensemble spread), the RUL
(fitted degradation line) and the contributing sensors (feature importances) all
land in one readable panel.

## The projected-degradation drill-down (/simulate)

The cycle slider only covers cycles the engine has actually run. To look *further
ahead*, the dashboard calls the `/simulate` endpoint (file 7), which extrapolates
the fitted degradation line beyond the last observed cycle
(`src/frontend/js/api.js:34`):

```javascript
export function getSimulate(id, toCycle) {
  return getJSON(`/api/engine/${id}/simulate?to_cycle=${toCycle}`);
}
```

This is wired into stage selection. When you pick a stage, `main.js` calls
`dashboard.projectStage` (`src/frontend/js/main.js:45`), which fetches a
projection 20 cycles past the current horizon and draws that component's future
health on the Health Trend chart (`src/frontend/js/dashboard.js:259`):

```javascript
export async function projectStage(engineId, stage) {
  projStage = stage === 'fan' ? 'overall' : stage;
  const sim = await getSimulate(engineId, maxCycle + 20);
  const comp = sim.components[projStage];
  ...
}
```

The chart shares one cycle→x mapping between the observed line and the projection,
so the projection continues seamlessly from where real data ends. The **future**
portion (beyond the last observed cycle) is drawn as a *dashed* line to signal "this
is a projection, not a measurement" (`src/frontend/js/dashboard.js:282`). The
backend supplies actual predictions where data exists and the linear extrapolation
beyond it (`src/backend/service.py:205`).

### The confidence band

Alongside the projection, the trend chart draws a **confidence band** — a faint
shaded region around the overall-health line whose half-width scales with
uncertainty, i.e. `1 − confidence` (`src/frontend/js/dashboard.js:225`):

```javascript
const uncertainty = Math.max(0, Math.min(1, 1 - conf));
const halfW = uncertainty * 14;   // wider band = less certain
```

Lower confidence (from a wider bootstrap-ensemble spread, file 6) produces a wider
band. So the same uncertainty the trees expressed as disagreement becomes a
visible cone on the dashboard — the reading and its trustworthiness shown together.

## Recap

- The **frontend** runs in the browser (HTML/CSS/JS) and only *displays* data; it
  fetches `EngineState` from the backend and paints it.
- **`api.js`** wraps the four endpoints; **`main.js`** fetches a state on every
  engine/cycle change and hands it to the renderer.
- **`controls.js`** provides the engine selector and the cycle **Play** loop that
  animates an engine aging in real time.
- **`scene.js`** colors each 3D stage by its real health (green → yellow → red).
- **`dashboard.js`** maps one `EngineState` onto every gauge, chart and log, and
  the **Stage Detail** drill-down surfaces health, confidence, RUL and the
  contributing sensors.
- Selecting a stage triggers the **`/simulate`** drill-down, which draws that
  component's projected future degradation as a dashed line, alongside a
  **confidence band** whose width grows with the ensemble's uncertainty.

Next: **file 9**, which zooms out to show how all nine pieces connect end to end.
