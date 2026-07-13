# Turbojet Digital Twin — Frontend

A vanilla-JS (ES modules) HUD dashboard that renders **real** health, performance
and RUL telemetry from the digital-twin backend over a procedural 3D turbojet.
No build step, no framework.

## Run

The frontend and backend are two separate servers.

1. **Start the backend** (serves the model on port 8000):

   ```bash
   cd ../backend
   uvicorn main:app --port 8000
   # wait until http://localhost:8000/api/health returns {"status":"ok",...}
   ```

2. **Serve the frontend** (any static server; it fetches the backend at
   `http://localhost:8000`):

   ```bash
   cd ../frontend          # this directory
   python -m http.server 8080
   ```

3. Open **http://localhost:8080/index.html** in a browser.

If the backend is down, the System Log panel shows `BACKEND UNREACHABLE`.

## Controls

- **Engine** dropdown — pick which engine's telemetry to view.
- **Cycle** slider — scrub through the engine's life (cycle 1 → max_cycle).
- **Play / Pause** — auto-advance ~1 s per cycle, looping the engine's life.
- **Stage buttons** (left) / **3D model** — select a stage to fill Stage Detail
  and highlight the mesh.

## File map

| File | Role |
| --- | --- |
| `index.html` | Markup (based on the teammate's HUD) + importmap; loads `styles.css` and `js/main.js`. |
| `styles.css` | Teammate's HUD CSS extracted verbatim, plus a controls block at the end. |
| `js/api.js` | Fetch wrappers: `getEngines`, `getState`, `getHistory`, `getSimulate`. |
| `js/scene.js` | Three.js procedural engine; exports `init`, `setStageHealth(stage,value)`, `selectStage(stage)`. |
| `js/dashboard.js` | Takes an `EngineState`, updates every DOM panel (metrics, health, stage detail, trend, radar, thermal, logs, alert ring). |
| `js/controls.js` | Engine dropdown + cycle slider + Play/Pause (auto-advance, looping). |
| `js/main.js` | Bootstrap: load engines, init scene, wire controls, orchestrate fetch → render. |

## Data mapping (backend → HUD)

- **Performance**: Thrust `thrust_n/1000` kN · Fuel `fuel_flow_kg_s*3600` kg/h ·
  EGT `t4_k-273.15` °C · N1 `n_pct` · N2 `n_pct*0.97` · Oil = cosmetic wobble
  (no real source).
- **Health bands** (dots, status text, alert ring, stage colours):
  `≥0.90` green/NOMINAL, `0.85–0.90` yellow/CAUTION, `<0.85` red/WARNING.
- **3D stages**: compressor/combustor/turbine coloured by their own health;
  fan by overall health (the model has no fan-specific head).
- **Stage Detail**: value, confidence %, RUL, recommendation and contributing
  sensor ratios for the selected component.
- **Health Trend** (renamed from "Vibration Trend"): `state.history` overall
  health polyline, scaled 0.70–1.0.
- **Structural radar**: 4 axes = compressor, combustor, turbine, overall health.
- **Thermal Map**: bar heights from station temps `t2/t3/t4` normalised over the
  gas-path range; turbine (t4) group reads hottest.
- **Cycle Log**: real rows from history (cycle → overall health, thrust kN).
- **System Log**: real `state.alerts`; when none, shows component recommendations.
- **Viewport SIG**: overall confidence × 100 %.
- Cosmetic ambient motion with no real counterpart is kept: airflow wave
  (`index.html`), shaft spin + auto-rotate (`js/scene.js`).
