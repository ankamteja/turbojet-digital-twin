# Architecture — Turbojet Digital Twin

End-to-end design and inter-module contracts. Build order: **model → backend → frontend**.
Every dashboard value traces to a trained model or a physics relation. No placeholders in the
final product.

---

## 0. Data facts (from `data/`)

- 10 engines × 30 cycles = 300 rows. `train.csv` 240, `test.csv` 60, `ground_truth.csv` 300.
- Split is **random cycles per engine** (both splits cover all 10 engines) → interpolation
  task, not future extrapolation. Ground truth exists for every cycle.
- Health degrades with cycle (monotone-ish). Ranges:
  Compressor 0.72–1.0, Combustor 0.89–1.0, Turbine 0.79–1.0, Overall 0.80–1.0.
  Thrust 13.3k–88.7k N, TSFC 0.0099–0.0381 g/N·s.
- **Physics anchor:** `TSFC_g_N_s ≈ 1000 · FuelFlow_kg_s / Thrust_N` (verified within ~1%).
  Couples the two performance targets — used as a hard-ish physics loss.

### Columns

Inputs (measured): `EngineID, Cycle, Altitude_m, Mach, Tamb_K, Pamb_Pa, RPM_rev_min,`
`FuelFlow_kg_s, P2_Pa, T2_K, P3_Pa, T3_K, P4_Pa, T4_K`
Stations: 2 = compressor inlet, 3 = compressor exit / combustor inlet, 4 = turbine inlet.

Targets: `CompressorHealth, CombustorHealth, TurbineHealth, OverallHealth` (0–1),
`Thrust_N, TSFC_g_N_s`.

---

## 1. Model (`src/model/`)

### Features

- **Drop `EngineID`** as a model input (identity, would leak; keep only as grouping key).
- **Keep `Cycle`** — real degradation driver.
- Raw sensors: `Altitude_m, Mach, Tamb_K, Pamb_Pa, RPM_rev_min, FuelFlow_kg_s, P2..T4` (12).
- **Derived physics features** (`physics.py`):
  - Compressor pressure ratio `PR_c = P3/P2`, temp ratio `TR_c = T3/T2`
  - Combustor `TR_b = T4/T3`, pressure drop `P4/P3`
  - Overall `TR = T4/T2`, `PR = P4/P2`
  - Corrected RPM `RPM/sqrt(T2)`, corrected fuel `FuelFlow/(P2·sqrt(T2))`
- Scale inputs with `StandardScaler`; scale Thrust with its own scaler (large magnitude).
  Health already 0–1 (sigmoid output, no scaling). TSFC derived from thrust+fuel.

### Network (`net.py`, PyTorch)

Shared MLP trunk → two heads:
- **Health head** → 4 outputs, `sigmoid` → [0,1].
- **Performance head** → `Thrust` (linear, scaled) ; TSFC derived via physics relation, plus a
  small learned residual head for correction.
- Dropout in trunk (feeds MC-dropout option). ~2–3 hidden layers, width 64–128. Small data,
  keep it compact + regularized (weight decay, early stop on val).

### Uncertainty

**Deep ensemble** — train N=5 nets with different seeds. Prediction = mean; confidence =
1 − normalized ensemble std. Cheap on this data, more robust than single-net MC-dropout.
(MC-dropout kept as fallback if ensemble too heavy.)

### Physics-informed losses (`physics.py`)

Added to base MSE (health + thrust):
1. **TSFC consistency** — `pred_TSFC` vs `1000·FuelFlow/pred_Thrust`. Ties thrust↔TSFC.
2. **Overall-health aggregation** — `OverallHealth` ≈ learned/weighted combo of the three
   subsystem healths; penalize divergence.
3. **Monotonic degradation** — within an engine, health should not increase with `Cycle`;
   penalize positive cycle-over-cycle deltas (soft hinge).
4. **Ratio ↔ health coupling** — degraded compressor achieves lower `PR_c` at given corrected
   RPM; soft penalty enforcing the correlation sign. (Weak weight — guardrail, not driver.)

Loss = `MSE_health + α·MSE_thrust + β·L_tsfc + γ·L_overall + δ·L_mono + ε·L_ratio`.
Weights tuned on validation split.

### Derived outputs (`predict.py`)

- **Degradation trajectory** — per engine, fit health(cycle) (linear or exponential decay) to
  predicted health over observed cycles.
- **RUL** — failure threshold on `OverallHealth` (default 0.80, configurable). Extrapolate the
  fitted trajectory, solve cycle where it crosses threshold; `RUL = cross_cycle − current_cycle`.
- **Recommendation** — rule table on health bands: >0.95 nominal / 0.90–0.95 monitor /
  0.85–0.90 inspect / <0.85 schedule maintenance. Per component.
- **Confidence** — from ensemble std per target.

### Artifacts

`src/model/artifacts/`: ensemble weights (`net_{i}.pt`), `scalers.pkl`, `config.json`,
`metrics.json`. Small — track in git (few hundred KB) so backend runs without retraining.

### Eval (`eval.py`)

On `test.csv` vs `ground_truth.csv`: per-target MAE / R². Report table for the technical report.

### Files

```
src/model/
  data.py       load csv, merge ground_truth, feature eng, scale, train/val split
  physics.py    derived features + physics loss terms + TSFC/RUL relations
  net.py        multi-head MLP
  train.py      ensemble training loop, early stop, save artifacts
  predict.py    load ensemble → mean+std, trajectory, RUL, recommendation
  eval.py       test metrics
  artifacts/    saved weights + scalers + config + metrics
```

---

## 2. Backend (`src/backend/`, FastAPI)

Loads ensemble + scalers once at startup (`service.py` wraps `model/predict.py`).
Precompute per-engine per-cycle states from `ground_truth`/complete dataset at boot; cache.
CORS open for frontend dev.

### Endpoints

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/health` | liveness `{status}` |
| GET | `/api/engines` | list of engine ids + max cycle |
| GET | `/api/engine/{id}/history` | all cycles: predicted 6 targets + uncertainty + ground truth |
| GET | `/api/engine/{id}/state?cycle=n` | full dashboard **EngineState** snapshot (schema below) |
| POST | `/api/predict` | body = one sensor row → EngineState (ad-hoc telemetry) |
| GET | `/api/engine/{id}/simulate?to_cycle=m` | projected future-cycle trajectory beyond data |

### Contract — `EngineState` (`schemas.py`, pydantic)

```json
{
  "engine_id": 1,
  "cycle": 10,
  "conditions": { "altitude_m": 7193.9, "mach": 0.14, "tamb_k": 240.1,
                  "pamb_pa": 39779, "rpm": 37691.7, "fuel_flow_kg_s": 0.302 },
  "health": {
    "compressor": { "value": 0.94, "confidence": 0.97, "trend": -0.004,
                    "rul_cycles": 42, "recommendation": "monitor",
                    "contributing": {"PR_c": 3.1, "TR_c": 1.28} },
    "combustor":  { ... },
    "turbine":    { ... },
    "overall":    { ... }
  },
  "performance": {
    "thrust_n": { "value": 21227, "confidence": 0.96 },
    "tsfc_g_n_s": { "value": 0.0143, "confidence": 0.95 }
  },
  "alerts": [ { "component": "compressor", "level": "warn", "msg": "..." } ],
  "history": [ { "cycle": 1, "overall": 0.99, "thrust_n": 21227 }, ... ]
}
```

Frontend depends only on this schema — model internals can change freely behind it.

### Files

```
src/backend/
  main.py       FastAPI app, routes, CORS, startup load
  schemas.py    pydantic contract models (EngineState, ...)
  service.py    wraps model/predict.py, builds EngineState, caches, alerts, simulate
```

---

## 3. Frontend (`src/frontend/`, React + Vite + R3F)

3D engine is the centerpiece. Data only from backend `EngineState`.

- **Engine3D** — R3F scene, 4 selectable meshes (compressor / combustor / turbine / overall
  casing). Mesh color mapped green→red by health. Click selects → drill-down.
- **DrillDown** — selected component: current health, degradation history chart, contributing
  sensor values, projected RUL, confidence, recommendation.
- **HealthPanel** — 4 health gauges, predicted thrust, TSFC, overall confidence.
- **Conditions** — live operating conditions (altitude, Mach, RPM, fuel flow, ambient).
- **Charts** — degradation trend + confidence band (Plotly or Chart.js).
- **Alerts** — auto health alerts from `EngineState.alerts`.
- **Controls** — engine selector + cycle slider / play button to step cycles (drives repeated
  `/state?cycle=n` calls → live-updating twin). Simulate button hits `/simulate`.

### Files

```
src/frontend/
  index.html, vite.config.js, package.json
  src/
    api.js               fetch wrappers → EngineState
    App.jsx              layout + state (selected engine/cycle/component)
    components/
      Engine3D.jsx  DrillDown.jsx  HealthPanel.jsx
      Conditions.jsx  Charts.jsx  Alerts.jsx  Controls.jsx
```

---

## 4. Milestones

1. **M1 Model** — `data.py`+`physics.py`+`net.py`+`train.py`+`eval.py`; ensemble trained,
   artifacts saved, test metrics table.
2. **M2 Backend** — FastAPI serving `EngineState` from artifacts; verify all endpoints.
3. **M3 Frontend** — Vite app, 3D engine + panels wired to backend; live cycle stepping.
4. **M4 Polish** — simulate/RUL projection, alerts, confidence bands, technical report + eval
   numbers, presentation.

## 5. Open decisions

- Uncertainty: **deep ensemble (N=5)** recommended — confirm vs MC-dropout.
- RUL failure threshold: default `OverallHealth = 0.80` — confirm.
- Charts lib: **Plotly** (richer, confidence bands) vs Chart.js (lighter).
- Track model artifacts in git (yes, small) vs regenerate on deploy.
