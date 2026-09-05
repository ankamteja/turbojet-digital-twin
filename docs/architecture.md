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
  Couples the two performance targets — TSFC is *derived* from predicted thrust, never a
  free output (hard constraint, see §1).

### Columns

Inputs (measured): `EngineID, Cycle, Altitude_m, Mach, Tamb_K, Pamb_Pa, RPM_rev_min,`
`FuelFlow_kg_s, P2_Pa, T2_K, P3_Pa, T3_K, P4_Pa, T4_K`
Stations: 2 = compressor inlet, 3 = compressor exit / combustor inlet, 4 = turbine inlet.

Targets: `CompressorHealth, CombustorHealth, TurbineHealth, OverallHealth` (0–1),
`Thrust_N, TSFC_g_N_s`.

---

## 1. Model (`src/model/`)

The surrogate is a **bootstrap ensemble of monotonic gradient-boosted trees**
(scikit-learn `HistGradientBoostingRegressor`), one ensemble per learned target. Trees were
chosen over a neural net because the data is small and tabular (240 rows) — they estimate the
subtle component-health signal far better (test health mean `R²` ~0.86 vs ~0.62 for an earlier
MLP) while staying interpretable and fast. Physics is enforced as **hard structural
constraints**, not soft penalties.

### Features

- **Drop `EngineID`** as a model input (identity, would leak; keep only as grouping key).
- **Keep `Cycle`** — real degradation driver.
- Raw sensors: `Altitude_m, Mach, Tamb_K, Pamb_Pa, RPM_rev_min, FuelFlow_kg_s, P2..T4` (12).
- **Derived physics features** (`add_physics_features`, `physics.py`) — 8 columns:
  - Compressor pressure ratio `PR_c = P3/P2`, temp ratio `TR_c = T3/T2`
  - Combustor `TR_b = T4/T3`, turbine-section `PR_t = P4/P3`
  - Overall `TR_overall = T4/T2`, `PR_overall = P4/P2`
  - Corrected RPM `RPM_corr = RPM/sqrt(T2)`, corrected fuel `Fuel_corr = FuelFlow/(P2·sqrt(T2))`
- Final feature vector = 13 raw + 8 derived = **21 dims**. **No scaling** — trees are
  scale-invariant, so there is no scaler to fit or leak.

### Base learner (`models.py`)

Each member is a `HistGradientBoostingRegressor`: `max_iter=400`, `learning_rate=0.05`,
`max_depth=3`, `l2_regularization=1.0`, `loss="squared_error"`. Shallow trees + L2 keep it
regularized on the small dataset.

### Hard physics constraints

1. **Monotonic degradation** — each health ensemble gets `monotonic_cst = -1` on `Cycle`, so
   predicted health can *never* rise as the engine ages. The boosting algorithm refuses any
   split that would violate this — rising health is structurally unrepresentable, not merely
   penalized.
2. **Derived TSFC** — TSFC is never learned. It is computed per ensemble member from predicted
   thrust via `TSFC = 1000·FuelFlow/Thrust` (`tsfc_from_fuel_thrust`, thrust floored at 1.0),
   so thrust and TSFC agree by construction and TSFC uncertainty is propagated from thrust.
3. **Overall-health aggregation** — `OverallHealth` is learned by its own ensemble, then
   checked against `mean(3 subsystem healths)` as an independent physics-consistency residual
   (`overall_aggregation_residual`) — a coherence test, reported, not a training penalty.

### Uncertainty

**Bootstrap ensemble** — `N=10` members per target, each fit on an independent bootstrap
resample of the training rows. Prediction = member mean; confidence = 1 − normalized ensemble
std (tight agreement = confident). No separate MC-dropout machinery needed.

### Interpretability

Permutation feature importance per target (`_compute_importances`, `importances.json`) — a
global explanation of which sensors/ratios drive each estimate. Health leans hardest on
`Cycle`; thrust on `FuelFlow_kg_s` and spool speed, matching physics.

### Derived outputs (`predict.py`)

- **Degradation trajectory** — per engine, fit health(cycle) (linear or exponential decay) to
  predicted health over observed cycles.
- **RUL** — failure threshold on `OverallHealth` (default 0.80, configurable). Extrapolate the
  fitted trajectory, solve cycle where it crosses threshold; `RUL = cross_cycle − current_cycle`.
- **Recommendation** — rule table on health bands: >0.95 nominal / 0.90–0.95 monitor /
  0.85–0.90 inspect / <0.85 schedule maintenance. Per component.
- **Confidence** — from ensemble std per target.

### Artifacts

`src/model/artifacts/` (git-ignored, regenerate via `python train.py`): the whole ensemble
serialized as a single `surrogate.joblib`, plus `config.json`, `metrics.json`,
`importances.json`, `generalization.json`. Loaded once by the backend at startup.

### Eval (`eval.py`, `generalize.py`)

- `eval.py` — on `test.csv` vs `ground_truth.csv`: per-target MAE / R², physics residuals,
  inference latency → `metrics.json`.
- `generalize.py` — **leave-one-engine-out** cross-validation (retrain on 9 engines, score the
  held-out one) → `generalization.json`. The harder *extrapolation* question the standard
  interpolation split can't answer.

### Files

```
src/model/
  data.py        load csv, merge ground_truth, feature eng, X/y split (no scaler — trees)
  physics.py     derived features + TSFC/aggregation/RUL relations (feature engineering)
  models.py      SurrogateEnsemble — boosted-tree bootstrap ensemble, hard constraints
  train.py       fit ensemble, save surrogate.joblib + config + importances
  predict.py     load ensemble → mean+std, trajectory, RUL, recommendation
  eval.py        test metrics + physics residuals + latency
  generalize.py  leave-one-engine-out generalization
  artifacts/     surrogate.joblib + config + metrics + importances + generalization (ignored)
```

---

## 2. Backend (`src/backend/`, FastAPI)

Loads the `surrogate.joblib` ensemble once at startup (`service.py` wraps `model/predict.py`).
Precompute per-engine per-cycle states from the complete dataset at boot; cache so dashboard
requests are constant-time lookups. CORS open for frontend dev. `main.py` caps
OMP/OPENBLAS/MKL/LOKY threads to 1 *before* importing sklearn — else the many small
startup predictions oversubscribe the CPU and boot stalls.

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
  "sensors": { "p2_pa": 294450, "t2_k": 455.2, "p3_pa": 274834, "t3_k": 2055.7,
               "p4_pa": 191656, "t4_k": 1842.1, "n_pct": 77.3 },
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
  main.py       FastAPI app, routes, CORS, lifespan model load
  schemas.py    pydantic contract models (EngineState, ...)
  service.py    wraps model/predict.py, builds EngineState, caches, alerts, simulate
```

---

## 3. Frontend (`src/frontend/`, vanilla JS + Three.js)

Plain ES modules — no framework, no build step; served as static files. Reuses the teammate's
HUD look (CSS/layout) but is driven entirely by the backend `EngineState`, with no placeholder
numbers. The 3D engine is the centerpiece.

- **`scene.js`** — procedural Three.js engine, 4 selectable stages (fan/compressor / combustor
  / turbine). `setStageHealth` colors a stage green→red by health; clicking a stage selects it.
- **`dashboard.js`** — maps `EngineState` onto the HUD: performance gauges (N1/N2, EGT, thrust,
  fuel flow), sensor readouts, system log, and the STAGE DETAIL drill-down (health, confidence,
  RUL, recommendation, contributing sensor ratios). `renderConfidenceBand` +
  `projectStage`/`drawProjection` draw the projected-degradation view from `/simulate`.
- **`controls.js`** — engine dropdown + cycle slider with Play/Pause that steps cycles, driving
  repeated `/state?cycle=n` calls into a live-updating twin.
- **`api.js`** — fetch wrappers (`getEngines`/`getState`/`getHistory`/`getSimulate`); backend
  origin hard-coded to `http://localhost:8000`.
- **`main.js`** — wires it together on load.

### Files

```
src/frontend/
  index.html          HUD markup
  styles.css          HUD styling (from teammate's UI, cleaned)
  js/
    api.js            fetch wrappers → EngineState
    scene.js          Three.js procedural engine + stage highlighting
    dashboard.js      EngineState → HUD panels + drill-down + projection/confidence band
    controls.js       engine selector + cycle play/pause
    main.js           bootstrap / wiring
```

---

## 4. Build order (as delivered)

1. **Model** — `data.py`+`physics.py`+`models.py`+`train.py`+`eval.py`; ensemble trained,
   `surrogate.joblib` saved, test-metrics table; `generalize.py` for LOEO.
2. **Backend** — FastAPI serving `EngineState` from the artifact; all endpoints verified.
3. **Frontend** — vanilla-JS HUD, 3D engine + panels wired to backend; live cycle stepping.
4. **Polish** — simulate/RUL projection, alerts, confidence bands, technical report with real
   eval numbers, concept docs, presentation.

## 5. Resolved decisions

- Model: **boosted-tree bootstrap ensemble (N=10)** — chosen over the earlier MLP after it
  raised health `R²` ~0.62→0.86.
- Uncertainty: **ensemble spread** (no MC-dropout).
- RUL failure threshold: **`OverallHealth = 0.80`**.
- No charting library — HUD panels drawn directly in the vanilla-JS frontend.
- Model artifacts **git-ignored**, regenerated via `train.py`.
