# 9. How It All Fits Together

Files 1–8 each covered one piece. This file zooms out to show the whole machine:
how a raw sensor reading travels all the way from a spinning engine to a colored
stage on the dashboard, with each stage consuming the output of the one before it.

## The one-paragraph story

A jet engine's health is hidden — you can't measure it in flight. So we read the
sensors we *can* measure, reshape them into physically meaningful ratios, feed
them to a bootstrap ensemble of physics-constrained boosted trees, and get back
health, thrust and fuel efficiency — each with a confidence from how much the
ensemble members agree. Physics guarantees the answers stay sane (health can only
fall with age; fuel efficiency is derived from thrust, not guessed). We fit each
engine's health trend and extrapolate it to a remaining-useful-life estimate and
a maintenance recommendation. A FastAPI backend packages all of this into a stable
`EngineState` object and serves it over the web. A browser dashboard fetches those
objects and paints them onto a live 3D engine you can scrub through cycle by cycle.

## The data-flow diagram

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                        MODEL  (src/model/)                            │
  │                                                                        │
  │  raw sensors            feature engineering        surrogate           │
  │  P2,T2,P3,T3,P4,T4,     add_physics_features()     SurrogateEnsemble   │
  │  RPM, FuelFlow,   ───▶  PR_c, TR_c, TR_b, PR_t,  ─▶  10 boosted-tree    │
  │  Cycle, Alt, Mach       corrected RPM/fuel …        members × 5 targets │
  │  (13 raw cols)          (+8 = 21 feature cols)      │                   │
  │                                                     ▼                   │
  │                                        ┌────────────────────────────┐   │
  │                                        │ HARD PHYSICS CONSTRAINTS   │   │
  │                                        │ • health ↓ with Cycle only │   │
  │                                        │ • TSFC = 1000·fuel/thrust  │   │
  │                                        └────────────┬───────────────┘   │
  │                                                     ▼                   │
  │   predictions + uncertainty         RUL + recommendation               │
  │   mean = prediction                 fit health(cycle) line,            │
  │   std across 10 members ─────────▶  extrapolate to 0.80 threshold,     │
  │   = confidence                      map health → action band           │
  └──────────────────────────────────────────┬───────────────────────────┘
                                              │  (precomputed & cached
                                              │   once at startup)
                                              ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │                     BACKEND  (src/backend/, FastAPI)                   │
  │   TwinService packs everything into the EngineState contract           │
  │   GET /api/engine/{id}/state?cycle=n  ──▶  { conditions, sensors,      │
  │                                              health, performance,      │
  │                                              alerts, history } (JSON)   │
  └──────────────────────────────────────────┬───────────────────────────┘
                                              │  HTTP + JSON
                                              ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │                    DASHBOARD  (src/frontend/, browser)                 │
  │   api.js fetches EngineState  ──▶  dashboard.js maps every field to a  │
  │   HUD panel; scene.js colors each 3D stage green→yellow→red by health; │
  │   controls.js scrubs/plays through cycles; /simulate projects ahead.   │
  └──────────────────────────────────────────────────────────────────────┘
```

## Stage by stage — how each consumes the last

**1. Raw sensors → features.** The engine gives 13 raw signals (station pressures
and temperatures, RPM, fuel flow, cycle, flight conditions). Absolute pressures
swing with altitude, so on their own they hide degradation. `add_physics_features`
(`src/model/physics.py:98`) reshapes them into 8 ratios and corrected quantities —
`PR_c = P3/P2`, corrected RPM, etc. — where wear shows up as a clean drift. Output:
21 feature columns. *(Files 2–3.)*

**2. Features → surrogate predictions.** Those 21 columns feed the
`SurrogateEnsemble` (`src/model/models.py:57`): for each of five learned targets
(4 healths + thrust), ten gradient-boosted-tree members each trained on a
bootstrap resample. Their **mean** is the prediction; their **spread** is the raw
uncertainty. *(File 4.)*

**3. Predictions → physics-guaranteed outputs.** Two hard constraints act here.
Health carries a monotonic-decreasing constraint on `Cycle`, so predicted health
can never rise as the engine ages. TSFC is never predicted — it's derived per
member from thrust via `TSFC = 1000·fuel/thrust`, so performance always stays
self-consistent and TSFC's uncertainty flows from thrust's. *(File 5.)*

**4. Outputs → confidence.** The spread across the ten members becomes a friendly
0–1 confidence via `_confidence` (`src/model/predict.py:44`) — tight agreement
near 1, wide disagreement near 0. Each dashboard number now arrives paired with a
trust level. *(File 6.)*

**5. Confidence-tagged health → RUL + recommendation.** For each engine,
`engine_trajectory` (`src/model/predict.py:83`) fits a straight line through its
per-cycle health, extrapolates to the 0.80 failure threshold for a
remaining-useful-life estimate, and maps current health to a plain-English action
("monitor," "schedule maintenance"). *(File 6.)*

**6. Everything → EngineState.** The backend's `TwinService` does all of the above
**once at startup** and caches it (`src/backend/service.py:57`), then packs each
engine-cycle into the `EngineState` contract (`src/backend/schemas.py:69`) —
conditions, sensors, per-component health with confidence/trend/RUL/recommendation,
performance, alerts and history. FastAPI serves it as JSON over HTTP. *(File 7.)*

**7. EngineState → dashboard.** `api.js` fetches the object; `dashboard.js` maps
every field to a gauge, chart or log; `scene.js` colors each 3D stage by its
health; `controls.js` lets you scrub or play through cycles, re-fetching a fresh
state each step; and `/simulate` projects degradation into the future with a
confidence band. *(File 8.)*

## Why the layering matters

Each layer only depends on the *shape* of the layer before it, not its internals:

- The backend depends on the predictor's function signatures, not on whether the
  model is trees or neural nets.
- The dashboard depends only on the `EngineState` contract, not on the backend's
  caching or the model at all.

That's why the model could be swapped from a neural network to a boosted-tree
ensemble (files 4–5) with **zero** changes to the backend or frontend — the
contract held, so the layers above never noticed. Clean seams like this are what
let a system grow without rewrites.

## Recap

- The pipeline is **sensors → features → surrogate → physics → predictions +
  uncertainty → RUL/recommendations → EngineState → dashboard.**
- Each stage **consumes the output of the previous one** and adds a layer of
  meaning, from raw numbers up to a colored 3D engine with maintenance advice.
- Well-defined seams (the predictor interface, the `EngineState` contract) keep
  the layers independent, so any one can change without breaking the others.

For the engineering-level write-up of methodology and results, see
[`docs/technical-report.md`](../technical-report.md).
