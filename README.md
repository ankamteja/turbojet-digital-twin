# Turbojet Digital Twin

A physics-informed digital twin for real-time health monitoring of a single-spool,
four-stage turbojet engine. The system reconstructs hidden component health and predicts
engine performance from a limited set of sensor measurements, and drives an interactive
dashboard that renders the engine as a live digital asset.

## Live demo

- **Dashboard:** https://turbojet-dashboard.onrender.com
- **API:** https://turbojet-api.onrender.com

Both run on Render's free tier, so the backend sleeps after ~15 min idle — the first
request after a nap takes ~30–60 s to wake and reload the model, then it's responsive.
Open the link a minute before you need it. See `DEPLOY.md` for one-click redeploy.

## Motivation

Turbojet components degrade over their operational life — compressor fouling, turbine
erosion, combustor efficiency loss. These mechanisms hurt thrust, fuel efficiency, and
reliability, yet the underlying health states cannot be measured directly in flight.
High-fidelity engine simulations capture this behavior but are far too expensive to run in
real time.

This project takes the surrogate-model approach: a computationally cheap, interpretable
model that approximates engine behavior, estimates hidden health indicators from available
telemetry, and updates continuously as new measurements arrive.

## What it does

- Estimates hidden subsystem health — compressor, combustor, turbine, and an overall index
- Predicts engine performance — thrust and thrust-specific fuel consumption (TSFC)
- Tracks degradation over cycles and projects remaining useful life
- Produces confidence estimates alongside predictions
- Feeds a live dashboard with an interactive 3D engine, per-component drill-downs, and
  automatic health alerts

## Dataset

A synthetic but physics-based dataset generated from a four-stage single-spool turbojet
model. It covers multiple virtual engines under varying flight conditions and progressive
degradation.

| File | Rows | Contents |
|------|------|----------|
| `data/train.csv` | 240 | Sensor inputs (training split) |
| `data/test.csv` | 60 | Sensor inputs (held-out split) |
| `data/ground_truth.csv` | 300 | Health + performance targets for all cycles |
| `data/turbojet_complete_dataset.csv` | 300 | Full table — inputs and targets combined |

### Inputs (measured)

`EngineID`, `Cycle`, `Altitude_m`, `Mach`, `Tamb_K`, `Pamb_Pa`, `RPM_rev_min`,
`FuelFlow_kg_s`, `P2_Pa`, `T2_K`, `P3_Pa`, `T3_K`, `P4_Pa`, `T4_K`

Station numbering follows the gas path: 2 — compressor inlet, 3 — compressor exit /
combustor inlet, 4 — turbine inlet.

### Targets (to estimate)

`CompressorHealth`, `CombustorHealth`, `TurbineHealth`, `OverallHealth` (each a normalized
0–1 index), `Thrust_N`, `TSFC_g_N_s`.

## Approach

```
Sensors / dataset
  → feature engineering
  → physics constraints
  → surrogate model
  → health estimation
  → performance prediction
  → dashboard
```

The surrogate is a bootstrap ensemble of monotonic gradient-boosted trees — one ensemble per
target — chosen over a neural network because the dataset is small and tabular. Physics is
enforced as **hard structural constraints**: health can never rise as the engine ages
(monotonic constraint on cycle), and TSFC is derived from predicted thrust rather than fit
independently, so the two performance numbers agree by construction. Uncertainty comes from
the ensemble's spread; interpretability from permutation feature importances.

## Dashboard

The engine is presented as a live 3D model rather than a wall of gauges. Four components are
selectable — compressor, combustor, turbine, and overall engine. Selecting one opens a
detail view: current health, degradation history, contributing sensor values, projected
remaining life, a confidence score, and a recommended action.

Alongside the model the dashboard shows live operating conditions, the four health scores,
predicted thrust, fuel efficiency, degradation trend, confidence intervals, and health
alerts that update automatically.

## Stack

- **Modeling:** Python, scikit-learn (gradient-boosted trees), pandas, numpy
- **Backend:** FastAPI + uvicorn
- **Frontend:** vanilla JavaScript (ES modules) + Three.js — no framework, no build step

## Quickstart

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. train the surrogate (writes src/model/artifacts/)
cd src/model && python train.py

# 3. run the backend API (from src/backend/, serves on :8000)
cd ../backend && uvicorn main:app --port 8000

# 4. serve the dashboard (from src/frontend/, in a second terminal)
cd ../frontend && python -m http.server 8080
# then open http://localhost:8080/index.html
```

Optional: `python src/model/eval.py` writes test metrics, and
`python src/model/generalize.py` runs the leave-one-engine-out generalization study.

## Repository layout

```
data/    dataset (inputs, targets, combined)
docs/    concept guides, technical report, architecture, presentation
src/
  model/     feature engineering, surrogate ensemble, training, evaluation
  backend/   FastAPI service exposing the EngineState contract
  frontend/  vanilla-JS + Three.js dashboard
```

## Documentation

`docs/concepts/` walks through every concept from scratch (digital twins → feature
engineering → the model → physics constraints → uncertainty → API → dashboard).
`docs/technical-report.md` is the results write-up; `docs/architecture.md` the design.

## License

MIT — see `LICENSE`.
