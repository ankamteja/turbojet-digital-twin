# Turbojet Digital Twin

A physics-informed digital twin for real-time health monitoring of a single-spool,
four-stage turbojet engine. The system reconstructs hidden component health and predicts
engine performance from a limited set of sensor measurements, and drives an interactive
dashboard that renders the engine as a live digital asset.

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

Physics-informed losses keep predictions physically meaningful — e.g. enforcing consistency
between pressure/temperature ratios and the estimated component health, so the model does
not fit the data at the expense of thermodynamic sense.

## Dashboard

The engine is presented as a live 3D model rather than a wall of gauges. Four components are
selectable — compressor, combustor, turbine, and overall engine. Selecting one opens a
detail view: current health, degradation history, contributing sensor values, projected
remaining life, a confidence score, and a recommended action.

Alongside the model the dashboard shows live operating conditions, the four health scores,
predicted thrust, fuel efficiency, degradation trend, confidence intervals, and health
alerts that update automatically.

## Planned stack

- **Frontend:** React, Three.js (React Three Fiber)
- **Backend:** FastAPI
- **Modeling:** Python, PyTorch / scikit-learn
- **Visualization:** Plotly / Chart.js

## Repository layout

```
data/    dataset (inputs, targets, combined)
docs/    project plan and design notes
src/     model + backend + frontend (in progress)
```

## Status

Early scaffold. Dataset and design are in place; modeling and dashboard are under
development. See `docs/project-plan.md`.

## License

MIT — see `LICENSE`.
