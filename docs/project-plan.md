# Project Plan — Turbojet Digital Twin

## Vision

Build a physics-informed digital twin of a single-spool, four-stage turbojet engine. The
twin estimates engine health in real time, predicts future degradation, and presents the
engine through an interactive 3D model. Every value shown in the interface comes from the
trained models and simulation — no placeholder numbers in the final product.

## Objectives

Reconstruct the operational and health state of the engine from limited sensor
measurements. Concretely, the twin will:

- Estimate hidden subsystem health (compressor, combustor, turbine, overall index)
- Predict engine performance (thrust, TSFC)
- Track degradation over time and project remaining useful life
- Produce confidence estimates for its predictions
- Continuously update its virtual representation from incoming telemetry

## Data pipeline

```
Sensors / dataset
  → feature engineering
  → physics constraints
  → surrogate model
  → health estimation
  → performance prediction
  → dashboard
```

Inputs: engine ID, cycle, altitude, Mach number, ambient temperature and pressure, shaft
RPM, fuel flow, and the compressor/combustor/turbine station pressures and temperatures.

## Models

The system learns compressor health, combustor health, turbine health, an overall health
index, thrust, fuel efficiency, a degradation trajectory, remaining useful life, and a
prediction uncertainty. Physics-informed losses keep the outputs thermodynamically
consistent.

## Dashboard

Continuously updated. Displays an interactive 3D engine, live operating conditions, the four
health scores, predicted thrust, fuel efficiency, degradation trend, confidence intervals,
health alerts, and mission status.

### 3D interaction

The engine model is the center of the application. Selecting a subsystem highlights it and
opens: current condition, historical degradation, contributing sensor values, projected
future degradation, an explanation of the estimate, and a maintenance recommendation.

## Simulation

The dashboard first runs on the supplied physics-based dataset. Once models are trained,
their predictions replace any placeholder values. A degradation simulator demonstrates
long-term behavior until live telemetry is available.

## Technology stack

- **Frontend:** React + Three.js (React Three Fiber)
- **Backend:** FastAPI
- **Modeling:** Python, PyTorch / scikit-learn
- **Visualization:** Plotly / Chart.js
- **Twin core:** physics-informed surrogate models

## Deliverables

- Source code
- Technical report
- Digital twin dashboard
- Presentation

## Long-term vision

Grow into a real-time aerospace digital twin supporting predictive maintenance, intelligent
diagnostics, autonomous health monitoring, and decision support for propulsion systems.
