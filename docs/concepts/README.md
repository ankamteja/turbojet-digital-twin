# The Turbojet Digital Twin, Explained From Scratch

A beginner's tour of the **whole system** behind the turbojet digital twin — the
model in `src/model/`, the backend API in `src/backend/`, and the dashboard in
`src/frontend/`. It assumes **no** background in jet engines, machine learning, or
web apps. Every term is defined the first time it appears, every concept gets a
plain-language analogy, and each idea is tied back to the exact line of code that
implements it.

## Big picture

A jet engine's parts wear out invisibly — you can't measure "how healthy the
compressor is" with any gauge. What you *can* measure is a handful of sensors:
pressures, temperatures, engine speed, fuel flow, altitude, and speed. This
project reshapes those sensors into physics-based ratios and feeds them to a
**surrogate model** — a bootstrap ensemble of physics-constrained gradient-boosted
decision trees — to infer the hidden health, along with thrust and fuel
efficiency. Physics is enforced as hard rules (health can only fall with age; fuel
efficiency is derived from thrust), the ensemble's *agreement* becomes a confidence
score, and each engine's health trend is extended into the future to estimate
remaining useful life. A FastAPI backend packages all of this into a stable data
contract, and a browser dashboard paints it onto a live 3D engine. The end result:
raw telemetry in, and a trustworthy dashboard of health scores, performance
numbers, confidence levels, maintenance recommendations, and time-to-service out.

## Read in order

| # | File | What you'll learn |
|---|------|-------------------|
| 1 | [What Is a Digital Twin?](01-what-is-a-digital-twin.md) | Digital twins, why an engine needs one, "hidden health," the surrogate-model idea. |
| 2 | [The Dataset and the Physics](02-the-dataset-and-physics.md) | The gas path (stations 2/3/4), every sensor column, the six prediction targets, and the key TSFC = 1000·Fuel/Thrust relation. |
| 3 | [Feature Engineering](03-feature-engineering.md) | Why raw sensors become ratios, corrected speed/fuel, and why `EngineID` is dropped (leakage). |
| 4 | [The Surrogate Model](04-the-surrogate-model.md) | Decision trees, gradient boosting, why trees beat a neural network on small tabular data, one ensemble per target, and the bootstrap ensemble for confidence. |
| 5 | [Physics Constraints](05-physics-constraints.md) | Hard structural constraints vs soft penalties: monotonic degradation, TSFC derived from thrust, and the overall-aggregation consistency check. |
| 6 | [Uncertainty and Remaining Useful Life](06-uncertainty-and-rul.md) | Confidence from bootstrap-ensemble spread, RUL from a fitted degradation line, recommendation bands, and how the model is evaluated. |
| 7 | [The Backend API](07-the-backend-api.md) | What a REST API is, FastAPI, the `EngineState` contract, each endpoint, and startup caching. |
| 8 | [The Dashboard](08-the-dashboard.md) | The frontend, the 3D engine, each panel, how it fetches `EngineState`, the engine selector + cycle play, and the projected-degradation drill-down. |
| 9 | [How It All Fits Together](09-how-it-all-fits-together.md) | The full end-to-end pipeline with an ASCII data-flow diagram, and how each stage consumes the previous one. |

## The code these docs explain

The model (`src/model/`):

- `physics.py` — engine knowledge: engineered features, physics relations (TSFC, aggregation), and RUL.
- `data.py` — loads the CSVs, merges ground truth, attaches physics features.
- `models.py` — the `SurrogateEnsemble`: bootstrap ensembles of monotonic gradient-boosted trees.
- `train.py` — fits the ensemble and saves artifacts + feature importances.
- `predict.py` — runs the ensemble; adds confidence, degradation fit, RUL, recommendations.
- `eval.py` — held-out test metrics + physics residuals + inference latency.
- `generalize.py` — leave-one-engine-out generalization test.

The backend (`src/backend/`): `schemas.py` (the `EngineState` contract),
`service.py` (predictions → `EngineState`, cached at startup), `main.py` (FastAPI
endpoints).

The frontend (`src/frontend/`): `js/api.js`, `js/controls.js`, `js/scene.js`,
`js/dashboard.js`, `js/main.js`, and `index.html`.

For an engineering-level write-up of methodology and results, see
[`docs/technical-report.md`](../technical-report.md).
