# Turbojet Digital Twin — Presentation Outline

A concise, slide-by-slide outline covering the engineering rationale, the
surrogate strategy, health-estimation methodology, physics integration, key
results, and the dashboard.

---

## Slide 1 — Title

**Turbojet Digital Twin**
Estimating hidden engine health, performance and remaining useful life from
gas-path telemetry, in real time.

---

## Slide 2 — The problem

- A turbojet's subsystem health (compressor / combustor / turbine) **cannot be
  measured directly in flight** — there is no "compressor health" gauge.
- The accurate way to compute it (a full thermodynamic simulation) is **too slow
  for real-time** use.
- Need: a fast, trustworthy estimate of hidden health from the sensors we *do*
  have.

---

## Slide 3 — Engineering rationale: a digital twin

- A **digital twin** mirrors a real engine from its live sensors.
- We infer hidden health with a **surrogate model** — a cheap, interpretable
  stand-in for the slow simulation.
- Design priorities: accuracy on the subtle health signal, hard physical
  guarantees, per-estimate confidence, interpretability, real-time speed.

---

## Slide 4 — The data

- Synthetic-but-physics-based: **10 engines × 30 cycles = 300 rows** (240 train /
  60 test), full ground truth.
- Inputs: cycle, flight conditions, RPM, fuel flow, and station pressures /
  temperatures (stations 2, 3, 4).
- Targets: 4 healths (0–1), thrust, TSFC (fuel efficiency).
- Anchor relation: `TSFC = 1000 · FuelFlow / Thrust` (holds within ~1%).

---

## Slide 5 — Feature engineering strategy

- Raw pressures/temps swing with altitude and hide degradation.
- Reshape into **station ratios** (`PR_c = P3/P2`, `TR_b = T4/T3`, …) and
  **corrected** speed/fuel — where wear shows as a clean drift.
- 13 raw + 8 engineered = **21 features**.
- **Drop `EngineID`** (identity → would leak); trees are scale-invariant, so no
  scaler needed.

---

## Slide 6 — Surrogate strategy: why gradient-boosted trees

- Small, tabular dataset (240 rows) → **boosted trees**, not a neural network.
- Gradient boosting = hundreds of shallow trees, each correcting the last.
- Swapping an earlier MLP for trees lifted mean health `R²` **~0.62 → ~0.86** and
  TSFC **~0.75 → ~0.98**, while staying fast and interpretable.
- One ensemble per learned target (4 healths + thrust); TSFC is derived.

---

## Slide 7 — Health-estimation methodology

- **Bootstrap ensemble:** 10 tree models per target, each on a random resample.
- **Mean** = the health estimate; **spread** = the uncertainty.
- Spread → a **0–1 confidence** (tight agreement = confident).
- Interpretability: **permutation importances** show each estimate's top drivers —
  health leans on `Cycle`, thrust on fuel flow / spool speed (physically expected).

---

## Slide 8 — Physics integration (hard constraints)

- Physics is enforced as **hard structural constraints**, stronger than soft
  penalties:
  1. **Monotonic degradation** — health can never rise with `Cycle`.
  2. **Derived TSFC** — computed from predicted thrust, never guessed → performance
     numbers always consistent.
- Plus an **aggregation check**: `OverallHealth` vs the mean of the three
  subsystems.

---

## Slide 9 — Remaining useful life

- Fit a straight line through each engine's predicted health over cycles.
- Extrapolate to the failure threshold (`OverallHealth = 0.80`) → **RUL in
  cycles**.
- Map health to a plain-English action: nominal / monitor / inspect / schedule
  maintenance.

---

## Slide 10 — Key results

| Metric | Value |
|--------|-------|
| Mean health R² (test) | ≈ 0.85 |
| Thrust / TSFC R² | 0.991 / 0.983 |
| Health MAE (0–1 scale) | ≤ 0.019 |
| TSFC physics residual | 2.3×10⁻⁵ (≈ 0, by construction) |
| Overall-aggregation residual | 0.0040 |
| Inference latency | ≈ 12.5 ms/row |
| Training time | ≈ 55 s (CPU) |

Generalization to unseen engines is validated with a leave-one-engine-out study
(see the technical report for the per-target table).

---

## Slide 11 — System architecture

- **Pipeline:** sensors → features → surrogate → physics → predictions +
  uncertainty → RUL / recommendations → `EngineState` → dashboard.
- **FastAPI backend** serves a stable `EngineState` contract; heavy model work is
  precomputed and cached at startup.
- Clean seams: the model was swapped (NN → trees) with **zero** backend/frontend
  changes.

---

## Slide 12 — The dashboard

- Real-time 3D turbojet; each stage **colors green → yellow → red** by its real
  health.
- Engine selector + **cycle Play** to watch an engine age; every step re-fetches a
  fresh state.
- Per-stage drill-down: health, confidence, RUL, recommendation, and the
  contributing sensors.
- **/simulate** projects degradation into the future with a confidence band.
- Every displayed number traces to a model output or a physics relation.

---

## Slide 13 — Summary

- Fast, interpretable surrogate that estimates hidden health from telemetry alone.
- **Hard physical guarantees** on degradation and fuel-efficiency consistency.
- Calibrated **confidence** and actionable **RUL** per component.
- Real-time inference behind a stable API and a live 3D dashboard.
