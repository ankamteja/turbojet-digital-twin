# Turbojet Digital Twin — Technical Report

A physics-constrained surrogate model that estimates the hidden health,
performance and remaining useful life of a turbojet engine from its gas-path
sensor telemetry, exposed through a REST API and a real-time 3D dashboard.

All quantitative results in this report are read directly from the model's
evaluation artifacts (`src/model/artifacts/metrics.json`,
`importances.json`, `generalization.json`).

---

## 1. Methodology

### 1.1 Problem

A turbojet's subsystem health — how efficiently the compressor, combustor and
turbine convert effort into thrust — cannot be measured directly in flight. A
high-fidelity thermodynamic simulation can compute it but is too slow for
real-time use. We therefore take the **surrogate-model** approach: a
computationally cheap, interpretable model that approximates engine behavior,
estimates the hidden health indicators from available telemetry, and updates as
new measurements arrive.

### 1.2 Data

The dataset is synthetic but physics-based (generated from a realistic turbojet
simulation that obeys real thermodynamics): **10 engines × 30 cycles = 300 rows**,
each a snapshot of one engine at one point in its life. It is split into a
240-row training set and a 60-row test set; ground truth (health + performance)
exists for every row.

The split assigns random cycles per engine to each side, so both splits cover all
10 engines — making the standard test split an *interpolation* task (unseen
cycles of seen engines). A harder *extrapolation* question — a previously unseen
engine — is addressed separately by the leave-one-engine-out study in §5.2.

**Inputs (measured):** `Cycle`, `Altitude_m`, `Mach`, `Tamb_K`, `Pamb_Pa`,
`RPM_rev_min`, `FuelFlow_kg_s`, and station pressures/temperatures `P2/T2`
(compressor inlet), `P3/T3` (compressor exit), `P4/T4` (turbine inlet).

**Targets:** `CompressorHealth`, `CombustorHealth`, `TurbineHealth`,
`OverallHealth` (each 0–1), `Thrust_N`, and `TSFC_g_N_s` (thrust-specific fuel
consumption).

### 1.3 Approach summary

Raw sensors are transformed into physically meaningful ratios and corrected
quantities, then fed to a bootstrap ensemble of monotonic gradient-boosted-tree
regressors — one ensemble per learned target. Physics is enforced as hard
structural constraints. Uncertainty comes from the ensemble's spread;
interpretability from permutation feature importances; remaining useful life from
extrapolating a per-engine degradation fit.

---

## 2. Feature engineering strategy

### 2.1 Rationale

Absolute pressures and temperatures vary strongly with flight conditions: at
altitude every station reading drops even on a perfectly healthy engine.
Degradation is therefore easier to detect in **ratios between stations** — a
compressor's pressure ratio is a property of the compressor itself and stays
meaningful across altitudes — and in **corrected** speed/fuel that remove inlet
dependence.

### 2.2 Engineered features

`add_physics_features` (`src/model/physics.py:98`) appends eight derived columns to
the 13 raw inputs:

| Feature | Definition | Targets it informs |
|---------|-----------|--------------------|
| `PR_c` | `P3 / P2` | compressor pressure ratio |
| `TR_c` | `T3 / T2` | compressor temperature ratio |
| `TR_b` | `T4 / T3` | combustor temperature rise |
| `PR_t` | `P4 / P3` | turbine-section pressure ratio |
| `TR_overall` | `T4 / T2` | end-to-end temperature ratio |
| `PR_overall` | `P4 / P2` | end-to-end pressure ratio |
| `RPM_corr` | `RPM / √T2` | corrected spool speed |
| `Fuel_corr` | `FuelFlow / (P2·√T2)` | corrected fuel flow |

The final feature vector is 21-dimensional (13 raw + 8 derived).

### 2.3 Leakage control

`EngineID` is deliberately excluded from the model inputs: it is an identity label
with no physical meaning, and including it would let the model memorize
per-engine answers rather than learn the physics. It is retained only as a
grouping key for per-engine degradation fitting. `Cycle` is kept because aging is
a genuine physical driver. No feature scaling is applied — the gradient-boosted
trees are scale-invariant, so there is no scaler to fit or leak.

The permutation-importance results (`importances.json`) confirm the features
behave as physics predicts: every health ensemble leans hardest on `Cycle`
(importance ≈ 1.5–1.8, dominating all other features), while the thrust ensemble
leans hardest on `FuelFlow_kg_s` (≈ 1.55) and spool speed `RPM_rev_min` /
`RPM_corr` (≈ 0.09 each) — the physically expected drivers of thrust.

---

## 3. Physics integration approach

Physics is enforced as **hard structural constraints** rather than soft training
penalties. A soft penalty only discourages a violation and can still be broken on
unusual inputs; a hard constraint makes the violating output structurally
impossible.

### 3.1 Monotonic degradation

Each health model is given a monotonic-decreasing constraint on the `Cycle`
feature (`src/model/models.py:41`), so predicted health can never rise as the
engine ages. The gradient-boosting algorithm refuses any split that would violate
this, making rising health unrepresentable rather than merely penalized.

### 3.2 Derived TSFC

TSFC is never a free output. It is derived from predicted thrust and measured fuel
flow via the identity

```
TSFC [g/(N·s)] = 1000 · FuelFlow [kg/s] / Thrust [N]
```

(`src/model/physics.py:123`), computed per ensemble member so its uncertainty is
propagated from the thrust ensemble. This guarantees the two performance numbers
can never disagree, and the identity was verified to hold within ~1% on the
dataset.

### 3.3 Overall-health aggregation (consistency check)

`OverallHealth` is learned by its own ensemble, then checked against the mean of
the three subsystem healths (`src/model/physics.py:133`) as a physics-consistency
residual — a genuine independent test that the four health estimates form a
coherent picture.

---

## 4. Model architecture

### 4.1 Base learner

Each member is a scikit-learn `HistGradientBoostingRegressor` — a fast
histogram-based gradient-boosting ensemble of shallow decision trees
(`src/model/models.py:46`):

- `max_iter = 400` (up to 400 boosting iterations / trees)
- `learning_rate = 0.05`
- `max_depth = 3`
- `l2_regularization = 1.0`
- `loss = "squared_error"`
- `monotonic_cst` = decreasing on `Cycle` for health targets, none for thrust

Gradient-boosted trees were selected over a neural network because the dataset is
small and tabular (240 training rows), where boosted trees estimate the subtle
component-health signal far more accurately while remaining interpretable and fast
to train and query. This replaced an earlier MLP-with-physics-losses approach;
mean health `R²` rose from ~0.62 to ~0.86 and TSFC `R²` from ~0.75 to ~0.98.

### 4.2 Bootstrap ensemble

For each of the five learned targets (four healths + thrust), **10 members** are
trained, each on an independent bootstrap resample of the training rows
(`src/model/models.py:73`). At inference:

- the member **mean** is the prediction,
- the member **standard deviation** is the uncertainty, normalized into a 0–1
  confidence by the target's own spread (`src/model/predict.py:44`).

TSFC (the sixth target) is derived, not learned, so it adds no separate model.

### 4.3 Remaining useful life

For each engine, predicted health is fit with a least-squares line over its
observed cycles (`fit_degradation`, `src/model/physics.py:148`). RUL is the number
of cycles until that line crosses the failure threshold of `OverallHealth = 0.80`
(`remaining_useful_life`, `src/model/physics.py:162`); a flat or improving trend
yields no finite RUL. Health values map to maintenance recommendations via a band
table (nominal / monitor / inspect / schedule maintenance).

### 4.4 Training cost

Fitting the full ensemble (10 members × 5 targets) takes ≈ **55 s** on CPU
(`artifacts/config.json`). The serialized model is a single `surrogate.joblib`
artifact loaded once by the backend at startup.

---

## 5. Validation methodology

### 5.1 Per-target test metrics (interpolation split)

Evaluated on the 60-row held-out test set (`src/model/eval.py`, results from
`metrics.json`):

| Target | MAE | R² |
|--------|-----|-----|
| CompressorHealth | 0.0145 | 0.941 |
| CombustorHealth | 0.0104 | 0.779 |
| TurbineHealth | 0.0190 | 0.733 |
| OverallHealth | 0.0080 | 0.954 |
| Thrust_N | 1235.5 N | 0.991 |
| TSFC_g_N_s | 0.00064 g/(N·s) | 0.983 |

Mean health `R²` across the four health targets ≈ **0.85**. The health MAEs are all
under 0.02 on the 0–1 scale (i.e. estimates within ~2 health points). Thrust and
the derived TSFC are estimated to `R²` ≥ 0.98.

### 5.2 Leave-one-engine-out generalization

To test generalization to a *previously unseen engine* (the harder extrapolation
question §1.2), each of the 10 engines is held out in turn, the model is retrained
on the other 9 (with a lighter 5-member ensemble for the cross-validation), and
the held-out engine is scored (`src/model/generalize.py`, results from
`generalization.json`):

| Target | Mean R² (across held-out engines) | Worst-case engine R² |
|--------|-----------------------------------|----------------------|
| CompressorHealth | 0.863 | 0.362 |
| CombustorHealth  | 0.459 | −2.575 |
| TurbineHealth    | 0.663 | −0.071 |
| OverallHealth    | 0.906 | 0.490 |
| Thrust_N         | 0.986 | 0.977 |
| TSFC_g_N_s       | 0.977 | 0.969 |

Thrust and TSFC generalize almost perfectly to unseen engines (R² ≥ 0.97 even in
the worst case), and `OverallHealth` remains strong (mean 0.91). Per-subsystem
health is harder: `CompressorHealth` holds up (0.86), while `CombustorHealth` is
the weakest and can degrade sharply on individual engines — the combustor signal
is the subtlest in the gas path (its health varies over only ~0.89–1.0), so a
single engine whose behaviour differs from the training nine is hard to
extrapolate. This is the model's main honest limitation and the clearest target
for future data collection.

### 5.3 Physics-consistency residuals

From `metrics.json`:

- **TSFC identity residual** (`|predicted TSFC − 1000·fuel/thrust|`): **2.3×10⁻⁵**
  g/(N·s). Effectively zero by construction, confirming TSFC is derived rather than
  independently estimated.
- **Overall-aggregation residual** (`|OverallHealth − mean(3 subsystems)|`):
  **0.0040** on the 0–1 scale. `OverallHealth` is learned independently, so this
  small residual is evidence the four health estimates form a coherent picture
  rather than four unrelated numbers.

### 5.4 Inference latency

Batch inference over the test set gives **≈ 12.5 ms per row** (`metrics.json`),
covering all six targets across the full 10-member ensembles. This is comfortably
real-time; in deployment the backend precomputes and caches every engine's
per-cycle state at startup, so dashboard requests are served as constant-time
lookups.

---

## 6. Summary

| Aspect | Result |
|--------|--------|
| Mean health R² (test, interpolation) | ≈ 0.85 |
| Thrust / TSFC R² (test) | 0.991 / 0.983 |
| Health MAE (test) | ≤ 0.019 (0–1 scale) |
| TSFC physics residual | 2.3×10⁻⁵ (≈ 0, by construction) |
| Overall-aggregation residual | 0.0040 |
| Inference latency | ≈ 12.5 ms/row |
| Training time | ≈ 55 s (CPU) |

The surrogate meets its design goals: accurate estimation of hidden subsystem
health and performance from telemetry alone, hard physical guarantees on
degradation and fuel-efficiency consistency, calibrated per-estimate confidence
from ensemble spread, interpretable feature attributions, and real-time inference
served behind a stable API to a live dashboard.
