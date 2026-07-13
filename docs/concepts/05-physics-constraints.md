# 5. Physics Constraints

File 4 introduced the surrogate — a bootstrap ensemble of gradient-boosted trees.
This file explains what makes it **physics-informed**: the engine physics baked
directly into the model so it can never produce a physically impossible answer.

The key idea is the difference between a *suggestion* and a *rule*.

## Soft penalties vs hard constraints

There are two ways to make a model respect physics.

- **Soft penalty (a suggestion).** During training you *punish* the model a
  little every time it breaks a physical relationship. It learns to mostly obey —
  but "mostly" is the catch. On an unusual input it can still drift into
  nonsense, because the penalty only *discouraged* the bad behavior, it didn't
  *forbid* it.

- **Hard constraint (a rule).** You change the model's *structure* so the bad
  answer is simply not representable. There's nothing to drift toward, because the
  impossible answer no longer exists in the model's vocabulary.

This project uses **hard constraints**. An earlier version used soft physics
penalties (added onto the training error of a neural network); the current
tree-based surrogate replaces them with structural rules, which are both stronger
*and* simpler. The file header states the philosophy directly
(`src/model/models.py:8`):

> *"Physics is enforced as **hard structural constraints**, which is stronger
> than soft training penalties."*

There are two hard constraints, plus one consistency *check* used for reporting.

## Constraint 1: monotonic degradation

**The physical truth.** An engine does not heal itself. As the number of flight
cycles goes up, each subsystem's health can only stay the same or fall — never
rise. (Real maintenance can restore an engine, but this model describes an engine
between overhauls, and the dataset degrades gradually.)

**How it's enforced.** Recall from file 4 that a gradient-boosting model can be
handed a **monotonic constraint** — a rule that its output must move in only one
direction as a chosen input increases. Each health model gets a
*monotonic-decreasing* constraint on the `Cycle` feature
(`src/model/models.py:41`):

```python
mono = None
if target in HEALTH_TARGETS:
    mono = [0] * len(MODEL_FEATURES)
    mono[CYCLE_IDX] = -1          # health non-increasing in Cycle
```

The list `mono` has one entry per feature: `0` means "no constraint," and `-1`
on the `Cycle` slot means "output must be non-increasing as `Cycle` grows." The
tree-building algorithm then *refuses to make any split* that would let health
rise with cycle. It is structurally impossible for the model to say an engine got
healthier as it aged.

Why this beats a soft penalty: a penalty would let the model occasionally output
a small health *increase* on a noisy input and just accept the penalty. The hard
constraint makes that answer unrepresentable, so the degradation curve you see on
the dashboard is always physically sensible — it can plateau, but it can never
tick upward.

## Constraint 2: TSFC derived from thrust, never predicted

**The physical truth.** Fuel efficiency isn't an independent quantity — it is a
*definition*. Thrust-Specific Fuel Consumption is fuel burned per unit of thrust
(file 2):

```
TSFC [g/(N·s)]  =  1000 × FuelFlow [kg/s]  ÷  Thrust [N]
```

Fuel flow is a sensor we already measure. So the moment we predict thrust, TSFC
is *fixed* — there is no freedom left to predict separately.

**How it's enforced.** The model never learns TSFC. Look back at the learned
targets (`src/model/physics.py:95`): they are the four healths plus `Thrust_N`
only. TSFC is computed *after* thrust is predicted, per ensemble member
(`src/model/models.py:104`):

```python
thrust = np.maximum(self._member_predict("Thrust_N", X), 0.0)
out["Thrust_N"] = {"mean": thrust.mean(0), "std": thrust.std(0)}

# derive TSFC per member, then aggregate
fuel = np.asarray(fuel_flow_kg_s, dtype=float)
tsfc = tsfc_from_fuel_thrust(np.broadcast_to(fuel, thrust.shape), thrust)
out["TSFC_g_N_s"] = {"mean": tsfc.mean(0), "std": tsfc.std(0)}
```

The closed-form relation lives in `physics.py` (`src/model/physics.py:123`):

```python
def tsfc_from_fuel_thrust(fuel_flow_kg_s, thrust_n):
    thrust = np.maximum(np.asarray(thrust_n, dtype=float), 1.0)
    return TSFC_FROM_FUEL_THRUST * np.asarray(fuel_flow_kg_s, dtype=float) / thrust
```

Two elegant consequences:

1. **The two performance numbers can never disagree.** Because TSFC is *computed
   from* thrust rather than guessed alongside it, the pair is always consistent
   with the fuel the engine is actually burning. The evaluation confirms this —
   the "TSFC consistency residual" is essentially zero (~2×10⁻⁵), which it must
   be by construction (file 6, technical report).

2. **Uncertainty flows through for free.** TSFC is derived *inside* each of the
   ten ensemble members before averaging, so its confidence is inherited from the
   spread of the thrust ensemble — no separate uncertainty model needed. The
   `np.maximum(..., 1.0)` floor just keeps the division finite if thrust ever came
   out near zero on a weird input.

## The consistency check: overall-health aggregation

The third piece is a *check*, not a hard constraint — a physics-based sanity test
we measure and report rather than force.

**The idea.** The whole engine should be about as healthy as the aggregate of its
three subsystems. If compressor, combustor and turbine are all around 0.90, an
`OverallHealth` of 0.55 would be suspicious. We quantify the gap
(`src/model/physics.py:133`):

```python
def overall_aggregation_residual(comp, comb, turb, overall):
    parts_mean = (np.asarray(comp) + np.asarray(comb) + np.asarray(turb)) / 3.0
    return np.abs(np.asarray(overall) - parts_mean)
```

This is the absolute distance between predicted `OverallHealth` and the average
of the three parts. `OverallHealth` is learned by its *own* ensemble (it isn't
forced to equal the mean), so this residual is a genuine independent check that
the four health predictions hang together. In evaluation it comes out tiny
(~0.004 on the 0–1 scale), confirming the model learned a coherent picture rather
than four unrelated numbers.

## Why hard constraints are the right call here

- They are **guarantees**, not tendencies. Monotonic health and consistent
  performance hold on *every* input, including odd ones the model never trained
  on.
- They are **simpler**. No penalty weights to tune, no balancing act between
  fitting the data and obeying physics — the physics is in the model's shape.
- They **improved accuracy**. Deriving TSFC instead of predicting it, and
  constraining health monotonicity, removed whole classes of possible error. The
  switch lifted TSFC accuracy to `R²` ~0.98 and health accuracy to ~0.86 (file 6,
  technical report).

## Recap

- Physics enters as **hard structural constraints**, stronger than soft training
  penalties: the impossible answer isn't merely discouraged, it can't be produced.
- **Constraint 1 — monotonic degradation:** health can never rise with `Cycle`,
  enforced by the trees' monotonic constraint.
- **Constraint 2 — derived TSFC:** fuel efficiency is computed from predicted
  thrust via `TSFC = 1000·fuel/thrust`, so thrust and TSFC always agree and
  TSFC's uncertainty is inherited from thrust's.
- **Aggregation check:** `OverallHealth` is verified against the mean of the three
  subsystems as a physics-consistency metric.

Next: **file 6**, on turning the ensemble's spread into a confidence score, and
projecting the health trend into a remaining-useful-life estimate.
