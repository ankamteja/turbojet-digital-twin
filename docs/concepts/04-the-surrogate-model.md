# 4. The Surrogate Model

File 3 left us with a table of prepared inputs: 13 raw sensors plus 8 engineered
physics features = 21 columns per row. This file covers the thing that finally
*reads* those columns and produces predictions — the **surrogate model**.

Remember from file 1 what "surrogate" means: a fast stand-in for a slow, accurate
physics simulation. The surrogate looks at sensor readings and estimates the
hidden health, thrust and fuel efficiency in milliseconds, instead of running a
full thermodynamic simulation. This file explains *what kind* of model it is, and
*why* that kind was chosen.

The code is all in `src/model/models.py`.

## What is a machine-learning model, in one breath?

A **machine-learning model** is a program that learns a pattern from examples
instead of being hand-coded. You show it many rows where the answer is known
("these sensor readings go with this much wear"), it finds the pattern, and then
it can answer *new* rows it has never seen. Our examples are the 240 training
rows from file 2; the answers are the true health, thrust and TSFC values.

## Decision trees: the building block

The surrogate is built out of **decision trees**. A decision tree is just a
flowchart of yes/no questions that ends in a number.

Imagine estimating compressor health with a flowchart:

```
   Is the compressor pressure ratio (PR_c) below 3.0?
         ├── yes → Is Cycle above 20?
         │           ├── yes → health ≈ 0.82
         │           └── no  → health ≈ 0.90
         └── no  → health ≈ 0.98
```

Each question splits the data; each leaf at the bottom is a prediction. A single
tree is crude, but it captures the basic idea: *follow the sensor clues down to
an estimate.*

## Gradient boosting: many small trees, each fixing the last

One tree is too simple. **Gradient boosting** builds a *team* of trees, but not
independent ones — each new tree is trained to correct the mistakes the previous
trees are still making. Tree 1 makes a rough guess, tree 2 learns "tree 1 tends
to be 0.03 too high in this region — nudge it down," tree 3 corrects what's still
wrong, and so on for hundreds of tiny trees. Add all their nudges together and
you get a very accurate estimate.

Analogy: a group of specialists reviewing an X-ray one after another. The first
gives a rough read; each next one only says "here's the small thing the others
missed." Their combined verdict is far sharper than any single read.

We use scikit-learn's `HistGradientBoostingRegressor` — a fast, modern
gradient-boosting implementation. One regressor is built like this
(`src/model/models.py:40`):

```python
return HistGradientBoostingRegressor(
    loss="squared_error",
    max_iter=400,          # up to 400 small trees
    learning_rate=0.05,    # each tree nudges gently
    max_depth=3,           # each tree asks at most 3 questions deep
    l2_regularization=1.0, # discourage overfitting
    monotonic_cst=mono,    # physics constraint (explained below + in file 5)
    random_state=seed,
)
```

`max_iter=400` is the number of trees; `learning_rate=0.05` keeps each tree's
correction small so the team learns gradually; `max_depth=3` keeps each tree
shallow (simple) so no single tree overreacts.

## Why trees instead of a neural network?

An earlier version of this project used a **neural network** (a different kind of
model — many layered arithmetic units tuned by trial and error). On this dataset
it was replaced by the boosted-tree surrogate. The reason is written at the top
of the file (`src/model/models.py:3`):

> *"On this small, tabular dataset (240 training rows) boosted trees estimate the
> subtle component-health signal far better than an MLP (test health R^2 ~0.86
> vs ~0.62), while staying interpretable (feature importances) and fast to train
> and query."*

Three concrete reasons, in plain language:

1. **Small, table-shaped data.** Neural networks are hungry — they shine on huge
   datasets (millions of images, sentences). With only 240 rows of numbers in a
   spreadsheet-like table, a neural net tends to either memorize or flounder.
   Boosted trees are the well-proven default for small tabular data.

2. **Accuracy on the hard signal.** The component healths are a *subtle* signal —
   a fouled compressor shifts a pressure ratio only a little. Swapping to trees
   lifted mean health accuracy dramatically (`R²` from ~0.62 to ~0.86; more
   numbers in file 6 and the technical report).

3. **Interpretability and speed.** Trees make it easy to ask "which inputs
   mattered most?" (see *feature importances* below) and they train and predict
   very fast. "Interpretable" means a human can inspect *why* the model answered
   the way it did — valuable when the answer drives a maintenance decision.

(`R²`, "R-squared", is a standard accuracy score from 0 to 1 where 1.0 is
perfect. File 6 defines it fully.)

## One ensemble per target

The model doesn't predict everything with one tree-team. It trains a **separate**
gradient-boosting ensemble for each thing it learns directly. Those are the four
healths plus thrust (`src/model/physics.py:95`):

```python
LEARNED_TARGETS = HEALTH_TARGETS + ["Thrust_N"]
```

Notice **TSFC is not learned.** Fuel efficiency is *derived* from predicted
thrust using the physics identity `TSFC = 1000 · fuel / thrust`, so it can never
disagree with thrust. That is a physics constraint, and file 5 is entirely about
it. For now just note: five things are learned by trees, the sixth is computed.

## The bootstrap ensemble: many copies for confidence

Here is a second, separate use of the word "ensemble." For **each** target we
don't train one gradient-boosting model — we train **ten**, each on a slightly
different sample of the data. This is a **bootstrap ensemble**.

"Bootstrap" is a statistics trick: from the 240 training rows, draw 240 rows *at
random with replacement* — so some rows appear twice, some not at all. Each of
the ten members sees a slightly different slice of reality, and so each becomes a
slightly different expert (`src/model/models.py:73`):

```python
for target in LEARNED_TARGETS:
    members = []
    for m in range(self.n_members):          # n_members = 10
        idx = rng.integers(0, n, n)          # bootstrap resample (with replacement)
        mod = _make_member(target, seed=1000 + m)
        mod.fit(X[idx], np.asarray(y[target])[idx])
        members.append(mod)
    self.models[target] = members
```

Why bother with ten? Because their **agreement** becomes a confidence signal. At
prediction time we ask all ten and look at both their average *and* their spread
(`src/model/models.py:100`):

```python
for target in HEALTH_TARGETS:
    preds = np.clip(self._member_predict(target, X), 0.0, 1.0)
    out[target] = {"mean": preds.mean(0), "std": preds.std(0)}
```

- `mean` (the average of the ten) → the **prediction**.
- `std` (the standard deviation, i.e. how spread out the ten are) → the
  **uncertainty**. Tight agreement = confident; wide disagreement = shaky.

Analogy: ask ten trained inspectors to rate the same engine. If all ten say
"0.90," you trust it. If they scatter from 0.80 to 0.98, you know the reading is
uncertain. File 6 turns this spread into a friendly 0–1 confidence score.

The `np.clip(..., 0.0, 1.0)` simply keeps health values inside their valid 0–1
range.

## The monotonic constraint (a first look)

One line in the model builder deserves a flag now, because it is where physics
enters the trees. For the health targets only, the model is given a **monotonic
constraint** on the `Cycle` feature (`src/model/models.py:41`):

```python
mono = None
if target in HEALTH_TARGETS:
    mono = [0] * len(MODEL_FEATURES)
    mono[CYCLE_IDX] = -1          # health non-increasing in Cycle
```

`-1` tells the tree: *as `Cycle` (the age of the engine) goes up, predicted
health may never go up.* An engine can't heal itself, and this hard-wires that
truth into the model's structure — it becomes *impossible* for the model to
output rising health as an engine ages. File 5 explains why a hard structural
constraint like this is stronger than merely nudging the model in training.

## Feature importances: which sensors mattered?

Because trees are interpretable, we can ask each ensemble *which inputs it leaned
on most*. The technique is **permutation importance**: take a trained model,
scramble one feature's values, and see how much accuracy drops. If scrambling
`PR_c` wrecks the compressor-health prediction, then `PR_c` was important
(`src/model/models.py:114`):

```python
r = permutation_importance(
    model, X, np.asarray(y[target]),
    n_repeats=n_repeats, random_state=0, scoring="r2",
)
```

These scores are saved to `artifacts/importances.json` and drive the "which
sensors are behind this call" readout on the dashboard. As a sanity check they
line up with physics: the health models lean hardest on `Cycle` (aging), and
thrust leans hardest on `FuelFlow_kg_s` and spool speed — exactly what you'd
expect.

## Recap

- The surrogate is a **gradient-boosting model**: hundreds of small decision
  trees, each correcting the last, per learned target.
- **Trees beat a neural network here** because the data is small and tabular; the
  swap roughly doubled the useful health accuracy and kept the model fast and
  interpretable.
- Five targets are learned directly (4 healths + thrust); **TSFC is derived**
  from thrust by physics (file 5).
- For each target we train a **bootstrap ensemble of 10 members**; their mean is
  the prediction and their spread is the confidence (file 6).
- Health carries a **hard monotonic constraint** on `Cycle` so it can never rise
  as the engine ages (file 5).

Next: **file 5**, on how physics is baked into the model as *hard constraints*
rather than gentle suggestions.
