# 3. Feature Engineering

We now have raw sensor columns (file 2). But we don't feed them to the model
exactly as they arrive. First we *reshape* them into forms that are easier to
learn from. That reshaping is called **feature engineering**: using human
knowledge to build better input columns out of the raw ones.

Analogy: if you're judging whether a car is fuel-efficient, "gallons used" and
"miles driven" are less useful separately than the single derived number "miles
per gallon." Same fuel data, but the *ratio* carries the meaning directly.
Feature engineering is building those "miles per gallon" columns for the engine.

## Why ratios instead of raw pressures and temperatures

The raw pressures and temperatures swing wildly depending on where the plane is
flying. At high altitude the air is thin and cold, so *every* pressure and
temperature drops — even on a perfectly healthy engine. If the model only saw raw
`P3_Pa`, it couldn't tell "low pressure because the compressor is worn" from "low
pressure because we're flying high."

The fix: look at **ratios** between stations instead of absolute values. How much
the compressor *multiplies* the pressure (exit ÷ inlet) is a property of the
compressor itself, and stays meaningful whether you're at sea level or 10 km up.

`physics.py` builds eight of these derived columns in `add_physics_features`
(`src/model/physics.py:90`):

```python
df["PR_c"] = df["P3_Pa"] / df["P2_Pa"]          # compressor pressure ratio
df["TR_c"] = df["T3_K"] / df["T2_K"]            # compressor temperature ratio
df["TR_b"] = df["T4_K"] / df["T3_K"]            # combustor temperature ratio
df["PR_t"] = df["P4_Pa"] / df["P3_Pa"]          # turbine-section pressure ratio
df["TR_overall"] = df["T4_K"] / df["T2_K"]      # overall temperature ratio
df["PR_overall"] = df["P4_Pa"] / df["P2_Pa"]    # overall pressure ratio
df["RPM_corr"] = df["RPM_rev_min"] / np.sqrt(df["T2_K"])
df["Fuel_corr"] = df["FuelFlow_kg_s"] / (df["P2_Pa"] * np.sqrt(df["T2_K"]))
```

Read the first one out loud: `PR_c = P3 / P2` is "pressure ratio across the
compressor" — how many times bigger the pressure got as air passed through it. A
healthy compressor squeezes hard (high ratio); a fouled one squeezes less (lower
ratio). The code comment captures the intuition:

> *"a fouled compressor delivers a lower pressure ratio for the same corrected
> speed, an eroded turbine changes the expansion across station 4, etc."*
> — `physics.py:93`

So each ratio is aimed at one subsystem's health:

- `PR_c`, `TR_c` → compressor
- `TR_b` → combustor (how much the burning raised the temperature)
- `PR_t` → turbine section
- `TR_overall`, `PR_overall` → the engine end to end

## "Corrected" speed and fuel

The last two features, `RPM_corr` and `Fuel_corr`, are **corrected** versions of
engine speed and fuel flow. "Corrected" is engineering jargon for "adjusted to
remove the effect of inlet conditions, so you can compare apples to apples."

Dividing RPM by `sqrt(T2)` cancels out how much of the speed reading is just due
to hot or cold incoming air. After correction, two readings taken on different
days at different altitudes become directly comparable. The comment says it
plainly:

> *"Corrected speed/fuel remove the dependence on inlet conditions so the model
> compares like with like."* — `physics.py:93`

You don't need the exact formulas. The point: these transformations let a
degradation show up as a *clean drift* in the feature, instead of being buried
under the noise of changing flight conditions.

## The full feature list

The model's final inputs are the raw sensors **plus** these engineered ones,
stitched together in order:

```python
MODEL_FEATURES = RAW_FEATURES + DERIVED_FEATURES
```
*(`src/model/physics.py:73`)*

That's 13 raw + 8 derived = 21 columns the network reads.

## Why `EngineID` is dropped: leakage

The raw data labels each row with which of the 10 engines it came from
(`EngineID`). That column is deliberately **left out** of the model's inputs.
Look back at `RAW_FEATURES` — it starts at `Cycle`, not `EngineID`. The reason is
in the comment:

```python
# Raw sensor columns fed to the model (EngineID is dropped — it is an identity
# label, not a physical signal, and keeping it would let the model memorise).
```
*(`src/model/physics.py:42`)*

This guards against **data leakage**: when a model accidentally learns a shortcut
that won't exist in real use. `EngineID` is just a name tag ("engine #7"). If the
model saw it, it could memorize "engine #7's answers are usually around here"
instead of actually learning to read the physics. It would look great on the
training data and fail on any engine it hadn't memorized. Since a name tag
carries no physical meaning, we remove the temptation entirely.

`Cycle` is *kept*, because unlike a name tag it is a genuine physical driver —
more cycles really does mean more wear. `EngineID` survives only as a *grouping
key* (to know which rows belong to the same engine, used later for per-engine
degradation), never as a model input. You can see it carried along separately in
`data.py` as `engine_train`/`engine_val` (`src/model/data.py:48`).

## StandardScaler: putting features on the same footing

The 21 features live on wildly different numeric scales. `Pamb_Pa` might be
~40,000 while `Mach` is ~0.14 and a ratio like `PR_c` is ~3. To a neural network,
a raw value of 40,000 *looks* enormously more important than 0.14 — purely
because of its size, not its meaning. That's a problem.

**StandardScaler** (from the scikit-learn library) fixes this. For each column it
computes the average and the typical spread, then rewrites every value as "how
many spreads above or below average am I." After scaling, every feature is
centered near 0 with a comparable range, so the model can weigh them on merit.

Analogy: it's like grading five exams on a curve so a 92 in an easy class and a
55 in a brutal one become comparable "how far from typical" scores.

In `data.py`:

```python
x_scaler = StandardScaler().fit(tr[MODEL_FEATURES].values)
```
*(`src/model/data.py:85`)*

Thrust gets its *own* separate scaler because it's a huge number (tens of
thousands of Newtons) and needs the same centering treatment before the model
predicts it:

```python
thrust_scaler = StandardScaler().fit(tr[["Thrust_N"]].values)
```
*(`src/model/data.py:86`)*

(Health targets are already 0–1, so they need no scaling; TSFC is small and left
as-is — see the note at `data.py:96`.)

## Why the scaler is fit on the *training* data only

This is subtle but important. Notice both `.fit(...)` calls above use `tr` — the
training rows only, not the validation or test rows. The comment in `data.py`
spells out why:

```python
# Fit scalers on training rows only.
```
*(`src/model/data.py:84`)*

"Fitting" the scaler means *measuring* each column's average and spread. If we
measured those using the test data too, we'd be sneaking information about the
test set into the model's preparation — another form of leakage. The test set is
supposed to stand in for *future, unseen* engine data, and in the future you
obviously can't peek at data you don't have yet.

So the rule is: **learn the scaling recipe from training data only, then apply
that same fixed recipe to everything else.** You can see the recipe being
*applied* (not re-learned) to new data at prediction time:

```python
X = self.x_scaler.transform(feats[MODEL_FEATURES].values)
```
*(`src/model/predict.py:81`)*

`.transform` uses the already-learned averages and spreads; it never re-measures.
This keeps the evaluation honest.

## Recap

- Raw pressures/temps get turned into **ratios** so degradation shows through
  changing flight conditions.
- Speed and fuel get **corrected** so different conditions are comparable.
- `EngineID` is **dropped** to prevent the model from memorizing name tags
  (leakage); `Cycle` is kept because aging is real physics.
- **StandardScaler** puts all features on a common scale, **fit on train only**
  so no future information leaks in.

Next: **file 4**, the neural network that finally consumes these 21 prepared
features.
