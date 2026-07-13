# 6. Uncertainty and Remaining Useful Life

The surrogate is trained (files 4–5). This final model file covers what happens
when we *use* it on new sensor data, and the two features that turn a bare number
into something you'd trust on a dashboard:

1. A **confidence score** — how sure is the model of this reading?
2. **RUL (Remaining Useful Life)** — how many flights until this engine needs
   maintenance?

It maps to `src/model/predict.py` and the life-estimate functions in
`src/model/physics.py`.

## The problem with a single prediction

A lone number like "compressor health = 0.88" tells you nothing about whether to
believe it. Is the model confident, or barely guessing? For anything
safety-related, a prediction without a confidence is close to useless. We want
the model to say "0.88, and I'm sure" versus "0.88, but honestly I'm shaky."

## Bootstrap-ensemble spread: confidence through agreement

File 4 introduced the **bootstrap ensemble**: for each target we trained not one
gradient-boosting model but **ten**, each on a different random resample of the
240 training rows. Each member ends up a slightly different expert. Their
*agreement* is the confidence signal:

- If all ten members **agree** closely → the region is well-understood →
  **high confidence**.
- If they **disagree** widely → the input is unfamiliar or ambiguous →
  **low confidence**.

Analogy: ask ten trained inspectors to rate the same engine from its readings. If
all ten say "healthy," you relax. If they split and argue, you get more opinions.
The *spread* of opinions is itself the confidence signal.

Note what this is **not**. It is not a deep neural-network ensemble, and it is not
MC-dropout (a neural-network uncertainty trick). It is simply the standard
deviation across ten bootstrap-trained tree models. The header of `predict.py`
says it plainly (`src/model/predict.py:7`):

> *"Confidence comes from the bootstrap-ensemble spread: agreement between members
> = high confidence, disagreement = low."*

### How it looks in code

At prediction time the surrogate runs all ten members and reports both the mean
and the spread for each target (`src/model/models.py:100`):

```python
for target in HEALTH_TARGETS:
    preds = np.clip(self._member_predict(target, X), 0.0, 1.0)
    out[target] = {"mean": preds.mean(0), "std": preds.std(0)}
```

- `mean` (average of the ten) → the **prediction**.
- `std` (**standard deviation** — a number for "how spread out are these ten
  values") → the **uncertainty**. Small = tight agreement; large = scattered.

### Turning spread into a 0–1 confidence

A raw spread isn't friendly to read, so `predict.py` converts it into a tidy
confidence between 0 and 1 — small spread → near 1, large spread → near 0
(`src/model/predict.py:44`):

```python
def _confidence(std, scale):
    """Map ensemble std to a 0–1 confidence, normalised by the target's spread."""
    return np.clip(1.0 - std / max(scale, 1e-9), 0.0, 1.0)
```

The `scale` sets "how much disagreement counts as a lot" for each target. It's
the natural spread of that target across the data, captured at training time
(`src/model/models.py:82`):

```python
self.conf_scale[target] = float(np.std(y[target])) or 1.0
```

`np.clip(..., 0, 1)` keeps the result inside 0–1. So each dashboard number arrives
paired with a confidence, e.g. `CompressorHealth = 0.94, CompressorHealth_conf =
0.97` (`src/model/predict.py:71`):

```python
for name in HEALTH_TARGETS:
    out[name] = pred[name]["mean"]
    out[f"{name}_conf"] = _confidence(pred[name]["std"], scale[name])
```

Because TSFC is *derived* per member from thrust (file 5), its spread — and hence
its confidence — is inherited from the thrust ensemble automatically. No separate
uncertainty machinery is needed for it.

## Remaining Useful Life (RUL)

Confidence tells you how much to trust *this moment's* reading. **RUL** looks to
the *future*: given how the engine has been degrading, how many more cycles until
it crosses the "needs maintenance" line?

### Step 1: fit a degradation line

For one engine, take its predicted health at each cycle it has run and draw the
**best straight line** through those points — a simple trend. The line's downward
tilt (its **slope**) is the health lost per cycle (`src/model/physics.py:148`):

```python
def fit_degradation(cycles, health):
    ...
    slope, intercept = np.polyfit(cycles, health, 1)
    return float(slope), float(intercept)
```

`np.polyfit(..., 1)` fits a degree-1 polynomial — a straight line. The comment
justifies the simplicity: *"Degradation over the observed window is close to
linear; the slope is the per-cycle health loss used for extrapolation."* The
**slope** is how fast health drops; the **intercept** is where the line starts.

### Step 2: extend the line to the failure threshold

We define a health level below which the engine is treated as "failed" for
planning. Here it's **0.80** on OverallHealth (`src/model/physics.py:40`):

```python
RUL_THRESHOLD = 0.80
```

Then we *extrapolate* — follow the fitted line forward — and solve for the cycle
where it drops to 0.80. RUL is that crossing cycle minus the current cycle
(`src/model/physics.py:162`):

```python
def remaining_useful_life(current_cycle, slope, intercept, threshold=RUL_THRESHOLD):
    if slope >= 0:
        return None  # not degrading — no finite RUL on current trend
    cross_cycle = (threshold - intercept) / slope
    rul = cross_cycle - current_cycle
    return max(int(round(rul)), 0)
```

In plain terms: *cycles until the fitted health line crosses the threshold.* Two
sensible guards: if the trend is flat or improving (`slope >= 0`) it returns
`None` — the engine isn't degrading, so there's no finite deadline — and
`max(..., 0)` prevents a negative RUL (never "you had −3 flights left").

Analogy: a phone battery draining from 100%. Note how fast it drops (slope), draw
the line to your "recharge now" mark (0.80), and read how long until you hit it.
RUL is exactly that for engine health.

The whole per-engine process is assembled in `engine_trajectory`
(`src/model/predict.py:83`): predict health across all the engine's cycles, fit
the line, compute slope and RUL for each component, and attach a recommendation.

## Recommendation bands

Numbers still need translating into an action a technician can take. `predict.py`
maps each health value to a plain-English recommendation using a small band table
(`src/model/predict.py:29`):

```python
RECOMMENDATIONS = [
    (0.95, "nominal — no action"),
    (0.90, "monitor"),
    (0.85, "inspect at next opportunity"),
    (0.00, "schedule maintenance"),
]
```

`recommend` walks down the table and returns the first band the health clears
(`src/model/predict.py:37`):

- health **≥ 0.95** → *nominal — no action*
- **0.90–0.95** → *monitor*
- **0.85–0.90** → *inspect at next opportunity*
- **below 0.85** → *schedule maintenance*

So a component at 0.88 gets "inspect at next opportunity" — a graduated, sensible
response, not a raw decimal.

## Evaluating the model: eval.py

Separately, `eval.py` measures how *accurate* the trained surrogate is, using the
held-out `test.csv` the model never trained on. For each of the six targets it
reports two standard scores (`src/model/eval.py:37`):

- **MAE** (Mean Absolute Error) — the average size of the miss, in the target's
  own units. Lower is better.
- **R²** (R-squared) — what fraction of the real variation the model captures, on
  a scale where 1.0 is perfect. Higher is better.

It also records the two **physics-consistency residuals** from file 5 (TSFC
identity ~0, overall-aggregation ~0.004) and the **inference latency** per row.
All of it is written to `artifacts/metrics.json` for the technical report — the
honesty check that the model works on data it has never seen.

## The full picture

Putting files 1–6 together, one row of raw sensor readings becomes a trustworthy
dashboard entry like this:

1. Raw sensors → engineered physics ratios + corrected features (files 2–3).
2. Fed to ten bootstrap-trained tree ensembles per target; their **mean** is the
   prediction, their **spread** is the confidence (file 4, this file).
3. Health, thrust and TSFC come out — kept physically sensible by the **hard
   constraints** (monotonic health, derived TSFC) from file 5.
4. Across an engine's cycles, the health trend is fitted and extended to give
   **RUL** and a plain-English **recommendation** (this file).

That's the complete model: hidden health, made visible, fast, with physics and a
confidence you can act on. Files 7–9 then carry these predictions out through the
backend API and onto the 3D dashboard.
