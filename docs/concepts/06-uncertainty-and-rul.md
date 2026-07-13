# 6. Uncertainty and Remaining Useful Life

The model is trained (file 5). This final file covers what happens when we
*use* it on new sensor data, and two features that turn a bare prediction into
something you'd actually trust on a dashboard:

1. A **confidence score** — how sure is the model?
2. **RUL (Remaining Useful Life)** — how many flights until this engine needs
   maintenance?

It maps to `predict.py` and the life-estimate functions in `physics.py`.

## The problem with a single prediction

A lone number like "compressor health = 0.88" tells you nothing about whether to
believe it. Is the model confident, or is it barely guessing? For anything
safety-related, a prediction without a confidence is close to useless. We need
the model to say "0.88, and I'm sure" versus "0.88, but honestly I'm shaky."

## Deep ensemble: measuring confidence through agreement

The trick this project uses is a **deep ensemble**. Instead of training one
network, we train **five** — identical in design but each starting from a
different random seed, so each ends up a slightly different "expert." At
prediction time we ask all five and compare their answers:

- If all five **agree** closely → the region is well-understood → **high
  confidence**.
- If they **disagree** widely → the input is unfamiliar or ambiguous → **low
  confidence**.

Analogy: ask five trained doctors to read the same scan. If all five say "looks
healthy," you relax. If they split three-to-two and argue, you get a sixth
opinion. The *spread* of opinions is itself the confidence signal. The header of
`train.py` states this directly:

> *"Deep ensemble = train N independent networks from different random seeds.
> Their mean is the prediction; their spread (standard deviation) is the
> uncertainty — a cheap, robust way to get calibrated confidence on a small
> dataset."* — `src/model/train.py:1`

### How it looks in code

Training saves five separate model files (`train.py:135`):

```python
for i in range(N_MODELS):                     # N_MODELS = 5
    net, val = train_one(bundle, seed=1000 + i, device=device)
    torch.save(net.state_dict(), ARTIFACTS / f"net_{i}.pt")
```

At prediction time, `predict.py` loads all five and runs every one on the input
(`predict.py:84`):

```python
for net in self.models:
    ph, pp = net(Xt)
    healths.append(ph.cpu().numpy())
    ...
return np.stack(healths), np.stack(thrusts), np.stack(tsfcs)
```

Then it takes the **mean** (the prediction) and the **standard deviation** (the
spread) across the five (`predict.py:99`):

```python
h_mean, h_std = H.mean(0), H.std(0)     # average and spread across models
```

**Standard deviation** is just a number for "how spread out are these five values"
— small means they clustered together (agreement), large means they scattered
(disagreement).

### Turning spread into a 0–1 confidence

A raw spread isn't friendly to read. `confidence_from_std` converts it into a
tidy confidence between 0 and 1 — small spread → near 1, large spread → near 0
(`predict.py:50`):

```python
def confidence_from_std(std: float, scale: float) -> float:
    return float(np.clip(1.0 - std / scale, 0.0, 1.0))
```

The `scale` sets "how much disagreement counts as a lot" for each kind of target
(e.g. `0.10` for the 0–1 healths, `predict.py:108`), and `np.clip(..., 0, 1)`
keeps the result inside 0–1. So each dashboard number arrives paired with a
confidence, e.g. `CompressorHealth = 0.88, CompressorHealth_conf = 0.96`.

## Remaining Useful Life (RUL)

Confidence tells you how much to trust *this moment's* reading. **RUL** looks to
the *future*: given how the engine has been degrading, how many more cycles
(flights) until it crosses the "needs maintenance" line?

### Step 1: fit a degradation line

For a single engine, we take its predicted health at each cycle it has run and
draw the **best straight line** through those points — a simple trend. The line's
downward tilt (its **slope**) is the health lost per cycle:

```python
def fit_degradation(cycles, health):
    ...
    slope, intercept = np.polyfit(cycles, health, 1)
    return float(slope), float(intercept)
```
*(`src/model/physics.py:191`)*

`np.polyfit(..., 1)` is "fit a degree-1 polynomial" — i.e. a straight line. The
comment justifies the simplicity: *"Degradation over the observed window is close
to linear; the slope is the per-cycle health loss used for extrapolation"*
(`physics.py:191`). The **slope** is how fast health drops; the **intercept** is
where the line starts.

### Step 2: extend the line to the failure threshold

We define a health level below which the engine is considered "failed" for
planning purposes. Here it's **0.80** on OverallHealth:

```python
RUL_THRESHOLD = 0.80
```
*(`src/model/physics.py:35`)*

Then we *extrapolate* — follow the fitted line forward — and solve for the cycle
where it drops to 0.80. RUL is that crossing cycle minus the current cycle:

```python
def remaining_useful_life(current_cycle, slope, intercept, threshold=RUL_THRESHOLD):
    if slope >= 0:
        return None  # not degrading — no finite RUL on current trend
    cross_cycle = (threshold - intercept) / slope
    rul = cross_cycle - current_cycle
    return max(int(round(rul)), 0)
```
*(`src/model/physics.py:205`)*

In plain terms: *"Cycles until the fitted health line crosses the threshold."* Two
sensible guards: if the trend is flat or improving (`slope >= 0`) it returns
`None` — the engine isn't degrading, so there's no finite deadline
(`physics.py:211`). And `max(..., 0)` prevents a negative RUL (never "you had −3
flights left").

Analogy: a phone battery draining from 100%. Note how fast it's dropping (slope),
draw the line to your "recharge now" mark (0.80), and read off how long until you
hit it. RUL is exactly that for engine health.

The whole per-engine process is assembled in `engine_trajectory`
(`predict.py:118`): predict health across all the engine's cycles, fit the line,
compute the slope and RUL for each component, and attach a recommendation.

## Recommendation bands

Numbers still need translating into an action a technician can take. `predict.py`
maps each health value to a plain-English recommendation using a small band table
(`predict.py:35`):

```python
RECOMMENDATIONS = [
    (0.95, "nominal — no action"),
    (0.90, "monitor"),
    (0.85, "inspect at next opportunity"),
    (0.00, "schedule maintenance"),
]
```

The `recommend` function walks down the table and returns the first band the
health clears (`predict.py:43`):

- health **≥ 0.95** → *nominal — no action*
- **0.90–0.95** → *monitor*
- **0.85–0.90** → *inspect at next opportunity*
- **below 0.85** → *schedule maintenance*

So a component at 0.88 gets "inspect at next opportunity" — a graded, sensible
response, not a raw decimal.

## Grading the model: eval.py

Separately, `eval.py` measures how *accurate* the trained ensemble is, using the
held-out `test.csv` the model never trained on (file 3). For each of the six
targets it reports two standard scores (`eval.py:33`):

- **MAE** (Mean Absolute Error) — the average size of the miss, in the target's
  own units. Lower is better.
- **R²** (R-squared) — what fraction of the real variation the model captures, on
  a scale where 1.0 is perfect. Higher is better.

These get written to `artifacts/metrics.json` for the project's technical report.
This is the honesty check: it proves the model works on data it has never seen,
not just on what it was trained on.

## The full picture

Putting files 1–6 together, one row of raw sensor readings becomes a trustworthy
dashboard entry like this:

1. Raw sensors → engineered ratios + corrected features, scaled (files 2–3).
2. Fed to five trained networks; their **mean** is the prediction, their
   **spread** is the confidence (this file).
3. Health, thrust, and TSFC come out — each kept physically sensible by the
   physics-informed training (file 5).
4. Across an engine's cycles, the health trend is fitted and extended to give
   **RUL** and a plain-English **recommendation** (this file).

That's the complete model: hidden health, made visible, fast, with physics and a
confidence you can act on.
