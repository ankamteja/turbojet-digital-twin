# The Model, Explained From Scratch

A beginner's tour of the **model** behind the turbojet digital twin — the code in
`src/model/` that turns raw engine sensor readings into a live picture of an
engine's hidden health. It assumes **no** background in jet engines, machine
learning, or math. Every term is defined the first time it appears, every concept
gets a plain-language analogy, and each idea is tied back to the exact line of
code that implements it.

## Big picture

A jet engine's parts wear out invisibly — you can't measure "how healthy the
compressor is" with any gauge. What you *can* measure is a handful of sensors:
pressures, temperatures, engine speed, fuel flow, altitude, and speed. This
project trains a small, fast **neural network** to read those sensors and infer
the hidden health, along with thrust and fuel efficiency. It reshapes the sensors
into physics-based ratios, trains five copies of the network under rules borrowed
from real thermodynamics (so it can't produce physically impossible answers),
and uses the five copies' *agreement* as a confidence score. It then extends each
engine's health trend into the future to estimate remaining useful life. The end
result: raw telemetry in, and a trustworthy dashboard of health scores,
performance numbers, confidence levels, maintenance recommendations, and time-to-
service out.

## Read in order

| # | File | What you'll learn |
|---|------|-------------------|
| 1 | [What Is a Digital Twin?](01-what-is-a-digital-twin.md) | Digital twins, why an engine needs one, "hidden health," the surrogate-model idea. |
| 2 | [The Dataset and the Physics](02-the-dataset-and-physics.md) | The gas path (stations 2/3/4), every sensor column, the six prediction targets, and the key TSFC = 1000·Fuel/Thrust relation. |
| 3 | [Feature Engineering](03-feature-engineering.md) | Why raw sensors become ratios, corrected speed/fuel, why `EngineID` is dropped (leakage), and what `StandardScaler` does. |
| 4 | [The Neural Network](04-the-neural-network.md) | What a neural net / MLP is, the shared-trunk-plus-two-heads design, why sigmoid for health, what dropout does. |
| 5 | [Physics-Informed Loss](05-physics-informed-loss.md) | Loss functions, how training works (gradient descent, epochs, validation, early stopping), and the four physics loss terms. |
| 6 | [Uncertainty and Remaining Useful Life](06-uncertainty-and-rul.md) | Deep ensembles and confidence, RUL from a fitted degradation line, recommendation bands, and how the model is graded. |

## The code these docs explain

All under `src/model/`:

- `physics.py` — engine knowledge: engineered features, physics loss terms, TSFC and RUL relations.
- `data.py` — loads the data, builds features, scales, splits into train/validation.
- `net.py` — the multi-head neural network.
- `train.py` — trains the ensemble with physics-informed loss and early stopping.
- `predict.py` — runs the ensemble on new data; adds confidence, RUL, recommendations.
- `eval.py` — grades accuracy on held-out test data.

For the broader system (backend + frontend dashboard), see
[`docs/architecture.md`](../architecture.md).
