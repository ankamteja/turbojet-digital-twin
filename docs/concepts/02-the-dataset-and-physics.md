# 2. The Dataset and the Physics

In file 1 we said the model reads *observable sensors* and infers *hidden
health*. This file makes both of those concrete: what exactly the sensors are,
what the engine looks like inside, what we're trying to predict, and one simple
physics equation that ties two of the predictions together.

## The gas path and its "stations"

Inside a turbojet, air flows front to back along a route called the **gas path**.
Engineers put numbered labels, called **stations**, at key points so everyone
agrees on where a measurement was taken. This project uses three:

```
   air in                                                 hot gas out
     │                                                         │
     ▼                                                         ▼
 ┌────────┐ station2  ┌────────────┐ station3  ┌───────────┐ station4 ┌─────────┐
 │ intake │──────────▶│ compressor │──────────▶│ combustor │─────────▶│ turbine │──▶ thrust
 └────────┘           └────────────┘           └───────────┘          └─────────┘
             inlet of        exit of /              inlet of
             compressor      inlet to combustor     turbine
```

- **Station 2** = compressor inlet (air just before it gets squeezed)
- **Station 3** = compressor exit = combustor inlet (air after squeezing, before burning)
- **Station 4** = turbine inlet (hot gas after burning, before it drives the turbine)

At each station the engine measures a **pressure** (`P`) and a **temperature**
(`T`). So `P3_Pa` is "pressure at station 3, in Pascals" and `T4_K` is
"temperature at station 4, in Kelvin." Watching how pressure and temperature
climb from station 2 → 3 → 4 tells you how well each part is doing its job. This
station scheme is documented right in the code header:

```
Station numbering follows the gas path:
    2 = compressor inlet, 3 = compressor exit / combustor inlet, 4 = turbine inlet.
```
*(`src/model/physics.py:21`)*

## The input sensors, column by column

These are the measurements the model is *allowed to see*. They're listed in
`physics.py` as `RAW_FEATURES` (`src/model/physics.py:49`). "Feature" is just the
machine-learning word for "an input column."

| Column | Plain meaning |
|--------|---------------|
| `Cycle` | How many flights/runs this engine has done. The clock of aging. Higher cycle → more worn. |
| `Altitude_m` | How high the plane is flying, in meters. |
| `Mach` | Speed as a fraction of the speed of sound (Mach 0.8 = 80% of sound speed). |
| `Tamb_K` | Ambient (outside) air temperature, in Kelvin. |
| `Pamb_Pa` | Ambient (outside) air pressure, in Pascals. |
| `RPM_rev_min` | How fast the engine's shaft spins (revolutions per minute). |
| `FuelFlow_kg_s` | How much fuel is being burned per second, in kg/s. |
| `P2_Pa`, `T2_K` | Pressure & temperature at station 2 (compressor inlet). |
| `P3_Pa`, `T3_K` | Pressure & temperature at station 3 (compressor exit). |
| `P4_Pa`, `T4_K` | Pressure & temperature at station 4 (turbine inlet). |

`Altitude_m`, `Mach`, `Tamb_K`, `Pamb_Pa` describe the *flight conditions* (where
and how the plane is flying). The rest describe what the *engine* is doing. There
is also an `EngineID` column in the raw data that names which of the 10 engines a
row belongs to — but it is deliberately **not** given to the model as an input.
File 3 explains why.

## The dataset itself

From the README and `docs/architecture.md`: the data is **synthetic but
physics-based** — generated from a realistic turbojet simulation rather than a
real jet, but it obeys real thermodynamics. It contains **10 engines × 30 cycles
= 300 rows**. Each row is one snapshot of one engine at one point in its life.

That generating simulation is **not part of this repository** — the data arrives
ready-made as the CSV files in `data/`, and this project's job starts from those
files. (Note: the word "simulate" *does* appear in our own code, in
`src/backend/service.py`, but it means something different — projecting an
engine's health into *future* cycles beyond the data, covered in file 6 — not
generating the dataset.)

The data is split into files:

- `train.csv` (240 rows) — used to teach the model.
- `test.csv` (60 rows) — hidden from training, used only to evaluate the model later.
- `ground_truth.csv` (300 rows) — the *true* answers (health + performance) for
  every row, used to teach and to evaluate.

`data.py` loads the sensor inputs and joins them to their true answers by
matching on `EngineID` and `Cycle`:

```python
merged = inputs.merge(truth[keep], on=["EngineID", "Cycle"], how="left")
```
*(`src/model/data.py:33`)*

## The six things we predict (the "targets")

A **target** is a value the model is trying to output. There are six, listed as
`TARGETS` in `physics.py:81`. They come in two groups.

### The four healths

```python
HEALTH_TARGETS = TARGETS[:4]   # CompressorHealth, CombustorHealth,
                               # TurbineHealth, OverallHealth
```
*(`src/model/physics.py:90`)*

Each health is a number between **0 and 1**: `1.0` means "as good as new," and
lower means "degraded." Three of them are per-subsystem (compressor, combustor,
turbine); the fourth, **OverallHealth**, is a single summary score for the whole
engine. In this dataset the values stay fairly high — e.g. compressor health
ranges about 0.72–1.0, overall 0.80–1.0 (from `docs/architecture.md`) — because
we're watching engines *degrade gradually*, not catastrophically fail.

### The two performance numbers

```python
PERF_TARGETS = TARGETS[4:]     # Thrust_N, TSFC_g_N_s
```
*(`src/model/physics.py:91`)*

- **`Thrust_N`** — the forward push the engine produces, in Newtons (N). More
  thrust = more power. A degraded engine makes less thrust for the same effort.
- **`TSFC_g_N_s`** — **Thrust-Specific Fuel Consumption**, in grams of fuel per
  Newton of thrust per second. It measures *fuel efficiency*: how much fuel you
  burn to get a unit of push. **Lower is better** (less fuel for the same
  thrust). A degrading engine's TSFC creeps up — it burns more to do the same
  work.

## Why health "can't be measured directly"

Notice the split: the six targets include the four healths, but **health is not
in the sensor list**. There is no thermometer or pressure gauge that reads out
"compressor health." Health is an abstract measure of *how efficiently a part
converts effort into result* — it only reveals itself indirectly, through subtle
shifts in the pressures, temperatures, speed, and fuel it takes to hit a given
thrust.

That's exactly why we need the model. In the training data we happen to *know*
the true health (because the simulation that generated the data computed it), so
the model can learn the mapping "these sensor patterns ↔ this much wear." On a
real engine, where we'd only have the sensors, the trained model fills in the
hidden health.

## The one physics equation to remember: TSFC

Two of the six targets — thrust and TSFC — are not independent. They're linked by
a definition that always holds:

```
TSFC [g/(N·s)]  =  1000 × FuelFlow [kg/s]  ÷  Thrust [N]
```

In the code:

```python
TSFC_FROM_FUEL_THRUST = 1000.0
...
def tsfc_from_fuel_thrust(fuel_flow_kg_s, thrust_n):
    """Closed-form TSFC [g/(N.s)] from fuel flow and thrust."""
    thrust = np.maximum(np.asarray(thrust_n, dtype=float), 1.0)
    return TSFC_FROM_FUEL_THRUST * np.asarray(fuel_flow_kg_s, dtype=float) / thrust
```
*(`src/model/physics.py:36`, `physics.py:123`)*

In plain words: **fuel efficiency = fuel burned divided by push produced.** If
you know how much fuel is going in (`FuelFlow`, a sensor we *do* have) and how
much thrust comes out, you *automatically* know the fuel efficiency — it's just
one divided by the other. The `1000` is only a unit conversion (kg → g) so the
number lands in convenient grams. The comment notes this relation was *"Verified
against the dataset to hold within ~1%"* (`physics.py:34`).

Why does this matter so much? Because it's a free, unbreakable fact about the
engine that doesn't depend on any labels. Later (file 5) the model doesn't even
*try* to predict TSFC independently — it predicts thrust and then *derives* TSFC
from this equation, so the two performance numbers can never disagree with each
other or with the fuel the engine is actually burning. It's a first taste of what
"physics-informed" means: we don't just fit numbers, we build the model so it
respects relationships that physics guarantees.

Next: **file 3**, on how these raw sensors get reshaped into more informative
inputs before the model ever sees them.
