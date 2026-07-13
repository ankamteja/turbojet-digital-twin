# 1. What Is a Digital Twin?

This is the first stop in a beginner's tour of the **model** — the part of the
project that turns raw engine sensor readings into a live, understandable
picture of an engine's health. You don't need to know anything about jets,
machine learning, or math to follow along. We'll build up every idea in plain
language first, then point to the exact code that does it.

## The one-sentence idea

A **digital twin** is a software copy of a real physical thing that stays in
sync with it. As the real object runs, the twin receives its measurements and
mirrors its state — so you can look at the twin instead of tearing the real
thing apart.

Think of a fitness watch. It doesn't cut you open to check your heart. It reads
a few signals from your wrist (pulse, motion, skin temperature) and builds a
running picture of your body: heart rate, steps, sleep quality, "you seem
stressed." That running picture is a tiny digital twin of *you*. This project
builds the same kind of thing for a jet engine.

## What is a turbojet, briefly?

A **turbojet** is a type of jet engine. Air is sucked in the front, squeezed
hard by a spinning **compressor**, mixed with fuel and set alight in the
**combustor**, and the hot high-pressure gas blasts out the back through a
**turbine**, producing forward push called **thrust**. That's the whole engine
for our purposes: three main moving parts (compressor, combustor, turbine) and
one job (make thrust).

## Why does a turbojet need a twin?

Two hard facts about real engines:

1. **Parts wear out invisibly.** Over hundreds of flights the compressor gets
   dirty ("fouling"), the turbine blades erode, the combustor burns less
   cleanly. These problems quietly steal thrust and waste fuel long before
   anything breaks. The README puts it directly: these health states
   *"cannot be measured directly in flight."* There is no dashboard gauge that
   reads "compressor is 84% healthy." You only have indirect clues.

2. **The honest way to know the health is too slow.** Engineers *can* compute an
   engine's exact behavior with a huge, detailed physics simulation — but it is,
   in the README's words, *"far too expensive to run in real time."* You can't
   run a supercomputer simulation for every second of every flight.

So we're stuck: the thing we care about (health) can't be measured, and the
accurate way to figure it out is too slow. A digital twin is the way out.

## "Hidden health": the core problem

The engine gives us **sensors** we *can* read — pressures and temperatures at a
few points, engine speed, fuel flow, altitude, speed. Call these the
**observable** signals.

What we actually want is **hidden**: how healthy each subsystem is right now.
"Hidden" doesn't mean secret; it means *not directly instrumented*. There's no
wire coming out of the engine carrying the number "turbine health = 0.91."

The whole trick of this project is: **infer the hidden health from the
observable sensors.** A dirty compressor leaves fingerprints in the pressures
and temperatures. If we learn to read those fingerprints, we can estimate the
health we can't measure. That estimation is the heart of the model.

## The surrogate-model idea

A **surrogate** is a fast stand-in for something slow. Instead of running the
giant, accurate physics simulation every time, we build a lightweight model that
*imitates* it well enough and runs instantly.

Analogy: a world-class chef (the slow, accurate simulation) can taste a soup and
tell you exactly what's in it — but you can't have that chef in your kitchen at
all times. So you train an apprentice by having them taste thousands of soups
next to the chef's verdicts. Eventually the apprentice gives near-chef answers in
a second. The apprentice is a *surrogate* for the chef.

In this project the "apprentice" is a small **machine-learning model** — an
ensemble of decision trees (explained in file 4). It is trained on a dataset
where, for many engine states, we already know the true hidden health — the
"chef's verdicts." After training it can look at brand-new sensor readings and
estimate the health on its own, fast enough to power a live dashboard.

The README names this exactly: *"This project takes the surrogate-model
approach: a computationally cheap, interpretable model that approximates engine
behavior, estimates hidden health indicators from available telemetry, and
updates continuously as new measurements arrive."*

## Where this lives in the code

The whole surrogate lives in `src/model/`. Each file is one stage of the tour:

| File | What it does | Covered in |
|------|--------------|-----------|
| `data.py` | Loads the data, prepares it, attaches features | files 2–3 |
| `physics.py` | Engine knowledge: features, physics rules, life estimates | files 2, 3, 5, 6 |
| `models.py` | The surrogate model itself (the "apprentice") | files 4–5 |
| `train.py` | Teaches the model from the data | files 4–5 |
| `predict.py` | Uses the trained model on new readings; adds confidence + life estimate | file 6 |
| `eval.py` | Measures how accurate the trained model is | file 6 |

The very top of `models.py` even names the model by its role:

```
"""The surrogate model — a physics-constrained gradient-boosting ensemble.
```
*(`src/model/models.py:1`)*

Don't worry about "gradient-boosting ensemble" yet — file 4 unpacks it. For now,
hold onto three ideas:

- A **digital twin** mirrors a real engine from its sensors.
- The engine's **health is hidden** and must be *inferred*, not read.
- We infer it with a **surrogate model**: a fast imitator of an accurate but slow
  physics simulation.

Next: **file 2**, where we look at the actual sensors and the actual things we're
trying to predict.
