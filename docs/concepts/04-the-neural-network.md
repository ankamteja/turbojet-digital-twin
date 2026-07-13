# 4. The Neural Network

We have 21 clean, scaled input features (file 3). Now for the "apprentice" itself
— the thing that maps those inputs to the six answers. It's a **neural network**.
This file explains what that is in plain terms and walks through the exact one in
`net.py`, which is short: about 25 lines.

## What is a neural network?

A **neural network** is a mathematical function with lots of adjustable knobs.
You feed numbers in one end, they flow through layers of simple operations, and
numbers come out the other end. During training (file 5) the knobs are tuned
automatically until the outputs match the correct answers. That's all it is: a
big adjustable function that you shape by example rather than by writing rules.

Analogy: imagine a wall of thousands of tiny volume dials wired between an input
and an output. At first they're set randomly and the output is nonsense. You show
it example after example and, each time, nudge the dials a hair toward the right
answer. After enough examples the dial settings encode a working input→output
relationship — even though nobody ever wrote down the rule.

### Neurons, layers, and weights

- A **neuron** takes several input numbers, multiplies each by a **weight** (one
  of the adjustable knobs), adds them up, and passes the total through a simple
  bend called an **activation function**.
- A **layer** is a row of neurons working in parallel.
- Stacking layers so each feeds the next lets the network combine simple patterns
  into complex ones. Early layers might notice "pressure ratio is low"; later
  layers combine such observations into "compressor looks worn."

### What "MLP" means

The header of `net.py` calls the model *"a compact multi-head MLP"*
(`src/model/net.py:1`). **MLP** stands for **Multi-Layer Perceptron** — the plain
vanilla neural network: a stack of fully connected layers, one after another,
input to output. "Fully connected" means every neuron in a layer receives from
every value in the previous layer. It's the simplest useful design, and for a
small dataset like this (240 training rows) simple is exactly what you want.

## The building blocks in this network

The layers are assembled with PyTorch (a machine-learning library). You'll see
four kinds:

- **`nn.Linear(a, b)`** — a fully connected layer taking `a` numbers in and
  producing `b` numbers out. This is where the weights live.
- **`nn.ReLU()`** — the activation function. ReLU ("Rectified Linear Unit") is a
  tiny rule: keep positive numbers as-is, turn negatives into 0. That little bend
  is what lets the network learn curved, non-straight-line relationships. Without
  it, stacking layers would collapse into one boring straight line.
- **`nn.Dropout(p)`** — a regularizer (explained below).
- **`nn.Sigmoid()`** — squashes any number into the range 0–1 (explained below).

## The shape of this model: a shared trunk with two heads

Here's the "multi-head" idea, and it's intuitive. The network is a **tree**:

```
             21 scaled features
                     │
              ┌──────────────┐
              │ SHARED TRUNK │   3 layers: 128 → 96 → 64 neurons
              └──────────────┘   (learns a general "engine state" summary)
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
 ┌─────────────┐          ┌──────────────┐
 │ HEALTH HEAD │          │  PERF HEAD   │
 │ 4 outputs   │          │  2 outputs   │
 │ 0–1 sigmoid │          │  thrust,TSFC │
 └─────────────┘          └──────────────┘
```

- The **shared trunk** is the common bottom of the tree. Every input passes
  through it, and it distills the 21 features into a compact internal summary of
  "what state is this engine in right now." As the header says: *"One shared
  trunk learns a general representation of the engine's operating state"*
  (`net.py:1`).
- The two **heads** branch off that shared summary, each specializing in one job.
  The **health head** outputs the four health scores; the **performance head**
  outputs thrust and TSFC.

Why share a trunk instead of building two separate networks? Because health and
performance depend on *the same underlying physics* — a worn compressor affects
both how healthy the engine is *and* how much thrust it makes. Sharing forces the
network to learn one honest picture of the engine and lets both jobs benefit from
it. It's like a single medical exam feeding both a "fitness score" and a
"performance estimate," rather than running two unrelated exams.

## Reading the actual code

Here is the whole model (`src/model/net.py:20`):

```python
class TurbojetNet(nn.Module):
    def __init__(self, n_features: int, hidden=(128, 96, 64), dropout=0.10):
        super().__init__()

        # shared trunk
        layers = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)

        # health head: 4 outputs squashed to [0, 1]
        self.health_head = nn.Sequential(
            nn.Linear(prev, 32), nn.ReLU(),
            nn.Linear(32, 4), nn.Sigmoid(),
        )

        # performance head: 2 outputs (scaled thrust, TSFC), unbounded
        self.perf_head = nn.Sequential(
            nn.Linear(prev, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        z = self.trunk(x)
        return self.health_head(z), self.perf_head(z)
```

Let's walk through it.

**The trunk.** The loop over `hidden=(128, 96, 64)` builds three blocks. Each
block is a `Linear` layer followed by `ReLU` and `Dropout`. So the data goes:
21 features → 128 neurons → 96 → 64. The widths *shrink* on purpose — the network
progressively compresses the raw inputs into a tighter, more meaningful 64-number
summary. `nn.Sequential(*layers)` just chains these blocks together in order.

**The health head** takes the trunk's 64-number summary, runs it through a small
32-neuron layer, then down to **4 outputs**, and finishes with **`nn.Sigmoid()`**.

**The performance head** is the same shape but ends at **2 outputs** with **no
sigmoid** — thrust and TSFC are left "unbounded."

**`forward`** is the recipe for one pass: send `x` through the `trunk` to get the
summary `z`, then feed `z` to both heads and return both results as a pair. This
is exactly what gets called during training and prediction: `ph, pp = net(Xtr)`
(you'll see that in file 5).

## Why sigmoid on the health head (but not on performance)

The four healths are, by definition, numbers between 0 and 1 (file 2). **Sigmoid**
is a function that takes *any* number and gently squashes it into the range 0 to
1 — big positives approach 1, big negatives approach 0. Putting a sigmoid at the
end of the health head means the model **physically cannot** output an impossible
health like 1.7 or −0.3. The valid range is baked into the network's shape. The
comment says it directly: *"4 outputs squashed to [0, 1]"* (`net.py:32`).

The performance head has **no** sigmoid because thrust (tens of thousands of
Newtons, in scaled form) and TSFC are not confined to 0–1 — they need to range
freely, so we leave them "unbounded" (`net.py:38`).

## What dropout does

**Dropout** fights **overfitting**. Overfitting is when a model memorizes the
training examples — including their random noise — instead of learning the real
pattern, so it aces the training data but flunks anything new. With only 240
training rows, this network is very prone to it.

During each training step, `nn.Dropout(0.10)` randomly *switches off* 10% of the
trunk's neurons, chosen fresh each time. The network is thus never allowed to
lean too hard on any single neuron; it has to spread its knowledge so it can
still perform when random pieces go missing. The result is a sturdier model that
generalizes better. Analogy: a student who studies from many partial sets of
notes learns the concepts, while one who memorizes a single perfect answer sheet
is lost the moment the questions change.

The header notes a bonus use: *"Dropout in the trunk regularises the small
dataset and also enables MC-dropout uncertainty as a fallback to the deep
ensemble"* (`net.py:1`). Don't worry about "MC-dropout" — file 6 explains the
main way this project measures confidence (the deep ensemble).

## Recap

- A **neural network** is an adjustable function tuned by examples; this one is a
  simple **MLP**.
- It has a **shared trunk** (learns a general engine-state summary) feeding two
  **heads** (health, performance) — that's "multi-head."
- **Sigmoid** on the health head guarantees valid 0–1 health scores; the
  performance head is left unbounded.
- **Dropout** randomly disables neurons during training to prevent memorizing the
  small dataset.

Next: **file 5**, how this network actually gets *trained* — and how physics gets
baked into the training itself.
