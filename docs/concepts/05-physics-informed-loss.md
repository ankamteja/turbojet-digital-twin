# 5. Physics-Informed Loss

We have a network with tunable knobs (file 4). Training is the process of tuning
those knobs so the outputs become correct. This file explains *how* training
works in plain terms, then explains the special ingredient that makes this
project "physics-informed": extra rules that teach the model to respect the laws
of the engine, not just match numbers. It maps to `train.py` and the loss
functions in `physics.py`.

## What is a loss function?

A **loss function** is a single number that measures *how wrong* the model
currently is. Big loss = very wrong; zero loss = perfect. Training is nothing
more than **turning the knobs to make that number as small as possible.**

The most common loss is **MSE**, Mean Squared Error: take each prediction, find
how far it is from the true answer, square that gap (so being off by 2 counts as
4× worse than being off by 1, and so negatives don't cancel positives), and
average over all examples. In the code, MSE is one line:

```python
mse = nn.MSELoss()
```
*(`src/model/train.py:57`)*

## How training actually works

Four ideas, each in plain language.

### Gradient descent

Imagine the loss as a hilly landscape and the model's knob settings as your
position on it. You want the lowest valley (smallest loss). **Gradient descent**
is the rule: feel which way is downhill from where you stand, take a small step
that way, repeat. Do this thousands of times and you roll down into a valley. The
"feel which way is downhill" part is done automatically by PyTorch computing
*gradients* (the slope of the loss with respect to each knob). The stepping is
done by an **optimizer**; this project uses one called Adam:

```python
opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
```
*(`src/model/train.py:56`)*

`LR` is the **learning rate** (`2e-3` = 0.002), the *size* of each downhill step.
Too big and you overshoot the valley; too small and you crawl. `weight_decay`
gently discourages the knobs from growing too large — another anti-overfitting
nudge, like dropout.

### Epochs

One **epoch** is one full pass through the training data. Since one step rarely
gets you to the valley, you repeat for many epochs. This model allows up to 600:

```python
EPOCHS = 600
```
*(`src/model/train.py:30`)*

You can see the loop and the three core moves of every step in `train.py:75`:

```python
for epoch in range(EPOCHS):
    net.train()
    opt.zero_grad()          # forget last step's slope
    ph, pp = net(Xtr)        # run the network → health(4), perf(2)
    ...
    loss.backward()          # compute the downhill direction
    opt.step()               # take one step
```

### Validation

We hold back a slice of the training data as a **validation set** — data the
model trains *on top of but is never taught from*. After each epoch we check the
loss on this held-out slice. Why? The training loss always keeps dropping (the
model can always memorize harder), but if it's *memorizing* rather than
*learning*, the validation loss stops improving. Validation is our honest
early-warning gauge of real-world performance. `data.py` carves out 20% of the
data for this (`load_dataset(val_fraction=0.2)`, `data.py:69`), and `train.py`
measures validation loss every epoch (`train.py:104`).

### Early stopping

If validation loss stops improving, continuing only overfits. **Early stopping**
watches the validation score and halts once it hasn't improved for a while,
keeping the best version seen. The waiting period is called **patience**:

```python
PATIENCE = 80
```
*(`src/model/train.py:34`)*

The logic (`train.py:112`): each epoch, if validation loss beat the previous best,
save this version as `best_state` and reset the counter; otherwise count up, and
if 80 epochs pass with no improvement, stop. At the end it restores the best
version: `net.load_state_dict(best_state)` (`train.py:121`). So even though 600
epochs are allowed, training usually quits earlier at its best point.

## The base (supervised) loss

"Supervised" means "learning from labeled examples where we know the right
answer." The base loss has three pieces, one per output group (`train.py:80`):

```python
l_health = mse(ph, yh)              # 4 healths vs their true values
l_thrust = mse(pp[:, 0], yp[:, 0])  # thrust vs true thrust (scaled)
l_tsfc_sup = mse(pp[:, 1], yp[:, 1])# TSFC vs true TSFC
```

TSFC is a tiny number (~0.01), so its raw error is tiny too and the model would
ignore it. To compensate, its loss is scaled up 5× (`W_TSFC = 5.0`,
`train.py:37`) so the model takes it seriously.

## What makes it *physics-informed*

Here's the key idea of the whole project. A purely data-driven model only knows
"match these 240 labeled examples." With so little data it can find answers that
fit the examples yet are **physically nonsensical** — e.g. an engine that
mysteriously heals over time, or a thrust and fuel-efficiency that contradict
each other.

**Physics-informed** training adds extra loss terms that punish physically
impossible answers — *even though those terms need no labels*. They encode
relationships that must hold no matter what the true values are. The comment in
`physics.py` states it well:

> *"They do not require ground-truth labels — they encode relationships that
> must hold regardless of the labels, which is what makes the model
> 'physics-informed' rather than purely data-driven."* — `physics.py:118`

Think of it as giving the apprentice not just answer keys but also *rules of the
trade*: "wear never reverses," "efficiency equals fuel over thrust." The rules
rule out whole families of wrong answers the tiny answer key alone couldn't.

This project adds **four** physics loss terms. Each is a function in `physics.py`,
combined into the total loss in `train.py:92`.

### 1. TSFC consistency

Recall the iron law from file 2: TSFC = 1000 × fuel ÷ thrust. This term forces
the model's *predicted* TSFC to agree with what its *own predicted thrust* implies
given the known fuel flow:

```python
def loss_tsfc_consistency(pred_thrust_n, pred_tsfc, fuel_flow_kg_s):
    implied = tsfc_from_fuel_thrust(fuel_flow_kg_s, pred_thrust_n)
    return torch.mean((pred_tsfc - implied) ** 2)
```
*(`src/model/physics.py:125`)*

Intuitively: the model *"cannot invent a thrust and a TSFC that disagree with the
fuel it was told the engine is burning"* (`physics.py:126`). It ties the two
performance outputs into one coherent story. (Note in `train.py:86` the thrust is
un-scaled back to real Newtons first, because the physics equation only holds in
real units.)

### 2. Overall-health aggregation

The engine's `OverallHealth` shouldn't float free of its three subsystems — an
engine is only as healthy as its parts. This term pushes overall health toward
the average of compressor, combustor, and turbine health:

```python
def loss_overall_aggregation(pred_health):
    parts_mean = pred_health[:, :3].mean(dim=1)   # avg of the 3 subsystems
    overall = pred_health[:, 3]
    return torch.mean((overall - parts_mean) ** 2)
```
*(`src/model/physics.py:135`)*

Plain version: *"The engine is only as healthy as its subsystems; this stops the
overall index floating free of them"* (`physics.py:136`). It links the fourth
health output to the other three.

### 3. Monotonic degradation

**Monotonic** means "only moves in one direction." Wear is one-directional:
engines degrade, they don't spontaneously heal. So within a single engine,
predicted health should never *rise* as the cycle count grows. This term sorts
each engine's readings by cycle and penalizes any step where health went *up*
over time:

```python
def loss_monotonic_degradation(pred_health, cycle, engine_id):
    ...
    for eid in torch.unique(engine_id):
        ...
        deltas = h[1:] - h[:-1]                 # step-to-step change
        total = total + torch.mean(torch.clamp(deltas, min=0.0) ** 2)
```
*(`src/model/physics.py:148`)*

`torch.clamp(deltas, min=0.0)` keeps only the *positive* steps (health going up)
and zeroes the rest — so only the physically-wrong direction is punished; genuine
downward wear is left alone. It's a **soft** penalty (a nudge, not a hard ban) so
that a little measurement noise is tolerated. Note this term *needs* the grouping
key `engine_id` and `cycle` — the very columns we kept aside in file 3, so we
know which readings belong to the same engine's timeline.

### 4. Ratio–health coupling

The weakest of the four — a gentle "sign check." A healthy compressor achieves a
higher pressure ratio (`PR_c`). This term encourages predicted compressor health
and the (scaled) `PR_c` feature to move *together* by penalizing them moving
*apart*:

```python
def loss_ratio_health_coupling(pred_health, pr_c_norm):
    ...
    cov = torch.mean(comp_c * pr_c)
    return torch.clamp(-cov, min=0.0)
```
*(`src/model/physics.py:171`)*

The comment calls it a *"Weak guardrail... a guardrail on the sign, not a hard
law"* (`physics.py:171`). It only insists the relationship point the right way,
nothing more.

## Adding it all up

Every physics term carries a **weight** deciding how much it counts, set at the
top of `train.py` (`L_TSFC=0.5`, `L_OVERALL=0.5`, `L_MONO=0.2`, `L_RATIO=0.05`).
Bigger weight = stronger enforcement. Notice the ratio guardrail's weight (0.05)
is tiny — that's what "weak guardrail" means in numbers. The total loss the
optimizer minimizes stacks the supervised and physics pieces together
(`train.py:92`):

```python
loss = (
    l_health
    + W_THRUST * l_thrust
    + W_TSFC * l_tsfc_sup
    + L_TSFC * pl_tsfc
    + L_OVERALL * pl_overall
    + L_MONO * pl_mono
    + L_RATIO * pl_ratio
)
```

One important detail: the **validation** score (used for early stopping) uses
*only the supervised part* (`train.py:104`), so we judge "is this model actually
accurate?" on plain error, while the physics terms do their guiding job during
training.

## Why physics losses help *especially* on small data

With only 240 rows, there simply isn't enough data to pin down the right answer
from examples alone — countless different knob settings fit those 240 points, and
most are physically absurd. The physics terms act as **free extra evidence** that
costs no labels: they eliminate the absurd solutions (healing engines,
contradictory thrust/TSFC) and steer the model toward the physically plausible
ones. In effect, decades of engineering knowledge substitute for the thousands of
extra data rows we don't have.

Next: **file 6**, using the trained models to predict on new data — with a
confidence score and a remaining-life estimate.
