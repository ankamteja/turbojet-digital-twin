"""Physics of the turbojet gas path.

Two jobs:

1. Turn raw station pressures/temperatures into *physically meaningful* features
   (pressure ratios, temperature ratios, corrected spool speed / fuel flow). A
   degrading component shows up as a drift in these ratios, so they are far more
   informative to the model than the raw numbers.

2. Provide the physics-informed loss terms and the closed-form relations
   (TSFC, RUL) that keep the model's predictions thermodynamically sensible.

Station numbering follows the gas path:
    2 = compressor inlet, 3 = compressor exit / combustor inlet, 4 = turbine inlet.
"""

from __future__ import annotations

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# TSFC (thrust-specific fuel consumption) is fuel burned per unit thrust.
#   TSFC [g/(N.s)] = 1000 * FuelFlow [kg/s] / Thrust [N]
# Verified against the dataset to hold within ~1%. This ties the two
# performance targets together and is used both as a feature-free check and as
# a physics loss.
TSFC_FROM_FUEL_THRUST = 1000.0

# Default end-of-life threshold on OverallHealth. Below this the engine is
# considered failed for remaining-useful-life (RUL) purposes.
RUL_THRESHOLD = 0.80


# --------------------------------------------------------------------------- #
# Feature engineering (numpy / pandas world)
# --------------------------------------------------------------------------- #

# Raw sensor columns fed to the model (EngineID is dropped — it is an identity
# label, not a physical signal, and keeping it would let the model memorise).
RAW_FEATURES = [
    "Cycle",
    "Altitude_m",
    "Mach",
    "Tamb_K",
    "Pamb_Pa",
    "RPM_rev_min",
    "FuelFlow_kg_s",
    "P2_Pa",
    "T2_K",
    "P3_Pa",
    "T3_K",
    "P4_Pa",
    "T4_K",
]

# Names of the engineered physics features (added by add_physics_features).
DERIVED_FEATURES = [
    "PR_c",       # compressor pressure ratio  P3/P2
    "TR_c",       # compressor temperature ratio T3/T2
    "TR_b",       # combustor temperature ratio  T4/T3
    "PR_t",       # turbine-section pressure ratio P4/P3
    "TR_overall", # overall temperature ratio T4/T2
    "PR_overall", # overall pressure ratio    P4/P2
    "RPM_corr",   # corrected spool speed  RPM / sqrt(T2)
    "Fuel_corr",  # corrected fuel flow    FuelFlow / (P2 * sqrt(T2))
]

# Full ordered feature list the model consumes.
MODEL_FEATURES = RAW_FEATURES + DERIVED_FEATURES

# Target columns the model predicts.
TARGETS = [
    "CompressorHealth",
    "CombustorHealth",
    "TurbineHealth",
    "OverallHealth",
    "Thrust_N",
    "TSFC_g_N_s",
]

# Split of targets by head.
HEALTH_TARGETS = TARGETS[:4]
PERF_TARGETS = TARGETS[4:]


def add_physics_features(df):
    """Return a copy of `df` with the DERIVED_FEATURES columns appended.

    Ratios are the physically informative signals: a fouled compressor delivers
    a lower pressure ratio for the same corrected speed, an eroded turbine
    changes the expansion across station 4, etc. Corrected speed/fuel remove the
    dependence on inlet conditions so the model compares like with like.
    """
    df = df.copy()
    df["PR_c"] = df["P3_Pa"] / df["P2_Pa"]
    df["TR_c"] = df["T3_K"] / df["T2_K"]
    df["TR_b"] = df["T4_K"] / df["T3_K"]
    df["PR_t"] = df["P4_Pa"] / df["P3_Pa"]
    df["TR_overall"] = df["T4_K"] / df["T2_K"]
    df["PR_overall"] = df["P4_Pa"] / df["P2_Pa"]
    df["RPM_corr"] = df["RPM_rev_min"] / np.sqrt(df["T2_K"])
    df["Fuel_corr"] = df["FuelFlow_kg_s"] / (df["P2_Pa"] * np.sqrt(df["T2_K"]))
    return df


def tsfc_from_fuel_thrust(fuel_flow_kg_s, thrust_n):
    """Closed-form TSFC [g/(N.s)] from fuel flow and thrust."""
    return TSFC_FROM_FUEL_THRUST * fuel_flow_kg_s / thrust_n


# --------------------------------------------------------------------------- #
# Physics-informed loss terms (torch world)
# --------------------------------------------------------------------------- #
#
# Each returns a scalar tensor. They are *soft* constraints added to the base
# supervised MSE, weighted in train.py. They do not require ground-truth labels
# — they encode relationships that must hold regardless of the labels, which is
# what makes the model "physics-informed" rather than purely data-driven.


def loss_tsfc_consistency(pred_thrust_n, pred_tsfc, fuel_flow_kg_s):
    """Predicted TSFC must match 1000*fuel/thrust computed from predicted thrust.

    Couples the two performance heads: the model cannot invent a thrust and a
    TSFC that disagree with the fuel it was told the engine is burning.

    The thrust head is an unbounded linear output, so early in training (or on
    out-of-distribution inputs) it can dip to zero or negative. We clamp the
    denominator to a small positive floor to keep the division — and therefore
    the gradient — finite. Real in-distribution thrust is tens of thousands of N,
    far above the floor, so the clamp never affects healthy predictions.
    """
    thrust_safe = torch.clamp(pred_thrust_n, min=1.0)
    implied = tsfc_from_fuel_thrust(fuel_flow_kg_s, thrust_safe)
    return torch.mean((pred_tsfc - implied) ** 2)


def loss_overall_aggregation(pred_health):
    """OverallHealth should track the aggregate of the three subsystem healths.

    `pred_health` columns are [compressor, combustor, turbine, overall].
    We penalise the overall drifting away from the mean of the three parts. The
    engine is only as healthy as its subsystems; this stops the overall index
    floating free of them.
    """
    parts_mean = pred_health[:, :3].mean(dim=1)
    overall = pred_health[:, 3]
    return torch.mean((overall - parts_mean) ** 2)


def loss_monotonic_degradation(pred_health, cycle, engine_id):
    """Within one engine, health should not *increase* as cycles accumulate.

    Degradation is one-directional (wear does not heal). For each engine we sort
    by cycle and apply a hinge penalty on any positive step in predicted health.
    Soft, so genuine measurement noise is tolerated.
    """
    device = pred_health.device
    total = torch.zeros((), device=device)
    count = 0
    for eid in torch.unique(engine_id):
        mask = engine_id == eid
        if mask.sum() < 2:
            continue
        order = torch.argsort(cycle[mask])
        h = pred_health[mask][order]           # (n_cycles, 4) sorted by cycle
        deltas = h[1:] - h[:-1]                # step-to-step change
        # penalise increases only (health going up over time)
        total = total + torch.mean(torch.clamp(deltas, min=0.0) ** 2)
        count += 1
    return total / max(count, 1)


def loss_ratio_health_coupling(pred_health, pr_c_norm):
    """Weak guardrail: higher compressor health ~ higher (normalised) PR_c.

    A healthy compressor achieves a higher pressure ratio for given conditions.
    We encourage positive correlation between predicted compressor health and
    the standardised PR_c feature by penalising their negative covariance. Kept
    at a small weight — a guardrail on the sign, not a hard law.
    """
    comp = pred_health[:, 0]
    comp_c = comp - comp.mean()
    pr_c = pr_c_norm - pr_c_norm.mean()
    cov = torch.mean(comp_c * pr_c)
    return torch.clamp(-cov, min=0.0)


# --------------------------------------------------------------------------- #
# Remaining useful life
# --------------------------------------------------------------------------- #


def fit_degradation(cycles, health):
    """Fit health(cycle) with a straight line (least squares).

    Returns (slope, intercept). Degradation over the observed window is close to
    linear; the slope is the per-cycle health loss used for extrapolation.
    """
    cycles = np.asarray(cycles, dtype=float)
    health = np.asarray(health, dtype=float)
    if len(cycles) < 2:
        return 0.0, float(health[0]) if len(health) else 1.0
    slope, intercept = np.polyfit(cycles, health, 1)
    return float(slope), float(intercept)


def remaining_useful_life(current_cycle, slope, intercept, threshold=RUL_THRESHOLD):
    """Cycles until the fitted health line crosses `threshold`.

    Returns an integer RUL (>= 0), or None if health is flat/improving and never
    reaches the threshold (effectively infinite life on current trend).
    """
    if slope >= 0:
        return None  # not degrading — no finite RUL on current trend
    cross_cycle = (threshold - intercept) / slope
    rul = cross_cycle - current_cycle
    return max(int(round(rul)), 0)
