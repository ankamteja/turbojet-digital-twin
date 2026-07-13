"""Physics of the turbojet gas path.

Two jobs:

1. Turn raw station pressures/temperatures into *physically meaningful* features
   (pressure ratios, temperature ratios, corrected spool speed / fuel flow). A
   degrading component shows up as a drift in these ratios, so they are far more
   informative to the model than the raw numbers.

2. Provide the closed-form physical relations the surrogate is constrained by:
   the TSFC ↔ thrust identity, the overall-health aggregation, monotonic
   degradation, and the remaining-useful-life projection.

The surrogate (models.py) enforces physics as *hard structural constraints*
rather than soft training penalties:
  * degradation monotonicity — gradient-boosting monotonic constraint on Cycle
  * TSFC = 1000·fuel/thrust — TSFC is derived from predicted thrust, never a
    free output, so the two performance numbers can never disagree.

Station numbering follows the gas path:
    2 = compressor inlet, 3 = compressor exit / combustor inlet, 4 = turbine inlet.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# TSFC (thrust-specific fuel consumption) is fuel burned per unit thrust:
#   TSFC [g/(N.s)] = 1000 * FuelFlow [kg/s] / Thrust [N]
# Verified against the dataset to hold within ~1%. It ties the two performance
# targets together; we predict thrust and *derive* TSFC from it.
TSFC_FROM_FUEL_THRUST = 1000.0

# Default end-of-life threshold on OverallHealth. Below this the engine is
# considered failed for remaining-useful-life (RUL) purposes.
RUL_THRESHOLD = 0.80


# --------------------------------------------------------------------------- #
# Feature engineering
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

# Target columns.
TARGETS = [
    "CompressorHealth",
    "CombustorHealth",
    "TurbineHealth",
    "OverallHealth",
    "Thrust_N",
    "TSFC_g_N_s",
]

HEALTH_TARGETS = TARGETS[:4]
PERF_TARGETS = TARGETS[4:]

# Targets the surrogate learns directly. TSFC is NOT here — it is derived from
# predicted thrust via the physics identity, guaranteeing consistency.
LEARNED_TARGETS = HEALTH_TARGETS + ["Thrust_N"]


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


# --------------------------------------------------------------------------- #
# Physical relations
# --------------------------------------------------------------------------- #


def tsfc_from_fuel_thrust(fuel_flow_kg_s, thrust_n):
    """Closed-form TSFC [g/(N.s)] from fuel flow and thrust.

    Thrust is floored to a small positive value so the division stays finite on
    out-of-distribution inputs (real thrust is tens of thousands of N).
    """
    thrust = np.maximum(np.asarray(thrust_n, dtype=float), 1.0)
    return TSFC_FROM_FUEL_THRUST * np.asarray(fuel_flow_kg_s, dtype=float) / thrust


def overall_aggregation_residual(comp, comb, turb, overall):
    """How far predicted OverallHealth sits from the mean of the three parts.

    Reported as a physics-consistency metric — the engine should be about as
    healthy as the aggregate of its subsystems.
    """
    parts_mean = (np.asarray(comp) + np.asarray(comb) + np.asarray(turb)) / 3.0
    return np.abs(np.asarray(overall) - parts_mean)


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
