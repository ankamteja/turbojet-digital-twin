"""Unit tests for the gas-path physics helpers.

These functions are the load-bearing part of the "physics-informed" claim: the
TSFC identity, the health aggregation check, and the linear degradation fit that
produces remaining useful life. They are pure, so they are cheap to pin down.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "model"))

import physics  # noqa: E402


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

def _sensor_row(**overrides):
    """One physically plausible cruise reading, as a single-row frame."""
    row = {
        "Cycle": 10, "Altitude_m": 8000.0, "Mach": 0.8,
        "Tamb_K": 236.0, "Pamb_Pa": 35_600.0,
        "RPM_rev_min": 14_400.0, "FuelFlow_kg_s": 0.62,
        "P2_Pa": 54_000.0, "T2_K": 266.0,
        "P3_Pa": 486_000.0, "T3_K": 545.0,
        "P4_Pa": 462_000.0, "T4_K": 1180.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_add_physics_features_adds_every_derived_column():
    out = physics.add_physics_features(_sensor_row())
    for col in physics.DERIVED_FEATURES:
        assert col in out.columns, f"{col} missing from engineered frame"


def test_add_physics_features_does_not_mutate_input():
    df = _sensor_row()
    before = list(df.columns)
    physics.add_physics_features(df)
    assert list(df.columns) == before


def test_pressure_and_temperature_ratios_match_their_definitions():
    df = _sensor_row()
    out = physics.add_physics_features(df).iloc[0]
    assert out["PR_c"] == pytest.approx(486_000.0 / 54_000.0)
    assert out["TR_c"] == pytest.approx(545.0 / 266.0)
    assert out["TR_b"] == pytest.approx(1180.0 / 545.0)
    assert out["PR_t"] == pytest.approx(462_000.0 / 486_000.0)
    assert out["PR_overall"] == pytest.approx(462_000.0 / 54_000.0)
    assert out["TR_overall"] == pytest.approx(1180.0 / 266.0)


def test_corrected_speed_and_fuel_use_inlet_conditions():
    out = physics.add_physics_features(_sensor_row()).iloc[0]
    assert out["RPM_corr"] == pytest.approx(14_400.0 / np.sqrt(266.0))
    assert out["Fuel_corr"] == pytest.approx(0.62 / (54_000.0 * np.sqrt(266.0)))


def test_ratios_are_dimensionless_under_a_rescaled_inlet():
    """Corrected speed removes inlet-temperature dependence.

    Two engines running the same corrected speed at different inlet temperatures
    should land on the same RPM_corr — that is the whole point of correcting.
    """
    cold = physics.add_physics_features(_sensor_row(T2_K=250.0, RPM_rev_min=14_000.0))
    hot_rpm = 14_000.0 * np.sqrt(300.0 / 250.0)
    hot = physics.add_physics_features(_sensor_row(T2_K=300.0, RPM_rev_min=hot_rpm))
    assert cold.iloc[0]["RPM_corr"] == pytest.approx(hot.iloc[0]["RPM_corr"])


def test_model_features_are_raw_plus_derived_with_no_duplicates():
    assert physics.MODEL_FEATURES == physics.RAW_FEATURES + physics.DERIVED_FEATURES
    assert len(set(physics.MODEL_FEATURES)) == len(physics.MODEL_FEATURES)


def test_engine_id_is_never_a_model_feature():
    """EngineID is an identity label — feeding it would let the model memorise."""
    assert "EngineID" not in physics.MODEL_FEATURES


def test_tsfc_is_not_learned_directly():
    """TSFC must be derived from thrust, so the two performance numbers agree."""
    assert "TSFC_g_N_s" in physics.TARGETS
    assert "TSFC_g_N_s" not in physics.LEARNED_TARGETS
    assert physics.LEARNED_TARGETS == physics.HEALTH_TARGETS + ["Thrust_N"]


# --------------------------------------------------------------------------- #
# TSFC identity
# --------------------------------------------------------------------------- #

def test_tsfc_matches_the_closed_form():
    # 0.5 kg/s of fuel for 25 kN of thrust -> 500 g / 25000 N / s
    assert physics.tsfc_from_fuel_thrust(0.5, 25_000.0) == pytest.approx(0.02)


def test_tsfc_is_vectorised():
    got = physics.tsfc_from_fuel_thrust([0.5, 1.0], [25_000.0, 25_000.0])
    assert got == pytest.approx([0.02, 0.04])


def test_tsfc_stays_finite_when_thrust_collapses():
    """Thrust is floored at 1 N so out-of-distribution inputs cannot divide by zero."""
    got = physics.tsfc_from_fuel_thrust(1.0, 0.0)
    assert np.isfinite(got)
    assert got == pytest.approx(physics.TSFC_FROM_FUEL_THRUST)


def test_tsfc_falls_when_thrust_rises_at_fixed_fuel():
    lean = physics.tsfc_from_fuel_thrust(0.6, 30_000.0)
    strained = physics.tsfc_from_fuel_thrust(0.6, 20_000.0)
    assert lean < strained


# --------------------------------------------------------------------------- #
# Health aggregation
# --------------------------------------------------------------------------- #

def test_aggregation_residual_is_zero_for_a_consistent_engine():
    resid = physics.overall_aggregation_residual(0.90, 0.94, 0.86, 0.90)
    assert resid == pytest.approx(0.0)


def test_aggregation_residual_grows_with_disagreement():
    resid = physics.overall_aggregation_residual(0.90, 0.90, 0.90, 0.70)
    assert resid == pytest.approx(0.20)


# --------------------------------------------------------------------------- #
# Degradation fit and remaining useful life
# --------------------------------------------------------------------------- #

def test_degradation_fit_recovers_a_known_line():
    cycles = np.arange(1, 31)
    health = 1.0 - 0.004 * cycles
    slope, intercept = physics.fit_degradation(cycles, health)
    assert slope == pytest.approx(-0.004)
    assert intercept == pytest.approx(1.0)


def test_degradation_fit_handles_a_single_observation():
    """One point defines no trend — report flat at the observed value."""
    slope, intercept = physics.fit_degradation([5], [0.93])
    assert slope == 0.0
    assert intercept == pytest.approx(0.93)


def test_degradation_fit_handles_no_observations():
    slope, intercept = physics.fit_degradation([], [])
    assert slope == 0.0
    assert intercept == pytest.approx(1.0)


def test_rul_counts_cycles_to_the_threshold():
    # health = 1.0 - 0.01*cycle crosses 0.80 at cycle 20; from cycle 12 that is 8 away
    rul = physics.remaining_useful_life(12, slope=-0.01, intercept=1.0)
    assert rul == 8


def test_rul_is_none_when_health_is_flat_or_improving():
    assert physics.remaining_useful_life(10, slope=0.0, intercept=1.0) is None
    assert physics.remaining_useful_life(10, slope=0.002, intercept=0.9) is None


def test_rul_clamps_to_zero_once_the_threshold_is_passed():
    """An engine already below threshold has no negative life left."""
    rul = physics.remaining_useful_life(40, slope=-0.01, intercept=1.0)
    assert rul == 0


def test_rul_honours_a_custom_threshold():
    strict = physics.remaining_useful_life(0, slope=-0.01, intercept=1.0, threshold=0.95)
    lenient = physics.remaining_useful_life(0, slope=-0.01, intercept=1.0, threshold=0.70)
    assert strict < lenient


# --------------------------------------------------------------------------- #
# The identity against the real dataset
# --------------------------------------------------------------------------- #

def test_tsfc_identity_holds_across_the_shipped_dataset():
    """The identity is a definition, so it should reproduce the shipped TSFC column.

    It does not do so exactly: the dataset carries ~0.8% mean relative error and
    up to ~3% on the worst row, presumably rounding in the generating
    simulation. Pinning both bounds here means a future data drop that breaks
    the relation fails loudly instead of silently skewing every TSFC estimate.
    """
    df = pd.read_csv(ROOT / "data" / "turbojet_complete_dataset.csv")
    derived = physics.tsfc_from_fuel_thrust(df["FuelFlow_kg_s"], df["Thrust_N"])
    rel_err = np.abs(derived - df["TSFC_g_N_s"]) / df["TSFC_g_N_s"]
    assert rel_err.mean() < 0.01
    assert rel_err.max() < 0.035
