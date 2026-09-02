"""Tests for the surrogate's hard physics constraints.

These go at the estimator directly rather than through the API, because the
guarantee is all-else-equal: hold a real sensor row fixed, advance only Cycle,
and health must never rise. Over a real engine's life the other sensors move
too, so the guarantee is only visible when you pin them.

Skipped without a trained surrogate (see tests/test_api.py).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "src" / "model" / "artifacts" / "surrogate.joblib"

pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="surrogate not trained — run `python src/model/train.py` first",
)

sys.path.insert(0, str(ROOT / "src" / "model"))


@pytest.fixture(scope="module")
def predictor():
    from predict import Predictor

    return Predictor()


@pytest.fixture(scope="module")
def aging_inputs():
    """One real sensor row replayed across cycles 1..30, nothing else changed."""
    raw = pd.read_csv(ROOT / "data" / "turbojet_complete_dataset.csv")
    row = raw[raw["EngineID"] == 1].iloc[0]
    df = pd.DataFrame([row] * 30).reset_index(drop=True)
    df["Cycle"] = range(1, 31)
    return df


@pytest.fixture(scope="module")
def aging_sweep(predictor, aging_inputs):
    return predictor.predict_frame(aging_inputs)


@pytest.mark.parametrize("target", [
    "CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth",
])
def test_health_never_rises_with_cycle_all_else_equal(aging_sweep, target):
    """The monotonic_cst=-1 constraint on Cycle — the model's core guarantee."""
    values = aging_sweep[target].to_numpy()
    assert np.all(np.diff(values) <= 1e-9), f"{target} rose as the engine aged"


@pytest.mark.parametrize("target", [
    "CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth",
])
def test_the_constraint_is_not_satisfied_by_a_flat_prediction(aging_sweep, target):
    """Monotonicity would hold trivially if the model ignored Cycle. It doesn't."""
    values = aging_sweep[target].to_numpy()
    assert values[0] - values[-1] > 0.01, f"{target} barely responds to age"


def test_every_health_estimate_stays_inside_the_unit_interval(aging_sweep):
    for target in ["CompressorHealth", "CombustorHealth",
                   "TurbineHealth", "OverallHealth"]:
        values = aging_sweep[target].to_numpy()
        assert values.min() >= 0.0 and values.max() <= 1.0


def test_tsfc_is_derived_from_predicted_thrust(aging_sweep, aging_inputs):
    """TSFC is never a free output — it comes out of the thrust prediction."""
    df = aging_sweep
    per_row = 1000.0 * aging_inputs["FuelFlow_kg_s"].to_numpy() / df["Thrust_N"]
    # Ensemble members derive TSFC individually before averaging, so allow the
    # small Jensen gap between mean(1000f/T_i) and 1000f/mean(T_i).
    assert np.allclose(df["TSFC_g_N_s"], per_row, rtol=5e-3)


def test_predictions_carry_confidence_for_every_target(aging_sweep):
    for target in ["CompressorHealth", "CombustorHealth", "TurbineHealth",
                   "OverallHealth", "Thrust_N", "TSFC_g_N_s"]:
        col = f"{target}_conf"
        assert col in aging_sweep.columns, f"missing {col}"
        assert aging_sweep[col].between(0.0, 1.0).all()
