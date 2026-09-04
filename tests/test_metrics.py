"""Accuracy floors for the trained surrogate.

Feature engineering and the ensemble config are easy to change and hard to
eyeball: a broken derived feature still trains, still serves, and just predicts
worse. These floors are set well below the numbers reported in
docs/technical-report.md, so they do not churn on ordinary retraining noise but
do catch a real regression.

Reads src/model/artifacts/metrics.json, written by `python src/model/eval.py`;
skipped when that has not been run.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "src" / "model" / "artifacts" / "metrics.json"

pytestmark = pytest.mark.skipif(
    not METRICS.exists(),
    reason="metrics not generated — run `python src/model/eval.py` first",
)

# Reported test-split R2 -> floor. Combustor and turbine health are the weakest
# targets (their degradation signal is the subtlest in the sensor set), so they
# get correspondingly looser floors rather than being left unguarded.
R2_FLOORS = {
    "CompressorHealth": 0.85,
    "CombustorHealth": 0.65,
    "TurbineHealth": 0.60,
    "OverallHealth": 0.85,
    "Thrust_N": 0.95,
    "TSFC_g_N_s": 0.95,
}


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS.read_text())


@pytest.mark.parametrize("target,floor", sorted(R2_FLOORS.items()))
def test_target_meets_its_accuracy_floor(metrics, target, floor):
    got = metrics["accuracy"][target]["r2"]
    assert got >= floor, f"{target} R2 {got:.3f} fell below {floor}"


def test_every_target_is_covered_by_a_floor(metrics):
    """A new target must not slip in unguarded."""
    assert set(metrics["accuracy"]) == set(R2_FLOORS)


def test_tsfc_stays_consistent_with_predicted_thrust(metrics):
    """The derivation should leave a residual near machine noise, not near zero
    only because both numbers happen to be small."""
    assert metrics["physics"]["tsfc_consistency_residual"] < 1e-3


def test_overall_health_agrees_with_its_subsystems(metrics):
    """OverallHealth should stay near the mean of the three component healths."""
    assert metrics["physics"]["overall_aggregation_residual"] < 0.05


def test_inference_stays_fast_enough_to_be_real_time(metrics):
    """The surrogate exists to be cheap — the whole point over a CFD run.

    Generous ceiling: this is a shared CI runner, and the guard is against an
    order-of-magnitude regression (a per-row Python loop creeping in), not
    against normal variance.
    """
    assert metrics["efficiency"]["inference_ms_per_row"] < 100.0
