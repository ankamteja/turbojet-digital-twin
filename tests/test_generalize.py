"""Tests for the leave-one-engine-out generalization study.

generalize.py answers the harder question the test split doesn't: can the
surrogate estimate an engine it never saw at all, from sensors alone? It had
no test coverage, and its own output (generalization.json) shows combustor
health going *negative* on its worst held-out engine — worse than predicting
the training mean. That is a real, reported weakness (see
docs/technical-report.md), not a bug, but nothing was pinning it down: a
further regression could go unnoticed because "negative" was already the
baseline nobody was checking against.

Skipped unless `python src/model/generalize.py` has been run — it retrains 10
times (5-member ensembles) and takes noticeably longer than the main suite.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERALIZATION = ROOT / "src" / "model" / "artifacts" / "generalization.json"

pytestmark = pytest.mark.skipif(
    not GENERALIZATION.exists(),
    reason="generalization.json missing — run `python src/model/generalize.py` first",
)

ALL_TARGETS = {
    "CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth",
    "Thrust_N", "TSFC_g_N_s",
}

# Mean-R2 floors for generalizing to a *held-out engine* — looser than the
# in-distribution floors in test_metrics.py, since this is the harder task.
MEAN_R2_FLOORS = {
    "CompressorHealth": 0.70,
    "CombustorHealth": 0.30,
    "TurbineHealth": 0.50,
    "OverallHealth": 0.80,
    "Thrust_N": 0.95,
    "TSFC_g_N_s": 0.95,
}

# Worst-single-engine floors. Deliberately loose, and in CombustorHealth's
# case negative: the point is not to demand every fold looks good, it's to
# catch a fold getting *worse* than the documented baseline.
MIN_R2_FLOORS = {
    "CompressorHealth": 0.0,
    "CombustorHealth": -3.0,
    "TurbineHealth": -0.5,
    "OverallHealth": 0.3,
    "Thrust_N": 0.9,
    "TSFC_g_N_s": 0.9,
}


@pytest.fixture(scope="module")
def summary():
    return json.loads(GENERALIZATION.read_text())


def test_every_target_has_a_loeo_score(summary):
    assert set(summary) == ALL_TARGETS


@pytest.mark.parametrize("target,floor", sorted(MEAN_R2_FLOORS.items()))
def test_mean_generalization_meets_its_floor(summary, target, floor):
    got = summary[target]["mean_r2"]
    assert got >= floor, f"{target} mean LOEO R2 {got:.3f} fell below {floor}"


@pytest.mark.parametrize("target,floor", sorted(MIN_R2_FLOORS.items()))
def test_worst_held_out_engine_does_not_get_worse(summary, target, floor):
    got = summary[target]["min_r2"]
    assert got >= floor, (
        f"{target} worst-engine LOEO R2 {got:.3f} fell below the documented "
        f"floor {floor} — the report's caveat may now be understating it"
    )


def test_combustor_health_is_flagged_as_the_weakest_target(summary):
    """The documented finding this file exists to guard.

    Combustor health's mean LOEO R2 should be visibly the worst of the four
    health targets — if it stops being the worst, the report's framing
    ("combustor health is the weakest target") needs updating, not just the
    number.
    """
    healths = {"CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth"}
    worst = min(healths, key=lambda t: summary[t]["mean_r2"])
    assert worst == "CombustorHealth"


def test_performance_targets_generalize_far_better_than_health_targets(summary):
    """Thrust/TSFC ride mostly on FuelFlow and RPM, which don't drift much
    across engines; health has to infer wear the model never labels directly."""
    perf_mean = min(summary["Thrust_N"]["mean_r2"], summary["TSFC_g_N_s"]["mean_r2"])
    worst_health_mean = min(
        summary[t]["mean_r2"]
        for t in ("CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth")
    )
    assert perf_mean > worst_health_mean
