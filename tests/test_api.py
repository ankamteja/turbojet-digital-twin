"""Contract tests for the FastAPI service.

The frontend is a separate deployment that talks to this API over HTTP, so the
EngineState shape is a real contract — these tests are what stop a model-side
refactor from quietly breaking the dashboard.

The whole suite is skipped when the trained surrogate is absent, so a clean
checkout can still run the physics tests without a 20-second training step.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "src" / "backend"
ARTIFACT = ROOT / "src" / "model" / "artifacts" / "surrogate.joblib"

pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="surrogate not trained — run `python src/model/train.py` first",
)

sys.path.insert(0, str(BACKEND))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import main

    # TestClient as a context manager runs the startup hook that builds the twin
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="module")
def first_engine(client):
    return client.get("/api/engines").json()[0]


# --------------------------------------------------------------------------- #
# Liveness and discovery
# --------------------------------------------------------------------------- #

def test_health_reports_loaded_engines(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["engines"] > 0


def test_engines_lists_ids_with_their_last_cycle(client):
    engines = client.get("/api/engines").json()
    assert len(engines) == 10
    assert {e["engine_id"] for e in engines} == set(range(1, 11))
    assert all(e["max_cycle"] > 0 for e in engines)


# --------------------------------------------------------------------------- #
# EngineState contract
# --------------------------------------------------------------------------- #

def test_state_returns_the_full_engine_state_contract(client, first_engine):
    eid, cycle = first_engine["engine_id"], first_engine["max_cycle"]
    body = client.get(f"/api/engine/{eid}/state", params={"cycle": cycle}).json()

    assert body["engine_id"] == eid
    assert body["cycle"] == cycle
    assert set(body["health"]) == {"compressor", "combustor", "turbine", "overall"}
    assert set(body["conditions"]) >= {"altitude_m", "mach", "rpm", "fuel_flow_kg_s"}
    assert set(body["sensors"]) >= {"p2_pa", "t2_k", "p3_pa", "t3_k", "p4_pa", "t4_k"}
    assert set(body["performance"]) == {"thrust_n", "tsfc_g_n_s"}


def test_health_values_and_confidences_stay_in_range(client, first_engine):
    eid, cycle = first_engine["engine_id"], first_engine["max_cycle"]
    body = client.get(f"/api/engine/{eid}/state", params={"cycle": cycle}).json()
    for key, comp in body["health"].items():
        assert 0.0 <= comp["value"] <= 1.0, f"{key} health out of range"
        assert 0.0 <= comp["confidence"] <= 1.0, f"{key} confidence out of range"
        assert comp["recommendation"]


def test_performance_obeys_the_tsfc_identity(client, first_engine):
    """The served TSFC must track 1000*fuel/thrust — it is never fit freely.

    Not bit-exact: each ensemble member derives its own TSFC from its own
    thrust, and the mean of 1000*f/T_i is not 1000*f/mean(T_i). The gap is the
    ensemble's thrust spread (well under 1% here), and averaging per member is
    what makes the reported TSFC confidence meaningful.
    """
    eid, cycle = first_engine["engine_id"], first_engine["max_cycle"]
    body = client.get(f"/api/engine/{eid}/state", params={"cycle": cycle}).json()
    fuel = body["conditions"]["fuel_flow_kg_s"]
    thrust = body["performance"]["thrust_n"]["value"]
    tsfc = body["performance"]["tsfc_g_n_s"]["value"]
    assert tsfc == pytest.approx(1000.0 * fuel / thrust, rel=2e-3)


def test_alerts_use_known_severity_levels(client, first_engine):
    eid, cycle = first_engine["engine_id"], first_engine["max_cycle"]
    body = client.get(f"/api/engine/{eid}/state", params={"cycle": cycle}).json()
    for alert in body["alerts"]:
        assert alert["level"] in {"info", "warn", "critical"}
        assert alert["msg"]


def test_unknown_engine_is_a_404(client):
    assert client.get("/api/engine/999/state", params={"cycle": 1}).status_code == 404
    assert client.get("/api/engine/999/history").status_code == 404


def test_out_of_range_cycle_is_a_404(client, first_engine):
    eid = first_engine["engine_id"]
    beyond = first_engine["max_cycle"] + 50
    assert client.get(f"/api/engine/{eid}/state",
                      params={"cycle": beyond}).status_code == 404


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #

def test_history_pairs_every_cycle_with_its_ground_truth(client, first_engine):
    eid = first_engine["engine_id"]
    points = client.get(f"/api/engine/{eid}/history").json()
    cycles = [p["cycle"] for p in points]
    assert cycles == sorted(cycles)
    assert len(cycles) == first_engine["max_cycle"]
    components = {"compressor", "combustor", "turbine", "overall"}
    for p in points:
        assert set(p["predicted"]) == components
        assert set(p["ground_truth"]) == components


def test_health_trends_downward_over_an_engine_life(client, first_engine):
    """Health must end the recorded life below where it started.

    Deliberately a trend check, not a step-by-step one: flight conditions change
    from cycle to cycle, so a single step can read slightly healthier. The
    all-else-equal monotonicity the model actually guarantees is pinned in
    test_model.py, against the estimator itself.
    """
    eid = first_engine["engine_id"]
    points = client.get(f"/api/engine/{eid}/history").json()
    overall = [p["predicted"]["overall"] for p in points]
    assert overall[-1] < overall[0]


# --------------------------------------------------------------------------- #
# Forward projection
# --------------------------------------------------------------------------- #

def test_simulate_projects_past_the_recorded_data(client, first_engine):
    eid = first_engine["engine_id"]
    body = client.get(f"/api/engine/{eid}/simulate", params={"to_cycle": 60}).json()
    assert body["cycles"][-1] == 60
    for key, comp in body["components"].items():
        assert len(comp["health"]) == 60, f"{key} projection length mismatch"
        assert all(0.0 <= v <= 1.0 for v in comp["health"])
        assert comp["projected_from_cycle"] == first_engine["max_cycle"]


def test_simulate_rejects_an_unbounded_horizon(client, first_engine):
    """An unbounded to_cycle would let one request build an arbitrary payload."""
    eid = first_engine["engine_id"]
    assert client.get(f"/api/engine/{eid}/simulate",
                      params={"to_cycle": 10_000_000}).status_code == 422
    assert client.get(f"/api/engine/{eid}/simulate",
                      params={"to_cycle": 0}).status_code == 422


# --------------------------------------------------------------------------- #
# Ad-hoc prediction — the path real telemetry would take
# --------------------------------------------------------------------------- #

def test_predict_scores_a_sensor_row_the_model_never_saw(client):
    row = {
        "EngineID": 0, "Cycle": 12, "Altitude_m": 8200.0, "Mach": 0.78,
        "Tamb_K": 235.0, "Pamb_Pa": 35_100.0,
        "RPM_rev_min": 14_350.0, "FuelFlow_kg_s": 0.61,
        "P2_Pa": 53_400.0, "T2_K": 265.0,
        "P3_Pa": 480_000.0, "T3_K": 543.0,
        "P4_Pa": 456_000.0, "T4_K": 1175.0,
    }
    res = client.post("/api/predict", json=row)
    assert res.status_code == 200
    body = res.json()
    assert body["cycle"] == 12
    assert 0.0 <= body["health"]["overall"]["value"] <= 1.0
    assert body["performance"]["thrust_n"]["value"] > 0


def test_predict_rejects_a_row_missing_a_sensor(client):
    assert client.post("/api/predict", json={"Cycle": 12}).status_code == 422
