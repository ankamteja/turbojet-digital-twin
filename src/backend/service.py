"""Turns model predictions into EngineState objects for the API.

Loads the trained ensemble once, precomputes every engine's per-cycle estimate
from the dataset at startup, and serves snapshots, history, alerts and a
forward degradation simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# the model package uses flat imports (from physics import ...); put it on path
MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_DIR))

from physics import fit_degradation, remaining_useful_life  # noqa: E402
from predict import Predictor, recommend  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# health-target column ↔ short component key used in the API
COMPONENTS = {
    "CompressorHealth": "compressor",
    "CombustorHealth": "combustor",
    "TurbineHealth": "turbine",
    "OverallHealth": "overall",
}

# which engineered ratios to surface as "contributing" per component
CONTRIBUTING = {
    "compressor": ["PR_c", "TR_c", "RPM_corr"],
    "combustor": ["TR_b", "Fuel_corr"],
    "turbine": ["PR_t", "TR_overall"],
    "overall": ["PR_overall", "TR_overall"],
}


def _alert_for(component: str, value: float, confidence: float):
    """Emit an alert if a component is unhealthy or the estimate is uncertain."""
    if value < 0.85:
        return {"component": component, "level": "critical",
                "msg": f"{component} health {value:.2f} — schedule maintenance"}
    if value < 0.90:
        return {"component": component, "level": "warn",
                "msg": f"{component} health {value:.2f} — inspect soon"}
    if confidence < 0.5:
        return {"component": component, "level": "info",
                "msg": f"{component} estimate uncertain (confidence {confidence:.2f})"}
    return None


class TwinService:
    def __init__(self):
        self.predictor = Predictor()
        # full sensor table with engineered ratios attached
        from physics import add_physics_features
        raw = pd.read_csv(DATA_DIR / "turbojet_complete_dataset.csv")
        self.raw = add_physics_features(raw)
        # reference max spool speed → express RPM as a 0–100% gauge value
        self.rpm_max = float(self.raw["RPM_rev_min"].max())
        # per-engine predicted frames + fitted trajectories, cached at startup
        self._pred_cache: dict[int, pd.DataFrame] = {}
        self._traj_cache: dict[int, dict] = {}
        for eid in sorted(self.raw["EngineID"].unique()):
            df_e = self.raw[self.raw["EngineID"] == eid]
            self._pred_cache[int(eid)] = self.predictor.predict_frame(df_e).sort_values("Cycle")
            self._traj_cache[int(eid)] = self.predictor.engine_trajectory(df_e)

    # ------------------------------------------------------------------ #
    def _sensors(self, row) -> dict:
        """Raw station readings + spool-speed percent for the HUD gauges."""
        return {
            "p2_pa": float(row["P2_Pa"]), "t2_k": float(row["T2_K"]),
            "p3_pa": float(row["P3_Pa"]), "t3_k": float(row["T3_K"]),
            "p4_pa": float(row["P4_Pa"]), "t4_k": float(row["T4_K"]),
            "n_pct": float(row["RPM_rev_min"]) / self.rpm_max * 100.0,
        }

    # ------------------------------------------------------------------ #
    def engines(self):
        out = []
        for eid, df in self._pred_cache.items():
            out.append({"engine_id": int(eid), "max_cycle": int(df["Cycle"].max())})
        return out

    def history(self, engine_id: int):
        df = self._pred_cache[engine_id]
        gt = self.raw[self.raw["EngineID"] == engine_id].set_index("Cycle")
        rows = []
        for _, r in df.iterrows():
            cyc = int(r["Cycle"])
            rows.append({
                "cycle": cyc,
                "predicted": {k: float(r[col]) for col, k in COMPONENTS.items()},
                "thrust_n": float(r["Thrust_N"]),
                "tsfc_g_n_s": float(r["TSFC_g_N_s"]),
                "ground_truth": {k: float(gt.loc[cyc, col]) for col, k in COMPONENTS.items()},
            })
        return rows

    # ------------------------------------------------------------------ #
    def state(self, engine_id: int, cycle: int):
        """Full EngineState snapshot for one engine at one cycle."""
        df = self._pred_cache[engine_id]
        traj = self._traj_cache[engine_id]
        row = df[df["Cycle"] == cycle]
        if row.empty:
            raise KeyError(f"engine {engine_id} has no cycle {cycle}")
        row = row.iloc[0]
        raw_row = self.raw[(self.raw["EngineID"] == engine_id)
                           & (self.raw["Cycle"] == cycle)].iloc[0]

        health, alerts = {}, []
        for col, key in COMPONENTS.items():
            value = float(row[col])
            conf = float(row[f"{col}_conf"])
            comp_traj = traj["components"][col]
            contributing = {r: float(raw_row[r]) for r in CONTRIBUTING[key]}
            health[key] = {
                "value": value,
                "confidence": conf,
                "trend": comp_traj["slope_per_cycle"],
                "rul_cycles": comp_traj["rul_cycles"],
                "recommendation": recommend(value),
                "contributing": contributing,
            }
            a = _alert_for(key, value, conf)
            if a:
                alerts.append(a)

        history = [{"cycle": int(r["Cycle"]),
                    "overall": float(r["OverallHealth"]),
                    "thrust_n": float(r["Thrust_N"])}
                   for _, r in df.iterrows() if int(r["Cycle"]) <= cycle]

        return {
            "engine_id": int(engine_id),
            "cycle": int(cycle),
            "conditions": {
                "altitude_m": float(raw_row["Altitude_m"]),
                "mach": float(raw_row["Mach"]),
                "tamb_k": float(raw_row["Tamb_K"]),
                "pamb_pa": float(raw_row["Pamb_Pa"]),
                "rpm": float(raw_row["RPM_rev_min"]),
                "fuel_flow_kg_s": float(raw_row["FuelFlow_kg_s"]),
            },
            "sensors": self._sensors(raw_row),
            "health": health,
            "performance": {
                "thrust_n": {"value": float(row["Thrust_N"]),
                             "confidence": float(row["Thrust_N_conf"])},
                "tsfc_g_n_s": {"value": float(row["TSFC_g_N_s"]),
                               "confidence": float(row["TSFC_g_N_s_conf"])},
            },
            "alerts": alerts,
            "history": history,
        }

    # ------------------------------------------------------------------ #
    def predict_row(self, sensor: dict):
        """Ad-hoc single-row prediction (POST /api/predict)."""
        df = pd.DataFrame([sensor])
        pred = self.predictor.predict_frame(df).iloc[0]
        from physics import add_physics_features
        raw_row = add_physics_features(df).iloc[0]
        health = {}
        alerts = []
        for col, key in COMPONENTS.items():
            value = float(pred[col])
            conf = float(pred[f"{col}_conf"])
            health[key] = {
                "value": value, "confidence": conf, "trend": 0.0,
                "rul_cycles": None, "recommendation": recommend(value),
                "contributing": {r: float(raw_row[r]) for r in CONTRIBUTING[key]},
            }
            a = _alert_for(key, value, conf)
            if a:
                alerts.append(a)
        return {
            "engine_id": int(sensor.get("EngineID", 0)),
            "cycle": int(sensor["Cycle"]),
            "conditions": {
                "altitude_m": float(sensor["Altitude_m"]), "mach": float(sensor["Mach"]),
                "tamb_k": float(sensor["Tamb_K"]), "pamb_pa": float(sensor["Pamb_Pa"]),
                "rpm": float(sensor["RPM_rev_min"]),
                "fuel_flow_kg_s": float(sensor["FuelFlow_kg_s"]),
            },
            "sensors": self._sensors(raw_row),
            "health": health,
            "performance": {
                "thrust_n": {"value": float(pred["Thrust_N"]),
                             "confidence": float(pred["Thrust_N_conf"])},
                "tsfc_g_n_s": {"value": float(pred["TSFC_g_N_s"]),
                               "confidence": float(pred["TSFC_g_N_s_conf"])},
            },
            "alerts": alerts,
            "history": [],
        }

    # ------------------------------------------------------------------ #
    def simulate(self, engine_id: int, to_cycle: int):
        """Project each component's health forward using its fitted trend.

        We have no future sensor readings, so we extrapolate the linear
        degradation fit — a demonstration of long-term behaviour and the RUL
        estimate until live telemetry is available.
        """
        traj = self._traj_cache[engine_id]
        df = self._pred_cache[engine_id]
        last_cycle = int(df["Cycle"].max())
        cycles = list(range(1, to_cycle + 1))
        projection = {"engine_id": int(engine_id), "cycles": cycles, "components": {}}

        for col, key in COMPONENTS.items():
            observed_cycles = df["Cycle"].values
            observed = df[col].values
            slope, intercept = fit_degradation(observed_cycles, observed)
            values = []
            for c in cycles:
                if c <= last_cycle:
                    # use actual prediction where we have data
                    match = df[df["Cycle"] == c]
                    values.append(float(match[col].iloc[0]) if not match.empty
                                  else float(slope * c + intercept))
                else:
                    values.append(float(np.clip(slope * c + intercept, 0.0, 1.0)))
            rul = remaining_useful_life(last_cycle, slope, intercept)
            projection["components"][key] = {
                "health": values,
                "slope_per_cycle": slope,
                "rul_cycles": rul,
                "projected_from_cycle": last_cycle,
            }
        return projection
