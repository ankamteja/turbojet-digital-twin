"""Inference: load the trained surrogate and turn sensor rows into estimates.

Public interface (unchanged, so the backend needs no edits):
  * Predictor.predict_frame(df)      -> per-row means + confidences
  * Predictor.engine_trajectory(df)  -> per-engine degradation fit + RUL

Confidence comes from the bootstrap-ensemble spread: agreement between members
= high confidence, disagreement = low.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from physics import (
    HEALTH_TARGETS,
    add_physics_features,
    fit_degradation,
    remaining_useful_life,
)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# health-band → recommendation rule table (applied per component)
RECOMMENDATIONS = [
    (0.95, "nominal — no action"),
    (0.90, "monitor"),
    (0.85, "inspect at next opportunity"),
    (0.00, "schedule maintenance"),
]


def recommend(health_value: float) -> str:
    for threshold, text in RECOMMENDATIONS:
        if health_value >= threshold:
            return text
    return RECOMMENDATIONS[-1][1]


def _confidence(std, scale):
    """Map ensemble std to a 0–1 confidence, normalised by the target's spread."""
    return np.clip(1.0 - std / max(scale, 1e-9), 0.0, 1.0)


class Predictor:
    """Loads the surrogate once; call `predict_frame` for estimates."""

    def __init__(self, artifacts=ARTIFACTS):
        self.surrogate = joblib.load(Path(artifacts) / "surrogate.joblib")

    # -------------------------------------------------------------- #
    def _prepare(self, df: pd.DataFrame):
        feats = add_physics_features(df)
        X = feats[self.surrogate.feature_names].values.astype(float)
        fuel = feats["FuelFlow_kg_s"].values.astype(float)
        return X, fuel

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a per-row DataFrame of mean predictions + confidences."""
        X, fuel = self._prepare(df)
        pred = self.surrogate.predict(X, fuel)
        scale = self.surrogate.conf_scale

        out = pd.DataFrame(index=df.index)
        out["EngineID"] = df["EngineID"].values
        out["Cycle"] = df["Cycle"].values
        for name in HEALTH_TARGETS:
            out[name] = pred[name]["mean"]
            out[f"{name}_conf"] = _confidence(pred[name]["std"], scale[name])
        out["Thrust_N"] = pred["Thrust_N"]["mean"]
        out["Thrust_N_conf"] = _confidence(pred["Thrust_N"]["std"], scale["Thrust_N"])
        # TSFC scale from thrust spread scaled by the physics factor magnitude
        tsfc_scale = float(np.std(pred["TSFC_g_N_s"]["mean"])) or 1e-3
        out["TSFC_g_N_s"] = pred["TSFC_g_N_s"]["mean"]
        out["TSFC_g_N_s_conf"] = _confidence(pred["TSFC_g_N_s"]["std"], tsfc_scale)
        return out

    # -------------------------------------------------------------- #
    def engine_trajectory(self, df_engine: pd.DataFrame) -> dict:
        """Fit degradation and RUL for one engine's full cycle history."""
        preds = self.predict_frame(df_engine).sort_values("Cycle")
        cycles = preds["Cycle"].values
        result = {"cycles": cycles.tolist(), "components": {}}

        for name in HEALTH_TARGETS:
            values = preds[name].values
            slope, intercept = fit_degradation(cycles, values)
            current_cycle = float(cycles[-1])
            rul = remaining_useful_life(current_cycle, slope, intercept)
            result["components"][name] = {
                "health": values.tolist(),
                "slope_per_cycle": slope,
                "rul_cycles": rul,
                "recommendation": recommend(float(values[-1])),
            }
        return result
