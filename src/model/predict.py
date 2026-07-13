"""Inference: load the trained ensemble and turn sensor rows into full estimates.

Produces, per row:
  * health (4) with confidence
  * thrust + TSFC with confidence
And, per engine trajectory:
  * fitted degradation slope, remaining useful life, recommendation.

Confidence comes from the ensemble spread: agreement between the 5 models = high
confidence, disagreement = low.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import load_scalers
from net import TurbojetNet
from physics import (
    HEALTH_TARGETS,
    MODEL_FEATURES,
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


def confidence_from_std(std: float, scale: float) -> float:
    """Map an ensemble std to a 0–1 confidence. `scale` normalises the target."""
    return float(np.clip(1.0 - std / scale, 0.0, 1.0))


class Predictor:
    """Loads the ensemble + scalers once; call `predict_frame` for estimates."""

    def __init__(self, artifacts=ARTIFACTS, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with open(Path(artifacts) / "config.json") as f:
            self.config = json.load(f)
        self.x_scaler, self.thrust_scaler = load_scalers(Path(artifacts) / "scalers.pkl")

        self.models = []
        for i in range(self.config["n_models"]):
            net = TurbojetNet(n_features=self.config["n_features"]).to(self.device)
            net.load_state_dict(torch.load(Path(artifacts) / f"net_{i}.pt",
                                           map_location=self.device))
            net.eval()
            self.models.append(net)

    # -------------------------------------------------------------- #
    def _raw_predict(self, df: pd.DataFrame):
        """Run every ensemble member. Returns stacked health & perf arrays.

        health: (n_models, n_rows, 4) in [0,1]
        thrust: (n_models, n_rows)   in real N
        tsfc:   (n_models, n_rows)   in g/(N.s)
        """
        feats = add_physics_features(df)
        X = self.x_scaler.transform(feats[MODEL_FEATURES].values)
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)

        healths, thrusts, tsfcs = [], [], []
        with torch.no_grad():
            for net in self.models:
                ph, pp = net(Xt)
                healths.append(ph.cpu().numpy())
                thrust_real = (pp[:, 0].cpu().numpy() * self.thrust_scaler.scale_[0]
                               + self.thrust_scaler.mean_[0])
                # thrust is physically non-negative; the unbounded linear head can
                # extrapolate below zero on out-of-distribution inputs. Floor it so
                # downstream TSFC (1000*fuel/thrust) stays well-defined.
                thrust_real = np.maximum(thrust_real, 0.0)
                thrusts.append(thrust_real)
                tsfcs.append(pp[:, 1].cpu().numpy())
        return np.stack(healths), np.stack(thrusts), np.stack(tsfcs)

    # -------------------------------------------------------------- #
    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a per-row DataFrame of mean predictions + confidences."""
        H, T, S = self._raw_predict(df)
        h_mean, h_std = H.mean(0), H.std(0)      # (n_rows, 4)
        t_mean, t_std = T.mean(0), T.std(0)      # (n_rows,)
        s_mean, s_std = S.mean(0), S.std(0)

        out = pd.DataFrame(index=df.index)
        out["EngineID"] = df["EngineID"].values
        out["Cycle"] = df["Cycle"].values
        for j, name in enumerate(HEALTH_TARGETS):
            out[name] = h_mean[:, j]
            out[f"{name}_conf"] = [confidence_from_std(s, 0.10) for s in h_std[:, j]]
        out["Thrust_N"] = t_mean
        # confidence relative to typical thrust magnitude
        out["Thrust_N_conf"] = [confidence_from_std(s, max(m, 1.0) * 0.10)
                                for s, m in zip(t_std, t_mean)]
        out["TSFC_g_N_s"] = s_mean
        out["TSFC_g_N_s_conf"] = [confidence_from_std(s, 0.005) for s in s_std]
        return out

    # -------------------------------------------------------------- #
    def engine_trajectory(self, df_engine: pd.DataFrame) -> dict:
        """Fit degradation and RUL for one engine's full cycle history.

        `df_engine` = all available cycles for a single engine (sensor rows).
        """
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
