"""Evaluate the trained surrogate on the held-out test split.

Reports, and writes to artifacts/metrics.json:
  * per-target MAE and R^2 (accuracy)
  * physics-consistency residuals (TSFC identity, overall-health aggregation)
  * inference latency per row (surrogate-efficiency evidence)

Run:  python eval.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from data import load_frame
from physics import TARGETS, overall_aggregation_residual, tsfc_from_fuel_thrust
from predict import ARTIFACTS, Predictor


def main():
    test = load_frame("test")
    predictor = Predictor()

    # timed batch inference → per-row latency
    t0 = time.perf_counter()
    pred = predictor.predict_frame(test)
    latency_ms = (time.perf_counter() - t0) / len(test) * 1000

    metrics = {"accuracy": {}, "physics": {}, "efficiency": {}}
    print(f"{'target':<18}{'MAE':>12}{'R2':>10}")
    print("-" * 40)
    for name in TARGETS:
        y_true, y_pred = test[name].values, pred[name].values
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics["accuracy"][name] = {"mae": float(mae), "r2": float(r2)}
        print(f"{name:<18}{mae:>12.4f}{r2:>10.4f}")

    health_r2 = np.mean([metrics["accuracy"][t]["r2"] for t in TARGETS[:4]])

    # physics consistency
    tsfc_implied = tsfc_from_fuel_thrust(test["FuelFlow_kg_s"].values, pred["Thrust_N"].values)
    tsfc_resid = float(np.mean(np.abs(pred["TSFC_g_N_s"].values - tsfc_implied)))
    agg_resid = float(np.mean(overall_aggregation_residual(
        pred["CompressorHealth"].values, pred["CombustorHealth"].values,
        pred["TurbineHealth"].values, pred["OverallHealth"].values)))
    metrics["physics"] = {
        "tsfc_consistency_residual": tsfc_resid,     # ~0 by construction
        "overall_aggregation_residual": agg_resid,
    }
    metrics["efficiency"] = {"inference_ms_per_row": float(latency_ms)}

    print(f"\nmean health R2: {health_r2:.3f}")
    print(f"TSFC consistency residual: {tsfc_resid:.2e} (derived → ~0)")
    print(f"overall-aggregation residual: {agg_resid:.4f}")
    print(f"inference latency: {latency_ms:.3f} ms/row")

    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {ARTIFACTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
