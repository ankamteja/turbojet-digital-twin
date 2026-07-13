"""Evaluate the trained ensemble on the held-out test split.

Reports per-target MAE and R^2 (test.csv vs ground_truth.csv) and writes
artifacts/metrics.json for the technical report.

Run:  python eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from data import DATA_DIR
from physics import TARGETS
from predict import ARTIFACTS, Predictor


def main():
    test = pd.read_csv(DATA_DIR / "test.csv")
    truth = pd.read_csv(DATA_DIR / "ground_truth.csv")
    keep = ["EngineID", "Cycle"] + TARGETS
    test = test.merge(truth[keep], on=["EngineID", "Cycle"], how="left")

    pred = Predictor().predict_frame(test)

    metrics = {}
    print(f"{'target':<18}{'MAE':>12}{'R2':>10}")
    print("-" * 40)
    for name in TARGETS:
        y_true = test[name].values
        y_pred = pred[name].values
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics[name] = {"mae": float(mae), "r2": float(r2)}
        print(f"{name:<18}{mae:>12.4f}{r2:>10.4f}")

    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nwrote {ARTIFACTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
