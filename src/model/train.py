"""Train the physics-constrained gradient-boosting surrogate ensemble.

Run:  python train.py     # fits on train.csv, saves to artifacts/
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib

from data import load_frame, xy
from models import SurrogateEnsemble
from physics import LEARNED_TARGETS, MODEL_FEATURES

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    train = load_frame("train")
    X, y = xy(train)
    print(f"training rows: {len(X)}  features: {X.shape[1]}")

    t0 = time.perf_counter()
    surrogate = SurrogateEnsemble().fit(X, y)
    train_s = time.perf_counter() - t0
    print(f"fit {surrogate.n_members} members x {len(LEARNED_TARGETS)} targets "
          f"in {train_s:.1f}s")

    joblib.dump(surrogate, ARTIFACTS / "surrogate.joblib")

    config = {
        "model": "SurrogateEnsemble (HistGradientBoosting, bootstrap x"
                 f"{surrogate.n_members})",
        "n_members": surrogate.n_members,
        "feature_names": list(MODEL_FEATURES),
        "learned_targets": list(LEARNED_TARGETS),
        "derived_targets": ["TSFC_g_N_s (= 1000*fuel/thrust)"],
        "physics_constraints": [
            "monotonic health decrease in Cycle",
            "TSFC derived from thrust",
        ],
        "train_seconds": train_s,
    }
    with open(ARTIFACTS / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # feature importances (interpretability) for the report
    importances = {t: surrogate.importances[t] for t in LEARNED_TARGETS}
    with open(ARTIFACTS / "importances.json", "w") as f:
        json.dump(importances, f, indent=2)

    print("top drivers per target:")
    for t in LEARNED_TARGETS:
        tops = ", ".join(f"{n}({v:.2f})" for n, v in surrogate.top_features(t, 3))
        print(f"  {t:18s} {tops}")

    print(f"saved surrogate + config + importances to {ARTIFACTS}")


if __name__ == "__main__":
    main()
