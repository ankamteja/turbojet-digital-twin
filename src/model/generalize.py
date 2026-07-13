"""Leave-one-engine-out generalization test.

The standard test split shares engines between train and test (random-cycle
interpolation). This is the harder question the evaluation cares about: can the
surrogate estimate a *previously unseen engine* from its sensors alone?

For each of the 10 engines we retrain on the other 9 and score the held-out
engine, then average. Writes artifacts/generalization.json.

Run:  python generalize.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from data import load_frame, xy
from models import SurrogateEnsemble
from physics import LEARNED_TARGETS, tsfc_from_fuel_thrust
from predict import ARTIFACTS

ALL = LEARNED_TARGETS + ["TSFC_g_N_s"]


def main():
    full = load_frame("complete")
    engines = sorted(full["EngineID"].unique())
    scores = {t: [] for t in ALL}

    for eid in engines:
        tr = full[full["EngineID"] != eid]
        te = full[full["EngineID"] == eid]
        X_tr, y_tr = xy(tr)
        surrogate = SurrogateEnsemble(n_members=5).fit(
            X_tr, y_tr, compute_importance=False)  # lighter for CV

        X_te = te[surrogate.feature_names].values.astype(float)
        pred = surrogate.predict(X_te, te["FuelFlow_kg_s"].values)
        for t in LEARNED_TARGETS:
            scores[t].append(r2_score(te[t].values, pred[t]["mean"]))
        scores["TSFC_g_N_s"].append(r2_score(te["TSFC_g_N_s"].values,
                                              pred["TSFC_g_N_s"]["mean"]))
        print(f"  held-out engine {eid:2d}: "
              f"overall R2 {scores['OverallHealth'][-1]:.3f}")

    summary = {t: {"mean_r2": float(np.mean(scores[t])),
                   "min_r2": float(np.min(scores[t]))}
               for t in ALL}
    print("\nleave-one-engine-out mean R2:")
    for t in ALL:
        print(f"  {t:18s} {summary[t]['mean_r2']:.3f}  (worst {summary[t]['min_r2']:.3f})")

    with open(ARTIFACTS / "generalization.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {ARTIFACTS / 'generalization.json'}")


if __name__ == "__main__":
    main()
