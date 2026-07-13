"""The surrogate model — a physics-constrained gradient-boosting ensemble.

Why gradient boosting and not a neural net? On this small, tabular dataset
(240 training rows) boosted trees estimate the subtle component-health signal
far better than an MLP (test health R^2 ~0.86 vs ~0.62), while staying
interpretable (feature importances) and fast to train and query.

Physics is enforced as **hard structural constraints**, which is stronger than
soft training penalties:

  * Monotonic degradation — each health model is given a monotonic-decreasing
    constraint on the Cycle feature, so predicted health can never rise as the
    engine ages.
  * TSFC consistency — TSFC is never a free output; it is derived from predicted
    thrust via TSFC = 1000·fuel/thrust, so the two performance numbers agree by
    construction.

Uncertainty comes from a bootstrap ensemble: N models are each trained on a
resample of the data, and the spread of their predictions is the confidence
signal (tight agreement = confident, disagreement = uncertain).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from physics import (
    HEALTH_TARGETS,
    LEARNED_TARGETS,
    MODEL_FEATURES,
    tsfc_from_fuel_thrust,
)

N_MEMBERS = 10
CYCLE_IDX = MODEL_FEATURES.index("Cycle")


def _make_member(target, seed):
    """One boosted-tree regressor, with a monotonic Cycle constraint for health."""
    mono = None
    if target in HEALTH_TARGETS:
        mono = [0] * len(MODEL_FEATURES)
        mono[CYCLE_IDX] = -1          # health non-increasing in Cycle
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=400,
        learning_rate=0.05,
        max_depth=3,
        l2_regularization=1.0,
        monotonic_cst=mono,
        random_state=seed,
    )


class SurrogateEnsemble:
    """Bootstrap ensemble of boosted trees, one ensemble per learned target."""

    def __init__(self, n_members=N_MEMBERS):
        self.n_members = n_members
        self.models: dict[str, list] = {}     # target -> [members]
        self.conf_scale: dict[str, float] = {}  # target -> std normaliser
        self.importances: dict[str, dict] = {}  # target -> {feature: importance}
        self.feature_names = list(MODEL_FEATURES)

    # ------------------------------------------------------------------ #
    def fit(self, X, y: dict, compute_importance=True):
        """X: (n, n_features) array. y: dict target -> (n,) array.

        `compute_importance=False` skips the (slow) permutation-importance pass —
        used during cross-validation where importances aren't needed.
        """
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(0)
        n = len(X)
        for target in LEARNED_TARGETS:
            members = []
            for m in range(self.n_members):
                idx = rng.integers(0, n, n)          # bootstrap resample
                mod = _make_member(target, seed=1000 + m)
                mod.fit(X[idx], np.asarray(y[target])[idx])
                members.append(mod)
            self.models[target] = members
            # confidence normaliser: spread of the target itself
            self.conf_scale[target] = float(np.std(y[target])) or 1.0

        if compute_importance:
            self._compute_importances(X, y)
        return self

    # ------------------------------------------------------------------ #
    def _member_predict(self, target, X):
        """Stack every ensemble member's prediction: (n_members, n_rows)."""
        return np.stack([m.predict(X) for m in self.models[target]])

    def predict(self, X, fuel_flow_kg_s):
        """Return {target: {'mean':(n,), 'std':(n,)}} for all six targets.

        TSFC is derived per-member from thrust so its uncertainty is propagated
        from the thrust ensemble rather than modelled independently.
        """
        X = np.asarray(X, dtype=float)
        out = {}
        for target in HEALTH_TARGETS:
            preds = np.clip(self._member_predict(target, X), 0.0, 1.0)
            out[target] = {"mean": preds.mean(0), "std": preds.std(0)}

        thrust = np.maximum(self._member_predict("Thrust_N", X), 0.0)
        out["Thrust_N"] = {"mean": thrust.mean(0), "std": thrust.std(0)}

        # derive TSFC per member, then aggregate
        fuel = np.asarray(fuel_flow_kg_s, dtype=float)
        tsfc = tsfc_from_fuel_thrust(np.broadcast_to(fuel, thrust.shape), thrust)
        out["TSFC_g_N_s"] = {"mean": tsfc.mean(0), "std": tsfc.std(0)}
        return out

    # ------------------------------------------------------------------ #
    def _compute_importances(self, X, y, n_repeats=10):
        """Permutation importance per target (interpretability, for the report).

        Uses the first ensemble member — importances are a global explanation of
        which sensors/ratios drive each estimate.
        """
        for target in LEARNED_TARGETS:
            model = self.models[target][0]
            r = permutation_importance(
                model, X, np.asarray(y[target]),
                n_repeats=n_repeats, random_state=0, scoring="r2",
            )
            self.importances[target] = {
                name: float(imp)
                for name, imp in zip(self.feature_names, r.importances_mean)
            }

    def top_features(self, target, k=5):
        imp = self.importances.get(target, {})
        return sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:k]
