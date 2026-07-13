"""Dataset loading, feature engineering and scaling.

Loads the CSVs, attaches engineered physics features, fits scalers on the
training split only (to avoid leaking test statistics), and hands back tensors
ready for training.

The public entry point is `load_dataset()`.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from physics import (
    HEALTH_TARGETS,
    MODEL_FEATURES,
    PERF_TARGETS,
    TARGETS,
    add_physics_features,
)

# Repo layout: this file is src/model/data.py → data dir is ../../data.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class Bundle:
    """Everything a training run needs, split into train / validation."""

    X_train: torch.Tensor
    X_val: torch.Tensor
    yh_train: torch.Tensor      # health targets (4)
    yh_val: torch.Tensor
    yp_train: torch.Tensor      # performance targets: thrust, tsfc (2)
    yp_val: torch.Tensor
    # raw context needed by physics losses (unscaled), aligned with X_train:
    fuel_train: torch.Tensor    # FuelFlow_kg_s
    fuel_val: torch.Tensor
    cycle_train: torch.Tensor
    cycle_val: torch.Tensor
    engine_train: torch.Tensor  # EngineID
    engine_val: torch.Tensor
    prc_train: torch.Tensor     # standardised PR_c column (for ratio loss)
    prc_val: torch.Tensor
    # fitted scalers + bookkeeping
    x_scaler: StandardScaler
    thrust_scaler: StandardScaler
    feature_names: list
    target_names: list


def _load_frame(split: str) -> pd.DataFrame:
    """Load train/test inputs and merge their ground-truth targets by key."""
    inputs = pd.read_csv(DATA_DIR / f"{split}.csv")
    truth = pd.read_csv(DATA_DIR / "ground_truth.csv")
    # keep only target columns from truth to avoid duplicate input columns
    keep = ["EngineID", "Cycle"] + [c for c in TARGETS if c in truth.columns]
    merged = inputs.merge(truth[keep], on=["EngineID", "Cycle"], how="left")
    return add_physics_features(merged)


def load_dataset(val_fraction: float = 0.2, seed: int = 0) -> Bundle:
    """Build the training Bundle.

    Training split is `train.csv` (240 rows), carved into train/validation. The
    held-out `test.csv` is used only by eval.py, never here.
    """
    df = _load_frame("train")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_val = int(len(df) * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    tr, va = df.iloc[train_idx], df.iloc[val_idx]

    # Fit scalers on training rows only.
    x_scaler = StandardScaler().fit(tr[MODEL_FEATURES].values)
    thrust_scaler = StandardScaler().fit(tr[["Thrust_N"]].values)

    def scale_x(frame):
        return torch.tensor(x_scaler.transform(frame[MODEL_FEATURES].values),
                            dtype=torch.float32)

    def health(frame):
        return torch.tensor(frame[HEALTH_TARGETS].values, dtype=torch.float32)

    def perf(frame):
        # performance head learns scaled thrust + raw tsfc; tsfc is small (~1e-2)
        thrust = thrust_scaler.transform(frame[["Thrust_N"]].values)
        tsfc = frame[["TSFC_g_N_s"]].values
        return torch.tensor(np.hstack([thrust, tsfc]), dtype=torch.float32)

    def col(frame, name):
        return torch.tensor(frame[name].values, dtype=torch.float32)

    # index of PR_c in the scaled feature matrix, for the ratio-coupling loss
    prc_i = MODEL_FEATURES.index("PR_c")
    Xtr, Xva = scale_x(tr), scale_x(va)

    return Bundle(
        X_train=Xtr,
        X_val=Xva,
        yh_train=health(tr),
        yh_val=health(va),
        yp_train=perf(tr),
        yp_val=perf(va),
        fuel_train=col(tr, "FuelFlow_kg_s"),
        fuel_val=col(va, "FuelFlow_kg_s"),
        cycle_train=col(tr, "Cycle"),
        cycle_val=col(va, "Cycle"),
        engine_train=col(tr, "EngineID"),
        engine_val=col(va, "EngineID"),
        prc_train=Xtr[:, prc_i],
        prc_val=Xva[:, prc_i],
        x_scaler=x_scaler,
        thrust_scaler=thrust_scaler,
        feature_names=list(MODEL_FEATURES),
        target_names=list(TARGETS),
    )


def save_scalers(path, x_scaler, thrust_scaler):
    with open(path, "wb") as f:
        pickle.dump({"x_scaler": x_scaler, "thrust_scaler": thrust_scaler}, f)


def load_scalers(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["x_scaler"], d["thrust_scaler"]
