"""Dataset loading and feature engineering.

Boosted trees are scale-invariant, so there is no scaler to fit — we just load
the CSVs, attach the engineered physics features, and hand back a frame plus
ready-to-use arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from physics import LEARNED_TARGETS, MODEL_FEATURES, TARGETS, add_physics_features

# Repo layout: this file is src/model/data.py → data dir is ../../data.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_frame(split: str) -> pd.DataFrame:
    """Load a split's inputs, merge its ground-truth targets, add physics feats.

    split ∈ {"train", "test"}. "complete" loads the full combined table.
    """
    if split == "complete":
        df = pd.read_csv(DATA_DIR / "turbojet_complete_dataset.csv")
        return add_physics_features(df)

    inputs = pd.read_csv(DATA_DIR / f"{split}.csv")
    truth = pd.read_csv(DATA_DIR / "ground_truth.csv")
    keep = ["EngineID", "Cycle"] + [c for c in TARGETS if c in truth.columns]
    merged = inputs.merge(truth[keep], on=["EngineID", "Cycle"], how="left")
    return add_physics_features(merged)


def xy(df: pd.DataFrame):
    """Split a frame into (feature matrix X, {target: values} dict)."""
    X = df[MODEL_FEATURES].values.astype(float)
    y = {t: df[t].values.astype(float) for t in LEARNED_TARGETS}
    return X, y
