"""Train a deep ensemble of physics-informed surrogate models.

Deep ensemble = train N independent networks from different random seeds. Their
mean is the prediction; their spread (standard deviation) is the uncertainty —
a cheap, robust way to get calibrated confidence on a small dataset.

Run:  python train.py            # trains N=5, saves to artifacts/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import load_dataset, save_scalers
from net import TurbojetNet
from physics import (
    loss_monotonic_degradation,
    loss_overall_aggregation,
    loss_ratio_health_coupling,
    loss_tsfc_consistency,
)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
N_MODELS = 5
EPOCHS = 600
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 80

# physics-loss weights (tuned on validation)
W_THRUST = 1.0     # scaled-thrust MSE vs health MSE
W_TSFC = 5.0       # tsfc supervised MSE (tsfc is ~1e-2, needs up-weighting)
L_TSFC = 0.5       # physics: tsfc consistency
L_OVERALL = 0.5    # physics: overall = aggregate of parts
L_MONO = 0.2       # physics: monotonic degradation
L_RATIO = 0.05     # physics: ratio ↔ health coupling (weak guardrail)


def _thrust_unscale(scaled, thrust_scaler, device):
    """Invert StandardScaler on the thrust column, in torch."""
    mean = torch.tensor(thrust_scaler.mean_[0], dtype=torch.float32, device=device)
    scale = torch.tensor(thrust_scaler.scale_[0], dtype=torch.float32, device=device)
    return scaled * scale + mean


def train_one(bundle, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = TurbojetNet(n_features=bundle.X_train.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()

    Xtr = bundle.X_train.to(device)
    yh = bundle.yh_train.to(device)
    yp = bundle.yp_train.to(device)
    fuel = bundle.fuel_train.to(device)
    cycle = bundle.cycle_train.to(device)
    engine = bundle.engine_train.to(device)
    prc = bundle.prc_train.to(device)

    Xva = bundle.X_val.to(device)
    yhv = bundle.yh_val.to(device)
    ypv = bundle.yp_val.to(device)

    best_val = float("inf")
    best_state = None
    since_improve = 0

    for epoch in range(EPOCHS):
        net.train()
        opt.zero_grad()
        ph, pp = net(Xtr)                       # health(4), perf(2: thrust_s, tsfc)

        # --- supervised losses ---
        l_health = mse(ph, yh)
        l_thrust = mse(pp[:, 0], yp[:, 0])
        l_tsfc_sup = mse(pp[:, 1], yp[:, 1])

        # --- physics losses (need real-unit thrust) ---
        thrust_real = _thrust_unscale(pp[:, 0], bundle.thrust_scaler, device)
        pl_tsfc = loss_tsfc_consistency(thrust_real, pp[:, 1], fuel)
        pl_overall = loss_overall_aggregation(ph)
        pl_mono = loss_monotonic_degradation(ph, cycle, engine)
        pl_ratio = loss_ratio_health_coupling(ph, prc)

        loss = (
            l_health
            + W_THRUST * l_thrust
            + W_TSFC * l_tsfc_sup
            + L_TSFC * pl_tsfc
            + L_OVERALL * pl_overall
            + L_MONO * pl_mono
            + L_RATIO * pl_ratio
        )
        loss.backward()
        opt.step()

        # --- validation (supervised only) ---
        net.eval()
        with torch.no_grad():
            phv, ppv = net(Xva)
            val = (mse(phv, yhv)
                   + W_THRUST * mse(ppv[:, 0], ypv[:, 0])
                   + W_TSFC * mse(ppv[:, 1], ypv[:, 1])).item()

        if val < best_val - 1e-6:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= PATIENCE:
                break

    net.load_state_dict(best_state)
    return net, best_val


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset()
    print(f"features: {bundle.X_train.shape[1]}  "
          f"train: {len(bundle.X_train)}  val: {len(bundle.X_val)}")

    val_scores = []
    for i in range(N_MODELS):
        net, val = train_one(bundle, seed=1000 + i, device=device)
        torch.save(net.state_dict(), ARTIFACTS / f"net_{i}.pt")
        val_scores.append(val)
        print(f"model {i}: best val loss {val:.5f}")

    save_scalers(ARTIFACTS / "scalers.pkl", bundle.x_scaler, bundle.thrust_scaler)

    config = {
        "n_models": N_MODELS,
        "n_features": bundle.X_train.shape[1],
        "feature_names": bundle.feature_names,
        "target_names": bundle.target_names,
        "hidden": [128, 96, 64],
        "dropout": 0.10,
        "val_scores": val_scores,
        "mean_val": float(np.mean(val_scores)),
    }
    with open(ARTIFACTS / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"saved {N_MODELS} models + scalers + config to {ARTIFACTS}")
    print(f"mean val loss: {np.mean(val_scores):.5f}")


if __name__ == "__main__":
    main()
