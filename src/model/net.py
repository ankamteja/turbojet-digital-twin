"""The surrogate model — a compact multi-head MLP.

One shared trunk learns a general representation of the engine's operating
state; two heads specialise:

  * health head       → 4 values in [0, 1] via sigmoid
                        (compressor, combustor, turbine, overall)
  * performance head  → 2 values (scaled thrust, TSFC)

Dropout in the trunk regularises the small dataset and also enables MC-dropout
uncertainty as a fallback to the deep ensemble.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TurbojetNet(nn.Module):
    def __init__(self, n_features: int, hidden=(128, 96, 64), dropout=0.10):
        super().__init__()

        # shared trunk
        layers = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)

        # health head: 4 outputs squashed to [0, 1]
        self.health_head = nn.Sequential(
            nn.Linear(prev, 32), nn.ReLU(),
            nn.Linear(32, 4), nn.Sigmoid(),
        )

        # performance head: 2 outputs (scaled thrust, TSFC), unbounded
        self.perf_head = nn.Sequential(
            nn.Linear(prev, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        z = self.trunk(x)
        return self.health_head(z), self.perf_head(z)
