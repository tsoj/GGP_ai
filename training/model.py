"""Small MLP that maps a pooled embedding to a scalar in [-1, +1]."""
from __future__ import annotations

import torch
from torch import nn


class OutcomeMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 2560,
        hidden: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
