"""MLP Block used by every GPT layer"""

from __future__ import annotations

import torch
import torch.nn as nn

from skyai.nn.layers import ResidualProjection


class MLP(nn.Module):
    """Position wise feed forward: project up, GELU, project down"""

    def __init__(self, n_embed: int, hidden_multiple: int = 4) -> None:
        super().__init__()
        hidden = hidden_multiple * n_embed
        self.c_fc = nn.Linear(n_embed, hidden)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = ResidualProjection(hidden, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x
    