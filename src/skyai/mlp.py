"""MLP Block used by every GPT layer"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyai.layers import Linear, ResidualProjection


class MLP(nn.Module):
    """SwiGLU position wise feed forward"""

    def __init__(self, n_embd: int, hidden_multiple: int = 4, align: int = 256) -> None:
        super().__init__()

        # 8/3 scaling keeps params equal to hidden_multiple x GELU MLP
        hidden = int(2 * hidden_multiple * n_embd / 3)

        # Round up to multiple of align for hardware friendly shape
        hidden = ((hidden + align - 1) // align) * align

        self.gate_proj = Linear(n_embd, hidden, bias=False)
        self.up_proj = Linear(n_embd, hidden, bias=False)
        self.down_proj = ResidualProjection(hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
