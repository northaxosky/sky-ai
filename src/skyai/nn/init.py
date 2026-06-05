"""Modern transformer weight initialization"""

from __future__ import annotations

import torch.nn as nn

from skyai.nn.layers import ResidualProjection


def init_weights(module: nn.Module, n_layer: int) -> None:
    """initialize a single module in place. designed for model.apply()"""
    if isinstance(module, ResidualProjection):
        std = 0.02 * (2 * n_layer) ** -0.5
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
