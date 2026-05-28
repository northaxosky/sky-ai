"""GPT-2 weight initialization"""

from __future__ import annotations

import torch.nn as nn

from skyai.nn.layers import ResidualProjection


def init_gpt2_weights(module: nn.Module, n_layer: int) -> None:
    """Apply GPT-2's init recipe to module

    Standard linears and embeddings get N(0, 0.02). Residual path projections
    scale by 1/sqrt(2 * n_layer) so the streams variance is bounded as depth grows
    """
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