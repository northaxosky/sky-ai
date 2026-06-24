from __future__ import annotations

import torch.nn as nn


def init_weights(module: nn.Mudle, n_layer: int) -> None:
    """GPT-2 init, applied via model.apply(lambda m: init_weights(m, n_layer))"""
    if isinstance(module, nn.Linear):
        std = 0.02
        if getattr(module, "_is_residual_projection", False):
            std *= (2 * n_layer) ** -0.5
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    # nn.LayerNorm intentionally untouched: stays at default (weight=1, bias=0)
