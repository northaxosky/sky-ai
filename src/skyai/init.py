"""Modern transformer weight initialization"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch.nn as nn

from skyai.layers import ResidualProjection


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


def init_sky_ai_weights(
    *,
    wte: nn.Embedding,
    lm_head: nn.Linear,
    blocks: Iterable[nn.Module],
    n_embed: int,
    tie_weights: bool,
) -> None:
    """Initialize the modern SkyAI architecture policy."""
    if tie_weights:
        nn.init.normal_(wte.weight, mean=0.0, std=0.02)
    else:
        nn.init.normal_(wte.weight, mean=0.0, std=0.8)
        nn.init.normal_(lm_head.weight, mean=0.0, std=0.001)
        _zero_bias(lm_head)

    attn_input_std = n_embed**-0.5
    mlp_input_std = 0.4 * attn_input_std

    for block in blocks:
        _uniform_with_std_(block.attn.c_q, attn_input_std)  # pyright: ignore
        _uniform_with_std_(block.attn.c_k, attn_input_std)  # pyright: ignore
        _uniform_with_std_(block.attn.c_v, attn_input_std)  # pyright: ignore

        _zero_weight_and_bias(block.attn.c_proj)  # pyright: ignore

        _uniform_with_std_(block.mlp.gate_proj, mlp_input_std)  # pyright: ignore
        _uniform_with_std_(block.mlp.up_proj, mlp_input_std)  # pyright: ignore

        _zero_weight_and_bias(block.mlp.down_proj)  # pyright: ignore


def _uniform_with_std_(module: nn.Linear, std: float) -> None:
    bound = math.sqrt(3.0) * std
    nn.init.uniform_(module.weight, -bound, bound)
    _zero_bias(module)


def _zero_weight_and_bias(module: nn.Linear) -> None:
    nn.init.zeros_(module.weight)
    _zero_bias(module)


def _zero_bias(module: nn.Linear) -> None:
    if module.bias is not None:
        nn.init.zeros_(module.bias)
