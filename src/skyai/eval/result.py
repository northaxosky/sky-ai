"""Shared types for skyai evals: result dataclass and call signature protocol"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import tiktoken
import torch
from torch import nn


@dataclass(frozen=True)
class EvalResult:
    """Result of running one eval on a model"""

    name: str
    metrics: dict[str, float]
    num_examples: int


class EvalFn(Protocol):
    """Call signature every eval kernel implements"""

    def __call__(
        self,
        model: nn.Module,
        *,
        encoder: tiktoken.Encoding,
        device: str | torch.device,
        rank: int,
        world_size: int,
        dtype: torch.dtype = ...,
    ) -> EvalResult: ...
