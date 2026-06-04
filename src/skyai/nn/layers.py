"""Shared building-block layers used across the nn/ package"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root mean square normalization"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim, ))


class Linear(nn.Linear):
    """Linear that casts weights to match input dtype at forward"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


class ResidualProjection(Linear):
    """Marker subclass for output projections that feed the residual stream"""
    