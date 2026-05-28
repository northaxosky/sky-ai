"""Shared building-block layers used across the nn/ package"""

from __future__ import annotations

import torch.nn as nn


class ResidualProjection(nn.Linear):
    """Marker subclass for output projections that feed the residual stream"""
    