"""Tests for primitive layers"""

import torch
import torch.nn as nn

from skyai.layers import Linear, ResidualProjection, RMSNorm


def test_rmsnorm_unit_rms():
    """RMSNorm should produce outputs with unit root mean square"""
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64) * 5.0
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rmsnorm_no_parameters():
    """RMSNorm has no parameters"""
    norm = RMSNorm(64)
    assert sum(p.numel() for p in norm.parameters()) == 0


def test_linear_casts_to_input_dtype():
    """Linear should run the matmul in the input dtype not weight dtype"""
    linear = Linear(8, 4, bias=False)
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    y = linear(x)
    assert y.dtype == torch.bfloat16
    assert linear.weight.dtype == torch.float32


def test_residual_projection_is_a_linear():
    """ResidualProjection should be both linear and nn.linear"""
    proj = ResidualProjection(8, 8, bias=False)
    assert isinstance(proj, Linear)
    assert isinstance(proj, nn.Linear)
    assert isinstance(proj, ResidualProjection)
