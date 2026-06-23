"""Tests for SwiGLU MLP"""

import torch
import torch.nn as nn

from skyai.mlp import MLP


def test_mlp_shape_preserving():
    """MLP should preserve (B, T, n_embed) shape"""
    mlp = MLP(n_embed=64)
    x = torch.randn(2, 8, 64)
    y = mlp(x)
    assert y.shape == x.shape


def test_mlp_no_biases():
    """All linears should be bias free"""
    mlp = MLP(n_embed=64)
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            assert m.bias is None, f"{type(m).__name__} has a bias"


def test_mlp_hidden_dim_8_3_with_alignment():
    """Hidden dim should be the 8/3 formula rounded up to multiple of align"""
    mlp = MLP(n_embed=768, hidden_multiple=4, align=256)
    assert mlp.gate_proj.out_features == 2048
    assert mlp.up_proj.out_features == 2048

    mlp_xl = MLP(n_embed=1600, hidden_multiple=4, align=256)
    assert mlp_xl.gate_proj.out_features == 4352


def test_mlp_gradient_flow():
    """Backward should populate gradients on all weights"""
    mlp = MLP(n_embed=64)
    x = torch.randn(2, 8, 64, requires_grad=True)
    y = mlp(x).sum()
    y.backward()
    for p in mlp.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_mlp_param_count_matched_to_gelu():
    """Swiglu at 8/3 hidden should have same params as 4xGELU"""
    n = 768
    mlp = MLP(n_embed=n, hidden_multiple=4, align=256)
    swiglu_params = sum(p.numel() for p in mlp.parameters())
    expected = 2 * n * (4 * n)
    assert swiglu_params == expected
