import torch
import torch.nn as nn

from skyai.nn.init import init_weights
from skyai.nn.layers import Linear, ResidualProjection


def test_linear_init_std_is_0_02():
    torch.manual_seed(0)
    layer = Linear(1024, 1024, bias=False)
    init_weights(layer, n_layer=12)
    measured_std = layer.weight.std().item()
    assert abs(measured_std - 0.02) < 0.001


def test_residual_projection_init_is_depth_scaled():
    torch.manual_seed(0)
    n_layer = 48
    layer = ResidualProjection(1024, 1024, bias=False)
    init_weights(layer, n_layer=n_layer)
    expected_std = 0.02 * (2 * n_layer) ** -0.5
    measured_std = layer.weight.std().item()
    assert abs(measured_std - expected_std) < 0.0005


def test_embedding_init_std_is_0_02():
    torch.manual_seed(0)
    emb = nn.Embedding(10000, 1024)
    init_weights(emb, n_layer=12)
    measured_std = emb.weight.std().item()
    assert abs(measured_std - 0.02) < 0.001


def test_bias_is_zeroed_when_present():
    layer = nn.Linear(128, 128, bias=True)
    nn.init.normal_(layer.bias)
    init_weights(layer, n_layer=12)
    assert torch.all(layer.bias == 0)


def test_rmsnorm_is_unaffected():
    from skyai.nn.layers import RMSNorm

    norm = RMSNorm(128)
    init_weights(norm, n_layer=12)
    assert len(list(norm.parameters())) == 0


def test_residual_branch_wins_over_linear_branch():
    """Because ResidualProjection inherits from Linear, the isinstance
    ordering matters; the depth-scaled branch must execute, not the
    standard branch."""
    torch.manual_seed(0)
    n_layer = 48
    res_proj = ResidualProjection(1024, 1024, bias=False)
    init_weights(res_proj, n_layer=n_layer)
    standard_std = 0.02
    depth_std = 0.02 * (2 * n_layer) ** -0.5
    measured = res_proj.weight.std().item()
    assert abs(measured - depth_std) < 0.0005
    assert abs(measured - standard_std) > 0.01
