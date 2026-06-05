import pytest
import torch
import torch.nn as nn

from skyai.nn.init import init_weights
from skyai.nn.layers import Linear, ResidualProjection
from skyai.nn.model import GPT, GPTConfig


def _tiny_config(**overrides) -> GPTConfig:
    base = dict(
        block_size=16,
        vocab_size=512,
        vocab_pad_multiple=128,
        n_layer=4,
        n_head=4,
        n_embed=128,
        logit_softcap=None,
    )
    base.update(overrides)
    return GPTConfig(**base)  # pyright: ignore


def _assert_std(tensor: torch.Tensor, expected: float, rel: float = 0.15) -> None:
    assert tensor.float().std().item() == pytest.approx(expected, rel=rel)


def test_gpt2_policy_keeps_current_model_init() -> None:
    torch.manual_seed(0)
    model = GPT(_tiny_config(init_policy="gpt2"))

    _assert_std(model.transformer.wte.weight, 0.02, rel=0.10)
    _assert_std(model.lm_head.weight, 0.02, rel=0.10)

    expected_resid = 0.02 * (2 * model.config.n_layer) ** -0.5
    _assert_std(model.transformer.h[0].attn.c_proj.weight, expected_resid)  # pyright: ignore
    _assert_std(model.transformer.h[0].mlp.down_proj.weight, expected_resid)  # pyright: ignore


def test_sky_ai_policy_uses_embedding_and_head_scales() -> None:
    torch.manual_seed(0)
    model = GPT(_tiny_config(init_policy="skyai", tie_weights=False))

    _assert_std(model.transformer.wte.weight, 0.8, rel=0.05)
    _assert_std(model.lm_head.weight, 0.001, rel=0.10)


def test_sky_ai_policy_keeps_tied_weights_on_compat_scale() -> None:
    torch.manual_seed(0)
    model = GPT(_tiny_config(init_policy="skyai", tie_weights=True))

    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()
    _assert_std(model.transformer.wte.weight, 0.02, rel=0.10)


def test_sky_ai_policy_uses_width_scaled_inputs() -> None:
    torch.manual_seed(0)
    cfg = _tiny_config(init_policy="skyai")
    model = GPT(cfg)
    block = model.transformer.h[0]

    _assert_std(block.attn.c_q.weight, cfg.n_embed**-0.5)  # pyright: ignore
    _assert_std(block.attn.c_k.weight, cfg.n_embed**-0.5)  # pyright: ignore
    _assert_std(block.attn.c_v.weight, cfg.n_embed**-0.5)  # pyright: ignore
    _assert_std(block.mlp.gate_proj.weight, 0.4 * cfg.n_embed**-0.5)  # pyright: ignore
    _assert_std(block.mlp.up_proj.weight, 0.4 * cfg.n_embed**-0.5)  # pyright: ignore


def test_sky_ai_policy_zeroes_residual_outputs() -> None:
    torch.manual_seed(0)
    model = GPT(_tiny_config(init_policy="skyai"))

    for block in model.transformer.h:
        assert torch.count_nonzero(block.attn.c_proj.weight) == 0  # pyright: ignore
        assert torch.count_nonzero(block.mlp.down_proj.weight) == 0  # pyright: ignore


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
