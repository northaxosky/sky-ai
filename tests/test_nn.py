"""Tests for the nn/ package"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from skyai.nn.model import GPT, GPTConfig
from skyai.nn.block import Block
from skyai.nn.init import init_gpt2_weights
from skyai.nn.mlp import MLP
from skyai.nn.attention import CausalSelfAttention
from skyai.nn.layers import ResidualProjection


class TestGPTConfig:
    def test_defualts_match_124m(self) -> None:
        config = GPTConfig()
        assert config.block_size == 1024
        assert config.vocab_size == 50257
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_embed == 768
        assert config.hidden_multiple == 4


class TestGPT:
    def _tiny_config(self) -> GPTConfig:
        return GPTConfig(
            block_size=32, vocab_size=100, n_layer=2, n_head=4, n_embed=64
        )
    
    def test_param_count_at_default_config_is_124m(self) -> None:
        model = GPT(GPTConfig())
        n_params = sum(p.numel() for p in model.parameters())

        assert 123_000_000 < n_params < 125_000_000

    def test_forward_shape_without_targets(self) -> None:
        config = self._tiny_config()
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (2, 16))
        targets = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss = model(idx, targets)

        assert logits.shape == (2, 16, config.vocab_size)
        assert loss is not None
        assert loss.dim() == 0
    
    def test_rejects_oversized_sequence(self) -> None:
        config = self._tiny_config()
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (1, config.block_size + 1))
        with pytest.raises(ValueError, match="exceeds block_size"):
            model(idx)

    def test_weight_tying_shares_tensor(self) -> None:
        model = GPT(self._tiny_config())
        assert model.transformer.wte.weight is model.lm_head.weight

    def test_gradients_flow_end_to_end(self) -> None:
        config = self._tiny_config()
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (2, 16))
        targets = torch.randint(0, config.vocab_size, (2, 16))
        _, loss = model(idx, targets)
        
        assert loss is not None
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
    

class TestBlock:
    def test_shape_preserved(self) -> None:
        block = Block(n_embed=64, n_head=4)
        x = torch.randn(2, 16, 64)
        y = block(x)

        assert y.shape == x.shape

    def test_gradients_flow(self) -> None:
        block = Block(n_embed=64, n_head=4)
        x = torch.randn(2, 16, 64, requires_grad=True)
        y = block(x).sum()
        y.backward()

        assert x.grad is not None
        for name, parameter in block.named_parameters():
            assert parameter.grad is not None, f"No gradient for {name}"

    def test_has_two_layernorms(self) -> None:
        block = Block(n_embed=64, n_head=4)
        children = dict(block.named_children())

        assert isinstance(children['ln_1'], nn.LayerNorm)
        assert isinstance(children['ln_2'], nn.LayerNorm)

    def test_residual_passes_input_through_when_sublayers_zeroed(self) -> None:
        block = Block(n_embed=64, n_head=4)
        for module in [block.attn, block.mlp]:
            for param in module.parameters():
                nn.init.zeros_(param)

        x = torch.randn(2, 8, 64)
        y = block(x)

        assert torch.allclose(y, x, atol=1e-5)


class TestInit:
    def test_linear_std_is_0_02(self) -> None:
        torch.manual_seed(0)
        linear = nn.Linear(4096, 4096)
        init_gpt2_weights(linear, n_layer=12)

        assert abs(linear.weight.std().item() - 0.02) < 0.001

    def test_embedding_std_is_0_02(self) -> None:
        torch.manual_seed(0)
        embedding = nn.Embedding(50257, 768)
        init_gpt2_weights(embedding, n_layer=12)

        assert abs(embedding.weight.std().item() - 0.02) < 0.001

    def test_residual_projection_std_is_scaled(self) -> None:
        torch.manual_seed(0)
        n_layer = 12
        proj = ResidualProjection(4096, 4096)
        init_gpt2_weights(proj, n_layer=n_layer)

        expected = 0.02 * (2 * n_layer) ** -0.5
        assert abs(proj.weight.std().item() - expected) < 0.0001

    def test_linear_bias_is_zero(self) -> None:
        linear = nn.Linear(4096, 4096)
        nn.init.normal_(linear.bias, mean=1.0, std=1.0)
        init_gpt2_weights(linear, n_layer=12)

        assert torch.all(linear.bias == 0)

    def test_residual_projection_bias_is_zero(self) -> None:
        proj = ResidualProjection(4096, 4096)
        nn.init.normal_(proj.bias, mean=1.0, std=1.0)
        init_gpt2_weights(proj, n_layer=12)

        assert torch.all(proj.bias == 0)


class TestMLP:
    def test_shape_preserved(self) -> None:
        mlp = MLP(n_embed=768)
        x = torch.randn(2, 16, 768)
        y = mlp(x)

        assert y.shape == x.shape

    def test_gradients_flow(self) -> None:
        mlp = MLP(n_embed=768)
        x = torch.randn(2, 16, 768, requires_grad=True)
        y = mlp(x).sum()
        y.backward()

        assert x.grad is not None
        for name, parameter in mlp.named_parameters():
            assert parameter.grad is not None, f"No gradient for {name}"

    def test_hidden_dim_uses_multiple(self) -> None:
        mlp = MLP(n_embed=64, hidden_multiple=4)
        assert mlp.c_fc.out_features == 256

        wide = MLP(n_embed=64, hidden_multiple=8)
        assert wide.c_fc.out_features == 512

    def test_c_proj_is_residual_projection(self) -> None:
        mlp = MLP(n_embed=768)
        assert isinstance(mlp.c_proj, ResidualProjection)
        assert not isinstance(mlp.c_fc, ResidualProjection)

    def test_no_buffered_state(self) -> None:
        mlp = MLP(n_embed=768)
        buffer_names = [name for name, _ in mlp.named_buffers()]
        assert buffer_names == [], f"Unexpected buffers: {buffer_names}"


class TestCausalSelfAttention:
    def test_shape_preserved(self) -> None:
        attn = CausalSelfAttention(n_embed=768, n_head=12)
        x = torch.randn(2, 16, 768)
        y = attn(x)

        assert y.shape == x.shape

    def test_gradients_flow(self) -> None:
        attn = CausalSelfAttention(n_embed=768, n_head=12)
        x = torch.randn(2, 16, 768, requires_grad=True)
        y = attn(x).sum()
        y.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        for name, parameter in attn.named_parameters():
            assert parameter.grad is not None, f"No gradient for {name}"

    def test_rejects_indivisible_head_count(self) -> None:
        with pytest.raises(ValueError, match="must be divisible by"):
            CausalSelfAttention(n_embed=768, n_head=7)

    def test_c_proj_is_residual_projection(self) -> None:
        attn = CausalSelfAttention(n_embed=768, n_head=12)
        assert isinstance(attn.c_proj, ResidualProjection)
        assert not isinstance(attn.c_attn, ResidualProjection)

    def test_no_buffered_mask(self) -> None:
        attn = CausalSelfAttention(n_embed=768, n_head=12)
        buffer_names = [name for name, _ in attn.named_buffers()]
        assert buffer_names == [], f"Unexpected buffers: {buffer_names}"        

