import torch

from gpt.model import GPT, GPTConfig


def test_gpt2_init_stds():
    cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=4, n_head=4, n_embd=64)
    model = GPT(cfg)

    # plain weights ~ 0.02
    assert abs(model.transformer.wte.weight.std().item() - 0.02) < 0.004
    assert abs(model.transformer.h[0].mlp.c_fc.weight.std().item() - 0.02) < 0.004

    # residual projections scaled by 1/sqrt(2 * n_layer)
    expected = 0.02 * (2 * cfg.n_layer) ** -0.5
    for block in model.transformer.h:
        assert abs(block.attn.c_proj.weight.std().item() - expected) < 0.003
        assert abs(block.mlp.c_proj.weight.std().item() - expected) < 0.003


def test_biases_zero_layernorm_default():
    cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=4, n_embd=64)
    model = GPT(cfg)
    assert torch.all(model.transformer.h[0].attn.c_proj.bias == 0)
    assert torch.all(model.transformer.ln_f.weight == 1)  # LayerNorm y left at 1
