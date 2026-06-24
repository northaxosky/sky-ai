"""GPT-2 (124M) model tests"""

import torch

from gpt.model import GPT, GPTConfig


def test_param_count_is_gpt2_124m():
    """Exact structural tripwire: Confirm model matches"""
    model = GPT(GPTConfig())  # default config = GPT-2 small
    assert sum(p.numel() for p in model.parameters()) == 124_439_808


def test_forward_shapes_and_loss():
    cfg = GPTConfig(block_size=16, vocab_size=256, n_layer=2, n_head=4, n_embd=64)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(idx, targets)

    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_no_targets_returns_none_loss():
    cfg = GPTConfig(block_size=16, vocab_size=256, n_layer=2, n_head=4, n_embd=64)
    logits, loss = GPT(cfg)(torch.randint(0, 256, (1, 8)))
    assert logits.shape == (1, 8, 256) and loss is None
