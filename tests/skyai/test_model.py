"""GPT should: shape-correct, loss-finite, GQA-aware, RoPE-table-correct"""

import pytest
import torch

from skyai.layers import RMSNorm
from skyai.model import GPT, GPTConfig


def _tiny_config(**overrides) -> GPTConfig:
    base = dict(
        block_size=32,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=64,
        logit_softcap=None,
    )
    base.update(overrides)
    return GPTConfig(**base)


def test_forward_shapes():
    cfg = _tiny_config()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_finite_loss():
    cfg = _tiny_config()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    tgt = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(idx, tgt)
    assert loss is not None and torch.isfinite(loss)


def test_vocab_is_padded_to_multiple():
    cfg = _tiny_config(vocab_size=100, vocab_pad_multiple=128)
    assert cfg.vocab_size_padded == 128
    model = GPT(cfg)
    assert model.lm_head.weight.shape == (128, cfg.n_embd)
    assert model.transformer.wte.weight.shape == (128, cfg.n_embd)


def test_forward_excludes_padded_vocab_from_logits():
    cfg = _tiny_config(vocab_size=100, vocab_pad_multiple=128)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))

    logits, _ = model(idx)

    assert logits.shape[-1] == cfg.vocab_size
    assert model.lm_head.weight.shape[0] == cfg.vocab_size_padded


def test_untied_weights_are_independent():
    cfg = _tiny_config(tie_weights=False)
    model = GPT(cfg)
    assert model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr()


def test_tied_weights_share_storage():
    cfg = _tiny_config(tie_weights=True)
    model = GPT(cfg)
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()


def test_token_embeddings_are_normalized_before_blocks():
    model = GPT(_tiny_config())
    assert isinstance(model.transformer.embed_norm, RMSNorm)


def test_rope_tables_match_config():
    cfg = _tiny_config()
    model = GPT(cfg)
    assert model.cos.shape == (1, cfg.block_size, 1, cfg.head_dim // 2)
    assert model.sin.shape == (1, cfg.block_size, 1, cfg.head_dim // 2)
    assert model.cos.dtype == torch.float32


def test_rope_tables_are_non_persistent():
    """Rotary tables shouldn't bloat the checkpoint."""
    cfg = _tiny_config()
    model = GPT(cfg)
    sd = model.state_dict()
    assert "cos" not in sd
    assert "sin" not in sd


def test_gqa_config_works():
    cfg = _tiny_config(n_head=8, n_kv_head=2, n_embd=64)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(idx, idx)
    assert torch.isfinite(loss)


def test_softcap_bounds_logit_magnitude():
    """With softcap=2, no logit should exceed |2| in magnitude."""
    cfg = _tiny_config(logit_softcap=2.0)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        logits, _ = model(idx)
    assert logits.abs().max() <= 2.0 + 1e-5


def test_too_long_sequence_raises():
    cfg = _tiny_config(block_size=16)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 17))
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(idx)
