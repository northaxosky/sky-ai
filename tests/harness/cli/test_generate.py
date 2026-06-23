"""Tests for the generate module"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from harness.generate import generate


class _ToyLM(nn.Module):
    """Minimal autoregressive model with the (logits, loss) signature"""

    def __init__(self, vocab_size: int = 10, n_embed: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        h = self.emb(x)
        return self.head(h), None


def _model(vocab_size: int = 10) -> _ToyLM:
    torch.manual_seed(0)
    return _ToyLM(vocab_size)


def _prompt(B: int = 2, T: int = 3, vocab_size: int = 10) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, vocab_size, (B, T))


class TestGenerate:
    def test_output_shape(self) -> None:
        out = generate(_model(), _prompt(), max_new_tokens=5)
        assert out.shape == (2, 3 + 5)

    def test_extends_not_replaces_prompt(self) -> None:
        p = _prompt()
        out = generate(_model(), p, max_new_tokens=5)
        assert torch.equal(out[:, : p.size(1)], p)

    def test_does_not_mutate_prompt(self) -> None:
        p = _prompt()
        p_copy = p.clone()
        generate(_model(), p, max_new_tokens=5)
        assert torch.equal(p, p_copy)

    def test_output_dtype_is_long(self) -> None:
        out = generate(_model(), _prompt(), max_new_tokens=3)
        assert out.dtype == torch.long

    def test_deterministic_with_same_generator_seed(self) -> None:
        m = _model()
        p = _prompt()
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        out1 = generate(m, p, max_new_tokens=10, generator=gen1)
        out2 = generate(m, p, max_new_tokens=10, generator=gen2)
        assert torch.equal(out1, out2)

    def test_different_seeds_diverge(self) -> None:
        m = _model()
        p = _prompt()
        gen1 = torch.Generator().manual_seed(1)
        gen2 = torch.Generator().manual_seed(2)
        out1 = generate(m, p, max_new_tokens=20, generator=gen1)
        out2 = generate(m, p, max_new_tokens=20, generator=gen2)
        # 20 draws from vocab=10 with different seeds: essentially zero chance of equality
        assert not torch.equal(out1[:, p.size(1) :], out2[:, p.size(1) :])

    def test_top_k_1_is_greedy(self) -> None:
        m = _model()
        p = _prompt()
        gen1 = torch.Generator().manual_seed(1)
        gen2 = torch.Generator().manual_seed(2)
        out1 = generate(m, p, max_new_tokens=5, top_k=1, generator=gen1)
        out2 = generate(m, p, max_new_tokens=5, top_k=1, generator=gen2)
        assert torch.equal(out1, out2)

    def test_top_k_larger_than_vocab_is_a_noop(self) -> None:
        m = _model(vocab_size=5)
        p = torch.randint(0, 5, (1, 3))
        gen1 = torch.Generator().manual_seed(0)
        gen2 = torch.Generator().manual_seed(0)
        out_no_topk = generate(m, p, max_new_tokens=5, top_k=None, generator=gen1)
        out_huge_topk = generate(m, p, max_new_tokens=5, top_k=1000, generator=gen2)
        assert torch.equal(out_no_topk, out_huge_topk)

    def test_truncates_long_context(self) -> None:
        # prompt is already 5x longer than max_context_len; would crash otherwise
        m = _model()
        p = torch.randint(0, 10, (1, 50))
        out = generate(m, p, max_new_tokens=3, max_context_len=10)
        assert out.shape == (1, 50 + 3)

    def test_restores_training_mode(self) -> None:
        m = _model()
        m.train()
        generate(m, _prompt(), max_new_tokens=3)
        assert m.training is True

    def test_leaves_eval_mode_eval(self) -> None:
        m = _model()
        m.eval()
        generate(m, _prompt(), max_new_tokens=3)
        assert m.training is False

    def test_rejects_1d_prompt(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            generate(_model(), torch.tensor([1, 2, 3]), max_new_tokens=3)

    def test_rejects_zero_temperature(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            generate(_model(), _prompt(), max_new_tokens=3, temperature=0.0)

    def test_rejects_top_k_zero(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            generate(_model(), _prompt(), max_new_tokens=3, top_k=0)

    def test_rejects_zero_max_new_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_new_tokens"):
            generate(_model(), _prompt(), max_new_tokens=0)
