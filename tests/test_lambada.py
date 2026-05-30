"""Tests for LAMBADA eval module: rendering, scoring, and the eval loop"""

from __future__ import annotations

import math

import pytest
import tiktoken
import torch
from torch import nn

from skyai.eval import lambada as lam
from skyai.eval.result import EvalResult


@pytest.fixture
def encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def passage() -> str:
    return "The quick brown fox jumps over the lazy dog"


class TestRenderLambada:
    def test_shapes(self, encoder: tiktoken.Encoding, passage: str) -> None:
        input_ids, gt_target_ids, target_len = lam._render_lambada(
            passage, encoder, block_size=1024
        )
        assert input_ids.ndim == 2
        assert input_ids.shape[0] == 1
        assert gt_target_ids.ndim == 1
        assert gt_target_ids.shape[0] == target_len
        assert input_ids.dtype == torch.long
        assert gt_target_ids.dtype == torch.long

    def test_input_plus_target_reconstructs_full(
        self, encoder: tiktoken.Encoding, passage: str
    ) -> None:
        """input_ids is full_ids[:-1]; gt_target_ids is full_ids[-target_len:]"""
        input_ids, gt_target_ids, target_len = lam._render_lambada(
            passage, encoder, block_size=1024
        )
        full_ids = encoder.encode(passage)
        assert input_ids.squeeze(0).tolist() == full_ids[:-1]
        assert gt_target_ids.tolist() == full_ids[-target_len:]

    def test_target_len_matches_last_word_encoding(
        self, encoder: tiktoken.Encoding, passage: str
    ) -> None:
        _, _, target_len = lam._render_lambada(passage, encoder, block_size=1024)
        assert target_len == len(encoder.encode(" dog"))

    def test_raises_when_no_whitespace(self, encoder: tiktoken.Encoding) -> None:
        with pytest.raises(ValueError, match="no whitespace boundary"):
            lam._render_lambada("oneword", encoder, block_size=1024)

    def test_front_truncates_when_over_block_size(self, encoder: tiktoken.Encoding) -> None:
        long_passage = " ".join(["word"] * 500) + " endword"
        full_len = len(encoder.encode(long_passage))
        assert full_len > 256, "test setup requires passage to exceed block_size"

        input_ids, gt_target_ids, target_len = lam._render_lambada(
            long_passage, encoder, block_size=256
        )
        assert input_ids.shape[1] == 256 - 1
        # Target span is the final tokens, preserved
        assert gt_target_ids.tolist() == encoder.encode(long_passage)[-target_len:]


class TestScoreLambadaLogits:
    def test_perfect_predictions(self) -> None:
        """One-hot logits at every gt target position â correct, zero NLL"""
        vocab = 100
        target_len = 3
        gt = torch.tensor([7, 42, 99])
        logits = torch.zeros(1, 5, vocab)
        for i, tid in enumerate(gt.tolist()):
            logits[0, -target_len + i, tid] = 100.0

        is_correct, sum_nll = lam._score_lambada_logits(logits, gt, target_len)
        assert is_correct is True
        assert sum_nll == pytest.approx(0.0, abs=1e-3)

    def test_one_wrong_token_marks_incorrect(self) -> None:
        vocab = 100
        target_len = 3
        gt = torch.tensor([7, 42, 99])
        logits = torch.zeros(1, 5, vocab)
        for i, tid in enumerate(gt.tolist()):
            logits[0, -target_len + i, tid] = 100.0
        # Sabotage the last target position
        logits[0, -1, 99] = 0.0
        logits[0, -1, 50] = 100.0

        is_correct, sum_nll = lam._score_lambada_logits(logits, gt, target_len)
        assert is_correct is False
        assert sum_nll > 1.0

    def test_uniform_logits_yield_max_entropy_nll(self) -> None:
        """Uniform logits over vocab V â per-token NLL = log(V); summed = target_len*log(V)"""
        vocab = 100
        target_len = 4
        gt = torch.tensor([1, 2, 3, 4])
        logits = torch.zeros(1, 6, vocab)

        is_correct, sum_nll = lam._score_lambada_logits(logits, gt, target_len)
        # Argmax of all-zeros is index 0 for every position; never equals gt
        assert is_correct is False
        assert sum_nll == pytest.approx(target_len * math.log(vocab), rel=1e-4)


class _StubModel(nn.Module):
    """Returns one-hot logits matching `gt_targets` at the trailing positions; zeros elsewhere"""

    def __init__(self, vocab_size: int, gt_targets: list[int] | None = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.gt_targets = gt_targets

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, None]:
        n_rows, seq = tokens.shape
        logits = torch.zeros(n_rows, seq, self.vocab_size)
        if self.gt_targets is not None:
            tl = len(self.gt_targets)
            for i, tid in enumerate(self.gt_targets):
                logits[0, -tl + i, tid] = 100.0
        return logits, None


class TestEvaluateLambada:
    def test_returns_eval_result_with_both_metrics(
        self, encoder: tiktoken.Encoding, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        passages = [
            "The cat sat on the mat",
            "Hello there friend of mine",
        ]
        monkeypatch.setattr(lam, "_load_examples", lambda: passages)

        model = _StubModel(vocab_size=encoder.n_vocab)
        result = lam.evaluate_lambada(
            model, encoder=encoder, device="cpu",
            rank=0, world_size=1, dtype=torch.float32,
        )
        assert isinstance(result, EvalResult)
        assert result.name == "lambada"
        assert set(result.metrics.keys()) == {"accuracy", "perplexity"}
        assert result.num_examples == 2
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert result.metrics["perplexity"] > 0.0

    def test_ddp_sharding(
        self, encoder: tiktoken.Encoding, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        passages = [f"passage number {i} ends here" for i in range(6)]
        monkeypatch.setattr(lam, "_load_examples", lambda: passages)
        monkeypatch.setattr(lam.dist, "all_reduce", lambda t, op=None: t)

        model = _StubModel(vocab_size=encoder.n_vocab)
        r0 = lam.evaluate_lambada(
            model, encoder=encoder, device="cpu",
            rank=0, world_size=2, dtype=torch.float32,
        )
        r1 = lam.evaluate_lambada(
            model, encoder=encoder, device="cpu",
            rank=1, world_size=2, dtype=torch.float32,
        )
        assert r0.num_examples == 3
        assert r1.num_examples == 3


class TestPinning:
    def test_lambada_revision_is_pinned_sha(self) -> None:
        """Defense against accidentally clearing the revision pin"""
        rev = lam.LAMBADA_REVISION
        assert isinstance(rev, str)
        assert len(rev) == 40
        assert all(c in "0123456789abcdef" for c in rev)