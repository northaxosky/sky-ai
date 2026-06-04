"""Tests for HellaSwag eval module: rendering, scoring, and the eval loop"""

from __future__ import annotations

from typing import Any

import pytest
import tiktoken
import torch
from torch import nn

from skyai.eval import hellaswag as hs
from skyai.eval.result import EvalResult


@pytest.fixture
def encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")

@pytest.fixture
def example() -> dict[str, Any]:
    """Synthetic hellaswag exampel with 4 completions of deliberately varying length"""
    return {
        "ctx": "The cat sat on",
        "label": 1,
        "endings": [
            "the mat.",
            "the windowsill watching the birds outside.",
            "fire.",
            "Tuesday in October."
        ]
    }


class TestRenderExample:
    def test_shapes(self, encoder: tiktoken.Encoding, example: dict[str, Any]) -> None:
        _, tokens, mask, label = hs.render_example(example, encoder=encoder)
        assert tokens.shape == mask.shape
        assert tokens.shape[0] == 4
        assert tokens.dtype == torch.long
        assert mask.dtype == torch.long
        assert label == 1

    def test_mask_zero_in_context(
        self, encoder: tiktoken.Encoding, example: dict[str, Any]
    ) -> None:
        _, _, mask, _ = hs.render_example(example, encoder=encoder)
        ctx_len = len(encoder.encode(example["ctx"]))
        assert torch.all(mask[:, :ctx_len] == 0)

    def test_mask_one_in_completion(
        self, encoder: tiktoken.Encoding, example: dict[str, Any]
    ) -> None:
        _, _, mask, _ = hs.render_example(example, encoder=encoder)
        ctx_len = len(encoder.encode(example["ctx"]))
        end_tokens = encoder.encode(" " + example["endings"][1])
        assert torch.all(mask[1, ctx_len : ctx_len + len(end_tokens)] == 1)

    def test_padding_does_not_pollute(
        self, encoder: tiktoken.Encoding, example: dict[str, Any]
    ) -> None:
        _, tokens, mask, _ = hs.render_example(example, encoder=encoder)
        ctx_len = len(encoder.encode(example["ctx"]))
        end_2 = encoder.encode(" " + example["endings"][2])  # "fire." â shortest
        real_end = ctx_len + len(end_2)
        assert real_end < tokens.shape[1], "test setup requires row 2 to be padded"
        assert torch.all(mask[2, real_end:] == 0)
        assert torch.all(tokens[2, real_end:] == 0)

    def test_encoder_is_parameterized(self, example: dict[str, Any]) -> None:
        """A different encoder works without hitting any module-level gpt2 fallback"""
        cl100k = tiktoken.get_encoding("cl100k_base")
        _, tokens, mask, label = hs.render_example(example, encoder=cl100k)
        assert tokens.shape == mask.shape
        assert tokens.shape[0] == 4
        assert label == 1
 
 
class TestComputeCompletionLosses:
    def test_shapes(self) -> None:
        tokens = torch.zeros((4, 6), dtype=torch.long)
        mask = torch.zeros((4, 6), dtype=torch.long)
        mask[:, 3:] = 1
        logits = torch.randn(4, 6, 10)
        sum_loss, avg_loss = hs.compute_completion_losses(tokens, mask, logits)
        assert sum_loss.shape == (4,)
        assert avg_loss.shape == (4,)

    def test_avg_equals_sum_over_completion_count(self) -> None:
        tokens = torch.randint(0, 10, (4, 6))
        mask = torch.zeros((4, 6), dtype=torch.long)
        mask[:, 3:] = 1  # 3 completion tokens per row; after shift, 3 ones in shift_mask
        logits = torch.randn(4, 6, 10)
        sum_loss, avg_loss = hs.compute_completion_losses(tokens, mask, logits)
        assert torch.allclose(avg_loss, sum_loss / 3.0)


class TestGetMostLikelyRow:
    def test_picks_lowest_avg_loss_row(self) -> None:
        tokens = torch.tensor([[0, 0, 1, 2], [0, 0, 3, 4], [0, 0, 5, 6], [0, 0, 7, 8]])
        mask = torch.zeros((4, 4), dtype=torch.long)
        mask[:, 2:] = 1
        # Row 2 predicts its own tokens perfectly at the two completion positions
        logits = torch.zeros(4, 4, 10)
        logits[2, 1, 5] = 100.0
        logits[2, 2, 6] = 100.0
        assert hs.get_most_likely_row(tokens, mask, logits) == 2


class _StubModel(nn.Module):
    """Deterministic logits: makes `winning_row` predict its own tokens perfectly"""

    def __init__(self, vocab_size: int, winning_row: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.winning_row = winning_row

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, None]:
        n_rows, seq = tokens.shape
        logits = torch.zeros(n_rows, seq, self.vocab_size)
        for t in range(seq - 1):
            target = int(tokens[self.winning_row, t + 1].item())
            logits[self.winning_row, t, target] = 100.0
        return logits, None


class TestEvaluateHellaswag:
    def test_returns_eval_result_with_both_metrics(
        self, encoder: tiktoken.Encoding, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        examples = [
            {"ctx": "Hello", "label": 0, "endings": ["world", "there", "friend", "stranger"]},
            {"ctx": "Goodbye", "label": 1, "endings": ["foe", "now", "soon", "later"]},
        ]
        monkeypatch.setattr(hs, "iterate_examples", lambda split: iter(examples))
        model = _StubModel(vocab_size=encoder.n_vocab, winning_row=0)
        result = hs.evaluate_hellaswag(
            model, encoder=encoder, device="cpu",
            rank=0, world_size=1, dtype=torch.float32,
        )
        assert isinstance(result, EvalResult)
        assert result.name == "hellaswag"
        assert set(result.metrics.keys()) == {"acc", "acc_norm"}
        assert result.num_examples == 2

    def test_ddp_sharding(
        self, encoder: tiktoken.Encoding, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        examples = [
            {"ctx": f"ex {i}", "label": 0, "endings": ["a", "b", "c", "d"]}
            for i in range(6)
        ]
        monkeypatch.setattr(hs, "iterate_examples", lambda split: iter(examples))
        # No process group exists; make all_reduce a no-op so we can read per-rank counts
        monkeypatch.setattr(hs.dist, "all_reduce", lambda t, op=None: t)

        model = _StubModel(vocab_size=encoder.n_vocab, winning_row=0)
        r0 = hs.evaluate_hellaswag(
            model, encoder=encoder, device="cpu",
            rank=0, world_size=2, dtype=torch.float32,
        )
        r1 = hs.evaluate_hellaswag(
            model, encoder=encoder, device="cpu",
            rank=1, world_size=2, dtype=torch.float32,
        )
        # Indices 0,2,4 rank 0 (3 examples); indices 1,3,5 rank 1 (3 examples)
        assert r0.num_examples == 3
        assert r1.num_examples == 3