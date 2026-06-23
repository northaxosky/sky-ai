"""Tests for harness.sample.sample (multi-sample generation helper)"""

from __future__ import annotations

import pytest
import tiktoken
import torch

from harness.sample import sample
from skyai.model import GPT, GPTConfig


@pytest.fixture(scope="module")
def encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def tiny_model() -> GPT:
    return GPT(
        GPTConfig(
            n_layer=1,
            n_head=2,
            n_embed=32,
            vocab_size=50257,
            block_size=16,
        )
    )


class TestSample:
    def test_returns_n_samples_strings(self, tiny_model: GPT, encoder: tiktoken.Encoding) -> None:
        out = sample(tiny_model, encoder, "Hello", n_samples=3, max_length=8, device="cpu")
        assert len(out) == 3
        assert all(isinstance(s, str) for s in out)

    def test_each_completion_starts_with_prompt(
        self, tiny_model: GPT, encoder: tiktoken.Encoding
    ) -> None:
        out = sample(tiny_model, encoder, "Hello", n_samples=2, max_length=8, device="cpu")
        assert all(s.startswith("Hello") for s in out)

    def test_max_length_translates_to_max_new_tokens(
        self,
        tiny_model: GPT,
        encoder: tiktoken.Encoding,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, int] = {}

        def fake_generate(model, x, *, max_new_tokens, **kw):
            captured["max_new_tokens"] = max_new_tokens
            captured["prompt_len"] = x.size(1)
            pad = torch.zeros((x.size(0), max_new_tokens), dtype=torch.long, device=x.device)
            return torch.cat([x, pad], dim=1)

        import harness.sample as sm

        monkeypatch.setattr(sm, "generate", fake_generate)

        sample(tiny_model, encoder, "Hello", n_samples=1, max_length=6, device="cpu")
        assert captured["max_new_tokens"] == 6 - captured["prompt_len"]

    def test_top_k_none_does_not_crash(self, tiny_model: GPT, encoder: tiktoken.Encoding) -> None:
        out = sample(tiny_model, encoder, "Hi", n_samples=1, max_length=4, device="cpu", top_k=None)
        assert len(out) == 1

    def test_prompt_longer_than_max_length_still_generates_one_token(
        self,
        tiny_model: GPT,
        encoder: tiktoken.Encoding,
    ) -> None:
        # "Hello, I'm a language model," is ~8 tokens; max_length=4 means
        # max_new_tokens = max(1, 4 - 8) = 1, not a crash.
        out = sample(
            tiny_model,
            encoder,
            "Hello, I'm a language model,",
            n_samples=1,
            max_length=4,
            device="cpu",
        )
        assert len(out) == 1
        assert out[0].startswith("Hello, I'm a language model,")

    def test_raises_on_invalid_n_samples(self, tiny_model: GPT, encoder: tiktoken.Encoding) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            sample(tiny_model, encoder, "Hi", n_samples=0, max_length=4, device="cpu")

    def test_seeded_generator_makes_output_deterministic(
        self,
        tiny_model: GPT,
        encoder: tiktoken.Encoding,
    ) -> None:
        rng1 = torch.Generator(device="cpu").manual_seed(123)
        out1 = sample(
            tiny_model, encoder, "Hi", n_samples=2, max_length=8, device="cpu", generator=rng1
        )
        rng2 = torch.Generator(device="cpu").manual_seed(123)
        out2 = sample(
            tiny_model, encoder, "Hi", n_samples=2, max_length=8, device="cpu", generator=rng2
        )
        assert out1 == out2
