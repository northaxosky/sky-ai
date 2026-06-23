"""Tests for eval runner dispatch and the EVALS registry"""

from __future__ import annotations

from typing import Any, get_args, get_type_hints

import pytest
import tiktoken
import torch
from torch import nn

from harness.eval import EVALS, run_evals
from harness.eval.result import EvalResult


@pytest.fixture
def encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def stub_model() -> nn.Module:
    return nn.Linear(1, 1)


def _stub_eval(name: str) -> tuple[Any, dict[str, Any]]:
    """Build an eval-shaped callable that records the kwargs it was called with"""
    record: dict[str, Any] = {}

    def fn(
        model: nn.Module,
        *,
        encoder: tiktoken.Encoding,
        device: str | torch.device,
        rank: int,
        world_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> EvalResult:
        record["model"] = model
        record["encoder"] = encoder
        record["device"] = device
        record["rank"] = rank
        record["world_size"] = world_size
        record["dtype"] = dtype
        return EvalResult(name=name, metrics={"x": 1.0}, num_examples=1)

    return fn, record


class TestRunEvals:
    def test_dispatches_by_name(
        self, encoder: tiktoken.Encoding, stub_model: nn.Module, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fn, _ = _stub_eval("hellaswag")
        monkeypatch.setitem(EVALS, "hellaswag", fn)

        results = run_evals(
            ["hellaswag"],
            stub_model,
            encoder=encoder,
            device="cpu",
            rank=0,
            world_size=1,
        )
        assert "hellaswag" in results
        assert results["hellaswag"].name == "hellaswag"

    def test_preserves_input_order(
        self, encoder: tiktoken.Encoding, stub_model: nn.Module, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fn_h, _ = _stub_eval("hellaswag")
        fn_l, _ = _stub_eval("lambada")
        monkeypatch.setitem(EVALS, "hellaswag", fn_h)
        monkeypatch.setitem(EVALS, "lambada", fn_l)

        results = run_evals(
            ["lambada", "hellaswag"],
            stub_model,
            encoder=encoder,
            device="cpu",
            rank=0,
            world_size=1,
        )
        assert list(results.keys()) == ["lambada", "hellaswag"]

    def test_unknown_name_raises_keyerror(
        self, encoder: tiktoken.Encoding, stub_model: nn.Module
    ) -> None:
        with pytest.raises(KeyError, match="Unknown eval"):
            run_evals(
                ["nonexistent"],
                stub_model,
                encoder=encoder,
                device="cpu",
                rank=0,
                world_size=1,
            )

    def test_empty_list_returns_empty_dict(
        self, encoder: tiktoken.Encoding, stub_model: nn.Module
    ) -> None:
        results = run_evals(
            [],
            stub_model,
            encoder=encoder,
            device="cpu",
            rank=0,
            world_size=1,
        )
        assert results == {}

    def test_passes_through_kwargs(
        self, encoder: tiktoken.Encoding, stub_model: nn.Module, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fn, record = _stub_eval("hellaswag")
        monkeypatch.setitem(EVALS, "hellaswag", fn)

        run_evals(
            ["hellaswag"],
            stub_model,
            encoder=encoder,
            device="cuda:3",
            rank=2,
            world_size=4,
            dtype=torch.float16,
        )
        assert record["model"] is stub_model
        assert record["encoder"] is encoder
        assert record["device"] == "cuda:3"
        assert record["rank"] == 2
        assert record["world_size"] == 4
        assert record["dtype"] == torch.float16


class TestEvalsRegistry:
    def test_every_evals_key_matches_schema_literal(self) -> None:
        """Catches drift between EVALS dict and EvalConfig.evals Literal type"""
        from harness.config.schema import EvalConfig

        hints = get_type_hints(EvalConfig)
        literal_type = get_args(hints["evals"])[0]  # list[Literal["..."]] â Literal["..."]
        schema_names = set(get_args(literal_type))

        assert set(EVALS.keys()) == schema_names
