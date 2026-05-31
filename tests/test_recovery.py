"""Tests for skyai.training.recovery"""

from __future__ import annotations

import torch
from torch import nn

from skyai.training.recovery import (
    NonFiniteGradError,
    detect_non_finite_grad,
    is_oom_error,
)


class TestDetectNonFiniteGrad:
    def _model_with_grads(self, grad_value: float | None) -> nn.Module:
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.Linear(8, 2),
        )
        x = torch.randn(3, 4)
        y = model(x).sum()
        y.backward()
        if grad_value is not None:
            # Poison the second layer's bias gradient
            model[1].bias.grad.fill_(grad_value) # pyright: ignore
        return model

    def test_all_finite_returns_none(self) -> None:
        model = self._model_with_grads(grad_value=None)
        assert detect_non_finite_grad(model) is None

    def test_nan_detected(self) -> None:
        model = self._model_with_grads(grad_value=float("nan"))
        bad = detect_non_finite_grad(model)
        assert bad is not None
        assert "bias" in bad

    def test_inf_detected(self) -> None:
        model = self._model_with_grads(grad_value=float("inf"))
        bad = detect_non_finite_grad(model)
        assert bad is not None
        assert "bias" in bad

    def test_neg_inf_detected(self) -> None:
        model = self._model_with_grads(grad_value=float("-inf"))
        assert detect_non_finite_grad(model) is not None

    def test_none_grad_skipped(self) -> None:
        """A param with grad=None (never seen backward) shouldn't trip detection."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        # No backward called -> all grads are None
        assert detect_non_finite_grad(model) is None

    def test_returns_first_offender_deterministically(self) -> None:
        """When multiple params are bad, the first iteration order wins."""
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        x = torch.randn(1, 2)
        model(x).sum().backward()
        # Poison both layers
        model[0].weight.grad.fill_(float("nan")) # pyright: ignore
        model[1].weight.grad.fill_(float("nan")) # pyright: ignore
        bad = detect_non_finite_grad(model)
        # named_parameters iterates in registration order; 0.weight comes first
        assert bad == "0.weight"


class TestIsOomError:
    def test_modern_torch_oom_class(self) -> None:
        err = torch.cuda.OutOfMemoryError("simulated")
        assert is_oom_error(err) is True

    def test_legacy_runtime_error_with_oom_message(self) -> None:
        err = RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB.")
        assert is_oom_error(err) is True

    def test_legacy_message_case_insensitive(self) -> None:
        assert is_oom_error(RuntimeError("OUT OF MEMORY")) is True

    def test_unrelated_runtime_error_not_oom(self) -> None:
        assert is_oom_error(RuntimeError("something else broke")) is False

    def test_unrelated_exception_not_oom(self) -> None:
        assert is_oom_error(ValueError("nope")) is False


class TestNonFiniteGradError:
    def test_is_runtime_error_subclass(self) -> None:
        """So users can catch it with `except RuntimeError` if they wish."""
        assert issubclass(NonFiniteGradError, RuntimeError)

    def test_carries_message(self) -> None:
        err = NonFiniteGradError("step 42: foo.bar")
        assert "step 42" in str(err)