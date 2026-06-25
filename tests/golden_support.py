"""Shared scaffolding for the golden short-run tests (one per model family).

The two golden modules — `test_golden.py` (skyai) and `test_golden_gpt2.py` (gpt2) —
keep their own family-specific `_golden_cfg`. Everything else is identical: writing
the deterministic shards, running the harness, and comparing metrics against the
committed fixture. That lives here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness.config.schema import RunConfig
from harness.training import loop


def make_golden_shards(data_root: Path, n_tokens: int = 2048) -> None:
    """Write deterministic uint16 train/val shards from a fixed RNG seed."""
    data_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    np.save(data_root / "train_000.npy", rng.integers(0, 32, size=n_tokens, dtype=np.uint16))
    np.save(data_root / "val_000.npy", rng.integers(0, 32, size=n_tokens, dtype=np.uint16))


def run_golden(cfg: RunConfig) -> dict:
    """Train through the harness and return the metrics dict (asserts master rank)."""
    metrics = loop.train(cfg)
    assert metrics is not None, "train() returned None on master rank"
    return metrics


def compare_metrics(actual: dict, expected: dict, atol: float = 1e-5) -> None:
    """Assert actual matches expected within tolerance; localize the mismatch."""
    assert len(actual["step_losses"]) == len(expected["step_losses"]), (
        f"step_losses length mismatch: "
        f"{len(actual['step_losses'])} vs {len(expected['step_losses'])}"
    )
    for i, (a, e) in enumerate(zip(actual["step_losses"], expected["step_losses"], strict=True)):
        assert a == pytest.approx(e, abs=atol), f"step {i} loss: {a} != {e} (atol={atol})"

    if expected["final_val_loss"] is None:
        assert actual["final_val_loss"] is None
    else:
        assert actual["final_val_loss"] == pytest.approx(expected["final_val_loss"], abs=atol), (
            f"final_val_loss: {actual['final_val_loss']} != {expected['final_val_loss']}"
        )

    assert actual["sample_text"] == expected["sample_text"], (
        f"sample_text mismatch:\n"
        f"  actual:   {actual['sample_text']}\n"
        f"  expected: {expected['sample_text']}"
    )

    ac, ec = actual["param_checksum"], expected["param_checksum"]
    assert ac["n_params"] == ec["n_params"], f"n_params: {ac['n_params']} != {ec['n_params']}"
    assert ac["sum"] == pytest.approx(ec["sum"], abs=1e-3), f"param sum: {ac['sum']} != {ec['sum']}"
    assert ac["norm"] == pytest.approx(ec["norm"], abs=1e-3), (
        f"param norm: {ac['norm']} != {ec['norm']}"
    )
