"""Golden fixture short run test"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from harness.config.schema import (
    CheckpointConfig,
    DataConfig,
    EvalConfig,
    LogConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    ScheduleConfig,
)
from harness.training import loop

pytestmark = pytest.mark.slow
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_harness_short.json"
REGEN = os.environ.get("REGENERATE_GOLDEN") == "1"


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    """Pin device to CPU regardless of host so the golden numerics are stable"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _make_golden_shards(data_root: Path, vocab_size: int, n_tokens: int = 2048) -> None:
    """Write deterministic train/val shards from a fixed RNG seed"""
    data_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    train = rng.integers(0, 32, size=n_tokens, dtype=np.uint16)
    val = rng.integers(0, 32, size=n_tokens, dtype=np.uint16)
    np.save(data_root / "train_000.npy", train)
    np.save(data_root / "val_000.npy", val)


def _golden_cfg(tmp_path: Path) -> RunConfig:
    """Canonical golden config: tiny, deterministic, CPU-only, no evals"""
    data_root = tmp_path / "data"
    vocab_size = 50257
    _make_golden_shards(data_root, vocab_size=vocab_size)
    return RunConfig(
        seed=42,
        dtype="float32",
        compile=False,
        grad_clip=1.0,
        total_batch_size=64,
        model=ModelConfig(
            n_layer=2,
            n_head=2,
            n_embed=16,
            vocab_size=vocab_size,
            block_size=16,
        ),
        data=DataConfig(root=data_root, batch_size=2),
        optim=OptimConfig(weight_decay=0.0),
        schedule=ScheduleConfig(
            max_lr=3e-3,
            min_lr=1e-4,
            warmup_steps=1,
            max_steps=50,
        ),
        eval=EvalConfig(
            interval=10,
            val_steps=1,
            evals=[],
            sample_prompt="Hello",
            sample_n=2,
            sample_max_length=12,
        ),
        log=LogConfig(dir=tmp_path / "logs"),
        checkpoint=CheckpointConfig(
            dir=tmp_path / "ckpts",
            every_n_steps=10,
            keep_last_n=1,
        ),
    )


def _run_golden(tmp_path: Path) -> dict:
    cfg = _golden_cfg(tmp_path)
    metrics = loop.train(cfg)
    assert metrics is not None, "train() returned None on master rank"
    return metrics


def _compare_metrics(actual: dict, expected: dict, atol: float = 1e-5) -> None:
    """Assert actual matches expected within tolerance; raise with localized message"""
    assert len(actual["step_losses"]) == len(expected["step_losses"]), (
        f"step_losses length mismatch: "
        f"{len(actual['step_losses'])} vs {len(expected['step_losses'])}"
    )
    for i, (a, e) in enumerate(zip(actual["step_losses"], expected["step_losses"], strict=True)):
        assert a == pytest.approx(e, abs=atol), f"step {i} loss: {a} != {e} (atol={atol})"

    if expected["final_val_loss"] is None:
        assert actual["final_val_loss"] is None
    else:
        assert actual["final_val_loss"] == pytest.approx(
            expected["final_val_loss"],
            abs=atol,
        ), f"final_val_loss: {actual['final_val_loss']} != {expected['final_val_loss']}"

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


@pytest.mark.skipif(REGEN, reason="Skipped during fixture regeneration")
def test_golden_matches_fixture(tmp_path: Path) -> None:
    """Harness output must match the committed golden fixture"""
    assert FIXTURE_PATH.exists(), (
        f"Fixture missing at {FIXTURE_PATH}. Generate with: "
        f"REGENERATE_GOLDEN=1 uv run pytest tests/test_golden.py"
    )
    with FIXTURE_PATH.open() as f:
        expected = json.load(f)
    actual = _run_golden(tmp_path)
    _compare_metrics(actual, expected)


@pytest.mark.skipif(not REGEN, reason="Set REGENERATE_GOLDEN=1 to regenerate")
def test_regenerate_golden_fixture(tmp_path: Path) -> None:
    """Write a fresh fixture from a successful run; gated on REGENERATE_GOLDEN=1"""
    metrics = _run_golden(tmp_path)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "generated_by": "tests/test_golden.py::test_regenerate_golden_fixture",
            "note": ("Regenerate with: REGENERATE_GOLDEN=1 uv run pytest tests/test_golden.py"),
        },
        **metrics,
    }
    with FIXTURE_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
