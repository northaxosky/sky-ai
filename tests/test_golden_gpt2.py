"""Golden short-run for the real GPT-2 model"""

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

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_gpt2_short.json"
REGEN = os.environ.get("REGENERATE_GOLDEN") == "1"
IN_CI = os.environ.get("CI") == "true"


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _make_golden_shards(data_root: Path, n_tokens: int = 2048) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    np.save(data_root / "train_000.npy", rng.integers(0, 32, size=n_tokens, dtype=np.uint16))
    np.save(data_root / "val_000.npy", rng.integers(0, 32, size=n_tokens, dtype=np.uint16))


def _golden_cfg(tmp_path: Path) -> RunConfig:
    data_root = tmp_path / "data"
    _make_golden_shards(data_root)
    return RunConfig(
        seed=42,
        dtype="float32",
        compile=False,
        grad_clip=1.0,
        total_batch_size=64,
        model=ModelConfig(
            family="gpt2", n_layer=2, n_head=2, n_embed=16, vocab_size=50257, block_size=16
        ),
        data=DataConfig(root=data_root, batch_size=2),
        optim=OptimConfig(weight_decay=0.0),
        schedule=ScheduleConfig(max_lr=3e-3, min_lr=1e-4, warmup_steps=1, max_steps=50),
        eval=EvalConfig(
            interval=10,
            val_steps=1,
            evals=[],
            sample_prompt="Hello",
            sample_n=2,
            sample_max_length=12,
        ),
        log=LogConfig(dir=tmp_path / "logs"),
        checkpoint=CheckpointConfig(dir=tmp_path / "ckpts", every_n_steps=10, keep_last_n=1),
    )


def _run_golden(tmp_path: Path) -> dict:
    metrics = loop.train(_golden_cfg(tmp_path))
    assert metrics is not None, "train() returned None on master rank"
    return metrics


def _compare(actual: dict, expected: dict, atol: float = 1e-5) -> None:
    assert len(actual["step_losses"]) == len(expected["step_losses"])
    for i, (a, e) in enumerate(zip(actual["step_losses"], expected["step_losses"], strict=True)):
        assert a == pytest.approx(e, abs=atol), f"step {i}: {a} != {e}"
    assert actual["final_val_loss"] == pytest.approx(expected["final_val_loss"], abs=atol)
    assert actual["sample_text"] == expected["sample_text"], "sample_text drift"
    ac, ec = actual["param_checksum"], expected["param_checksum"]
    assert ac["n_params"] == ec["n_params"]
    assert ac["sum"] == pytest.approx(ec["sum"], abs=1e-3)
    assert ac["norm"] == pytest.approx(ec["norm"], abs=1e-3)


@pytest.mark.slow
@pytest.mark.skipif(REGEN, reason="Skipped during fixture regeneration")
@pytest.mark.skipif(
    IN_CI,
    reason="Golden training numerics are host-specific (float reduction order varies across CPUs/BLAS); run locally",
)
def test_gpt2_golden_matches_fixture(tmp_path: Path) -> None:
    assert FIXTURE_PATH.exists(), (
        "Generate: REGENERATE_GOLDEN=1 uv run pytest tests/test_golden_gpt2.py"
    )
    with FIXTURE_PATH.open() as f:
        expected = json.load(f)
    _compare(_run_golden(tmp_path), expected)


@pytest.mark.skipif(not REGEN, reason="Set REGENERATE_GOLDEN=1 to regenerate")
def test_regenerate_gpt2_golden(tmp_path: Path) -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w") as f:
        json.dump(_run_golden(tmp_path), f, indent=2)
