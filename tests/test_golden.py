"""Golden fixture short run test"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

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
from tests.golden_support import compare_metrics, make_golden_shards, run_golden

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_harness_short.json"
REGEN = os.environ.get("REGENERATE_GOLDEN") == "1"
IN_CI = os.environ.get("CI") == "true"

pytestmark = pytest.mark.usefixtures("force_cpu")


def _golden_cfg(tmp_path: Path) -> RunConfig:
    """Canonical golden config: tiny, deterministic, CPU-only, no evals"""
    data_root = tmp_path / "data"
    make_golden_shards(data_root)
    return RunConfig(
        seed=42,
        dtype="float32",
        compile=False,
        grad_clip=1.0,
        total_batch_size=64,
        model=ModelConfig(
            n_layer=2,
            n_head=2,
            n_embd=16,
            vocab_size=50257,
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


@pytest.mark.slow
@pytest.mark.skipif(REGEN, reason="Skipped during fixture regeneration")
@pytest.mark.skipif(
    IN_CI,
    reason="Golden training numerics are host-specific (float reduction order varies across CPUs/BLAS); run locally",
)
def test_golden_matches_fixture(tmp_path: Path) -> None:
    """Harness output must match the committed golden fixture"""
    assert FIXTURE_PATH.exists(), (
        f"Fixture missing at {FIXTURE_PATH}. Generate with: "
        f"REGENERATE_GOLDEN=1 uv run pytest tests/test_golden.py"
    )
    with FIXTURE_PATH.open() as f:
        expected = json.load(f)
    compare_metrics(run_golden(_golden_cfg(tmp_path)), expected)


@pytest.mark.skipif(not REGEN, reason="Set REGENERATE_GOLDEN=1 to regenerate")
def test_regenerate_golden_fixture(tmp_path: Path) -> None:
    """Write a fresh fixture from a successful run; gated on REGENERATE_GOLDEN=1"""
    metrics = run_golden(_golden_cfg(tmp_path))
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
