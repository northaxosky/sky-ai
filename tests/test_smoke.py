"""End to end smoke test for the training harness"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from skyai.config.schema import (
    CheckpointConfig,
    DataConfig,
    EvalConfig,
    LogConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    ScheduleConfig,
)
from skyai.training import loop


def _make_smoke_shards(data_root: Path, vocab_size: int, n_tokens: int = 4096) -> None:
    """Write train_000.npy + val_000.npy with random token ids in [0, vocab_size)"""
    data_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    train = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    val = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    np.save(data_root / "train_000.npy", train)
    np.save(data_root / "val_000.npy", val)


def _smoke_cfg(tmp_path: Path, *, max_steps: int = 5, evals: list[str] | None = None) -> RunConfig:
    data_root = tmp_path / "data"
    # vocab_size must accept gpt2 tokenizer ids (used by the loop's hardcoded
    # sampling block); block_size must fit the sample prompt + new tokens.
    vocab_size = 50257
    _make_smoke_shards(data_root, vocab_size=vocab_size)
    return RunConfig(
        seed=42,
        dtype="float32",
        compile=False,
        grad_clip=1.0,
        total_batch_size=256,
        model=ModelConfig(n_layer=2, n_head=2, n_embed=32, vocab_size=vocab_size, block_size=64),
        data=DataConfig(root=data_root, batch_size=4),
        optim=OptimConfig(weight_decay=0.0),
        schedule=ScheduleConfig(max_lr=1e-3, min_lr=1e-4, warmup_steps=1, max_steps=max_steps),
        eval=EvalConfig(interval=2, val_steps=1, evals=evals if evals is not None else []),  # pyright: ignore
        log=LogConfig(dir=tmp_path / "logs", wandb=False),
        checkpoint=CheckpointConfig(
            dir=tmp_path / "ckpt",
            every_n_steps=2,
            keep_last_n=3,
            best_metric="val_loss",
            best_direction="min",
        ),
    )


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin smoke to CPU regardless of CUDA availability on the host"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


class TestEndToEndSmoke:
    def test_train_completes_and_writes_checkpoints(self, tmp_path: Path) -> None:
        cfg = _smoke_cfg(tmp_path, max_steps=5)
        loop.train(cfg, resume=False)

        ckpts = sorted(cfg.checkpoint.dir.glob("step_*.pt"))
        steps = [int(p.stem.split("_")[1]) for p in ckpts]
        assert 2 in steps, f"expected checkpoint at step 2, got {steps}"
        assert 4 in steps, f"expected checkpoint at step 4 (last_step), got {steps}"

        latest = json.loads((cfg.checkpoint.dir / "latest.json").read_text())
        assert latest["step"] == 4
        assert (cfg.checkpoint.dir / "best.json").is_file()

    def test_resume_continues_past_original_max_steps(self, tmp_path: Path) -> None:
        cfg = _smoke_cfg(tmp_path, max_steps=3)
        loop.train(cfg, resume=False)
        first_latest = json.loads((cfg.checkpoint.dir / "latest.json").read_text())["step"]
        assert first_latest == 2  # last step under max_steps=3 (step 2 is last_step)

        cfg2 = cfg.model_copy(
            update={
                "schedule": cfg.schedule.model_copy(update={"max_steps": 6}),
            }
        )
        loop.train(cfg2, resume=True)
        second_latest = json.loads((cfg.checkpoint.dir / "latest.json").read_text())["step"]
        assert second_latest == 5
        assert second_latest > first_latest

    def test_eval_block_dispatches_to_run_evals(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[dict] = []

        def fake_run_evals(names, model, *, encoder, device, rank, world_size, dtype):
            calls.append(
                {
                    "names": list(names),
                    "rank": rank,
                    "world_size": world_size,
                }
            )
            return {}

        monkeypatch.setattr(loop, "run_evals", fake_run_evals)

        cfg = _smoke_cfg(tmp_path, max_steps=3, evals=["hellaswag"])
        loop.train(cfg, resume=False)

        # eval fires at step 0 and step 2 (interval=2, max_steps=3 -> last_step=2)
        assert len(calls) == 2
        assert all(c["names"] == ["hellaswag"] for c in calls)
        assert all(c["rank"] == 0 and c["world_size"] == 1 for c in calls)
