"""Tests for skyai training loop helpers"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from skyai.checkpoint import save_checkpoint
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
from skyai.nn.model import GPT, GPTConfig
from skyai.training import loop
from skyai.training.optimizer import build_optimizer
from skyai.training.schedule import CosineSchedule

# ---- helpers --------------------------------------------------------------

def _tiny_gpt(vocab_size: int = 128) -> GPT:
    return GPT(GPTConfig(
        n_layer=2, n_head=2, n_embed=32,
        vocab_size=vocab_size, block_size=16,
    ))


def _make_shards(data_root: Path) -> None:
    """Write minimal train/val shards so DataConfig validators pass and DataLoader could load"""
    import numpy as np
    data_root.mkdir(parents=True, exist_ok=True)
    np.save(data_root / "train_000.npy", np.arange(1024, dtype=np.uint16))
    np.save(data_root / "val_000.npy", np.arange(1024, dtype=np.uint16))


def _tiny_cfg(tmp_path: Path) -> RunConfig:
    """A minimal RunConfig satisfying validators; used for _maybe_resume tests"""
    data_root = tmp_path / "data"
    _make_shards(data_root)
    return RunConfig(
        seed=42, dtype="float32", grad_clip=1.0, total_batch_size=16,
        model=ModelConfig(n_layer=2, n_head=2, n_embed=32, vocab_size=128, block_size=4),
        data=DataConfig(root=data_root, batch_size=4),
        optim=OptimConfig(weight_decay=0.0),
        schedule=ScheduleConfig(max_lr=1e-3, min_lr=1e-4, warmup_steps=1, max_steps=10),
        eval=EvalConfig(interval=5, val_steps=1, evals=[]),
        log=LogConfig(wandb=False),
        checkpoint=CheckpointConfig(
            dir=tmp_path / "ckpt", every_n_steps=5, keep_last_n=2,
            best_metric="val_loss", best_direction="min",
        ),
    )


class _StubLoader:
    """Minimal stand-in for DataLoader state_dict round-trip"""
    def __init__(self) -> None:
        self._state: dict = {"position": 0}
    def state_dict(self) -> dict: return dict(self._state)
    def load_state_dict(self, state: dict) -> None: self._state = dict(state)
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError  # not exercised by _maybe_resume tests
    def reset(self) -> None: pass


class _BatchLoader:
    """DataLoader stand-in that yields a fixed (x, y) for every next_batch() call.

    Same batch every time = overfitting target; loss must drop after a few steps.
    """
    def __init__(self, vocab_size: int, batch_size: int, block_size: int) -> None:
        g = torch.Generator().manual_seed(0)
        self.x = torch.randint(0, vocab_size, (batch_size, block_size), generator=g)
        self.y = torch.randint(0, vocab_size, (batch_size, block_size), generator=g)
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x, self.y
    def reset(self) -> None: pass
    def state_dict(self) -> dict: return {}
    def load_state_dict(self, state: dict) -> None: pass


# ---- tests ----------------------------------------------------------------

class TestDistInfo:
    def test_single_process(self) -> None:
        d = loop.DistInfo(rank=0, local_rank=0, world_size=1)
        assert d.is_ddp is False
        assert d.is_master is True

    def test_ddp_master(self) -> None:
        d = loop.DistInfo(rank=0, local_rank=0, world_size=4)
        assert d.is_ddp is True
        assert d.is_master is True

    def test_ddp_worker(self) -> None:
        d = loop.DistInfo(rank=2, local_rank=2, world_size=4)
        assert d.is_ddp is True
        assert d.is_master is False


class TestInitDistributed:
    def test_no_env_returns_single_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            monkeypatch.delenv(var, raising=False)
        assert loop._init_distributed() == loop.DistInfo(rank=0, local_rank=0, world_size=1)

    def test_with_env_initializes_and_returns_dist_info(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "8")
        calls: list[dict] = []
        monkeypatch.setattr(
            loop, "init_process_group",
            lambda backend: calls.append({"init": backend}),
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "set_device", lambda dev: calls.append({"set_device": dev}))

        d = loop._init_distributed()
        assert d == loop.DistInfo(rank=3, local_rank=1, world_size=8)
        assert calls == [{"init": "nccl"}, {"set_device": 1}]

    def test_raises_when_ddp_requested_without_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            loop._init_distributed()


class TestResolveDevice:
    def test_no_cuda_returns_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert loop._resolve_device(0) == "cpu"

    def test_cuda_uses_local_rank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert loop._resolve_device(2) == "cuda:2"


class TestComputeGradAccum:
    def test_happy_single_process(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path)
        cfg = cfg.model_copy(update={"total_batch_size": 64})  # 4 * 4 * 1 * 4 accum
        assert loop._compute_grad_accum(cfg, world_size=1) == 4

    def test_happy_with_world_size(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path)
        cfg = cfg.model_copy(update={"total_batch_size": 64})  # 4 * 4 * 2 = 32, accum=2
        assert loop._compute_grad_accum(cfg, world_size=2) == 2

    def test_raises_when_indivisible_by_world_size(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path)
        cfg = cfg.model_copy(update={"total_batch_size": 64})  # 4 * 4 * 5 = 80, 64 % 80 != 0
        with pytest.raises(ValueError, match="divisible"):
            loop._compute_grad_accum(cfg, world_size=5)


class TestSetSeeds:
    def test_same_seed_same_tensor(self) -> None:
        loop._set_seeds(1337)
        a = torch.randn(5)
        loop._set_seeds(1337)
        b = torch.randn(5)
        assert torch.equal(a, b)


class TestMaybeResume:
    def test_no_checkpoint_returns_zero(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path)
        model = _tiny_gpt()
        optim = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0, device_type="cpu")
        loader = _StubLoader()
        start_step, run_id = loop._maybe_resume(cfg, model, optim, loader)  # pyright: ignore
        assert start_step == 0
        assert run_id is None

    def test_restores_step_run_id_and_loader_state(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path)
        model = _tiny_gpt()
        optim = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0, device_type="cpu")

        saver_loader = _StubLoader()
        saver_loader.load_state_dict({"position": 1234})
        save_checkpoint(
            cfg.checkpoint.dir, step=7,
            model=model, optimizer=optim, data_loader=saver_loader,  # pyright: ignore
            config=cfg, metrics={"val_loss": 4.2},
            wandb_run_id="run-abc", rank=0, keep_last_n=2,
            best_metric="val_loss", best_direction="min",
        )

        fresh_loader = _StubLoader()
        start_step, run_id = loop._maybe_resume(cfg, model, optim, fresh_loader)  # pyright: ignore
        assert start_step == 8
        assert run_id == "run-abc"
        assert fresh_loader.state_dict() == {"position": 1234}


class TestRunTrainStep:
    def test_returns_finite_metrics(self) -> None:
        model = _tiny_gpt()
        optim = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0, device_type="cpu")
        loader = _BatchLoader(vocab_size=128, batch_size=2, block_size=8)
        sched = CosineSchedule(max_lr=1e-3, min_lr=1e-4, warmup_steps=1, max_steps=10)
        dist_info = loop.DistInfo(rank=0, local_rank=0, world_size=1)

        loss, grad_norm, lr = loop._run_train_step(
            model, loader, optim, sched, dist_info,  # pyright: ignore
            step=0, grad_accum_steps=2, grad_clip=1.0,
            device="cpu", device_type="cpu", dtype=torch.float32,
        )
        assert torch.isfinite(torch.tensor(loss))
        assert torch.isfinite(torch.tensor(grad_norm))
        assert lr == pytest.approx(sched.lr_for(0))

    def test_overfits_a_fixed_batch(self) -> None:
        model = _tiny_gpt()
        optim = build_optimizer(model, learning_rate=1e-2, weight_decay=0.0, device_type="cpu")
        loader = _BatchLoader(vocab_size=128, batch_size=2, block_size=8)
        sched = CosineSchedule(max_lr=1e-2, min_lr=1e-3, warmup_steps=1, max_steps=20)
        dist_info = loop.DistInfo(rank=0, local_rank=0, world_size=1)

        loss0, *_ = loop._run_train_step(
            model, loader, optim, sched, dist_info,  # pyright: ignore
            step=0, grad_accum_steps=1, grad_clip=1.0,
            device="cpu", device_type="cpu", dtype=torch.float32,
        )
        for step in range(1, 15):
            loop._run_train_step(
                model, loader, optim, sched, dist_info,  # pyright: ignore
                step=step, grad_accum_steps=1, grad_clip=1.0,
                device="cpu", device_type="cpu", dtype=torch.float32,
            )
        loss_after, *_ = loop._run_train_step(
            model, loader, optim, sched, dist_info,  # pyright: ignore
            step=15, grad_accum_steps=1, grad_clip=1.0,
            device="cpu", device_type="cpu", dtype=torch.float32,
        )
        assert loss_after < loss0 * 0.5  # generous: same batch should overfit hard


class TestRunValLoss:
    def test_returns_finite_loss(self) -> None:
        model = _tiny_gpt()
        loader = _BatchLoader(vocab_size=128, batch_size=2, block_size=8)
        dist_info = loop.DistInfo(rank=0, local_rank=0, world_size=1)
        loss = loop._run_val_loss(
            model, loader, dist_info,  # pyright: ignore
            val_steps=3, device="cpu", device_type="cpu", dtype=torch.float32,
        )
        assert torch.isfinite(torch.tensor(loss))
        assert loss > 0

    def test_leaves_model_in_eval_mode(self) -> None:
        model = _tiny_gpt()
        model.train()
        loader = _BatchLoader(vocab_size=128, batch_size=2, block_size=8)
        dist_info = loop.DistInfo(rank=0, local_rank=0, world_size=1)
        loop._run_val_loss(
            model, loader, dist_info,  # pyright: ignore
            val_steps=1, device="cpu", device_type="cpu", dtype=torch.float32,
        )
        assert model.training is False
