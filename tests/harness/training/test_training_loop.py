"""Tests for skyai training loop helpers"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from harness.checkpoint import save_checkpoint
from harness.config.schema import (
    CheckpointConfig,
    DataConfig,
    EvalConfig,
    LogConfig,
    ModelConfig,
    OptimConfig,
    ProfilingConfig,
    RecoveryConfig,
    RunConfig,
    ScheduleConfig,
)
from harness.training import loop
from harness.training.optimizer import build_optimizer
from harness.training.profiler import Profiler
from harness.training.schedule import CosineSchedule
from skyai.model import GPT, GPTConfig

# ---- helpers --------------------------------------------------------------


def _noop_profiler() -> Profiler:
    """Disabled profiler so call sites can pass the new positional arg without timing anything."""
    return Profiler(ProfilingConfig(enabled=False), device="cpu", rank=0)


def _default_recovery() -> RecoveryConfig:
    """Halt-mode recovery; matches production default. Suite has no NaNs so it never trips."""
    return RecoveryConfig()


def _tiny_gpt(vocab_size: int = 50304) -> GPT:
    return GPT(
        GPTConfig(
            n_layer=2,
            n_head=2,
            n_embed=32,
            vocab_size=vocab_size,
            block_size=16,
        )
    )


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
        seed=42,
        dtype="float32",
        grad_clip=1.0,
        total_batch_size=16,
        # vocab_size must satisfy the gpt2 tokenizer bound (>= 50257, <= 50257+1024)
        model=ModelConfig(n_layer=2, n_head=2, n_embed=32, vocab_size=50304, block_size=4),
        data=DataConfig(root=data_root, batch_size=4),
        optim=OptimConfig(weight_decay=0.0),
        schedule=ScheduleConfig(max_lr=1e-3, min_lr=1e-4, warmup_steps=1, max_steps=10),
        eval=EvalConfig(interval=5, val_steps=1, evals=[]),
        log=LogConfig(wandb=False),
        checkpoint=CheckpointConfig(
            dir=tmp_path / "ckpt",
            every_n_steps=5,
            keep_last_n=2,
            best_metric="val_loss",
            best_direction="min",
        ),
    )


class _StubLoader:
    """Minimal stand-in for DataLoader state_dict round-trip"""

    def __init__(self) -> None:
        self._state: dict = {"position": 0}

    def state_dict(self) -> dict:
        return dict(self._state)

    def load_state_dict(self, state: dict) -> None:
        self._state = dict(state)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError  # not exercised by _maybe_resume tests

    def reset(self) -> None:
        pass


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

    def reset(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class _NaNGradModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loss = self.weight * torch.tensor(float("nan"), device=x.device)
        return x.float(), loss


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
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "8")
        calls: list[dict] = []
        monkeypatch.setattr(
            loop,
            "init_process_group",
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


class TestFormatLrGroups:
    def test_single_group(self) -> None:
        optim = type("Opt", (), {"param_groups": [{"lr": 1e-3}]})()
        assert loop._format_lr_groups(optim) == "lr=1.0000e-03"

    def test_split_groups(self) -> None:
        optim = type(
            "Opt",
            (),
            {
                "param_groups": [
                    {"name": "embed", "lr": 0.3},
                    {"name": "lm_head", "lr": 0.008},
                    {"name": "muon_64x64", "optimizer_type": "muon", "lr": 0.02},
                ]
            },
        )()
        summary = loop._format_lr_groups(optim)
        assert "lr/max=3.0000e-01" in summary
        assert "lr/embed=3.0000e-01" in summary
        assert "lr/lm_head=8.0000e-03" in summary
        assert "lr/muon=2.0000e-02" in summary


class TestBuildModel:
    def test_threads_modern_arch_fields(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path)
        model_cfg = cfg.model.model_copy(
            update={
                "init_policy": "sky-ai",
                "n_kv_head": 1,
                "hidden_multiple": 8,
                "rope_theta": 10_000.0,
                "vocab_pad_multiple": 256,
                "tie_weights": True,
                "logit_softcap": None,
            }
        )
        cfg = cfg.model_copy(update={"model": model_cfg, "compile": False})

        forward_model, raw_model = loop._build_model(
            cfg,
            device="cpu",
            dist_info=loop.DistInfo(rank=0, local_rank=0, world_size=1),
        )

        assert forward_model is raw_model
        assert raw_model.config.init_policy == "sky-ai"
        assert raw_model.config.n_kv_head == 1
        assert raw_model.config.hidden_multiple == 8
        assert raw_model.config.rope_theta == 10_000.0
        assert raw_model.config.vocab_pad_multiple == 256
        assert raw_model.config.tie_weights is True
        assert raw_model.config.logit_softcap is None
        assert raw_model.config.vocab_size == 50257
        assert raw_model.config.vocab_size_padded == 50432
        assert raw_model.transformer.wte.weight.data_ptr() == raw_model.lm_head.weight.data_ptr()


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
            cfg.checkpoint.dir,
            step=7,
            model=model,
            optimizer=optim,
            data_loader=saver_loader,  # pyright: ignore
            config=cfg,
            metrics={"val_loss": 4.2},
            wandb_run_id="run-abc",
            rank=0,
            keep_last_n=2,
            best_metric="val_loss",
            best_direction="min",
        )

        fresh_loader = _StubLoader()
        start_step, run_id = loop._maybe_resume(cfg, model, optim, fresh_loader)  # pyright: ignore
        assert start_step == 8
        assert run_id == "run-abc"
        assert fresh_loader.state_dict() == {"position": 1234}

    def test_rejects_resume_when_batch_size_changed(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path)
        model = _tiny_gpt()
        optim = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0, device_type="cpu")
        save_checkpoint(
            cfg.checkpoint.dir,
            step=3,
            model=model,
            optimizer=optim,
            data_loader=_StubLoader(),  # pyright: ignore
            config=cfg,
            metrics={"val_loss": 4.2},
            wandb_run_id=None,
            rank=0,
            keep_last_n=2,
            best_metric="val_loss",
            best_direction="min",
        )
        # Bump batch_size and keep total_batch_size divisible by new microbatch
        new_data = cfg.data.model_copy(update={"batch_size": 2})
        new_cfg = cfg.model_copy(update={"data": new_data})
        with pytest.raises(RuntimeError, match="data.batch_size"):
            loop._maybe_resume(new_cfg, model, optim, _StubLoader())  # pyright: ignore

    def test_rejects_resume_when_block_size_changed(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path)
        model = _tiny_gpt()
        optim = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0, device_type="cpu")
        save_checkpoint(
            cfg.checkpoint.dir,
            step=3,
            model=model,
            optimizer=optim,
            data_loader=_StubLoader(),  # pyright: ignore
            config=cfg,
            metrics={"val_loss": 4.2},
            wandb_run_id=None,
            rank=0,
            keep_last_n=2,
            best_metric="val_loss",
            best_direction="min",
        )
        new_model = cfg.model.model_copy(update={"block_size": 8})
        new_cfg = cfg.model_copy(update={"model": new_model})
        with pytest.raises(RuntimeError, match="block_size"):
            loop._maybe_resume(new_cfg, _tiny_gpt(), optim, _StubLoader())  # pyright: ignore

    def test_rejects_resume_when_vocab_changed(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path)
        model = _tiny_gpt()
        optim = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0, device_type="cpu")
        save_checkpoint(
            cfg.checkpoint.dir,
            step=3,
            model=model,
            optimizer=optim,
            data_loader=_StubLoader(),  # pyright: ignore
            config=cfg,
            metrics={"val_loss": 4.2},
            wandb_run_id=None,
            rank=0,
            keep_last_n=2,
            best_metric="val_loss",
            best_direction="min",
        )
        # 50257 is tokenizer vocab without tensor core padding, differs from checkpoints 50304
        new_model = cfg.model.model_copy(update={"vocab_size": 50257})
        new_cfg = cfg.model_copy(update={"model": new_model})
        with pytest.raises(RuntimeError, match="vocab_size"):
            loop._maybe_resume(new_cfg, _tiny_gpt(), optim, _StubLoader())  # pyright: ignore

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("init_policy", "sky-ai"),
            ("n_kv_head", 1),
            ("hidden_multiple", 8),
            ("rope_theta", 10_000.0),
            ("vocab_pad_multiple", 256),
            ("tie_weights", True),
            ("logit_softcap", None),
        ],
    )
    def test_rejects_resume_when_modern_arch_field_changed(
        self,
        tmp_path: Path,
        field: str,
        value: object,
    ) -> None:
        cfg = _tiny_cfg(tmp_path)
        model = _tiny_gpt()
        optim = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0, device_type="cpu")
        save_checkpoint(
            cfg.checkpoint.dir,
            step=3,
            model=model,
            optimizer=optim,
            data_loader=_StubLoader(),  # pyright: ignore
            config=cfg,
            metrics={"val_loss": 4.2},
            wandb_run_id=None,
            rank=0,
            keep_last_n=2,
            best_metric="val_loss",
            best_direction="min",
        )

        new_model = cfg.model.model_copy(update={field: value})
        new_cfg = cfg.model_copy(update={"model": new_model})
        with pytest.raises(RuntimeError, match=field):
            loop._maybe_resume(new_cfg, model, optim, _StubLoader())  # pyright: ignore


class TestRunTrainStep:
    def test_returns_finite_metrics(self) -> None:
        model = _tiny_gpt()
        optim = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0, device_type="cpu")
        loader = _BatchLoader(vocab_size=128, batch_size=2, block_size=8)
        sched = CosineSchedule(max_lr=1e-3, min_lr=1e-4, warmup_steps=1, max_steps=10)
        dist_info = loop.DistInfo(rank=0, local_rank=0, world_size=1)

        loss, grad_norm, lr = loop._run_train_step(
            model,
            loader,
            optim,
            sched,
            dist_info,
            _noop_profiler(),
            _default_recovery(),  # pyright: ignore
            step=0,
            grad_accum_steps=2,
            grad_clip=1.0,
            device="cpu",
            device_type="cpu",
            dtype=torch.float32,
        )
        assert torch.isfinite(torch.tensor(loss))
        assert torch.isfinite(torch.tensor(grad_norm))
        assert lr == pytest.approx(sched.lr_for(0))

    def test_non_finite_grad_raises_project_error(self) -> None:
        model = _NaNGradModel()
        optim = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0, device_type="cpu")
        loader = _BatchLoader(vocab_size=128, batch_size=2, block_size=8)
        sched = CosineSchedule(max_lr=1e-3, min_lr=1e-4, warmup_steps=1, max_steps=10)
        dist_info = loop.DistInfo(rank=0, local_rank=0, world_size=1)

        with pytest.raises(loop.NonFiniteGradError, match="Non-finite gradient"):
            loop._run_train_step(
                model,
                loader,
                optim,
                sched,
                dist_info,
                _noop_profiler(),
                RecoveryConfig(nan_grad_action="halt"),  # pyright: ignore
                step=0,
                grad_accum_steps=1,
                grad_clip=1.0,
                device="cpu",
                device_type="cpu",
                dtype=torch.float32,
            )

    def test_overfits_a_fixed_batch(self) -> None:
        model = _tiny_gpt()
        optim = build_optimizer(model, learning_rate=1e-2, weight_decay=0.0, device_type="cpu")
        loader = _BatchLoader(vocab_size=128, batch_size=2, block_size=8)
        sched = CosineSchedule(max_lr=1e-2, min_lr=1e-3, warmup_steps=1, max_steps=20)
        dist_info = loop.DistInfo(rank=0, local_rank=0, world_size=1)

        loss0, *_ = loop._run_train_step(
            model,
            loader,
            optim,
            sched,
            dist_info,
            _noop_profiler(),
            _default_recovery(),  # pyright: ignore
            step=0,
            grad_accum_steps=1,
            grad_clip=1.0,
            device="cpu",
            device_type="cpu",
            dtype=torch.float32,
        )
        for step in range(1, 25):
            loop._run_train_step(
                model,
                loader,
                optim,
                sched,
                dist_info,
                _noop_profiler(),
                _default_recovery(),  # pyright: ignore
                step=step,
                grad_accum_steps=1,
                grad_clip=1.0,
                device="cpu",
                device_type="cpu",
                dtype=torch.float32,
            )
        loss_after, *_ = loop._run_train_step(
            model,
            loader,
            optim,
            sched,
            dist_info,
            _noop_profiler(),
            _default_recovery(),  # pyright: ignore
            step=25,
            grad_accum_steps=1,
            grad_clip=1.0,
            device="cpu",
            device_type="cpu",
            dtype=torch.float32,
        )
        assert loss_after < loss0 * 0.5  # generous: same batch should overfit hard


class TestRunValLoss:
    def test_returns_finite_loss(self) -> None:
        model = _tiny_gpt()
        loader = _BatchLoader(vocab_size=128, batch_size=2, block_size=8)
        dist_info = loop.DistInfo(rank=0, local_rank=0, world_size=1)
        loss = loop._run_val_loss(
            model,
            loader,
            dist_info,
            _noop_profiler(),  # pyright: ignore
            val_steps=3,
            device="cpu",
            device_type="cpu",
            dtype=torch.float32,
        )
        assert torch.isfinite(torch.tensor(loss))
        assert loss > 0

    def test_leaves_model_in_eval_mode(self) -> None:
        model = _tiny_gpt()
        model.train()
        loader = _BatchLoader(vocab_size=128, batch_size=2, block_size=8)
        dist_info = loop.DistInfo(rank=0, local_rank=0, world_size=1)
        loop._run_val_loss(
            model,
            loader,
            dist_info,
            _noop_profiler(),  # pyright: ignore
            val_steps=1,
            device="cpu",
            device_type="cpu",
            dtype=torch.float32,
        )
        assert model.training is False
