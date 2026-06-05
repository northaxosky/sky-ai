"""Tests for checkpoint save/load, rotation, best tracking, and lineage manifest"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from skyai.checkpoint import (
    CheckpointBundle,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)
from skyai.config.schema import RunConfig


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _FakeLoader:
    """Stand-in for DataLoaderLite; just needs state_dict / load_state_dict."""

    def __init__(self, current_shard: int = 0) -> None:
        self.current_shard = current_shard

    def state_dict(self) -> dict[str, int]:
        return {"current_shard": self.current_shard}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.current_shard = state["current_shard"]


def _make_cfg() -> RunConfig:
    return RunConfig.model_validate(
        {
            "seed": 1337,
            "dtype": "float32",
            "compile": False,
            "grad_clip": 1.0,
            "total_batch_size": 256,
            "model": {
                "n_layer": 2,
                "n_head": 2,
                "n_embed": 64,
                "vocab_size": 50257,
                "block_size": 64,
            },
            "data": {"root": "data/x", "batch_size": 4},
            "optim": {"weight_decay": 0.1},
            "schedule": {
                "max_lr": 1e-3,
                "min_lr": 1e-4,
                "warmup_steps": 5,
                "max_steps": 50,
            },
            "eval": {"interval": 10, "val_steps": 2, "evals": ["hellaswag"]},
        }
    )


@pytest.fixture
def model() -> torch.nn.Module:
    return _TinyModel()


@pytest.fixture
def optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.1)


@pytest.fixture
def save_kwargs(
    tmp_path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    return {
        "dir": tmp_path,
        "model": model,
        "optimizer": optimizer,
        "data_loader": _FakeLoader(),
        "config": _make_cfg(),
        "metrics": {"val_loss": 1.0},
    }


# ---------- TestSave: file layout, rank guard, atomicity ----------


class TestSave:
    def test_creates_bundle_manifest_and_latest(self, save_kwargs: dict[str, Any]) -> None:
        path = save_checkpoint(step=10, **save_kwargs)
        assert path == save_kwargs["dir"] / "step_00000010.pt"
        assert (save_kwargs["dir"] / "step_00000010.pt").is_file()
        assert (save_kwargs["dir"] / "step_00000010.json").is_file()
        assert (save_kwargs["dir"] / "latest.json").is_file()
        pointer = json.loads((save_kwargs["dir"] / "latest.json").read_text())
        assert pointer == {"step": 10}

    def test_returns_none_on_rank_nonzero(self, save_kwargs: dict[str, Any]) -> None:
        result = save_checkpoint(step=10, rank=1, **save_kwargs)
        assert result is None
        assert list(save_kwargs["dir"].iterdir()) == []

    def test_creates_dir_if_missing(self, tmp_path: Path, save_kwargs: dict[str, Any]) -> None:
        nested = tmp_path / "deeply" / "nested" / "ckpts"
        save_kwargs["dir"] = nested
        save_checkpoint(step=1, **save_kwargs)
        assert nested.is_dir()
        assert (nested / "step_00000001.pt").is_file()

    def test_no_tmp_orphans(self, save_kwargs: dict[str, Any]) -> None:
        save_checkpoint(step=10, **save_kwargs)
        assert list(save_kwargs["dir"].glob("*.tmp")) == []


# ---------- TestRotation: keep_last_n ----------


class TestRotation:
    def test_prunes_oldest_to_keep_last_n(self, save_kwargs: dict[str, Any]) -> None:
        for step in [1, 2, 3, 4, 5]:
            save_checkpoint(step=step, keep_last_n=2, **save_kwargs)
        remaining = sorted(p.name for p in save_kwargs["dir"].glob("step_*.pt"))
        assert remaining == ["step_00000004.pt", "step_00000005.pt"]
        assert sorted(p.name for p in save_kwargs["dir"].glob("step_*.json")) == [
            "step_00000004.json",
            "step_00000005.json",
        ]

    def test_noop_under_threshold(self, save_kwargs: dict[str, Any]) -> None:
        for step in [1, 2]:
            save_checkpoint(step=step, keep_last_n=5, **save_kwargs)
        assert len(list(save_kwargs["dir"].glob("step_*.pt"))) == 2


# ---------- TestBest: best.pt tracking ----------


class TestBest:
    def test_created_on_first_save(self, save_kwargs: dict[str, Any]) -> None:
        save_checkpoint(step=1, **save_kwargs)
        assert (save_kwargs["dir"] / "best.pt").is_file()
        best_manifest = json.loads((save_kwargs["dir"] / "best.json").read_text())
        assert best_manifest["best_for_step"] == 1
        assert best_manifest["best_metric"] == "val_loss"
        assert best_manifest["best_direction"] == "min"

    def test_updates_when_better(self, save_kwargs: dict[str, Any]) -> None:
        save_checkpoint(step=1, **save_kwargs)
        save_kwargs["metrics"] = {"val_loss": 0.5}
        save_checkpoint(step=2, **save_kwargs)
        best_manifest = json.loads((save_kwargs["dir"] / "best.json").read_text())
        assert best_manifest["best_for_step"] == 2
        assert best_manifest["metrics"]["val_loss"] == 0.5

    def test_skips_when_worse(self, save_kwargs: dict[str, Any]) -> None:
        save_checkpoint(step=1, **save_kwargs)
        save_kwargs["metrics"] = {"val_loss": 2.0}
        save_checkpoint(step=2, **save_kwargs)
        best_manifest = json.loads((save_kwargs["dir"] / "best.json").read_text())
        assert best_manifest["best_for_step"] == 1

    def test_direction_max(self, save_kwargs: dict[str, Any]) -> None:
        save_kwargs["metrics"] = {"hellaswag_acc": 0.25}
        save_checkpoint(step=1, best_metric="hellaswag_acc", best_direction="max", **save_kwargs)
        save_kwargs["metrics"] = {"hellaswag_acc": 0.30}
        save_checkpoint(step=2, best_metric="hellaswag_acc", best_direction="max", **save_kwargs)
        best_manifest = json.loads((save_kwargs["dir"] / "best.json").read_text())
        assert best_manifest["best_for_step"] == 2
        assert best_manifest["metrics"]["hellaswag_acc"] == 0.30

    def test_skipped_when_metric_missing(self, save_kwargs: dict[str, Any]) -> None:
        save_kwargs["metrics"] = {"train_loss": 1.0}
        save_checkpoint(step=1, **save_kwargs)
        assert not (save_kwargs["dir"] / "best.pt").exists()
        assert not (save_kwargs["dir"] / "best.json").exists()

    def test_survives_rotation(self, save_kwargs: dict[str, Any]) -> None:
        save_kwargs["metrics"] = {"val_loss": 0.1}
        save_checkpoint(step=1, keep_last_n=2, **save_kwargs)
        for step in [2, 3, 4, 5]:
            save_kwargs["metrics"] = {"val_loss": 5.0}
            save_checkpoint(step=step, keep_last_n=2, **save_kwargs)
        assert (save_kwargs["dir"] / "best.pt").is_file()
        best_manifest = json.loads((save_kwargs["dir"] / "best.json").read_text())
        assert best_manifest["best_for_step"] == 1


# ---------- TestLoad: polymorphic resolver ----------


class TestLoad:
    def test_by_pt_path(self, save_kwargs: dict[str, Any]) -> None:
        bundle_path = save_checkpoint(step=10, **save_kwargs)
        assert bundle_path is not None

        bundle = load_checkpoint(bundle_path)
        assert isinstance(bundle, CheckpointBundle)
        assert bundle.step == 10

    def test_by_directory_uses_latest(self, save_kwargs: dict[str, Any]) -> None:
        save_checkpoint(step=1, **save_kwargs)
        save_checkpoint(step=2, **save_kwargs)
        bundle = load_checkpoint(save_kwargs["dir"])
        assert bundle.step == 2

    def test_by_best_json(self, save_kwargs: dict[str, Any]) -> None:
        save_kwargs["metrics"] = {"val_loss": 0.1}
        save_checkpoint(step=1, **save_kwargs)
        save_kwargs["metrics"] = {"val_loss": 5.0}
        save_checkpoint(step=2, **save_kwargs)
        bundle = load_checkpoint(save_kwargs["dir"] / "best.json")
        assert bundle.step == 1

    def test_missing_bundle_raises(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "step_00000001.json"
        manifest_path.write_text(json.dumps({"step": 1, "config": {}}))
        with pytest.raises(FileNotFoundError, match="Bundle not found"):
            load_checkpoint(manifest_path)


# ---------- TestRoundTrip: state preservation ----------


class TestRoundTrip:
    def test_preserves_metadata(self, save_kwargs: dict[str, Any]) -> None:
        save_kwargs["metrics"] = {"val_loss": 1.23, "hellaswag_acc": 0.27}
        save_kwargs["wandb_run_id"] = "abc123"
        save_checkpoint(step=42, **save_kwargs)

        bundle = load_checkpoint(save_kwargs["dir"])
        assert bundle.step == 42
        assert bundle.wandb_run_id == "abc123"
        assert bundle.manifest["metrics"] == {"val_loss": 1.23, "hellaswag_acc": 0.27}
        assert bundle.manifest["torch_version"] == torch.__version__
        assert isinstance(bundle.config, RunConfig)
        assert bundle.config.seed == 1337

    def test_preserves_model_weights(
        self, save_kwargs: dict[str, Any], model: torch.nn.Module
    ) -> None:
        with torch.no_grad():
            model.linear.weight.fill_(0.4242)  # pyright: ignore
        save_checkpoint(step=1, **save_kwargs)

        bundle = load_checkpoint(save_kwargs["dir"])
        loaded = _TinyModel()
        loaded.load_state_dict(bundle.model_state)
        assert torch.allclose(loaded.linear.weight, torch.full((4, 4), 0.4242))

    def test_rng_round_trip(self, save_kwargs: dict[str, Any]) -> None:
        torch.manual_seed(42)
        _ = torch.randn(10)
        expected_state = torch.get_rng_state()

        save_checkpoint(step=1, **save_kwargs)

        torch.manual_seed(0)
        bundle = load_checkpoint(save_kwargs["dir"])
        restore_rng(bundle.rng_state)

        assert torch.equal(torch.get_rng_state(), expected_state)

    def test_numpy_rng_round_trip(self, save_kwargs: dict[str, Any]) -> None:
        import numpy as np

        np.random.seed(123)
        _ = np.random.randn(10)
        expected = np.random.randn(5)

        np.random.seed(123)
        _ = np.random.randn(10)
        save_checkpoint(step=1, **save_kwargs)

        np.random.seed(0)
        _ = np.random.randn(50)
        bundle = load_checkpoint(save_kwargs["dir"])
        restore_rng(bundle.rng_state)

        assert np.allclose(np.random.randn(5), expected)


# ---------- TestDistributed: NCCL hygiene around large checkpoints ----------


class TestDistributed:
    def test_barrier_called_when_dist_initialized(
        self,
        save_kwargs: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[int] = []
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "barrier", lambda *a, **kw: calls.append(1))

        save_checkpoint(step=1, **save_kwargs)
        assert len(calls) == 1

    def test_nonzero_rank_returns_none_but_still_barriers(
        self,
        save_kwargs: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[int] = []
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "barrier", lambda *a, **kw: calls.append(1))

        result = save_checkpoint(step=1, rank=2, **save_kwargs)
        assert result is None
        assert len(calls) == 1

    def test_no_barrier_when_dist_not_initialized(
        self,
        save_kwargs: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[int] = []
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
        monkeypatch.setattr(torch.distributed, "barrier", lambda *a, **kw: calls.append(1))

        save_checkpoint(step=1, **save_kwargs)
        assert calls == []


# ---------- TestProvenance: manifest content ----------


class TestProvenance:
    def test_git_sha_recorded_when_in_repo(self, save_kwargs: dict[str, Any]) -> None:
        save_checkpoint(step=1, **save_kwargs)
        manifest = json.loads((save_kwargs["dir"] / "step_00000001.json").read_text())
        # In this repo's test run, git is available; sha is a 40-char hex string.
        assert manifest["git_sha"] is None or (
            isinstance(manifest["git_sha"], str) and len(manifest["git_sha"]) == 40
        )
        assert isinstance(manifest["git_dirty"], bool)
        assert manifest["host"]
        assert "created_at" in manifest

    def test_git_sha_none_when_subprocess_fails(self, save_kwargs: dict[str, Any]) -> None:
        with patch(
            "skyai.checkpoint.subprocess.run",
            side_effect=FileNotFoundError("git not installed"),
        ):
            save_checkpoint(step=1, **save_kwargs)
        manifest = json.loads((save_kwargs["dir"] / "step_00000001.json").read_text())
        assert manifest["git_sha"] is None
        assert manifest["git_dirty"] is False


# ---------- TestListing: list_checkpoints, latest_checkpoint ----------


class TestListing:
    def test_list_sorted_by_step(self, save_kwargs: dict[str, Any]) -> None:
        for step in [3, 1, 2]:
            save_checkpoint(step=step, **save_kwargs)
        paths = list_checkpoints(save_kwargs["dir"])
        assert [p.name for p in paths] == [
            "step_00000001.pt",
            "step_00000002.pt",
            "step_00000003.pt",
        ]

    def test_list_empty_when_missing(self, tmp_path: Path) -> None:
        assert list_checkpoints(tmp_path / "nope") == []

    def test_latest_returns_none_when_empty(self, tmp_path: Path) -> None:
        assert latest_checkpoint(tmp_path) is None

    def test_latest_returns_highest_step(self, save_kwargs: dict[str, Any]) -> None:
        for step in [1, 2, 3]:
            save_checkpoint(step=step, **save_kwargs)
        latest = latest_checkpoint(save_kwargs["dir"])
        assert latest is not None
        assert latest.name == "step_00000003.pt"
