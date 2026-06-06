"""Tests for skyai doctor command + check functions"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from typer.testing import CliRunner

from skyai.cli import doctor as doctor_module
from skyai.cli.doctor import (
    _check_bf16,
    _check_checkpoint_dir,
    _check_cuda,
    _check_data_shards,
    _check_ddp_env,
    _check_git,
    _check_gpu,
    _check_python,
    _check_torch,
    _check_visible_devices,
    _check_wandb,
    _check_wandb_auth,
    _check_world_size_divisibility,
    _estimate_model_params,
    run_doctor,
)
from skyai.cli.main import app
from skyai.config.loader import load_config

runner = CliRunner()


class TestPython:
    def test_returns_ok_on_312_plus(self):
        status, msg = _check_python()
        assert status == "OK"
        assert msg.startswith("Python ")


class TestTorch:
    def test_returns_ok_when_importable(self):
        status, _ = _check_torch()
        assert status == "OK"


class TestCuda:
    def test_no_cuda_warns(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        status, _ = _check_cuda()
        assert status == "WARN"

    def test_with_cuda_ok(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        status, msg = _check_cuda()
        assert status == "OK"
        assert "2 devices" in msg


class TestGpu:
    def test_no_cuda_warns(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        status, _ = _check_gpu()
        assert status == "WARN"


class TestBF16:
    def test_no_cuda_warns(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        status, _ = _check_bf16()
        assert status == "WARN"

    def test_unsupported_warns(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
        status, _ = _check_bf16()
        assert status == "WARN"

    def test_supported_ok(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        status, _ = _check_bf16()
        assert status == "OK"


class TestVisibleDevices:
    def test_unset_ok(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        status, _ = _check_visible_devices()
        assert status == "OK"

    def test_mismatch_warns(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        status, msg = _check_visible_devices()
        assert status == "WARN"
        assert "0,1,2" in msg

    def test_match_ok(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        status, _ = _check_visible_devices()
        assert status == "OK"


class TestDDPEnv:
    def test_no_vars_ok(self, monkeypatch):
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            monkeypatch.delenv(k, raising=False)
        status, _ = _check_ddp_env()
        assert status == "OK"

    def test_partial_fails(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        status, _ = _check_ddp_env()
        assert status == "FAIL"

    def test_inconsistent_fails(self, monkeypatch):
        monkeypatch.setenv("RANK", "5")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        status, _ = _check_ddp_env()
        assert status == "FAIL"

    def test_non_integer_fails(self, monkeypatch):
        monkeypatch.setenv("RANK", "zero")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        status, _ = _check_ddp_env()
        assert status == "FAIL"

    def test_all_consistent_ok(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        status, _ = _check_ddp_env()
        assert status == "OK"

    def test_world_gt_1_without_master_addr_fails(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.delenv("MASTER_ADDR", raising=False)
        monkeypatch.setenv("MASTER_PORT", "29500")
        status, msg = _check_ddp_env()
        assert status == "FAIL"
        assert "MASTER_ADDR" in msg

    def test_world_gt_1_without_master_port_fails(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.delenv("MASTER_PORT", raising=False)
        status, msg = _check_ddp_env()
        assert status == "FAIL"
        assert "MASTER_PORT" in msg

    def test_master_port_non_integer_fails(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", "not-a-port")
        status, msg = _check_ddp_env()
        assert status == "FAIL"
        assert "MASTER_PORT" in msg

    def test_world_gt_1_with_master_addr_port_ok(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", "29500")
        status, _ = _check_ddp_env()
        assert status == "OK"


class TestGit:
    def test_in_repo_ok(self):
        status, _ = _check_git()
        assert status == "OK"

    def test_outside_repo_warns(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        status, _ = _check_git()
        assert status == "WARN"


class TestWandb:
    def test_no_key_warns(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        status, _ = _check_wandb()
        assert status == "WARN"

    def test_offline_ok(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_MODE", "offline")
        status, _ = _check_wandb()
        assert status == "OK"

    def test_key_set_ok(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake-key")
        status, _ = _check_wandb()
        assert status == "OK"


def _minimal_cfg(tmp_path: Path, data_root: Path) -> Path:
    cfg = {
        "total_batch_size": 64,
        "model": {"n_layer": 2, "n_head": 4, "n_embed": 64, "vocab_size": 50304, "block_size": 16},
        "data": {"root": str(data_root), "batch_size": 2},
        "optim": {"weight_decay": 0.1},
        "schedule": {"max_lr": 6e-4, "min_lr": 6e-5, "warmup_steps": 1, "max_steps": 10},
        "eval": {"interval": 5},
        "checkpoint": {"dir": str(tmp_path / "ckpts")},
        "log": {"dir": str(tmp_path / "logs")},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


class TestDataShards:
    def test_missing_root_fails(self, tmp_path):
        cfg_path = _minimal_cfg(tmp_path, tmp_path / "nonexistent")
        cfg = load_config(cfg_path, overrides=[])
        status, _ = _check_data_shards(cfg)
        assert status == "FAIL"

    def test_no_matching_files_fails(self, tmp_path):
        data_root = tmp_path / "data"
        data_root.mkdir()
        (data_root / "unrelated.bin").touch()
        cfg_path = _minimal_cfg(tmp_path, data_root)
        cfg = load_config(cfg_path, overrides=[])
        status, _ = _check_data_shards(cfg)
        assert status == "FAIL"

    def test_only_train_no_val_fails(self, tmp_path):
        data_root = tmp_path / "data"
        data_root.mkdir()
        (data_root / "train_0.bin").touch()
        cfg_path = _minimal_cfg(tmp_path, data_root)
        cfg = load_config(cfg_path, overrides=[])
        status, _ = _check_data_shards(cfg)
        assert status == "FAIL"

    def test_train_and_val_present_ok(self, tmp_path):
        data_root = tmp_path / "data"
        data_root.mkdir()
        (data_root / "train_0.bin").touch()
        (data_root / "val_0.bin").touch()
        cfg_path = _minimal_cfg(tmp_path, data_root)
        cfg = load_config(cfg_path, overrides=[])
        status, msg = _check_data_shards(cfg)
        assert status == "OK"
        assert "1 train" in msg and "1 val" in msg


class TestCheckpointDir:
    def test_model_param_estimate_matches_modern_shapes(self, tmp_path):
        cfg_path = _minimal_cfg(tmp_path, tmp_path / "data")
        cfg = load_config(cfg_path, overrides=[])

        assert _estimate_model_params(cfg.model) == 6_569_984

    def test_creates_and_passes(self, tmp_path):
        cfg_path = _minimal_cfg(tmp_path, tmp_path / "data")
        cfg = load_config(cfg_path, overrides=[])
        status, _ = _check_checkpoint_dir(cfg)
        assert status in {"OK", "WARN"}
        assert (tmp_path / "ckpts").exists()

    def test_fails_when_free_disk_below_model_estimate(self, tmp_path, monkeypatch):
        import shutil as _shutil

        cfg_path = _minimal_cfg(tmp_path, tmp_path / "data")
        cfg = load_config(cfg_path, overrides=[])
        # Pretend almost no free disk; the model is small but >> 1 byte
        FakeDU = type("DU", (), {})
        fake = FakeDU()
        fake.total = 100  # type: ignore[attr-defined]
        fake.used = 99  # type: ignore[attr-defined]
        fake.free = 1  # type: ignore[attr-defined]
        monkeypatch.setattr(_shutil, "disk_usage", lambda p: fake)
        status, msg = _check_checkpoint_dir(cfg)
        assert status == "FAIL"
        assert "need" in msg.lower()


class TestWorldSizeDivisibility:
    def test_unset_world_skipped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        cfg_path = _minimal_cfg(tmp_path, tmp_path / "data")
        cfg = load_config(cfg_path, overrides=[])
        status, _ = _check_world_size_divisibility(cfg)
        assert status == "OK"

    def test_divisible_ok(self, tmp_path, monkeypatch):
        # _minimal_cfg: total_batch_size=64, batch_size=2, block_size=16 -> B*T=32
        # world=2 -> 32*2=64, total/64 = 1
        monkeypatch.setenv("WORLD_SIZE", "2")
        cfg_path = _minimal_cfg(tmp_path, tmp_path / "data")
        cfg = load_config(cfg_path, overrides=[])
        status, msg = _check_world_size_divisibility(cfg)
        assert status == "OK"
        assert "grad_accum" in msg

    def test_indivisible_fails(self, tmp_path, monkeypatch):
        # world=3 -> 32*3=96; 64 % 96 != 0
        monkeypatch.setenv("WORLD_SIZE", "3")
        cfg_path = _minimal_cfg(tmp_path, tmp_path / "data")
        cfg = load_config(cfg_path, overrides=[])
        status, msg = _check_world_size_divisibility(cfg)
        assert status == "FAIL"
        assert "divisible" in msg

    def test_non_integer_fails(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "eight")
        cfg_path = _minimal_cfg(tmp_path, tmp_path / "data")
        cfg = load_config(cfg_path, overrides=[])
        status, _ = _check_world_size_divisibility(cfg)
        assert status == "FAIL"


class TestWandbAuth:
    def test_skipped_when_wandb_disabled(self, tmp_path):
        cfg_path = _minimal_cfg(tmp_path, tmp_path / "data")
        cfg = load_config(cfg_path, overrides=[])
        status, _ = _check_wandb_auth(cfg)
        assert status == "OK"

    def test_offline_mode_ok(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WANDB_MODE", "offline")
        cfg_path = _minimal_cfg(tmp_path, tmp_path / "data")
        cfg = load_config(cfg_path, overrides=["log.wandb=true", "log.wandb_project=p"])
        status, _ = _check_wandb_auth(cfg)
        assert status == "OK"

    def test_missing_key_fails_when_wandb_on(self, tmp_path, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        cfg_path = _minimal_cfg(tmp_path, tmp_path / "data")
        cfg = load_config(cfg_path, overrides=["log.wandb=true", "log.wandb_project=p"])
        status, msg = _check_wandb_auth(cfg)
        assert status == "FAIL"
        assert "WANDB_API_KEY" in msg

    def test_invalid_key_fails(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "bogus")
        monkeypatch.delenv("WANDB_MODE", raising=False)
        cfg_path = _minimal_cfg(tmp_path, tmp_path / "data")
        cfg = load_config(cfg_path, overrides=["log.wandb=true", "log.wandb_project=p"])
        import wandb

        class _BoomApi:
            def __init__(self, *args, **kwargs) -> None:
                pass

            @property
            def viewer(self):
                raise RuntimeError("401 Unauthorized")

        monkeypatch.setattr(wandb, "Api", _BoomApi)
        status, msg = _check_wandb_auth(cfg)
        assert status == "FAIL"
        assert "auth probe failed" in msg

    def test_valid_key_ok(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "valid")
        monkeypatch.delenv("WANDB_MODE", raising=False)
        cfg_path = _minimal_cfg(tmp_path, tmp_path / "data")
        cfg = load_config(cfg_path, overrides=["log.wandb=true", "log.wandb_project=p"])
        import wandb

        class _FakeViewer:
            username = "tester"

        class _FakeApi:
            def __init__(self, *args, **kwargs) -> None:
                pass

            viewer = _FakeViewer()

        monkeypatch.setattr(wandb, "Api", _FakeApi)
        status, msg = _check_wandb_auth(cfg)
        assert status == "OK"
        assert "tester" in msg


class TestRunDoctor:
    def test_no_config_returns_int(self, capsys):
        rc = run_doctor(config_path=None)
        assert rc in (0, 1)
        out = capsys.readouterr().out
        assert "Summary:" in out

    def test_invalid_config_returns_one(self, tmp_path, capsys):
        bad = tmp_path / "bad.yaml"
        bad.write_text("not: [a: valid")  # malformed yaml
        rc = run_doctor(config_path=bad)
        assert rc == 1
        assert "FAIL" in capsys.readouterr().out

    def test_check_raises_caught_as_fail(self, monkeypatch, capsys):
        def boom() -> tuple[str, str]:
            raise RuntimeError("simulated breakage")

        monkeypatch.setattr(doctor_module, "_ENV_CHECKS", [("boom", boom)])
        rc = run_doctor(config_path=None)
        assert rc == 1
        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "simulated breakage" in out


class TestCLI:
    def test_doctor_command_runs(self):
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code in (0, 1)
        assert "Summary:" in result.output

    def test_doctor_with_invalid_config_exits_one(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("not: [a: valid")
        result = runner.invoke(app, ["doctor", "--config", str(bad)])
        assert result.exit_code == 1

    def test_doctor_with_valid_config_no_shards_exits_one(self, tmp_path):
        cfg_path = _minimal_cfg(tmp_path, tmp_path / "data_does_not_exist")
        result = runner.invoke(app, ["doctor", "--config", str(cfg_path)])
        assert result.exit_code == 1
        assert "FAIL" in result.output
