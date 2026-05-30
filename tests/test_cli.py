"""Tests for the skyai CLI surface"""

from __future__ import annotations

import logging
from importlib.metadata import version as _pkg_version
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from skyai.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Save/restore root logger handlers around each test so state doesnt leak"""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    root.handlers.clear
    yield

    for handler in list(root.handlers):
        try:
            handler.close()
        except Exception:
            pass
    root.handlers.clear()
    root.handlers.extend(saved_handlers)
    root.setLevel(saved_level)

def _minimal_yaml(tmp_path: Path) -> Path:
    """Write a minimal but valid RunCOnfig YAML and return its path"""
    cfg = {
        "total_batch_size": 64,
            "model": {
                "n_layer": 2,
                "n_head": 4,
                "n_embed": 64,
                "vocab_size": 100,
                "block_size": 16,
            },
            "data": {
                "root": str(tmp_path / "shards"),
                "batch_size": 2,
            },
            "optim": {"weight_decay": 0.1},
            "schedule": {
                "max_lr": 1e-3,
                "min_lr": 1e-4,
                "warmup_steps": 10,
                "max_steps": 100,
            },
            "eval": {"interval": 50},
            "log": {"dir": str(tmp_path / "logs")},
            "checkpoint": {"dir": str(tmp_path / "ckpts")},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


class TestVersion:
    def test_version_prints_package_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert _pkg_version("skyai") in result.output


class TestHelp:
    def test_root_help_lists_all_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for name in ("version", "train", "eval", "sample", "doctor"):
            assert name in result.output    

    def test_train_help_mentions_config_and_resume(self) -> None:
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--resume" in result.output

    def test_eval_help_mentions_config_and_checkpoint(self) -> None:
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--checkpoint" in result.output

    def test_sample_help_mentions_checkpoint_and_prompt(self) -> None:
        result = runner.invoke(app, ["sample", "--help"])
        assert result.exit_code == 0
        assert "--checkpoint" in result.output
        assert "--prompt" in result.output

    def test_sample_help_mentions_new_flags(self) -> None:
        result = runner.invoke(app, ["sample", "--help"])
        assert result.exit_code == 0
        for flag in ("--num-samples", "--max-new-tokens", "--temperature",
                     "--top-k", "--seed", "--device"):
            assert flag in result.output, f"missing {flag} in sample --help"


class TestSampleEndToEnd:
    """Sample against a real (tiny) checkpoint built in-test"""

    def _build_checkpoint(self, tmp_path: Path) -> Path:
        import yaml

        from skyai.checkpoint import save_checkpoint
        from skyai.config.loader import load_config
        from skyai.nn.model import GPT, GPTConfig
        from skyai.training.optimizer import build_optimizer

        # write a tiny but valid YAML config (vocab matches gpt2 for the CLI's encoder)
        cfg_dict = {
            "total_batch_size": 256,
            "model": {
                "n_layer": 2, "n_head": 2, "n_embed": 32,
                "vocab_size": 50257, "block_size": 64,
                "tokenizer": "gpt2",
            },
            "data": {"root": str(tmp_path / "shards"), "batch_size": 4},
            "optim": {"weight_decay": 0.0},
            "schedule": {"max_lr": 1e-3, "min_lr": 1e-4,
                         "warmup_steps": 1, "max_steps": 10},
            "eval": {"interval": 5},
            "log": {"dir": str(tmp_path / "logs")},
            "checkpoint": {"dir": str(tmp_path / "ckpts")},
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_dict))
        cfg = load_config(cfg_path)

        model = GPT(GPTConfig(
            n_layer=cfg.model.n_layer, n_head=cfg.model.n_head,
            n_embed=cfg.model.n_embed, vocab_size=cfg.model.vocab_size,
            block_size=cfg.model.block_size,
        ))
        optim = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0, device_type="cpu")

        class _StubLoader:
            def state_dict(self) -> dict: return {}
            def load_state_dict(self, state: dict) -> None: pass

        ckpt_path = save_checkpoint(
            cfg.checkpoint.dir, step=0,
            model=model, optimizer=optim, data_loader=_StubLoader(),  # pyright: ignore
            config=cfg, metrics={"val_loss": 10.0},
        )
        assert ckpt_path is not None
        return ckpt_path

    def test_sample_runs_and_prints_prompt_prefix(self, tmp_path: Path) -> None:
        ckpt = self._build_checkpoint(tmp_path)
        result = runner.invoke(app, [
            "sample", "--checkpoint", str(ckpt),
            "--prompt", "Hello",
            "--max-new-tokens", "4",
            "--device", "cpu",
            "--seed", "1",
        ])
        assert result.exit_code == 0, result.output
        assert "Hello" in result.output

    def test_sample_with_num_samples_emits_separators(self, tmp_path: Path) -> None:
        ckpt = self._build_checkpoint(tmp_path)
        result = runner.invoke(app, [
            "sample", "--checkpoint", str(ckpt),
            "--prompt", "Hi",
            "--num-samples", "3",
            "--max-new-tokens", "2",
            "--device", "cpu",
            "--seed", "1",
        ])
        assert result.exit_code == 0, result.output
        assert "--- sample 1/3 ---" in result.output
        assert "--- sample 2/3 ---" in result.output
        assert "--- sample 3/3 ---" in result.output

    def test_sample_top_k_zero_disables_top_k(self, tmp_path: Path) -> None:
        ckpt = self._build_checkpoint(tmp_path)
        result = runner.invoke(app, [
            "sample", "--checkpoint", str(ckpt),
            "--prompt", "Hi",
            "--max-new-tokens", "2",
            "--top-k", "0",
            "--device", "cpu",
            "--seed", "1",
        ])
        assert result.exit_code == 0, result.output


class TestErrors:
    def test_train_missing_config_errors(self) -> None:
        result = runner.invoke(app, ["train"])
        assert result.exit_code != 0

    def test_eval_missing_checkpoint_errors(self, tmp_path: Path) -> None:
        cfg = _minimal_yaml(tmp_path)
        result = runner.invoke(app, ["eval", "--config", str(cfg), "--checkpoint", str(tmp_path / "missing.pt")])
        assert result.exit_code != 0

    def test_train_nonexistent_config_errors(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["train", "--config", str(tmp_path / "missing.yaml")])
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)

    def test_train_invalid_override_errors(self, tmp_path: Path) -> None:
        cfg = _minimal_yaml(tmp_path)
        result = runner.invoke(app, ["train", "--config", str(cfg), "--override", "no-equals-here"])
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
