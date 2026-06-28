"""Tests for harness.wandb_logger"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from harness.config.schema import LogConfig
from harness.wandb_logger import WandbLogger


@pytest.fixture
def mock_wandb(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the wandb module with a MagicMock and provide a fake key so the auth guard passes"""
    mock = MagicMock()
    mock.util.generate_id.return_value = "auto-id-123"
    monkeypatch.setattr("harness.wandb_logger.wandb", mock)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    return mock


def _enabled_cfg() -> LogConfig:
    return LogConfig(wandb=True, wandb_project="test-project")


class TestDisabled:
    def test_cfg_off_is_noop(self, mock_wandb: MagicMock) -> None:
        cfg = LogConfig(wandb=False)
        wb = WandbLogger(cfg=cfg, rank=0)
        wb.log_metrics({"loss": 1.0}, step=0)
        wb.update_config({"x": 1})
        wb.finish()

        assert wb.run_id is None
        mock_wandb.init.assert_not_called()
        mock_wandb.log.assert_not_called()
        mock_wandb.finish.assert_not_called()

    def test_rank_nonzero_is_noop(self, mock_wandb: MagicMock) -> None:
        wb = WandbLogger(cfg=_enabled_cfg(), rank=1)
        wb.log_metrics({"loss": 1.0}, step=0)
        wb.finish()

        assert wb.run_id is None
        mock_wandb.init.assert_not_called()


class TestEnabled:
    def test_init_generates_run_id_when_no_resume(self, mock_wandb: MagicMock) -> None:
        wb = WandbLogger(cfg=_enabled_cfg(), rank=0)
        mock_wandb.util.generate_id.assert_called_once()
        mock_wandb.init.assert_called_once_with(
            project="test-project",
            entity=None,
            id="auto-id-123",
            resume="allow",
            config=None,
        )
        assert wb.run_id == "auto-id-123"

    def test_init_uses_resume_id_when_provided(self, mock_wandb: MagicMock) -> None:
        wb = WandbLogger(cfg=_enabled_cfg(), rank=0, resume_id="my-old-id")
        mock_wandb.util.generate_id.assert_not_called()
        mock_wandb.init.assert_called_once()
        kwargs = mock_wandb.init.call_args.kwargs

        assert kwargs["id"] == "my-old-id"
        assert kwargs["resume"] == "must"
        assert wb.run_id == "my-old-id"

    def test_init_forwards_config_dict(self, mock_wandb: MagicMock) -> None:
        run_cfg: dict[str, Any] = {"seed": 42, "n_layer": 12}
        WandbLogger(cfg=_enabled_cfg(), rank=0, config=run_cfg)
        kwargs = mock_wandb.init.call_args.kwargs
        assert kwargs["config"] == run_cfg

    def test_resume_mode_must_when_resume_id_given(self, mock_wandb: MagicMock) -> None:
        """Resume should be 'must' so wandb errors out if the prior run is gone,
        instead of silently splitting metrics across a new run."""
        WandbLogger(cfg=_enabled_cfg(), rank=0, resume_id="prior-run")
        assert mock_wandb.init.call_args.kwargs["resume"] == "must"

    def test_resume_mode_allow_when_no_resume_id(self, mock_wandb: MagicMock) -> None:
        """Fresh runs use 'allow' so the first init doesn't fail on missing prior."""
        WandbLogger(cfg=_enabled_cfg(), rank=0)
        assert mock_wandb.init.call_args.kwargs["resume"] == "allow"

    def test_log_metrics_forwards_to_wandb(self, mock_wandb: MagicMock) -> None:
        wb = WandbLogger(cfg=_enabled_cfg(), rank=0)
        wb.log_metrics({"train/loss": 1.23, "lr": 1e-4}, step=10)
        mock_wandb.log.assert_called_once_with({"train/loss": 1.23, "lr": 1e-4}, step=10)

    def test_update_config_forwards(self, mock_wandb: MagicMock) -> None:
        wb = WandbLogger(cfg=_enabled_cfg(), rank=0)
        wb.update_config({"world_size": 8})
        mock_wandb.config.update.assert_called_once_with({"world_size": 8}, allow_val_change=True)

    def test_finish_forward(self, mock_wandb: MagicMock) -> None:
        wb = WandbLogger(cfg=_enabled_cfg(), rank=0)
        wb.finish()
        mock_wandb.finish.assert_called_once()

    def test_finish_is_idempotent(self, mock_wandb: MagicMock) -> None:
        wb = WandbLogger(cfg=_enabled_cfg(), rank=0)
        wb.finish()
        wb.finish()
        wb.finish()
        mock_wandb.finish.assert_called_once()


class TestAuthGuard:
    def test_raises_when_enabled_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Enabled + no WANDB_API_KEY must fail fast, not hang on wandb.init's login prompt."""
        monkeypatch.setattr("harness.wandb_logger.wandb", MagicMock())
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
            WandbLogger(cfg=_enabled_cfg(), rank=0)

    def test_offline_mode_needs_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WANDB_MODE=offline authenticates without a key, so init still proceeds."""
        mock = MagicMock()
        monkeypatch.setattr("harness.wandb_logger.wandb", mock)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_MODE", "offline")
        WandbLogger(cfg=_enabled_cfg(), rank=0)
        mock.init.assert_called_once()


class TestContextManager:
    def test_exit_calls_finish(self, mock_wandb: MagicMock) -> None:
        with WandbLogger(cfg=_enabled_cfg(), rank=0) as wb:
            wb.log_metrics({"x": 1.0}, step=0)
        mock_wandb.finish.assert_called_once()
