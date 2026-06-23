"""Tests for skyai logger: rank aware setup logging and helpers"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Literal

import pytest

from harness.config.schema import LogConfig
from harness.log import _RankFilter, get_logger, setup_logging


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Save/restore root logger handlers around each test so state doesnt leak"""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    root.handlers.clear()
    yield

    for handler in list(root.handlers):
        with contextlib.suppress(Exception):
            handler.close()
    root.handlers.clear()
    root.handlers.extend(saved_handlers)
    root.setLevel(saved_level)


def _make_cfg(
    tmp_path: Path, level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
) -> LogConfig:
    return LogConfig(dir=tmp_path, level=level, wandb=False, wandb_project=None)


def _skyai_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if getattr(h, "_skyai", False)]


class TestSetupLogging:
    def test_rank0_attaches_console_and_file(self, tmp_path: Path) -> None:
        setup_logging(_make_cfg(tmp_path), rank=0)
        types = [type(h).__name__ for h in _skyai_handlers()]
        assert "StreamHandler" in types
        assert "FileHandler" in types

    def test_rank_nonzero_skips_file(self, tmp_path: Path) -> None:
        setup_logging(_make_cfg(tmp_path), rank=3)
        types = [type(h).__name__ for h in _skyai_handlers()]
        assert "StreamHandler" in types
        assert "FileHandler" not in types

    def test_rank_nonzero_console_at_warning(self, tmp_path: Path) -> None:
        setup_logging(_make_cfg(tmp_path, level="DEBUG"), rank=1)
        # FileHandler subclasses StreamHandler; exclude it explicitly.
        console = next(
            h
            for h in _skyai_handlers()
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        )
        assert console.level == logging.WARNING

    def test_rank0_console_respects_cfg_level(self, tmp_path: Path) -> None:
        setup_logging(_make_cfg(tmp_path, level="DEBUG"), rank=0)
        console = next(
            h
            for h in _skyai_handlers()
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        )
        assert console.level == logging.DEBUG

    def test_rank0_file_is_debug(self, tmp_path: Path) -> None:
        # File should capture DEBUG even if console is INFO.
        setup_logging(_make_cfg(tmp_path, level="INFO"), rank=0)
        file_handler = next(h for h in _skyai_handlers() if isinstance(h, logging.FileHandler))
        assert file_handler.level == logging.DEBUG

    def test_idempotent(self, tmp_path: Path) -> None:
        setup_logging(_make_cfg(tmp_path), rank=0)
        first = len(_skyai_handlers())
        setup_logging(_make_cfg(tmp_path), rank=0)
        second = len(_skyai_handlers())
        assert first == second

    def test_idempotent_leaves_foreign_handlers_alone(self, tmp_path: Path) -> None:
        foreign = logging.StreamHandler()  # no _skyai marker
        logging.getLogger().addHandler(foreign)
        setup_logging(_make_cfg(tmp_path), rank=0)
        assert foreign in logging.getLogger().handlers

    def test_file_written_to_disk(self, tmp_path: Path) -> None:
        setup_logging(_make_cfg(tmp_path), rank=0)
        get_logger("skyai.test").info("hello world")
        for h in _skyai_handlers():
            h.flush()
        log_file = tmp_path / "run.log"
        assert log_file.exists()
        contents = log_file.read_text()
        assert "hello world" in contents
        assert "[rank 0]" in contents
        assert "skyai.test" in contents

    def test_file_records_rank_for_higher_levels(self, tmp_path: Path) -> None:
        # Rank > 0 doesn't get a file handler, but if a future caller passes
        # log_path explicitly we still want rank in the format. Verify on rank 0.
        setup_logging(_make_cfg(tmp_path), rank=0)
        get_logger("skyai.test").warning("careful")
        for h in _skyai_handlers():
            h.flush()
        contents = (tmp_path / "run.log").read_text()
        assert "WARNING" in contents
        assert "[rank 0]" in contents

    def test_custom_log_path(self, tmp_path: Path) -> None:
        custom = tmp_path / "subdir" / "custom.log"
        custom.parent.mkdir(parents=True, exist_ok=True)
        setup_logging(_make_cfg(tmp_path), rank=0, log_path=custom)
        get_logger("skyai.test").info("custom path")
        for h in _skyai_handlers():
            h.flush()
        assert custom.exists()
        assert "custom path" in custom.read_text()

    def test_log_dir_created_if_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "does" / "not" / "exist"
        cfg = LogConfig(dir=target, level="INFO", wandb=False, wandb_project=None)
        setup_logging(cfg, rank=0)
        assert target.exists()
        assert target.is_dir()


class TestRankFilter:
    def test_attaches_rank_to_record(self) -> None:
        filt = _RankFilter(rank=2)
        record = logging.LogRecord("x", logging.INFO, "f", 1, "msg", None, None)
        assert filt.filter(record) is True
        assert record.rank == 2  # pyright: ignore

    def test_never_drops_records(self) -> None:
        filt = _RankFilter(rank=0)
        for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
            record = logging.LogRecord("x", level, "f", 1, "msg", None, None)
            assert filt.filter(record) is True


class TestGetLogger:
    def test_returns_named_logger(self) -> None:
        assert get_logger("harness.training").name == "harness.training"

    def test_returns_same_logger_for_same_name(self) -> None:
        # Stdlib guarantees this; we're documenting the guarantee.
        assert get_logger("skyai.x") is get_logger("skyai.x")
