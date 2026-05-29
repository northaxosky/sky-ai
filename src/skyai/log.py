"""Stdlib logging setup for SkyAI training and eval"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from skyai.config.schema import LogConfig


_LOG_FORMAT = "%(asctime)s [%(levelname)s] [rank %(rank)d] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _RankFilter(logging.Filter):
    """Attach a 'rank' attribute to every LogRecord so the formatter can use it"""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self._rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self._rank
        return True # Dont drop, only annotate
    

def setup_logging(cfg: LogConfig, rank: int = 0, log_path: Path | None = None) -> None:
    """Configure the root logger for a SkyAI process"""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG) # pass everything, let handler do filtering

    for handler in list(root.handlers):
        if getattr(handler, "_skyai", False):
            handler.close()
            root.removeHandler(handler)

    rank_filter = _RankFilter(rank)
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(cfg.level if rank == 0 else "WARNING")
    console.setFormatter(formatter)
    console.addFilter(rank_filter)
    console._skyai = True # pyright: ignore
    root.addHandler(console)

    if rank == 0:
        cfg.dir.mkdir(parents=True, exist_ok=True)
        path = log_path if log_path is not None else cfg.dir / "run.log"
        
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(rank_filter)
        file_handler._skyai = True # pyright: ignore
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Thin wrapper so call sites improt from skyai.log, not stdlib logging"""
    return logging.getLogger(name)
