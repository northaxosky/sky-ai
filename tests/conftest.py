"""Pytest Configuration: shared fixtures and environment setup"""

import contextlib
import logging
import os
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def synthetic_shards(tmp_path: Path) -> Path:
    """Create 2 train shards + 1 val shard with predictable token values"""
    data_root = tmp_path / "data"
    data_root.mkdir()

    layout = [("train", 2), ("val", 1)]
    for split, n_shards in layout:
        for i in range(n_shards):
            offset = i * 1_000_000
            tokens = np.arange(offset + 1000, dtype=np.uint16)
            np.save(data_root / f"shard_{split}_{i:04d}.npy", tokens)

    return data_root


@pytest.fixture
def force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin to CPU regardless of host CUDA so e2e/golden numerics stay stable.

    Not autouse: a global mock would break tests/test_environment.py's CUDA
    asserts. Opt in per module with `pytestmark = pytest.mark.usefixtures("force_cpu")`.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def reset_root_logger():
    """Save/restore root logger handlers around each test so state doesn't leak."""
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


REPO_ROOT = Path(__file__).resolve().parent.parent

# Default HF Cache to a repo-local directory
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache"))
