"""Pytest Configuration: shared fixtures and environment setup"""

import os
from pathlib import Path

import numpy as np
import pytest


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


REPO_ROOT = Path(__file__).resolve().parent.parent

# Default HF Cache to a repo-local directory
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache"))
