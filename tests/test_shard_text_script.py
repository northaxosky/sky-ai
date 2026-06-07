"""Tests for the HF text sharding script helpers."""

from __future__ import annotations

import numpy as np

from scripts import shard_text


def test_shard_path_uses_prefix_and_val_first(tmp_path):
    assert shard_text.shard_path(tmp_path, 0, prefix="climbmix").name == "climbmix_val_000000"
    assert shard_text.shard_path(tmp_path, 1, prefix="climbmix").name == "climbmix_train_000001"


def test_tokenize_uses_configured_text_column():
    shard_text._init_worker("gpt2", "body")
    tokens = shard_text.tokenize({"body": "hello"})

    assert tokens.dtype == np.uint16
    assert tokens[0] == shard_text.eot
    assert len(tokens) > 1
