"""Sharded .npy token loader with per-rank striding and resumable state"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

_SPLITS = ("train", "val")


def load_tokens(path: str | Path) -> torch.Tensor:
    tokens = np.load(path)
    # Shards are uint16 (gpt2) or uint32 (cl100k)
    tokens = tokens.astype(np.int32)
    return torch.tensor(tokens, dtype=torch.long)


class DataLoader:
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        batch_size: int,
        block_size: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if split not in _SPLITS:
            raise ValueError(f"Split must be one of {_SPLITS}, got {split!r}")
        if not (0 <= rank < world_size):
            raise ValueError(f"Rank must be in [0, {world_size}), got {rank}")

        self.B = batch_size
        self.T = block_size
        self.rank = rank
        self.world_size = world_size

        data_root = Path(data_root)
        self.shards = sorted(
            p for p in data_root.iterdir() if p.suffix == ".npy" and split in p.name
        )
        if not self.shards:
            raise FileNotFoundError(f"No shards matching split {split!r} in {data_root}")
        self.reset()

    @property
    def _stride(self) -> int:
        # Tokens consumed across ALL ranks per step
        return self.B * self.T * self.world_size

    def reset(self) -> None:
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.position = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        start = self.position + self.rank * B * T
        buf = self.tokens[start : start + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.position += self._stride
        if self.position + self._stride + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.position = 0
        return x, y

    def state_dict(self) -> dict[str, Any]:
        # Rank independent on purpose, any rank can resume from this
        return {"current_shard": self.current_shard, "position": self.position}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.current_shard = state["current_shard"]
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.position = state["position"]
