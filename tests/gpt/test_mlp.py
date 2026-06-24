import torch

from gpt.mlp import MLP
from gpt.model import GPTConfig


def _cfg() -> GPTConfig:
    return GPTConfig(block_size=16, vocab_size=50257, n_layer=1, n_head=4, n_embd=32)


def test_mlp_preserves_shape():
    x = torch.randn(2, 8, 32)
    assert MLP(_cfg())(x).shape == (2, 8, 32)


def test_mlp_expands_4x():
    mlp = MLP(_cfg())
    assert mlp.c_fc.out_features == 4 * 32 and mlp.c_proj.in_features == 4 * 32
