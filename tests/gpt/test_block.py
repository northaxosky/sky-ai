import torch

from gpt.block import Block
from gpt.model import GPTConfig


def test_block_preserves_shape_and_flows_gradient():
    cfg = GPTConfig(block_size=16, vocab_size=50257, n_layer=1, n_head=4, n_embd=32)
    block = Block(cfg)
    x = torch.randn(2, 8, 32, requires_grad=True)
    y = block(x)

    assert y.shape == (2, 8, 32)  # residual keeps the shape
    y.sum().backward()
    assert x.grad is not None  # gradient flows through the residual path
