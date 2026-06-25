"""Block should: shape-preserve, accept cos/sin, gradient-flow, residual identity"""

import torch

from skyai.block import Block


def _make_cos_sin(seq_len: int, head_dim: int, base: float = 100000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq_len).float()
    angles = torch.outer(pos, inv_freq)
    return angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]


def test_block_shape_preserving():
    n_embd, n_head = 128, 4
    block = Block(n_embd, n_head)
    head_dim = n_embd // n_head
    B, T = 2, 16
    x = torch.randn(B, T, n_embd)
    cos, sin = _make_cos_sin(T, head_dim)
    out = block(x, cos, sin)
    assert out.shape == (B, T, n_embd)


def test_block_accepts_gqa():
    n_embd, n_head, n_kv_head = 128, 8, 2
    block = Block(n_embd, n_head, n_kv_head=n_kv_head)
    head_dim = n_embd // n_head
    B, T = 2, 16
    x = torch.randn(B, T, n_embd)
    cos, sin = _make_cos_sin(T, head_dim)
    out = block(x, cos, sin)
    assert out.shape == (B, T, n_embd)


def test_block_gradient_flow():
    n_embd, n_head = 64, 4
    block = Block(n_embd, n_head)
    head_dim = n_embd // n_head
    B, T = 2, 8
    x = torch.randn(B, T, n_embd, requires_grad=True)
    cos, sin = _make_cos_sin(T, head_dim)
    out = block(x, cos, sin)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    for name, p in block.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"


def test_block_residual_is_additive():
    """If we zero the sublayer outputs, the block should be the identity in x."""
    n_embd, n_head = 64, 4
    block = Block(n_embd, n_head)
    head_dim = n_embd // n_head
    B, T = 1, 4
    x = torch.randn(B, T, n_embd)
    cos, sin = _make_cos_sin(T, head_dim)

    with torch.no_grad():
        block.attn.c_proj.weight.zero_()
        block.mlp.down_proj.weight.zero_()

    out = block(x, cos, sin)
    assert torch.allclose(out, x, atol=1e-5)
