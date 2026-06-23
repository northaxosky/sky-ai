"""flash.attention should: shape-preserve, support GQA, respect causality"""

import torch

from skyai.flash import attention


def test_attention_shape_preserves_mha():
    B, T, H, D = 2, 16, 4, 32
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)
    v = torch.randn(B, T, H, D)
    out = attention(q, k, v, is_causal=True)
    assert out.shape == (B, T, H, D)


def test_attention_supports_gqa():
    """H_kv < H_q with H_q divisible by H_kv should work transparently"""
    B, T, D = 2, 16, 32
    H_q, H_kv = 8, 2
    q = torch.randn(B, T, H_q, D)
    k = torch.randn(B, T, H_kv, D)
    v = torch.randn(B, T, H_kv, D)
    out = attention(q, k, v, is_causal=True)
    assert out.shape == (B, T, H_q, D)


def test_attention_is_causal():
    """Modifying position T-1 must not change outputs at positions 0..T-2"""
    torch.manual_seed(0)
    B, T, H, D = 1, 8, 2, 16
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)
    v = torch.randn(B, T, H, D)

    out_a = attention(q, k, v, is_causal=True)

    q2 = q.clone()
    k2 = k.clone()
    v2 = v.clone()
    q2[:, -1] = torch.randn_like(q2[:, -1])
    k2[:, -1] = torch.randn_like(k2[:, -1])
    v2[:, -1] = torch.randn_like(v2[:, -1])

    out_b = attention(q2, k2, v2, is_causal=True)

    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-5)


def test_attention_gradient_flow():
    B, T, H, D = 2, 8, 4, 16
    q = torch.randn(B, T, H, D, requires_grad=True)
    k = torch.randn(B, T, H, D, requires_grad=True)
    v = torch.randn(B, T, H, D, requires_grad=True)
    out = attention(q, k, v, is_causal=True)
    out.sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert k.grad is not None and k.grad.abs().sum() > 0
    assert v.grad is not None and v.grad.abs().sum() > 0
