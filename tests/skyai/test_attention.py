import pytest
import torch

from skyai.attention import CausalSelfAttention, apply_rotary_emb
from tests.skyai._rope import make_cos_sin


def test_apply_rotary_emb_preserves_magnitude():
    """RoPE is a rotation; it must not change vector norms."""
    torch.manual_seed(0)
    B, T, H, D = 1, 8, 2, 32
    x = torch.randn(B, T, H, D)
    cos, sin = make_cos_sin(T, D)
    y = apply_rotary_emb(x, cos, sin)
    x_norms = x.norm(dim=-1)
    y_norms = y.norm(dim=-1)
    assert torch.allclose(x_norms, y_norms, atol=1e-5)


def test_attention_shape_preserves_mha():
    n_embd, n_head = 128, 4
    attn = CausalSelfAttention(n_embd, n_head)
    head_dim = n_embd // n_head
    B, T = 2, 16
    x = torch.randn(B, T, n_embd)
    cos, sin = make_cos_sin(T, head_dim)
    out = attn(x, cos, sin)
    assert out.shape == (B, T, n_embd)


def test_attention_shape_preserves_gqa():
    """n_kv_head < n_head should still produce (B, T, n_embd) output."""
    n_embd, n_head, n_kv_head = 128, 8, 2
    attn = CausalSelfAttention(n_embd, n_head, n_kv_head=n_kv_head)
    head_dim = n_embd // n_head
    B, T = 2, 16
    x = torch.randn(B, T, n_embd)
    cos, sin = make_cos_sin(T, head_dim)
    out = attn(x, cos, sin)
    assert out.shape == (B, T, n_embd)


def test_attention_is_causal():
    """Modifying input at position T-1 must not change outputs at positions 0..T-2."""
    torch.manual_seed(0)
    n_embd, n_head = 64, 4
    attn = CausalSelfAttention(n_embd, n_head)
    head_dim = n_embd // n_head
    B, T = 1, 8
    x = torch.randn(B, T, n_embd)
    cos, sin = make_cos_sin(T, head_dim)

    out_a = attn(x, cos, sin)
    x2 = x.clone()
    x2[:, -1] = torch.randn_like(x2[:, -1])
    out_b = attn(x2, cos, sin)

    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-5)


def test_attention_gradient_flow():
    n_embd, n_head = 64, 4
    attn = CausalSelfAttention(n_embd, n_head)
    head_dim = n_embd // n_head
    B, T = 2, 8
    x = torch.randn(B, T, n_embd, requires_grad=True)
    cos, sin = make_cos_sin(T, head_dim)
    out = attn(x, cos, sin)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    for name, p in attn.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert p.grad.abs().sum() > 0, f"{name} gradient is all zero"


def test_attention_rejects_bad_head_config():
    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(n_embd=100, n_head=7)
    with pytest.raises(ValueError, match="<= n_head"):
        CausalSelfAttention(n_embd=128, n_head=4, n_kv_head=8)
    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(n_embd=128, n_head=8, n_kv_head=3)


def test_attention_qk_norm_disabled():
    """With QK-Norm off, the module should still produce a finite forward."""
    attn = CausalSelfAttention(n_embd=64, n_head=4, use_qk_norm=False)
    head_dim = 64 // 4
    B, T = 1, 4
    x = torch.randn(B, T, 64)
    cos, sin = make_cos_sin(T, head_dim)
    out = attn(x, cos, sin)
    assert torch.isfinite(out).all()
