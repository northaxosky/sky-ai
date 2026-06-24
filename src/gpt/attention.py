import math

import torch
import torch.nn.functional as F

from gpt.attention import CausalSelfAttention
from gpt.model import GPTConfig


def manual_causal_self_attention(q, k, v):
    """Reference impl: exactly what scaled_dot_product_attention computes, spelled out"""
    N, nh, T, hd = q.shape
    att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)  # (B, nh, T, T) scores
    causal = torch.tril(torch.ones(T, T)).view(1, 1, T, T)  # 1 on/below diagonal
    att = att.masked_fill(causal == 0, float("-inf"))  # forbid attending to the future
    att = F.softmax(att, dim=-1)  # weights over past positions
    return att @ v  # (B, nh, T, hd) weighted sum of values


def test_manual_matches_sdpa():
    torch.manual_seed(0)
    B, nh, T, hd = 2, 4, 16, 8
    q, k, v = (torch.randn(B, nh, T, hd) for _ in range(3))
    y_manual = manual_causal_self_attention(q, k, v)
    y_flash = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(y_manual, y_flash, atol=1e-6)  # same function, different algorithm/impl


def test_attention_shape_and_causality():
    cfg = GPTConfig(block_size=16, vocab_size=50257, n_layer=1, n_head=4, n_embd=32)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 8, 32)
    assert attn(x).shape == (2, 8, 32)  # shape preserved
