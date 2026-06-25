"""Shared RoPE cos/sin table builder for the skyai attention/block tests."""

import torch


def make_cos_sin(seq_len: int, head_dim: int, base: float = 100000.0):
    """Build broadcastable (1, T, 1, head_dim/2) cos/sin tables for RoPE."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq_len).float()
    angles = torch.outer(pos, inv_freq)
    return angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]
