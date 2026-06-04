"""Flast attention 3 selector with SDPA fallback"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from flash_attn_interface import flash_attn_func as _fa3_func  # pyright: ignore
    _HAS_FA3 = True
except ImportError:
    _HAS_FA3 = False


def _fa3_available(q: torch.Tensor) -> bool:
    if not _HAS_FA3 or not q.is_cuda:
        return False
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False
    major, _ = torch.cuda.get_device_capability(q.device)
    return major == 9

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
    """Multi head attention with FA3 fast path and SDPA fallback"""
    if _fa3_available(q):
        return _fa3_func(q, k, v, causal=is_causal)

    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=is_causal, enable_gqa=True)
    return out.transpose(1, 2)