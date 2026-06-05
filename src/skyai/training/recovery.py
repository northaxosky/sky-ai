"""Fault detection and graceful failure for the training loop"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from skyai.log import get_logger

if TYPE_CHECKING:
    from skyai.config.schema import RunConfig

logger = get_logger(__name__)


class NonFiniteGradError(RuntimeError):
    """Raised when a parameter gradient contains NaN or Inf"""


def detect_non_finite_grad(model: nn.Module) -> str | None:
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return name
    return None


def is_oom_error(e: BaseException) -> bool:
    """Modern PyTorch raises torch.cuda.OutOfMemoryError, older raises RuntimeError"""
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def diagnose_oom(e: BaseException, *, step: int, cfg: RunConfig, world_size: int) -> None:
    """Log VRAM stats + batch geometry"""
    logger.error(f"OOM at step {step}: {type(e).__name__}: {e}")
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                peak_alloc = torch.cuda.max_memory_allocated(i) / 1024**3
                peak_reserved = torch.cuda.max_memory_reserved(i) / 1024**3
                logger.error(
                    f"  cuda:{i} alloc={alloc:.2f}GB reserved={reserved:.2f}GB "
                    f"peak_alloc={peak_alloc:.2f}GB peak_reserved={peak_reserved:.2f}GB"
                )
        logger.error(
            f"  cfg: total_batch={cfg.total_batch_size} micro_batch={cfg.data.batch_size} "
            f"block_size{cfg.model.block_size} {world_size=} dtype={cfg.dtype}"
        )
    except Exception as diag_error:
        logger.error(f"  (diagnostics dump itself failed: {diag_error})")
