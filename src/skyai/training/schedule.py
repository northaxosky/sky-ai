"""Learning rate schedules"""

from __future__ import annotations

import math
from dataclasses import dataclass

@dataclass
class CosineSchedule:
    """Linear Warmup, cosine decay to a floor, then hold"""
    max_lr: float
    min_lr: float
    warmup_steps: int
    max_steps: int

    def __post_init__(self) -> None:
        if self.max_lr <= 0 or self.min_lr < 0:
            raise ValueError("max_lr must be positive; min_lr must be non-negative")
        if self.min_lr > self.max_lr:
            raise ValueError(f"min_lr ({self.min_lr}) must be <= max_lr ({self.max_lr})")
        if self.warmup_steps < 0 or self.max_steps < self.warmup_steps:
            raise ValueError(
                f"Need 0 <= warmup_steps ({self.warmup_steps}) "
                f"<= max_steps ({self.max_steps})"
            )
        
    def lr_for(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        if step > self.max_steps:
            return self.min_lr
        
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        return self.min_lr + coeff * (self.max_lr - self.min_lr)
    