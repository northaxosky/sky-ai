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
                f"Need 0 <= warmup_steps ({self.warmup_steps}) <= max_steps ({self.max_steps})"
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


@dataclass
class WarmupStableDecaySchedule:
    """LR multiplier schedule: linear warmup, stable plateau, linear warmdown."""

    warmup_steps: int
    max_steps: int
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.warmup_steps > self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be <= max_steps ({self.max_steps})"
            )
        if not 0.0 <= self.warmdown_ratio <= 1.0:
            raise ValueError("warmdown_ratio must be in [0, 1]")
        if not 0.0 <= self.final_lr_frac <= 1.0:
            raise ValueError("final_lr_frac must be in [0, 1]")

    def multiplier_for(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps

        warmdown_steps = round(self.warmdown_ratio * self.max_steps)
        warmdown_start = self.max_steps - warmdown_steps
        if warmdown_steps > 0 and step >= warmdown_start:
            progress = (step - warmdown_start + 1) / warmdown_steps
            progress = min(max(progress, 0.0), 1.0)
            return 1.0 - progress * (1.0 - self.final_lr_frac)

        return 1.0

    def lr_for(self, step: int) -> float:
        return self.multiplier_for(step)

    def muon_momentum_for(self, step: int) -> float:
        warmdown_steps = round(self.warmdown_ratio * self.max_steps)
        warmdown_start = self.max_steps - warmdown_steps

        if step < 400:
            return 0.85 + (0.97 - 0.85) * step / 400
        if warmdown_steps > 0 and step >= warmdown_start:
            progress = (step - warmdown_start + 1) / warmdown_steps
            progress = min(max(progress, 0.0), 1.0)
            return 0.97 + (0.90 - 0.97) * progress
        return 0.97

    def muon_weight_decay_for(self, step: int, base_weight_decay: float) -> float:
        progress = min(max(step / self.max_steps, 0.0), 1.0)
        return base_weight_decay * 0.5 * (1.0 + math.cos(math.pi * progress))
