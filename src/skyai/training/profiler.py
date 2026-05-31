"""Profiler for the training loop"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from skyai.config.schema import ProfilingConfig
from skyai.log import get_logger

logger = get_logger(__name__)


class Profiler:
    """Times labeled regions in the training loop"""

    def __init__(self, cfg: ProfilingConfig, *, device: torch.device | str, rank: int = 0) -> None:
        self.cfg = cfg
        self.rank = rank
        self.device = torch.device(device) if isinstance(device, str) else device
        self._is_cuda = self.device.type == "cuda"

        self._samples: dict[str, list[float]] = defaultdict(list)
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled and self.rank == 0
    
    @contextmanager
    def region(self, name: str) -> Iterator[None]:
        """Time everything inside the with block"""
        if not self.enabled:
            yield
            return
        
        if self._is_cuda:
            if self.cfg.cuda_sync:
                torch.cuda.synchronize(self.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                if self.cfg.cuda_sync:
                    torch.cuda.synchronize(self.device)
                    self._samples[name].append(start.elapsed_time(end))
                else:
                    self._pending.append((name, start, end))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self._samples[name].append((time.perf_counter() - t0) * 1000.0)

    def should_log(self, step: int) -> bool:
        if not self.enabled:
            return False
        return step > 0 and step % self.cfg.log_every == 0
    
    def _drain_pending(self) -> None:
        if not self._pending:
            return
        
        torch.cuda.synchronize(self.device)
        for name, start, end in self._pending:
            self._samples[name].append(start.elapsed_time(end))
        self._pending.clear()

    def log_and_reset(self, step: int) -> dict[str, float]:
        """Emit a breakdown line and return a flat dict"""
        if not self.enabled:
            return {}
        self._drain_pending()

        summary: dict[str, float] = {}
        parts: list[str] = []
        total = 0.0
        
        for name in sorted(self._samples):
            times = self._samples[name]
            mean = sum(times) / len(times)
            region_total = sum(times)

            summary[f"profile/{name}_mean_ms"] = mean
            summary[f"profile/{name}_total_ms"] = region_total
            summary[f"profile/{name}_count"] = float(len(times))
            parts.append(f"{name}={mean:.2f}ms")
            total += region_total
        
        summary["profile/total_ms"] = total
        summary["profile/window_steps"] = float(self.cfg.log_every)

        logger.info(
            f"profile step={step} window={self.cfg.log_every}steps "
            f"| {' '.join(parts)} | total={total:.1f}ms"
        )
        self._samples.clear()
        return summary
    
    def flush(self, step: int) -> dict[str, float]:
        """Final drain at end of training"""
        if not self.enabled:
            return {}
        if not self._samples and not self._pending:
            return {}
        return self.log_and_reset(step)