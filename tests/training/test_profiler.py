"""Tests for the training-region Profiler"""

from __future__ import annotations

import time

import pytest
import torch

from skyai.config.schema import ProfilingConfig
from skyai.training.profiler import Profiler


def _disabled_cfg() -> ProfilingConfig:
    return ProfilingConfig(enabled=False, log_every=10, cuda_sync=False)


def _enabled_cfg(log_every: int = 5) -> ProfilingConfig:
    return ProfilingConfig(enabled=True, log_every=log_every, cuda_sync=False)


class TestDisabled:
    def test_region_is_noop(self) -> None:
        prof = Profiler(_disabled_cfg(), device="cpu", rank=0)
        with prof.region("anything"):
            pass
        assert prof._samples == {}
        assert prof._pending == []

    def test_log_and_reset_returns_empty(self) -> None:
        prof = Profiler(_disabled_cfg(), device="cpu", rank=0)
        assert prof.log_and_reset(step=10) == {}

    def test_should_log_returns_false(self) -> None:
        prof = Profiler(_disabled_cfg(), device="cpu", rank=0)
        assert prof.should_log(step=10) is False

    def test_flush_returns_empty(self) -> None:
        prof = Profiler(_disabled_cfg(), device="cpu", rank=0)
        assert prof.flush(step=10) == {}


class TestRankGating:
    def test_nonzero_rank_is_disabled_even_if_cfg_enabled(self) -> None:
        prof = Profiler(_enabled_cfg(), device="cpu", rank=1)
        assert prof.enabled is False
        with prof.region("x"):
            pass
        assert prof._samples == {}


class TestCpuTiming:
    def test_region_records_one_sample(self) -> None:
        prof = Profiler(_enabled_cfg(), device="cpu", rank=0)
        with prof.region("sleep_5ms"):
            time.sleep(0.005)
        assert len(prof._samples["sleep_5ms"]) == 1
        elapsed = prof._samples["sleep_5ms"][0]
        # 5ms sleep with generous CI tolerance
        assert 3.0 < elapsed < 100.0

    def test_multiple_samples_aggregate(self) -> None:
        prof = Profiler(_enabled_cfg(), device="cpu", rank=0)
        for _ in range(3):
            with prof.region("x"):
                pass
        assert len(prof._samples["x"]) == 3

    def test_log_and_reset_summary_keys(self) -> None:
        prof = Profiler(_enabled_cfg(), device="cpu", rank=0)
        with prof.region("foo"):
            time.sleep(0.001)
        with prof.region("bar"):
            time.sleep(0.001)
        summary = prof.log_and_reset(step=10)
        for k in (
            "profile/foo_mean_ms", "profile/foo_total_ms", "profile/foo_count",
            "profile/bar_mean_ms", "profile/total_ms", "profile/window_steps",
        ):
            assert k in summary
        # samples cleared after the call
        assert prof._samples == {}

    def test_total_equals_sum_of_region_totals(self) -> None:
        prof = Profiler(_enabled_cfg(), device="cpu", rank=0)
        with prof.region("a"):
            time.sleep(0.001)
        with prof.region("b"):
            time.sleep(0.001)
        summary = prof.log_and_reset(step=10)
        assert summary["profile/total_ms"] == pytest.approx(
            summary["profile/a_total_ms"] + summary["profile/b_total_ms"]
        )

    def test_region_records_on_exception(self) -> None:
        """If the wrapped block raises, we still record the partial elapsed."""
        prof = Profiler(_enabled_cfg(), device="cpu", rank=0)
        with pytest.raises(RuntimeError), prof.region("boom"):
            raise RuntimeError("boom")
        assert "boom" in prof._samples


class TestShouldLog:
    def test_not_at_step_zero(self) -> None:
        prof = Profiler(_enabled_cfg(log_every=5), device="cpu", rank=0)
        assert prof.should_log(step=0) is False

    def test_at_interval(self) -> None:
        prof = Profiler(_enabled_cfg(log_every=5), device="cpu", rank=0)
        assert prof.should_log(step=5) is True
        assert prof.should_log(step=10) is True
        assert prof.should_log(step=12) is False


class TestFlush:
    def test_flush_emits_pending_samples(self) -> None:
        prof = Profiler(_enabled_cfg(), device="cpu", rank=0)
        with prof.region("trailing"):
            time.sleep(0.001)
        # max_steps < log_every situation; flush still drains
        summary = prof.flush(step=3)
        assert "profile/trailing_mean_ms" in summary

    def test_flush_with_nothing_buffered_is_empty(self) -> None:
        prof = Profiler(_enabled_cfg(), device="cpu", rank=0)
        assert prof.flush(step=10) == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda regions need a GPU")
class TestCudaTiming:
    def test_region_defers_then_drains(self) -> None:
        prof = Profiler(_enabled_cfg(), device="cuda", rank=0)
        x = torch.randn(512, 512, device="cuda")
        with prof.region("matmul"):
            _ = x @ x
        # cuda_sync=False mode defers sample to log_and_reset
        assert len(prof._pending) == 1
        assert prof._samples == {}
        summary = prof.log_and_reset(step=5)
        assert "profile/matmul_mean_ms" in summary
        assert summary["profile/matmul_mean_ms"] > 0

    def test_cuda_sync_mode_records_immediately(self) -> None:
        cfg = ProfilingConfig(enabled=True, log_every=5, cuda_sync=True)
        prof = Profiler(cfg, device="cuda", rank=0)
        x = torch.randn(512, 512, device="cuda")
        with prof.region("matmul"):
            _ = x @ x
        # cuda_sync=True mode records the elapsed_time inside the with-block
        assert len(prof._samples["matmul"]) == 1
        assert prof._pending == []