"""Tests for the training/ package"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from harness.config.schema import ModelConfig
from harness.training.loop import build_model
from harness.training.optimizer import Muon, OptimizerChain, build_optimizer
from harness.training.schedule import CosineSchedule, WarmupStableDecaySchedule
from skyai.model import GPT, GPTConfig


class TestCosineSchedule:
    def _default(self) -> CosineSchedule:
        return CosineSchedule(max_lr=6e-4, min_lr=6e-5, warmup_steps=10, max_steps=100)

    def test_lr_at_step_0_is_first_warmup_increment(self) -> None:
        sch = self._default()
        # step=0 returns max_lr * (0+1)/warmup = max_lr/warmup
        assert sch.lr_for(0) == pytest.approx(6e-4 * 1 / 10)

    def test_lr_at_warmup_boundary_equals_max_lr(self) -> None:
        sch = self._default()
        # step=warmup_steps-1 is the last warmup step, hits max_lr exactly
        assert sch.lr_for(9) == pytest.approx(6e-4)

    def test_lr_at_max_steps_equals_min_lr(self) -> None:
        sch = self._default()
        # cosine fully decayed: cos(pi) = -1, coeff = 0, returns min_lr
        assert sch.lr_for(100) == pytest.approx(6e-5)

    def test_lr_beyond_max_steps_holds_at_min(self) -> None:
        sch = self._default()
        for step in [101, 500, 10_000]:
            assert sch.lr_for(step) == pytest.approx(6e-5)

    def test_cosine_phase_is_monotonically_decreasing(self) -> None:
        sch = self._default()
        prev = sch.lr_for(10)
        for step in range(11, 101):
            curr = sch.lr_for(step)
            assert curr <= prev, f"LR increased at step {step}: {prev} -> {curr}"
            prev = curr

    def test_midpoint_lr_is_average_of_max_and_min(self) -> None:
        # at the midpoint of the cosine phase, cos(pi/2) = 0, coeff = 0.5
        # so lr = min + 0.5 * (max - min) = (max + min) / 2
        sch = CosineSchedule(max_lr=1.0, min_lr=0.0, warmup_steps=0, max_steps=100)
        assert sch.lr_for(50) == pytest.approx(0.5)

    def test_rejects_min_above_max(self) -> None:
        with pytest.raises(ValueError, match="min_lr"):
            CosineSchedule(max_lr=1e-4, min_lr=1e-3, warmup_steps=10, max_steps=100)

    def test_rejects_warmup_above_max(self) -> None:
        with pytest.raises(ValueError, match="warmup_steps"):
            CosineSchedule(max_lr=1e-4, min_lr=1e-5, warmup_steps=200, max_steps=100)


class _ToyModel(nn.Module):
    """Small module that has both 2D (matrix) and 1D (bias, LayerNorm) params"""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.ln = nn.LayerNorm(8)


class TestBuildOptimizer:
    def test_returns_adamw(self) -> None:
        opt = build_optimizer(_ToyModel(), learning_rate=1e-3, weight_decay=0.1)
        assert isinstance(opt, torch.optim.AdamW)

    def test_decay_group_holds_only_matrix_params(self) -> None:
        opt = build_optimizer(_ToyModel(), learning_rate=1e-3, weight_decay=0.1)
        decay_group = opt.param_groups[0]
        for param in decay_group["params"]:
            assert param.dim() >= 2, "decay group should only contain 2D+ params"

    def test_nodecay_group_holds_only_vector_params(self) -> None:
        opt = build_optimizer(_ToyModel(), learning_rate=1e-3, weight_decay=0.1)
        nodecay_group = opt.param_groups[1]
        for param in nodecay_group["params"]:
            assert param.dim() < 2, "no-decay group should only contain 1D params"

    def test_weight_decay_values_set_correctly(self) -> None:
        opt = build_optimizer(_ToyModel(), learning_rate=1e-3, weight_decay=0.1)
        assert opt.param_groups[0]["weight_decay"] == 0.1
        assert opt.param_groups[1]["weight_decay"] == 0.0

    def test_all_trainable_params_assigned_to_a_group(self) -> None:
        model = _ToyModel()
        opt = build_optimizer(model, learning_rate=1e-3, weight_decay=0.1)
        grouped = sum((g["params"] for g in opt.param_groups), start=[])
        model_params = list(model.parameters())
        assert len(grouped) == len(model_params)

    def test_tied_weights_appear_once(self) -> None:
        # Mimic GPT's weight tying: same tensor lives under two names
        model = _ToyModel()
        fc2 = nn.Linear(8, 8, bias=False)
        fc2.weight = model.fc.weight
        model.fc2 = fc2
        opt = build_optimizer(model, learning_rate=1e-3, weight_decay=0.1)

        decay_params = opt.param_groups[0]["params"]
        # The tied weight should appear exactly once across the optimizer
        n_tied = sum(1 for p in decay_params if p is model.fc.weight)
        assert n_tied == 1

    def test_rejects_model_with_no_trainable_params(self) -> None:
        model = _ToyModel()
        for p in model.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="no trainable parameters"):
            build_optimizer(model, learning_rate=1e-3, weight_decay=0.1)

    def test_respects_custom_betas_and_eps(self) -> None:
        opt = build_optimizer(
            _ToyModel(),
            learning_rate=1e-3,
            weight_decay=0.1,
            betas=(0.8, 0.99),
            eps=1e-9,
        )
        assert opt.defaults["betas"] == (0.8, 0.99)
        assert opt.defaults["eps"] == 1e-9

    def test_no_fused_kwarg_when_unsupported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If torch.optim.AdamW signature has no 'fused' param, don't pass it."""
        import inspect

        real_signature = inspect.signature

        def fake_signature(obj):  # type: ignore[no-untyped-def]
            if obj is torch.optim.AdamW:
                sig = real_signature(obj)
                params = [p for k, p in sig.parameters.items() if k != "fused"]
                return inspect.Signature(parameters=params)
            return real_signature(obj)

        monkeypatch.setattr(inspect, "signature", fake_signature)
        # Even with device_type="cuda", must not raise if fused is unsupported
        opt = build_optimizer(
            _ToyModel(),
            learning_rate=1e-3,
            weight_decay=0.1,
            device_type="cuda",
        )
        assert isinstance(opt, torch.optim.AdamW)

    def test_no_fused_kwarg_when_cpu(self) -> None:
        """Even with fused available, don't request it on cpu."""
        opt = build_optimizer(
            _ToyModel(),
            learning_rate=1e-3,
            weight_decay=0.1,
            device_type="cpu",
        )
        # On torch versions that expose 'fused', the default is None when we
        # didn't pass it. Either way we must not have opted into True.
        assert opt.defaults.get("fused") is not True

    def test_muon_rejects_non_matrix_param(self) -> None:
        param = nn.Parameter(torch.ones(8))
        param.grad = torch.ones_like(param)
        opt = Muon([param], lr=1e-3)
        with pytest.raises(ValueError, match="2D"):
            opt.step()

    def test_muon_updates_matrix_param(self) -> None:
        param = nn.Parameter(torch.randn(8, 8))
        param.grad = torch.randn_like(param)
        before = param.detach().clone()
        opt = Muon([param], lr=1e-3)
        opt.step()
        assert not torch.equal(param, before)

    def test_muon_split_returns_optimizer_chain(self) -> None:
        model = GPT(
            GPTConfig(
                block_size=16,
                vocab_size=128,
                n_layer=2,
                n_head=4,
                n_embd=64,
                logit_softcap=None,
            )
        )
        opt = build_optimizer(
            model,
            optimizer_type="muon-split",
            learning_rate=1e-3,
            weight_decay=0.28,
            device_type="cpu",
        )
        assert isinstance(opt, OptimizerChain)
        assert any(group.get("optimizer_type") == "muon" for group in opt.param_groups)

        grouped = [p for group in opt.param_groups for p in group["params"]]
        assert len({id(p) for p in grouped}) == len(grouped)
        assert {id(p) for p in grouped} == {id(p) for p in model.parameters() if p.requires_grad}

    def test_muon_split_omits_lm_head_when_tied(self) -> None:
        model = GPT(
            GPTConfig(
                block_size=16,
                vocab_size=128,
                n_layer=2,
                n_head=4,
                n_embd=64,
                tie_weights=True,
                logit_softcap=None,
            )
        )
        opt = build_optimizer(
            model,
            optimizer_type="muon-split",
            learning_rate=1e-3,
            weight_decay=0.28,
            device_type="cpu",
        )
        grouped = [p for group in opt.param_groups for p in group["params"]]
        assert sum(1 for p in grouped if p is model.transformer.wte.weight) == 1

    def test_muon_split_builds_on_gpt2(self) -> None:
        """gpt2 hardcodes weight tying and has no tie_weights config flag; muon-split
        must still build, with the shared embedding counted exactly once."""
        model = build_model(
            ModelConfig(
                family="gpt2", n_layer=2, n_head=4, n_embd=64, vocab_size=50257, block_size=16
            )
        )
        opt = build_optimizer(
            model,
            optimizer_type="muon-split",
            learning_rate=1e-3,
            weight_decay=0.28,
            device_type="cpu",
        )
        assert isinstance(opt, OptimizerChain)
        grouped = [p for group in opt.param_groups for p in group["params"]]
        assert sum(1 for p in grouped if p is model.transformer.wte.weight) == 1
        assert {id(p) for p in grouped} == {id(p) for p in model.parameters() if p.requires_grad}


class TestWarmupStableDecaySchedule:
    def test_warmup_starts_at_first_increment(self) -> None:
        sched = WarmupStableDecaySchedule(warmup_steps=10, max_steps=100)
        assert sched.multiplier_for(0) == pytest.approx(0.1)

    def test_plateau_holds_at_one(self) -> None:
        sched = WarmupStableDecaySchedule(warmup_steps=10, max_steps=100, warmdown_ratio=0.2)
        assert sched.multiplier_for(20) == pytest.approx(1.0)

    def test_final_step_hits_final_fraction(self) -> None:
        sched = WarmupStableDecaySchedule(
            warmup_steps=10, max_steps=100, warmdown_ratio=0.2, final_lr_frac=0.05
        )
        assert sched.multiplier_for(99) == pytest.approx(0.05)

    def test_muon_momentum_warms_and_decays(self) -> None:
        sched = WarmupStableDecaySchedule(warmup_steps=10, max_steps=1000, warmdown_ratio=0.2)
        assert sched.muon_momentum_for(0) == pytest.approx(0.85)
        assert sched.muon_momentum_for(400) == pytest.approx(0.97)
        assert sched.muon_momentum_for(999) == pytest.approx(0.90)

    def test_muon_weight_decay_cosines_to_zero(self) -> None:
        sched = WarmupStableDecaySchedule(warmup_steps=10, max_steps=100)
        assert sched.muon_weight_decay_for(0, 0.28) == pytest.approx(0.28)
        assert sched.muon_weight_decay_for(100, 0.28) == pytest.approx(0.0)
