"""Tests for the ablation spec parser, variant generator, and end-to-end runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from harness.ablation import (
    AblationSpec,
    Variant,
    VariantResult,
    _format_markdown_table,
    _format_value,
    _overrides_to_strings,
    _slug_for_overrides,
    generate_variants,
    load_ablation_spec,
    run_ablation,
)


def _write_base_config(tmp_path: Path) -> Path:
    """Minimal but valid RunConfig YAML; data.root need not exist for spec validation."""
    cfg = {
        "total_batch_size": 256,
        "model": {
            "n_layer": 2,
            "n_head": 2,
            "n_embed": 32,
            "vocab_size": 50257,
            "block_size": 64,
        },
        "data": {"root": str(tmp_path / "shards"), "batch_size": 4},
        "optim": {"weight_decay": 0.1},
        "schedule": {
            "max_lr": 1e-3,
            "min_lr": 1e-4,
            "warmup_steps": 1,
            "max_steps": 4,
        },
        "eval": {"interval": 2, "val_steps": 1, "evals": []},
        "log": {"dir": str(tmp_path / "logs")},
        "checkpoint": {"dir": str(tmp_path / "ckpts"), "every_n_steps": 2},
    }
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _write_spec(tmp_path: Path, base_config: Path, **kwargs: Any) -> Path:
    body: dict[str, Any] = {"base_config": str(base_config)}
    body.update(kwargs)
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(body))
    return path


class TestFormatValue:
    def test_int(self) -> None:
        assert _format_value(42) == "42"

    def test_float_replaces_dot(self) -> None:
        assert _format_value(0.1) == "0p1"
        assert _format_value(3e-4) == "0p0003"

    def test_bool_lowercase(self) -> None:
        assert _format_value(True) == "true"
        assert _format_value(False) == "false"

    def test_string_sanitised(self) -> None:
        assert _format_value("foo bar") == "foo_bar"
        assert _format_value("a/b") == "a_b"


class TestSlugForOverrides:
    def test_single_key(self) -> None:
        assert _slug_for_overrides({"optim.weight_decay": 0.1}) == "optim_weight_decay_0p1"

    def test_two_keys_joined_with_double_underscore(self) -> None:
        slug = _slug_for_overrides({"schedule.max_lr": 3e-4, "optim.weight_decay": 0.0})
        assert slug == "schedule_max_lr_0p0003__optim_weight_decay_0p0"

    def test_empty_returns_default(self) -> None:
        assert _slug_for_overrides({}) == "default"


class TestOverridesToStrings:
    def test_scalars(self) -> None:
        out = _overrides_to_strings({"a.b": 1, "c.d": 0.5, "e.f": "hi"})
        assert out == ["a.b=1", "c.d=0.5", "e.f=hi"]

    def test_bool_lowercased(self) -> None:
        out = _overrides_to_strings({"x": True, "y": False})
        assert out == ["x=true", "y=false"]


class TestAblationSpec:
    def test_happy_path(self, tmp_path: Path) -> None:
        base = _write_base_config(tmp_path)
        spec_path = _write_spec(tmp_path, base, sweep={"optim.weight_decay": [0.0, 0.1]})
        spec, abs_path = load_ablation_spec(spec_path)
        assert isinstance(spec, AblationSpec)
        assert abs_path == base.resolve()
        assert spec.sweep == {"optim.weight_decay": [0.0, 0.1]}

    def test_missing_sweep_fails(self, tmp_path: Path) -> None:
        base = _write_base_config(tmp_path)
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump({"base_config": str(base)}))
        with pytest.raises(Exception, match="sweep"):
            load_ablation_spec(spec_path)

    def test_empty_sweep_values_fails(self, tmp_path: Path) -> None:
        base = _write_base_config(tmp_path)
        spec_path = _write_spec(tmp_path, base, sweep={"optim.weight_decay": []})
        with pytest.raises(Exception, match="at least one value"):
            load_ablation_spec(spec_path)

    def test_nested_base_override_fails(self, tmp_path: Path) -> None:
        base = _write_base_config(tmp_path)
        spec_path = _write_spec(
            tmp_path,
            base,
            base_overrides={"schedule": {"max_steps": 1000}},
            sweep={"optim.weight_decay": [0.0]},
        )
        with pytest.raises(Exception, match="must be a scalar"):
            load_ablation_spec(spec_path)

    def test_nonexistent_base_config_fails(self, tmp_path: Path) -> None:
        spec_path = _write_spec(
            tmp_path, tmp_path / "does_not_exist.yaml", sweep={"optim.weight_decay": [0.0]}
        )
        with pytest.raises(FileNotFoundError):
            load_ablation_spec(spec_path)


class TestGenerateVariants:
    def test_single_key(self, tmp_path: Path) -> None:
        spec = AblationSpec(
            base_config=Path("dummy.yaml"),
            sweep={"optim.weight_decay": [0.0, 0.1, 0.2]},
        )
        variants = generate_variants(spec)
        assert len(variants) == 3
        assert variants[0] == Variant("optim_weight_decay_0p0", {"optim.weight_decay": 0.0})
        assert variants[2] == Variant("optim_weight_decay_0p2", {"optim.weight_decay": 0.2})

    def test_cartesian_product(self, tmp_path: Path) -> None:
        spec = AblationSpec(
            base_config=Path("dummy.yaml"),
            sweep={"a": [1, 2], "b": [10, 20, 30]},
        )
        variants = generate_variants(spec)
        assert len(variants) == 6
        # First key varies slowest in itertools.product
        assert variants[0].overrides == {"a": 1, "b": 10}
        assert variants[1].overrides == {"a": 1, "b": 20}
        assert variants[3].overrides == {"a": 2, "b": 10}

    def test_slugs_are_unique(self) -> None:
        spec = AblationSpec(
            base_config=Path("dummy.yaml"),
            sweep={"a": [1, 2, 3], "b": [0.1, 0.2]},
        )
        variants = generate_variants(spec)
        slugs = [v.slug for v in variants]
        assert len(set(slugs)) == len(slugs)


class TestMarkdownTable:
    def test_renders_ok_and_failed(self) -> None:
        results = [
            VariantResult(
                slug="a", overrides={"x": 1}, status="ok", final_val_loss=3.5, wall_seconds=12.5
            ),
            VariantResult(
                slug="b", overrides={"x": 2}, status="failed", wall_seconds=1.0, error="boom"
            ),
        ]
        md = _format_markdown_table(results)
        assert "| a | ok | 3.5000 | 12.5 | x=1 |" in md
        assert "| b | failed | - | 1.0 | x=2 |" in md


class TestRunAblation:
    def test_dry_run_does_not_train(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        base = _write_base_config(tmp_path)
        spec_path = _write_spec(tmp_path, base, sweep={"optim.weight_decay": [0.0, 0.1]})

        called: list[Any] = []

        def fake_train(cfg, resume=False):  # pragma: no cover - should not run
            called.append(cfg)
            return {"final_val_loss": 1.0}

        monkeypatch.setattr("harness.training.loop.train", fake_train)

        results = run_ablation(spec_path, tmp_path / "out", dry_run=True)
        assert len(results) == 2
        assert all(r.status == "planned" for r in results)
        assert called == []

    def test_runs_each_variant_and_aggregates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _write_base_config(tmp_path)
        spec_path = _write_spec(tmp_path, base, sweep={"optim.weight_decay": [0.0, 0.1]})

        seen_weight_decays: list[float] = []

        def fake_train(cfg, resume=False):
            seen_weight_decays.append(cfg.optim.weight_decay)
            return {"final_val_loss": 3.0 + cfg.optim.weight_decay}

        monkeypatch.setattr("harness.training.loop.train", fake_train)

        out = tmp_path / "out"
        results = run_ablation(spec_path, out)

        assert sorted(seen_weight_decays) == [0.0, 0.1]
        assert all(r.status == "ok" for r in results)
        assert results[0].final_val_loss == pytest.approx(3.0)
        assert results[1].final_val_loss == pytest.approx(3.1)

        assert (out / "results.json").exists()
        assert (out / "results.md").exists()
        assert (out / "optim_weight_decay_0p0" / "result.json").exists()
        assert (out / "optim_weight_decay_0p1" / "result.json").exists()

    def test_failed_variant_does_not_abort_sweep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _write_base_config(tmp_path)
        spec_path = _write_spec(tmp_path, base, sweep={"optim.weight_decay": [0.0, 0.1, 0.2]})

        def fake_train(cfg, resume=False):
            if cfg.optim.weight_decay == 0.1:
                raise RuntimeError("simulated NaN halt")
            return {"final_val_loss": 3.0}

        monkeypatch.setattr("harness.training.loop.train", fake_train)

        results = run_ablation(spec_path, tmp_path / "out")
        statuses = [r.status for r in results]
        assert statuses == ["ok", "failed", "ok"]
        assert "simulated NaN halt" in results[1].error

    def test_skip_when_result_exists(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        base = _write_base_config(tmp_path)
        spec_path = _write_spec(tmp_path, base, sweep={"optim.weight_decay": [0.0, 0.1]})

        # Pre-populate one variant's result as ok
        existing_dir = tmp_path / "out" / "optim_weight_decay_0p0"
        existing_dir.mkdir(parents=True)
        existing = VariantResult(
            slug="optim_weight_decay_0p0",
            overrides={"optim.weight_decay": 0.0},
            status="ok",
            final_val_loss=2.5,
            wall_seconds=1.0,
        )
        (existing_dir / "result.json").write_text(json.dumps(existing.__dict__))

        train_calls: list[float] = []

        def fake_train(cfg, resume=False):
            train_calls.append(cfg.optim.weight_decay)
            return {"final_val_loss": 3.0}

        monkeypatch.setattr("harness.training.loop.train", fake_train)

        results = run_ablation(spec_path, tmp_path / "out")
        # First variant skipped, second ran fresh
        assert train_calls == [0.1]
        assert results[0].status == "skipped"
        assert results[1].status == "ok"

    def test_force_reruns_existing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        base = _write_base_config(tmp_path)
        spec_path = _write_spec(tmp_path, base, sweep={"optim.weight_decay": [0.0]})

        existing_dir = tmp_path / "out" / "optim_weight_decay_0p0"
        existing_dir.mkdir(parents=True)
        (existing_dir / "result.json").write_text(
            json.dumps(
                VariantResult(
                    slug="optim_weight_decay_0p0",
                    overrides={"optim.weight_decay": 0.0},
                    status="ok",
                    final_val_loss=99.9,
                ).__dict__
            )
        )

        train_calls: list[float] = []

        def fake_train(cfg, resume=False):
            train_calls.append(cfg.optim.weight_decay)
            return {"final_val_loss": 1.0}

        monkeypatch.setattr("harness.training.loop.train", fake_train)

        results = run_ablation(spec_path, tmp_path / "out", force=True)
        assert train_calls == [0.0]
        assert results[0].final_val_loss == pytest.approx(1.0)

    def test_overrides_threaded_into_per_variant_cfg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """base_overrides + variant overrides + checkpoint.dir all reach the loaded cfg."""
        base = _write_base_config(tmp_path)
        spec_path = _write_spec(
            tmp_path,
            base,
            base_overrides={"schedule.max_steps": 8},
            sweep={"optim.weight_decay": [0.0]},
        )

        seen_cfg: list[Any] = []

        def fake_train(cfg, resume=False):
            seen_cfg.append(cfg)
            return {"final_val_loss": 1.0}

        monkeypatch.setattr("harness.training.loop.train", fake_train)

        out = tmp_path / "out"
        run_ablation(spec_path, out)

        cfg = seen_cfg[0]
        assert cfg.schedule.max_steps == 8
        assert cfg.optim.weight_decay == 0.0
        assert cfg.checkpoint.dir == (out / "optim_weight_decay_0p0" / "ckpts").resolve()
