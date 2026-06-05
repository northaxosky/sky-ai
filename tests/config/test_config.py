"""Tests for the config system: pydantic schema, YAML loader, CLI overrides."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from skyai.config.loader import load_config
from skyai.config.schema import RunConfig

# ---------- shared fixtures ----------


def _valid_run_dict() -> dict:
    """Minimal valid RunConfig dict. Tests mutate and re-validate."""
    return {
        "seed": 1337,
        "dtype": "bfloat16",
        "compile": False,
        "grad_clip": 1.0,
        "total_batch_size": 256,
        "model": {
            "n_layer": 2,
            "n_head": 2,
            "n_embed": 64,
            "vocab_size": 50257,
            "block_size": 64,
        },
        "data": {
            "root": "data/x",
            "train_split": "train",
            "val_split": "val",
            "batch_size": 4,
        },
        "optim": {
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
        },
        "schedule": {
            "max_lr": 1.0e-3,
            "min_lr": 1.0e-4,
            "warmup_steps": 5,
            "max_steps": 50,
        },
        "eval": {"interval": 10, "val_steps": 2, "evals": ["hellaswag"]},
        "log": {"dir": "logs", "level": "INFO", "wandb": False, "wandb_project": None},
        "checkpoint": {"dir": "ckpt", "every_n_steps": 25, "keep_last_n": 3},
    }


def _full_yaml_body() -> str:
    return """\
seed: 1337
dtype: bfloat16
compile: false
grad_clip: 1.0
total_batch_size: 256
model:
    n_layer: 2
    n_head: 2
    n_embed: 64
    vocab_size: 50257
    block_size: 64
data:
    root: data/x
    train_split: train
    val_split: val
    batch_size: 4
optim:
    weight_decay: 0.1
    betas: [0.9, 0.95]
    eps: 1.0e-8
schedule:
    max_lr: 1.0e-3
    min_lr: 1.0e-4
    warmup_steps: 5
    max_steps: 50
eval:
    interval: 10
    val_steps: 2
    evals: ["hellaswag"]
log:
    dir: logs
    level: INFO
    wandb: false
    wandb_project: null
checkpoint:
    dir: ckpt
    every_n_steps: 25
    keep_last_n: 3
"""


def _write_yaml(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body).lstrip())
    return p


# ---------- TestSchema: pydantic validators ----------


class TestSchema:
    def test_minimal_valid(self) -> None:
        cfg = RunConfig.model_validate(_valid_run_dict())
        assert cfg.model.n_embed == 64
        assert cfg.optim.betas == (0.9, 0.95)

    def test_model_defaults_include_modern_arch_fields(self) -> None:
        cfg = RunConfig.model_validate(_valid_run_dict())
        assert cfg.model.init_policy == "gpt2"
        assert cfg.model.n_kv_head is None
        assert cfg.model.hidden_multiple == 4
        assert cfg.model.rope_theta == 100_000.0
        assert cfg.model.vocab_pad_multiple == 128
        assert cfg.model.tie_weights is False
        assert cfg.model.logit_softcap == 15.0

    def test_model_modern_arch_fields_accepted(self) -> None:
        d = _valid_run_dict()
        d["model"].update(
            {
                "n_kv_head": 1,
                "init_policy": "sky-ai",
                "hidden_multiple": 8,
                "rope_theta": 10_000.0,
                "vocab_pad_multiple": 256,
                "tie_weights": True,
                "logit_softcap": None,
            }
        )
        cfg = RunConfig.model_validate(d)
        assert cfg.model.n_kv_head == 1
        assert cfg.model.init_policy == "sky-ai"
        assert cfg.model.hidden_multiple == 8
        assert cfg.model.rope_theta == 10_000.0
        assert cfg.model.vocab_pad_multiple == 256
        assert cfg.model.tie_weights is True
        assert cfg.model.logit_softcap is None

    def test_n_kv_head_must_divide_n_head(self) -> None:
        d = _valid_run_dict()
        d["model"]["n_head"] = 6
        d["model"]["n_embed"] = 60
        d["model"]["n_kv_head"] = 4
        with pytest.raises(ValidationError, match="n_kv_head"):
            RunConfig.model_validate(d)

    def test_path_coercion(self) -> None:
        cfg = RunConfig.model_validate(_valid_run_dict())
        assert isinstance(cfg.data.root, Path)
        assert isinstance(cfg.log.dir, Path)
        assert isinstance(cfg.checkpoint.dir, Path)

    def test_model_n_embed_not_divisible_by_n_head(self) -> None:
        d = _valid_run_dict()
        d["model"]["n_embed"] = 65
        with pytest.raises(ValidationError, match="n_embed"):
            RunConfig.model_validate(d)

    def test_schedule_min_lr_above_max_lr(self) -> None:
        d = _valid_run_dict()
        d["schedule"]["min_lr"] = 1.0e-2
        with pytest.raises(ValidationError, match="min_lr"):
            RunConfig.model_validate(d)

    def test_schedule_warmup_above_max_steps(self) -> None:
        d = _valid_run_dict()
        d["schedule"]["warmup_steps"] = 100
        with pytest.raises(ValidationError, match="warmup"):
            RunConfig.model_validate(d)

    def test_wandb_enabled_without_project(self) -> None:
        d = _valid_run_dict()
        d["log"]["wandb"] = True
        d["log"]["wandb_project"] = None
        with pytest.raises(ValidationError, match="wandb_project"):
            RunConfig.model_validate(d)

    def test_total_batch_not_divisible_by_microbatch(self) -> None:
        d = _valid_run_dict()
        # batch_size=4, block_size=64 -> product=256; 300 not divisible
        d["total_batch_size"] = 300
        with pytest.raises(ValidationError, match="total_batch_size"):
            RunConfig.model_validate(d)

    def test_top_level_typo_rejected(self) -> None:
        d = _valid_run_dict()
        d["grad_clpip"] = 1.0
        with pytest.raises(ValidationError):
            RunConfig.model_validate(d)

    def test_nested_typo_in_model_rejected(self) -> None:
        d = _valid_run_dict()
        d["model"]["n_layre"] = 24
        with pytest.raises(ValidationError):
            RunConfig.model_validate(d)

    def test_nested_typo_in_schedule_rejected(self) -> None:
        d = _valid_run_dict()
        d["schedule"]["max_step"] = 100
        with pytest.raises(ValidationError):
            RunConfig.model_validate(d)

    def test_nested_typo_in_log_rejected(self) -> None:
        d = _valid_run_dict()
        d["log"]["wnadb"] = True
        with pytest.raises(ValidationError):
            RunConfig.model_validate(d)

    def test_vocab_smaller_than_tokenizer_rejected(self) -> None:
        d = _valid_run_dict()
        d["model"]["vocab_size"] = 10000  # gpt2 has 50257
        with pytest.raises(ValidationError, match="vocab_size"):
            RunConfig.model_validate(d)

    def test_vocab_way_above_tokenizer_rejected(self) -> None:
        d = _valid_run_dict()
        # gpt2 has 50257; 100277 is cl100k_base size, > 50257 + 1024
        d["model"]["vocab_size"] = 100277
        with pytest.raises(ValidationError, match="vocab_size"):
            RunConfig.model_validate(d)

    def test_vocab_with_alignment_padding_accepted(self) -> None:
        d = _valid_run_dict()
        d["model"]["vocab_size"] = 50304  # gpt2 n_vocab + 47 (pad-to-128)
        cfg = RunConfig.model_validate(d)
        assert cfg.model.vocab_size == 50304
        assert cfg.model.tokenizer_vocab_size == 50257

    def test_unknown_tokenizer_rejected(self) -> None:
        d = _valid_run_dict()
        d["model"]["tokenizer"] = "definitely-not-an-encoding"
        with pytest.raises(ValidationError, match="tokenizer"):
            RunConfig.model_validate(d)

    def test_unknown_init_policy_rejected(self) -> None:
        d = _valid_run_dict()
        d["model"]["init_policy"] = "nanochat"
        with pytest.raises(ValidationError, match="init_policy"):
            RunConfig.model_validate(d)

    def test_vocab_above_alignment_padding_rejected(self) -> None:
        d = _valid_run_dict()
        d["model"]["vocab_size"] = 50305  # one above gpt2 pad-to-128
        with pytest.raises(ValidationError, match="vocab_size"):
            RunConfig.model_validate(d)

    def test_cl100k_vocab_with_alignment_padding_accepted(self) -> None:
        d = _valid_run_dict()
        d["model"]["tokenizer"] = "cl100k_base"
        d["model"]["vocab_size"] = 100352  # cl100k_base padded to 128
        cfg = RunConfig.model_validate(d)
        assert cfg.model.vocab_size == 100352


# ---------- TestLoader: YAML reading, extends, deep merge ----------


class TestLoader:
    def test_loads_single_file(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "cfg.yaml", _full_yaml_body())
        cfg = load_config(path)
        assert cfg.seed == 1337
        assert cfg.model.n_layer == 2

    def test_extends_merges_parent(self, tmp_path: Path) -> None:
        _write_yaml(tmp_path, "base.yaml", _full_yaml_body())
        _write_yaml(
            tmp_path,
            "child.yaml",
            """
            extends: base.yaml
            seed: 42
        """,
        )
        cfg = load_config(tmp_path / "child.yaml")
        assert cfg.seed == 42  # overridden
        assert cfg.model.n_layer == 2  # inherited

    def test_deep_merge_into_nested_subconfig(self, tmp_path: Path) -> None:
        _write_yaml(tmp_path, "base.yaml", _full_yaml_body())
        _write_yaml(
            tmp_path,
            "child.yaml",
            """
            extends: base.yaml
            model:
                n_embed: 128
        """,
        )
        cfg = load_config(tmp_path / "child.yaml")
        assert cfg.model.n_embed == 128  # overridden
        assert cfg.model.n_layer == 2  # inherited (deep merge, not replace)
        assert cfg.model.n_head == 2  # inherited

    def test_extends_chain_three_levels(self, tmp_path: Path) -> None:
        _write_yaml(tmp_path, "a.yaml", _full_yaml_body())
        _write_yaml(
            tmp_path,
            "b.yaml",
            """
            extends: a.yaml
            seed: 7
        """,
        )
        _write_yaml(
            tmp_path,
            "c.yaml",
            """
            extends: b.yaml
            grad_clip: 2.0
        """,
        )
        cfg = load_config(tmp_path / "c.yaml")
        assert cfg.seed == 7  # from b
        assert cfg.grad_clip == 2.0  # from c
        assert cfg.model.n_layer == 2  # from a

    def test_cycle_detected(self, tmp_path: Path) -> None:
        _write_yaml(tmp_path, "a.yaml", "extends: b.yaml\nseed: 1\n")
        _write_yaml(tmp_path, "b.yaml", "extends: a.yaml\nseed: 2\n")
        with pytest.raises(ValueError, match="cycle|cyclic"):
            load_config(tmp_path / "a.yaml")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_missing_parent_raises(self, tmp_path: Path) -> None:
        _write_yaml(tmp_path, "child.yaml", "extends: nope.yaml\nseed: 1\n")
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "child.yaml")

    def test_non_dict_yaml_raises(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "bad.yaml", "- a\n- list\n- not a dict\n")
        with pytest.raises(ValueError, match="dict|mapping"):
            load_config(path)


# ---------- TestOverrides: --override key=value parsing ----------


class TestOverrides:
    def test_top_level_override(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "cfg.yaml", _full_yaml_body())
        cfg = load_config(path, overrides=["seed=999"])
        assert cfg.seed == 999

    def test_nested_override(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "cfg.yaml", _full_yaml_body())
        cfg = load_config(path, overrides=["model.n_embed=128"])
        assert cfg.model.n_embed == 128

    def test_deeply_nested_override(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "cfg.yaml", _full_yaml_body())
        cfg = load_config(path, overrides=["schedule.max_steps=100"])
        assert cfg.schedule.max_steps == 100

    def test_bool_parse_true(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "cfg.yaml", _full_yaml_body())
        cfg = load_config(path, overrides=["compile=true"])
        assert cfg.compile is True

    def test_bool_parse_false(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "cfg.yaml", _full_yaml_body())
        cfg = load_config(path, overrides=["compile=false"])
        assert cfg.compile is False

    def test_none_parse(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "cfg.yaml", _full_yaml_body())
        cfg = load_config(path, overrides=["log.wandb_project=null"])
        assert cfg.log.wandb_project is None

    def test_int_parse(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "cfg.yaml", _full_yaml_body())
        cfg = load_config(path, overrides=["seed=2024"])
        assert cfg.seed == 2024
        assert isinstance(cfg.seed, int)

    def test_float_parse(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "cfg.yaml", _full_yaml_body())
        cfg = load_config(path, overrides=["grad_clip=0.5"])
        assert cfg.grad_clip == 0.5

    def test_string_fallback(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "cfg.yaml", _full_yaml_body())
        cfg = load_config(path, overrides=["log.level=DEBUG"])
        assert cfg.log.level == "DEBUG"

    def test_multiple_overrides(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "cfg.yaml", _full_yaml_body())
        cfg = load_config(
            path,
            overrides=["seed=42", "compile=true", "grad_clip=2.0"],
        )
        assert cfg.seed == 42
        assert cfg.compile is True
        assert cfg.grad_clip == 2.0

    def test_missing_equals_raises(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "cfg.yaml", _full_yaml_body())
        with pytest.raises(ValueError, match="="):
            load_config(path, overrides=["seed999"])

    def test_cli_beats_child_yaml(self, tmp_path: Path) -> None:
        _write_yaml(tmp_path, "base.yaml", _full_yaml_body())
        _write_yaml(tmp_path, "child.yaml", "extends: base.yaml\nseed: 5\n")
        cfg = load_config(tmp_path / "child.yaml", overrides=["seed=99"])
        assert cfg.seed == 99

    def test_modern_arch_overrides(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "cfg.yaml", _full_yaml_body())
        cfg = load_config(
            path,
            overrides=[
                "model.n_kv_head=1",
                "model.init_policy=sky-ai",
                "model.hidden_multiple=8",
                "model.rope_theta=10000.0",
                "model.vocab_pad_multiple=256",
                "model.tie_weights=true",
                "model.logit_softcap=null",
            ],
        )
        assert cfg.model.n_kv_head == 1
        assert cfg.model.init_policy == "sky-ai"
        assert cfg.model.hidden_multiple == 8
        assert cfg.model.rope_theta == 10_000.0
        assert cfg.model.vocab_pad_multiple == 256
        assert cfg.model.tie_weights is True
        assert cfg.model.logit_softcap is None
