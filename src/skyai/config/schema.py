"""Pydantic schemas for SkyAI run configuration"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, model_validator

# tiktoken's get_encoding hits disk/network on first call; cache per name.
_TOKENIZER_VOCAB_CACHE: dict[str, int] = {}


def _tokenizer_vocab(name: str) -> int:
    if name not in _TOKENIZER_VOCAB_CACHE:
        try:
            enc = tiktoken.get_encoding(name)
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"Unknown tokenizer '{name}'; "
                f"must be a tiktoken encoding name (e.g. gpt2, cl100k_base, o200k_base)"
            ) from e
        _TOKENIZER_VOCAB_CACHE[name] = enc.n_vocab
    return _TOKENIZER_VOCAB_CACHE[name]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    init_policy: Literal["gpt2", "sky-ai"] = Field(
        default="gpt2", description="Weight initialization policy"
    )
    n_layer: int = Field(gt=0, description="Number of transformer blocks")
    n_head: int = Field(gt=0, description="Number of attention heads")
    n_kv_head: int | None = Field(default=None, gt=0, description="Number of KV heads for GQA")
    n_embed: int = Field(gt=0, description="Hidden dim, must be divisible by n_head")
    hidden_multiple: int = Field(default=4, gt=0, description="MLP hidden size multiplier")
    vocab_size: int = Field(gt=0, description="Tokenizer vocabulary size")
    vocab_pad_multiple: int = Field(
        default=128, gt=0, description="Internal vocab padding multiple"
    )
    block_size: int = Field(gt=0, description="Max sequence length / context window")
    rope_theta: float = Field(default=100_000.0, gt=0.0, description="RoPE frequency base")
    tokenizer: str = Field(
        default="gpt2", description="tiktoken encoding name; must match the model's vocab"
    )
    tie_weights: bool = Field(default=False, description="Tie token embedding and lm_head weights")
    logit_softcap: float | None = Field(
        default=15.0, gt=0.0, description="Optional tanh logit softcap"
    )

    @property
    def tokenizer_vocab_size(self) -> int:
        return _tokenizer_vocab(self.tokenizer)

    @model_validator(mode="after")
    def _embed_divisible_by_head(self) -> ModelConfig:
        if self.n_embed % self.n_head != 0:
            raise ValueError(
                f"n_embed ({self.n_embed}) must be divisible by n_head ({self.n_head})"
            )
        head_dim = self.n_embed // self.n_head
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")
        return self

    @model_validator(mode="after")
    def _kv_heads_divide_query_heads(self) -> ModelConfig:
        if self.n_kv_head is None:
            return self
        if self.n_kv_head > self.n_head:
            raise ValueError(f"n_kv_head ({self.n_kv_head}) must be <= n_head ({self.n_head})")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
            )
        return self

    @model_validator(mode="after")
    def _vocab_matches_tokenizer(self) -> ModelConfig:
        encoder_vocab = _tokenizer_vocab(self.tokenizer)
        if self.vocab_size < encoder_vocab:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) is smaller than tokenizer "
                f"'{self.tokenizer}' n_vocab ({encoder_vocab}); valid token ids "
                f"would be out of range of the lm_head"
            )
        max_vocab_size = (
            (encoder_vocab + self.vocab_pad_multiple - 1) // self.vocab_pad_multiple
        ) * self.vocab_pad_multiple
        if self.vocab_size > max_vocab_size:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) exceeds tokenizer '{self.tokenizer}' "
                f"n_vocab ({encoder_vocab}) padded to vocab_pad_multiple "
                f"{self.vocab_pad_multiple} ({max_vocab_size})"
            )
        return self


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path = Field(description="Directory containing token shards")
    train_split: str = "train"
    val_split: str = "val"
    batch_size: int = Field(gt=0, description="Per-rank micro batch size")


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weight_decay: float = Field(ge=0.0)
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = Field(default=1e-8, gt=0.0)


class ScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_lr: float = Field(gt=0.0)
    min_lr: float = Field(ge=0.0)
    warmup_steps: int = Field(ge=0)
    max_steps: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordering(self) -> ScheduleConfig:
        if self.min_lr > self.max_lr:
            raise ValueError(f"min_lr ({self.min_lr}) must be <= max_lr ({self.max_lr})")
        if self.warmup_steps > self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) mustbe <= max_steps ({self.max_steps})"
            )
        return self


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interval: int = Field(gt=0, description="Run eval every n training steps")
    val_steps: int = Field(default=20, gt=0, description="Microbatches per val pass")
    evals: list[Literal["hellaswag", "lambada"]] = Field(
        default_factory=lambda: ["hellaswag"], description="Names of evals to run, order preserved"
    )
    sample_prompt: str = Field(
        default="Hello, I'm a language model,",
        description="Prompt fed to the periodic sampler",
    )
    sample_n: int = Field(default=4, gt=0, description="Number of completions per sample step")
    sample_max_length: int = Field(
        default=32, gt=0, description="Max total length (prompt + new tokens)"
    )


class LogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: Path = Path("logs")
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None

    @model_validator(mode="after")
    def _wandb_requires_project(self) -> LogConfig:
        if self.wandb and not self.wandb_project:
            raise ValueError("wandb=true requires wandb_project to be set")
        return self


class ProfilingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    log_every: int = Field(default=100, gt=0, description="Emit breakdown every N steps")
    cuda_sync: bool = Field(
        default=False, description="Force torch.cuda.synchronize for precise timing (expensive)"
    )


class RecoveryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nan_grad_action: Literal["halt", "skip"] = Field(
        default="halt", description="What to do when a parameter gradient is NaN/Inf"
    )
    oom_dump_diagnostics: bool = Field(
        default=True,
        description="On torch.cuda.OutOfMemoryError, log VRAM stats and batch geometry",
    )


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: Path = Path("checkpoints")
    every_n_steps: int = Field(default=1000, gt=0)
    keep_last_n: int = Field(default=3, ge=1, description="Rolling window size for step_*.pt")
    best_metric: str | None = Field(
        default="val_loss", description="Name of metric to track for best.pt"
    )
    best_direction: Literal["min", "max"] = "min"


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    compile: bool = False
    grad_clip: float = Field(default=1.0, gt=0.0)
    total_batch_size: int = Field(gt=0, description="Effective batch size in TOKENS")

    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    schedule: ScheduleConfig
    eval: EvalConfig
    log: LogConfig = LogConfig()
    profiling: ProfilingConfig = ProfilingConfig()
    recovery: RecoveryConfig = RecoveryConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()

    @model_validator(mode="after")
    def _total_batch_divides_microbatch(self) -> RunConfig:
        micro = self.data.batch_size * self.model.block_size
        if self.total_batch_size % micro != 0:
            raise ValueError(
                f"total_batch_size ({self.total_batch_size}) must be divisible by data.batch_size"
                f" * model.block_size ({micro}); otherwise grad_accum is not an int"
            )
        return self
