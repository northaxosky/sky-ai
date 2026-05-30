"""Pydantic schemas for SkyAI run configuration"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    n_layer: int = Field(gt=0, description="Number of transformer blocks")
    n_head: int = Field(gt=0, description="Number of attention heads")
    n_embed: int = Field(gt=0, description="Hidden dim, must be divisible by n_head")
    vocab_size: int = Field(gt=0, description="Tokenizer vocabulary size")
    block_size: int = Field(gt=0, description="Max sequence length / context window")
    tokenizer: str = Field(default="gpt2", description="tiktoken encoding name; must match the model's vocab")

    @model_validator(mode="after")
    def _embed_divisible_by_head(self) -> ModelConfig:
        if self.n_embed % self.n_head != 0:
            raise ValueError(f"n_embed ({self.n_embed}) must be divisible by n_head ({self.n_head})")
        return self
    

class DataConfig(BaseModel):
    root: Path = Field(description="Directory containing token shards")
    train_split: str = "train"
    val_split: str = "val"
    batch_size: int = Field(gt=0, description="Per-rank micro batch size")


class OptimConfig(BaseModel):
    weight_decay: float = Field(ge=0.0)
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = Field(default=1e-8, gt=0.0)


class ScheduleConfig(BaseModel):
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
                f"warmup_steps ({self.warmup_steps}) must"
                f"be <= max_steps ({self.max_steps})"
                )
        return self
    
class EvalConfig(BaseModel):
    interval: int = Field(gt=0, description="Run eval every n training steps")
    val_steps: int = Field(default=20, gt=0, description="Microbatches per val pass")
    evals: list[Literal["hellaswag", "lambada"]] = Field(
        default_factory=lambda: ["hellaswag"],
        description="Names of evals to run, order preserved"
    )
    sample_prompt: str = Field(
        default="Hello, I'm a language model,",
        description="Prompt fed to the periodic sampler",
    )
    sample_n: int = Field(default=4, gt=0, description="Number of completions per sample step")
    sample_max_length: int = Field(default=32, gt=0, description="Max total length (prompt + new tokens)")


class LogConfig(BaseModel):
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
    

class CheckpointConfig(BaseModel):
    dir: Path = Path("checkpoints")
    every_n_steps: int = Field(default=1000, gt=0)
    keep_last_n: int = Field(default=3, ge=1, description="Rolling window size for step_*.pt")
    best_metric: str | None = Field(default="val_loss", description="Name of metric to track for best.pt")
    best_direction: Literal["min", "max"] = "min"

class RunConfig(BaseModel):
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
    checkpoint: CheckpointConfig = CheckpointConfig()

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _total_batch_divides_microbatch(self) -> RunConfig:
        micro = self.data.batch_size * self.model.block_size
        if self.total_batch_size % micro != 0:
            raise ValueError(
                f"total_batch_size ({self.total_batch_size}) must be divisible by data.batch_size"
                f" * model.block_size ({micro}); otherwise grad_accum is not an int"
            )
        return self

        