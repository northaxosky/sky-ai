"""GPT-2 training loop"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from harness.checkpoint import load_checkpoint, restore_rng, save_checkpoint
from harness.config.schema import ModelConfig, RecoveryConfig, RunConfig
from harness.data.loader import DataLoader
from harness.eval import run_evals
from harness.log import get_logger
from harness.sample import sample
from harness.training.optimizer import build_optimizer
from harness.training.profiler import Profiler
from harness.training.recovery import (
    NonFiniteGradError,
    detect_non_finite_grad,
    diagnose_oom,
    is_oom_error,
)
from harness.training.schedule import CosineSchedule, WarmupStableDecaySchedule
from harness.wandb_logger import WandbLogger
from skyai.model import GPT, GPTConfig

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

logger = get_logger(__name__)


@dataclass(frozen=True)
class DistInfo:
    """Distributed runtime info"""

    rank: int
    local_rank: int
    world_size: int

    @property
    def is_ddp(self) -> bool:
        return self.world_size > 1

    @property
    def is_master(self) -> bool:
        return self.rank == 0


def _init_distributed() -> DistInfo:
    """Detect torchrun env, init process group when WORLD_SIZE > 1"""
    if "RANK" not in os.environ:
        return DistInfo(rank=0, local_rank=0, world_size=1)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requested (RANK is set) but CUDA is not available")

    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    logger.info(f"DDP initialized: {rank=}, {local_rank=}, {world_size=}")

    return DistInfo(rank=rank, local_rank=local_rank, world_size=world_size)


def _resolve_device(local_rank: int) -> str:
    """cuda:LOCAL_RANK when cuda is avail, else cpu"""
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"


def _compute_grad_accum(cfg: RunConfig, world_size: int) -> int:
    """Validate total_batch_size divides cleanly into (B * T * world_size), return accum count"""
    tokens_per_step = cfg.data.batch_size * cfg.model.block_size * world_size
    if cfg.total_batch_size % tokens_per_step != 0:
        raise ValueError(
            f"total_batch_size ({cfg.total_batch_size}) must be divisible by B * T * world_size "
            f"({cfg.data.batch_size} * {cfg.model.block_size} * {world_size} = {tokens_per_step})"
        )
    return cfg.total_batch_size // tokens_per_step


def _set_seeds(seed: int) -> None:
    """Seed python, numpy, torch cpu and Cuda"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_gpt_config(model_cfg: ModelConfig) -> GPTConfig:
    """Translate run config into GPT's logical-vocab config."""
    return GPTConfig(
        init_policy=model_cfg.init_policy,
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head,
        n_kv_head=model_cfg.n_kv_head,
        n_embed=model_cfg.n_embed,
        hidden_multiple=model_cfg.hidden_multiple,
        vocab_size=model_cfg.tokenizer_vocab_size,
        vocab_pad_multiple=model_cfg.vocab_pad_multiple,
        block_size=model_cfg.block_size,
        rope_theta=model_cfg.rope_theta,
        tie_weights=model_cfg.tie_weights,
        logit_softcap=model_cfg.logit_softcap,
    )


def build_model(model_cfg: ModelConfig) -> nn.Module:
    """Construct the model for the configured architecture family"""
    if model_cfg.family == "modern":
        return GPT(build_gpt_config(model_cfg))
    if model_cfg.family == "gpt2":
        from gpt.model import GPT as GPT2
        from gpt.model import GPTConfig as Gpt2Config

        return GPT2(
            Gpt2Config(
                n_layer=model_cfg.n_layer,
                n_head=model_cfg.n_head,
                n_embd=model_cfg.n_embed,
                vocab_size=model_cfg.vocab_size,
                block_size=model_cfg.block_size,
            )
        )

    raise ValueError(f"Unknown model family {model_cfg.family!r}")


def _build_model(cfg: RunConfig, device: str, dist_info: DistInfo) -> tuple[nn.Module, nn.Module]:
    """Build GPT, move to device, optionally compile + DDP"""
    raw_model = build_model(cfg.model)
    raw_model.to(device)

    forward_model: nn.Module = raw_model
    if cfg.compile:
        forward_model = cast(nn.Module, torch.compile(forward_model))
    if dist_info.is_ddp:
        forward_model = DDP(forward_model, device_ids=[dist_info.local_rank])
    return forward_model, raw_model


def _build_components(
    cfg: RunConfig, dist_info: DistInfo, device: str
) -> tuple[nn.Module, nn.Module, Any, Any, DataLoader, DataLoader, int]:
    """Build forward model, raw model, optimizer, schedule, train and val loader, grad_accum_steps"""
    grad_accum_steps = _compute_grad_accum(cfg, dist_info.world_size)
    forward_model, raw_model = _build_model(cfg, device, dist_info)

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    optimizer = build_optimizer(
        raw_model,
        optimizer_type=cfg.optim.type,
        learning_rate=cfg.schedule.max_lr,
        weight_decay=cfg.optim.weight_decay,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
        device_type=device_type,
        total_batch_size=cfg.total_batch_size,
        embedding_lr=cfg.optim.embedding_lr,
        unembedding_lr=cfg.optim.unembedding_lr,
        matrix_lr=cfg.optim.matrix_lr,
        muon_momentum=cfg.optim.muon_momentum,
        muon_beta2=cfg.optim.muon_beta2,
        muon_ns_steps=cfg.optim.muon_ns_steps,
    )
    if cfg.schedule.type == "cosine":
        schedule = CosineSchedule(
            max_lr=cfg.schedule.max_lr,
            min_lr=cfg.schedule.min_lr,
            warmup_steps=cfg.schedule.warmup_steps,
            max_steps=cfg.schedule.max_steps,
        )
    elif cfg.schedule.type == "warmup-stable-decay":
        schedule = WarmupStableDecaySchedule(
            warmup_steps=cfg.schedule.warmup_steps,
            max_steps=cfg.schedule.max_steps,
            warmdown_ratio=cfg.schedule.warmdown_ratio,
            final_lr_frac=cfg.schedule.final_lr_frac,
        )
    else:
        raise ValueError(f"Unknown schedule type {cfg.schedule.type!r}")
    train_loader = DataLoader(
        data_root=cfg.data.root,
        split=cfg.data.train_split,
        batch_size=cfg.data.batch_size,
        block_size=cfg.model.block_size,
        rank=dist_info.rank,
        world_size=dist_info.world_size,
    )
    val_loader = DataLoader(
        data_root=cfg.data.root,
        split=cfg.data.val_split,
        batch_size=cfg.data.batch_size,
        block_size=cfg.model.block_size,
        rank=dist_info.rank,
        world_size=dist_info.world_size,
    )
    return forward_model, raw_model, optimizer, schedule, train_loader, val_loader, grad_accum_steps


_RESUME_CRITICAL_FIELDS: list[tuple[str, Any]] = [
    ("model.init_policy", lambda c: c.model.init_policy),
    ("total_batch_size", lambda c: c.total_batch_size),
    ("model.n_layer", lambda c: c.model.n_layer),
    ("model.n_head", lambda c: c.model.n_head),
    ("model.n_kv_head", lambda c: c.model.n_kv_head),
    ("model.n_embed", lambda c: c.model.n_embed),
    ("model.hidden_multiple", lambda c: c.model.hidden_multiple),
    ("model.vocab_size", lambda c: c.model.vocab_size),
    ("model.vocab_pad_multiple", lambda c: c.model.vocab_pad_multiple),
    ("model.block_size", lambda c: c.model.block_size),
    ("model.rope_theta", lambda c: c.model.rope_theta),
    ("model.tie_weights", lambda c: c.model.tie_weights),
    ("model.logit_softcap", lambda c: c.model.logit_softcap),
    ("model.tokenizer", lambda c: c.model.tokenizer),
    ("data.batch_size", lambda c: c.data.batch_size),
]


def _assert_resume_compatible(current: RunConfig, saved: RunConfig) -> None:
    """Hard-fail when resume-critical fields have drifted since the checkpoint.

    These fields define the optimization trajectory (token budget per step,
    architecture, micro-batch geometry). Resuming with different values silently
    yields a different training run.
    """
    mismatches: list[str] = []
    for name, getter in _RESUME_CRITICAL_FIELDS:
        cur, old = getter(current), getter(saved)
        if cur != old:
            mismatches.append(f"  {name}: checkpoint={old!r}, current={cur!r}")
    if mismatches:
        raise RuntimeError(
            "Cannot resume: config has drifted on resume-critical fields:\n"
            + "\n".join(mismatches)
            + "\nResuming with different values would not be the same run; "
            "either revert the config or start a fresh run."
        )


def _maybe_resume(
    cfg: RunConfig, raw_model: nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader
) -> tuple[int, str | None]:
    """Restore from latest potentially"""
    latest = cfg.checkpoint.dir / "latest.json"
    if not latest.is_file():
        logger.info(f"No checkpoint to resume from at {latest}; starting fresh")
        return 0, None

    bundle = load_checkpoint(latest)
    _assert_resume_compatible(cfg, bundle.config)
    raw_model.load_state_dict(bundle.model_state)
    optimizer.load_state_dict(bundle.optim_state)
    train_loader.load_state_dict(bundle.data_loader_state)

    restore_rng(bundle.rng_state)
    start_step = bundle.step + 1
    logger.info(f"Resumed from {latest}: ckpt step={bundle.step}, next step={start_step}")
    return start_step, bundle.wandb_run_id


def _run_train_step(
    forward_model: nn.Module,
    train_loader: DataLoader,
    optimizer: Any,
    schedule: CosineSchedule | WarmupStableDecaySchedule,
    dist_info: DistInfo,
    profiler: Profiler,
    recovery: RecoveryConfig,
    *,
    step: int,
    grad_accum_steps: int,
    grad_clip: float | None,
    device: str,
    device_type: str,
    dtype: torch.dtype,
) -> tuple[float, float, float]:
    """One training step: grad-accum micro batches, all reduce, clip, optimize"""
    forward_model.train()
    optimizer.zero_grad()
    loss_accum = torch.zeros((), device=device)

    for micro_step in range(grad_accum_steps):
        with profiler.region("data"):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

        # DDP sync skip: only sync grads on the final micro step
        if dist_info.is_ddp:
            forward_model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

        with profiler.region("forward"):
            with torch.autocast(device_type=device_type, dtype=dtype):
                _, loss = forward_model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()

        with profiler.region("backward"):
            loss.backward()

    if dist_info.is_ddp:
        with profiler.region("ddp_loss_reduce"):
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    with profiler.region("grad_clip"):
        max_norm = float("inf") if grad_clip is None else grad_clip
        grad_norm = torch.nn.utils.clip_grad_norm_(forward_model.parameters(), max_norm)

    bad_param = detect_non_finite_grad(forward_model)
    if bad_param is not None:
        msg = (
            f"Non-finite gradient at step {step} in '{bad_param}'. "
            f"loss={float(loss_accum.item()):.4f}, "
            f"grad_norm={float(grad_norm.item()):.4f}"
        )
        if recovery.nan_grad_action == "halt":
            raise NonFiniteGradError(msg)
        logger.warning(f"SKIP {msg}")
        optimizer.zero_grad()
        return float("nan"), float("nan"), schedule.lr_for(step)

    if isinstance(schedule, WarmupStableDecaySchedule):
        lr_mult = schedule.multiplier_for(step)
        for pg in optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * lr_mult
            if pg.get("optimizer_type") == "muon":
                pg["momentum"] = schedule.muon_momentum_for(step)
                pg["weight_decay"] = schedule.muon_weight_decay_for(
                    step, base_weight_decay=pg["initial_weight_decay"]
                )
        lr = max(pg["lr"] for pg in optimizer.param_groups)
    else:
        lr = schedule.lr_for(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    with profiler.region("optimizer"):
        optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize()

    return float(loss_accum.item()), float(grad_norm.item()), lr


def _run_val_loss(
    forward_model: nn.Module,
    val_loader: DataLoader,
    dist_info: DistInfo,
    profiler: Profiler,
    *,
    val_steps: int,
    device: str,
    device_type: str,
    dtype: torch.dtype,
) -> float:
    """Reset val_loader, score val_steps micro-batches in eval mode; return averaged loss"""
    forward_model.eval()
    val_loader.reset()

    val_loss_accum = torch.zeros((), device=device)
    with torch.no_grad():
        for _ in range(val_steps):
            with profiler.region("val_data"):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
            with profiler.region("val_forward"):
                with torch.autocast(device_type=device_type, dtype=dtype):
                    _, loss = forward_model(x, y)
                val_loss_accum += loss.detach() / val_steps

    if dist_info.is_ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    return float(val_loss_accum.item())


def _format_lr_groups(optimizer: Any) -> str:
    """Compact LR summary that keeps split optimizers interpretable."""
    groups = getattr(optimizer, "param_groups", [])
    if not groups:
        return "lr=n/a"
    if len(groups) == 1:
        return f"lr={groups[0]['lr']:.4e}"

    max_lr = max(group["lr"] for group in groups)
    parts = [f"lr/max={max_lr:.4e}"]
    wanted = ("embed", "lm_head")
    for name in wanted:
        group = next((g for g in groups if g.get("name") == name), None)
        if group is not None:
            parts.append(f"lr/{name}={group['lr']:.4e}")
    muon_lrs = [g["lr"] for g in groups if g.get("optimizer_type") == "muon"]
    if muon_lrs:
        parts.append(f"lr/muon={max(muon_lrs):.4e}")
    return " ".join(parts)


def _build_metrics(
    model: nn.Module,
    step_losses: list[float],
    final_val_loss: float | None,
    sample_text: list[str] | None,
) -> dict[str, Any]:
    """Bundle the runs quantitative output for golden test comparison"""
    # Stream per-parameter so we don't concatenate every weight into a single
    # multi-GB tensor at 1.5B scale. Cast to float32 for the square reduction
    # to avoid bf16 overflow on large norms.
    total_sum = 0.0
    total_sq = 0.0
    n_params = 0
    for p in model.parameters():
        pd = p.detach()
        total_sum += float(pd.sum(dtype=torch.float32).item())
        total_sq += float((pd.to(torch.float32) ** 2).sum().item())
        n_params += pd.numel()
    return {
        "step_losses": step_losses,
        "final_val_loss": final_val_loss,
        "sample_text": sample_text,
        "param_checksum": {
            "sum": total_sum,
            "norm": total_sq**0.5,
            "n_params": n_params,
        },
    }


def train(cfg: RunConfig, *, resume: bool = False) -> dict[str, Any] | None:
    """End-to-end training loop, single-process or DDP via torchrun"""
    dist_info = _init_distributed()
    device = _resolve_device(dist_info.local_rank)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    dtype = _DTYPE_MAP[cfg.dtype]

    _set_seeds(cfg.seed)
    torch.set_float32_matmul_precision("high")

    (
        forward_model,
        raw_model,
        optimizer,
        schedule,
        train_loader,
        val_loader,
        grad_accum_steps,
    ) = _build_components(cfg, dist_info, device)

    start_step = 0
    wandb_run_id: str | None = None
    if resume:
        start_step, wandb_run_id = _maybe_resume(cfg, raw_model, optimizer, train_loader)

    wb = WandbLogger(
        cfg.log,
        rank=dist_info.rank,
        resume_id=wandb_run_id,
        config=cfg.model_dump(mode="json"),
    )
    encoder = tiktoken.get_encoding(cfg.model.tokenizer)
    last_val_loss: float | None = None
    last_samples: list[str] | None = None
    step_losses: list[float] = []
    profiler = Profiler(cfg.profiling, device=device, rank=dist_info.rank)

    if dist_info.is_master:
        logger.info(
            f"train: start_step={start_step}, max_steps={cfg.schedule.max_steps}, "
            f"world_size={dist_info.world_size}, grad_accum_steps={grad_accum_steps}, "
            f"total_batch_size={cfg.total_batch_size}, device={device}, dtype={cfg.dtype}"
        )

    try:
        for step in range(start_step, cfg.schedule.max_steps):
            t0 = time.time()
            last_step = step == cfg.schedule.max_steps - 1

            # ---- Training step ----
            loss, grad_norm, lr = _run_train_step(
                forward_model,
                train_loader,
                optimizer,
                schedule,
                dist_info,
                profiler,
                cfg.recovery,
                step=step,
                grad_accum_steps=grad_accum_steps,
                grad_clip=cfg.grad_clip,
                device=device,
                device_type=device_type,
                dtype=dtype,
            )

            # ---- Periodic eval (val loss + eval suite) ----
            if step % cfg.eval.interval == 0 or last_step:
                with profiler.region("eval_val_loss"):
                    val_loss = _run_val_loss(
                        forward_model,
                        val_loader,
                        dist_info,
                        profiler,
                        val_steps=cfg.eval.val_steps,
                        device=device,
                        device_type=device_type,
                        dtype=dtype,
                    )
                last_val_loss = val_loss

                with profiler.region("eval_suite"):
                    eval_results = run_evals(
                        cfg.eval.evals,
                        raw_model,
                        encoder=encoder,
                        device=device,
                        rank=dist_info.rank,
                        world_size=dist_info.world_size,
                        dtype=dtype,
                    )

                if dist_info.is_master:
                    logger.info(f"step {step}: val_loss={val_loss:.4f}")
                    eval_metrics: dict[str, float] = {"val/loss": val_loss}
                    for name, result in eval_results.items():
                        for metric, value in result.metrics.items():
                            eval_metrics[f"eval/{name}/{metric}"] = value
                            logger.info(f"step {step}: {name}/{metric}={value:.4f}")
                    wb.log_metrics(eval_metrics, step=step)

            # ---- Periodic sampling (skip step 0: the first update is still noise) ----
            if step > 0 and (step % cfg.eval.interval == 0 or last_step):
                with profiler.region("sample"):
                    rng = torch.Generator(device=device).manual_seed(42 + dist_info.rank)
                    samples = sample(
                        raw_model,
                        encoder,
                        cfg.eval.sample_prompt,
                        n_samples=cfg.eval.sample_n,
                        max_length=cfg.eval.sample_max_length,
                        device=device,
                        generator=rng,
                    )
                if dist_info.is_master:
                    last_samples = samples
                    for i, s in enumerate(samples):
                        logger.info(f"Step {step}: sample {i + 1}: {s}")

            dt = time.time() - t0
            tokens_per_sec = cfg.total_batch_size / dt
            if dist_info.is_master:
                if math.isfinite(loss):
                    lr_summary = _format_lr_groups(optimizer)
                    logger.info(
                        f"step {step}: loss={loss:.6f} {lr_summary} "
                        f"norm={grad_norm:.4f} dt={dt * 1000:.1f}ms tok/s={tokens_per_sec:.0f}"
                    )
                    wb.log_metrics(
                        {
                            "train/loss": loss,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/dt_ms": dt * 1000,
                        },
                        step=step,
                    )
                    step_losses.append(loss)
                else:
                    logger.warning(f"step {step}: SKIPPED (non-finite grad)")
                    wb.log_metrics({"train/skipped": 1.0}, step=step)

            # ---- Periodic checkpoint (after training step is complete) ----
            if step > 0 and (step % cfg.checkpoint.every_n_steps == 0 or last_step):
                with profiler.region("checkpoint"):
                    metrics_for_ckpt: dict[str, float] = {"train/loss": loss}
                    if last_val_loss is not None:
                        metrics_for_ckpt["val_loss"] = last_val_loss
                    save_checkpoint(
                        cfg.checkpoint.dir,
                        step,
                        model=raw_model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        config=cfg,
                        metrics=metrics_for_ckpt,
                        wandb_run_id=wb.run_id,
                        rank=dist_info.rank,
                        keep_last_n=cfg.checkpoint.keep_last_n,
                        best_metric=cfg.checkpoint.best_metric,
                        best_direction=cfg.checkpoint.best_direction,
                    )
            if profiler.should_log(step):
                wb.log_metrics(profiler.log_and_reset(step), step=step)

        if dist_info.is_master:
            metrics = _build_metrics(raw_model, step_losses, last_val_loss, last_samples)
        else:
            metrics = None

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not is_oom_error(e):
            raise
        current_step = step if "step" in locals() else start_step
        if cfg.recovery.oom_dump_diagnostics:
            diagnose_oom(e, step=current_step, cfg=cfg, world_size=dist_info.world_size)
        raise

    finally:
        profiler.flush(step if "step" in locals() else 0)
        wb.finish()
        if dist_info.is_ddp:
            destroy_process_group()
    return metrics
