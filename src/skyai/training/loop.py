"""GPT-2 training loop"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import cast

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from skyai.checkpoint import load_checkpoint, restore_rng, save_checkpoint
from skyai.config.schema import RunConfig
from skyai.data.loader import DataLoader
from skyai.eval import run_evals
from skyai.generate import generate
from skyai.log import get_logger
from skyai.nn.model import GPT, GPTConfig
from skyai.training.optimizer import build_optimizer
from skyai.training.schedule import CosineSchedule
from skyai.wandb_logger import WandbLogger

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

def _build_model(cfg: RunConfig, device: str, dist_info: DistInfo) -> tuple[nn.Module, GPT]:
    """Build GPT, move to device, optionally compile + DDP"""
    gpt_cfg = GPTConfig(
        n_layer=cfg.model.n_layer, n_head=cfg.model.n_head, n_embed=cfg.model.n_embed,
        vocab_size=cfg.model.vocab_size, block_size=cfg.model.block_size,
    )
    raw_model = GPT(gpt_cfg)
    raw_model.to(device)

    forward_model: nn.Module = raw_model
    if cfg.compile:
        forward_model = cast(nn.Module, torch.compile(forward_model))
    if dist_info.is_ddp:
        forward_model = DDP(forward_model, device_ids=[dist_info.local_rank])
    return forward_model, raw_model

def _build_components(cfg: RunConfig, dist_info: DistInfo, device: str
) -> tuple[nn.Module, GPT, torch.optim.Optimizer, CosineSchedule, DataLoader, DataLoader, int]:
    """Build forward model, raw model, optimizer, schedule, train and val loader, grad_accum_steps"""
    grad_accum_steps = _compute_grad_accum(cfg, dist_info.world_size)
    forward_model, raw_model = _build_model(cfg, device, dist_info)

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    optimizer = build_optimizer(
        raw_model, learning_rate=cfg.schedule.max_lr, weight_decay=cfg.optim.weight_decay,
        betas=cfg.optim.betas, eps=cfg.optim.eps, device_type=device_type,
    )
    schedule = CosineSchedule(
        max_lr=cfg.schedule.max_lr, min_lr=cfg.schedule.min_lr,
        warmup_steps=cfg.schedule.warmup_steps, max_steps=cfg.schedule.max_steps
    )
    train_loader = DataLoader(
        data_root=cfg.data.root, split=cfg.data.train_split, batch_size=cfg.data.batch_size,
        block_size=cfg.model.block_size, rank=dist_info.rank, world_size=dist_info.world_size,
    )
    val_loader = DataLoader(
        data_root=cfg.data.root, split=cfg.data.val_split, batch_size=cfg.data.batch_size,
        block_size=cfg.model.block_size, rank=dist_info.rank, world_size=dist_info.world_size
    )
    return forward_model, raw_model, optimizer, schedule, train_loader, val_loader, grad_accum_steps

def _maybe_resume(cfg: RunConfig, raw_model: GPT, optimizer: torch.optim.Optimizer, train_loader: DataLoader
                  ) -> tuple[int, str | None]:
    """Restore from latest potentially"""
    latest = cfg.checkpoint.dir / "latest.json"
    if not latest.is_file():
        logger.info(f"No checkpoint to resume from at {latest}; starting fresh")
        return 0, None

    bundle = load_checkpoint(latest)
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
        optimizer: torch.optim.Optimizer,
        schedule: CosineSchedule,
        dist_info: DistInfo, *,
        step: int,
        grad_accum_steps: int,
        grad_clip: float,
        device: str,
        device_type: str,
        dtype: torch.dtype,
) -> tuple[float, float, float]:
    """One training step: grad-accum micro batches, all reduce, clip, optimize"""
    forward_model.train()
    optimizer.zero_grad()
    loss_accum = torch.zeros((), device=device)

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # DDP sync skip: only sync grads on the final micro step
        if dist_info.is_ddp:
            forward_model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # pyright: ignore

        with torch.autocast(device_type=device_type, dtype=dtype):
            _, loss = forward_model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    if dist_info.is_ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    grad_norm = torch.nn.utils.clip_grad_norm_(forward_model.parameters(), grad_clip)

    lr = schedule.lr_for(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    optimizer.step()

    if device_type =="cuda":
        torch.cuda.synchronize()

    return float(loss_accum.item()), float(grad_norm.item()), lr

def _run_val_loss(
    forward_model: nn.Module,
    val_loader: DataLoader,
    dist_info: DistInfo,
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
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=dtype):
                _, loss = forward_model(x, y)
            val_loss_accum += loss.detach() / val_steps

    if dist_info.is_ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    return float(val_loss_accum.item())

def _sample_text(
    raw_model: GPT,
    encoder: tiktoken.Encoding,
    *,
    prompt: str,
    n_samples: int,
    max_length: int,
    device: str,
    rank: int,
) -> list[str]:
    """Generate n_samples completions of `prompt` up to max_length tokens"""
    prompt_ids = encoder.encode(prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    x = x.repeat(n_samples, 1)

    rng = torch.Generator(device=device).manual_seed(42 + rank)
    out = generate(
        raw_model, x,
        max_new_tokens=max_length - x.size(1),
        max_context_len=raw_model.config.block_size,
        temperature=1.0,
        top_k=50,
        generator=rng,
    )
    return [encoder.decode(out[i].tolist()) for i in range(n_samples)]

def train(cfg: RunConfig, *, resume: bool = False) -> None:
    """End-to-end training loop, single-process or DDP via torchrun"""
    dist_info = _init_distributed()
    device = _resolve_device(dist_info.local_rank)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    dtype = _DTYPE_MAP[cfg.dtype]

    _set_seeds(cfg.seed)
    torch.set_float32_matmul_precision("high")

    (
        forward_model, raw_model, optimizer, schedule,
        train_loader, val_loader, grad_accum_steps,
    ) = _build_components(cfg, dist_info, device)

    start_step = 0
    wandb_run_id: str | None = None
    if resume:
        start_step, wandb_run_id = _maybe_resume(cfg, raw_model, optimizer, train_loader)

    wb = WandbLogger(
        cfg.log, rank=dist_info.rank, resume_id=wandb_run_id,
        config=cfg.model_dump(mode="json"),
    )
    encoder = tiktoken.get_encoding(cfg.model.tokenizer)
    last_val_loss: float | None = None

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

            # ---- Periodic eval (val loss + eval suite); dt below includes this time ----
            if step % cfg.eval.interval == 0 or last_step:
                val_loss = _run_val_loss(
                    forward_model, val_loader, dist_info,
                    val_steps=cfg.eval.val_steps, device=device,
                    device_type=device_type, dtype=dtype,
                )
                last_val_loss = val_loss

                eval_results = run_evals(
                    cfg.eval.evals, raw_model, # pyright: ignore
                    encoder=encoder, device=device,
                    rank=dist_info.rank, world_size=dist_info.world_size, dtype=dtype,
                )

                if dist_info.is_master:
                    logger.info(f"step {step}: val_loss={val_loss:.4f}")
                    eval_metrics: dict[str, float] = {"val/loss": val_loss}
                    for name, result in eval_results.items():
                        for metric, value in result.metrics.items():
                            eval_metrics[f"eval/{name}/{metric}"] = value
                            logger.info(f"step {step}: {name}/{metric}={value:.4f}")
                    wb.log_metrics(eval_metrics, step=step)

            # ---- Periodic sampling (skip step 0: untrained model produces noise) ----
            if step > 0 and (step % cfg.eval.interval == 0 or last_step):
                samples = _sample_text(
                    raw_model, encoder,
                    prompt=cfg.eval.sample_prompt,
                    n_samples=cfg.eval.sample_n, max_length=cfg.eval.sample_max_length,
                    device=device, rank=dist_info.rank,
                )
                if dist_info.is_master:
                    for i, s in enumerate(samples):
                        logger.info(f"step {step}: sample {i + 1}: {s}")

            # ---- Training step ----
            loss, grad_norm, lr = _run_train_step(
                forward_model, train_loader, optimizer, schedule, dist_info,
                step=step, grad_accum_steps=grad_accum_steps,
                grad_clip=cfg.grad_clip, device=device,
                device_type=device_type, dtype=dtype,
            )

            dt = time.time() - t0
            tokens_per_sec = cfg.total_batch_size / dt
            if dist_info.is_master:
                logger.info(
                    f"step {step}: loss={loss:.6f} lr={lr:.4e} "
                    f"norm={grad_norm:.4f} dt={dt * 1000:.1f}ms tok/s={tokens_per_sec:.0f}"
                )
                wb.log_metrics({
                    "train/loss": loss,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/dt_ms": dt * 1000,
                }, step=step)

            # ---- Periodic checkpoint (after training step is complete) ----
            if step > 0 and (step % cfg.checkpoint.every_n_steps == 0 or last_step):
                metrics_for_ckpt: dict[str, float] = {"train/loss": loss}
                if last_val_loss is not None:
                    metrics_for_ckpt["val_loss"] = last_val_loss
                save_checkpoint(
                    cfg.checkpoint.dir, step,
                    model=raw_model, optimizer=optimizer,
                    data_loader=train_loader, config=cfg,
                    metrics=metrics_for_ckpt,
                    wandb_run_id=wb.run_id,
                    rank=dist_info.rank,
                    keep_last_n=cfg.checkpoint.keep_last_n,
                    best_metric=cfg.checkpoint.best_metric,
                    best_direction=cfg.checkpoint.best_direction,
                )
    finally:
        wb.finish()
        if dist_info.is_ddp:
            destroy_process_group()