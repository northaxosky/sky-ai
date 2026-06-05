"""Sanity checks for env + project state, surfaced via `skyai doctor`"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import typer

from skyai.config.loader import load_config
from skyai.config.schema import RunConfig

Status = Literal["OK", "WARN", "FAIL"]
CheckResult = tuple[Status, str]
Check = Callable[[], CheckResult]

_STATUS_COLOR: dict[Status, str] = {
    "OK": typer.colors.GREEN,
    "WARN": typer.colors.YELLOW,
    "FAIL": typer.colors.RED,
}


def _check_python() -> CheckResult:
    v = sys.version_info
    msg = f"Python {v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < (3, 12):
        return ("FAIL", f"{msg} (need >= 3.12)")
    return ("OK", msg)


def _check_torch() -> CheckResult:
    try:
        import torch
    except ImportError as e:
        return ("FAIL", f"torch not importable: {e}")
    return ("OK", torch.__version__)


def _check_cuda() -> CheckResult:
    import torch

    if not torch.cuda.is_available():
        return ("WARN", "CUDA not available (CPU-only host)")
    n = torch.cuda.device_count()
    return ("OK", f"Available ({n} device{'s' if n != 1 else ''})")


def _check_gpu() -> CheckResult:
    import torch

    if not torch.cuda.is_available():
        return ("WARN", "skipped (no CUDA)")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    return ("OK", f"{name} (sm_{major}{minor})")


def _check_bf16() -> CheckResult:
    import torch

    if not torch.cuda.is_available():
        return ("WARN", "skipped (no CUDA)")
    if torch.cuda.is_bf16_supported():
        return ("OK", "Supported")
    return ("WARN", "Not supported on this GPU")


def _check_visible_devices() -> CheckResult:
    import torch

    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if raw is None:
        return (
            "OK",
            f"CUDA_VISIBLE_DEVICES not set ({n_visible} device{'s' if n_visible != 1 else ''} visible)",
        )
    requested = [s for s in raw.split(",") if s.strip()]
    if len(requested) != n_visible:
        return (
            "WARN",
            f"CUDA_VISIBLE_DEVICES={raw!r} requested {len(requested)} devices, torch sees {n_visible}",
        )
    return ("OK", f"CUDA_VISIBLE_DEVICES={raw!r}, {n_visible} visible")


def _check_ddp_env() -> CheckResult:
    keys = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    vars_ = {k: os.environ.get(k) for k in keys}
    set_vars = {k: v for k, v in vars_.items() if v is not None}
    if not set_vars:
        return ("OK", "Not in DDP mode (single process)")
    missing = [k for k, v in vars_.items() if v is None]
    if missing:
        return ("FAIL", f"DDP partially set: {set_vars}; missing {missing}")
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as e:
        return ("FAIL", f"DDP vars non-integer or missing: {vars_} ({e})")
    if rank >= world or local_rank >= world:
        return (
            "FAIL",
            f"DDP vars inconsistent: RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world}",
        )

    # missing values will hang at init rather than fail loudly.
    if world > 1:
        master_addr = os.environ.get("MASTER_ADDR")
        master_port = os.environ.get("MASTER_PORT")
        if not master_addr:
            return ("FAIL", f"WORLD_SIZE={world} but MASTER_ADDR not set; NCCL init will hang")
        if not master_port:
            return ("FAIL", f"WORLD_SIZE={world} but MASTER_PORT not set; NCCL init will hang")
        try:
            port = int(master_port)
        except ValueError:
            return ("FAIL", f"MASTER_PORT={master_port!r} is not an integer")
        if not (1024 <= port <= 65535):
            return ("WARN", f"MASTER_PORT={port} outside typical range [1024, 65535]")
    return ("OK", f"RANK={rank}/{world} LOCAL_RANK={local_rank}")


def _check_git() -> CheckResult:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return ("WARN", "Not a git repo (or git not installed)")
    dirty_out = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    state = "dirty" if dirty_out else "clean"
    return ("OK", f"{sha} ({state}) on {branch}")


def _check_wandb() -> CheckResult:
    try:
        import wandb  # noqa: F401
    except ImportError as e:
        return ("WARN", f"wandb not importable: {e}")
    if os.environ.get("WANDB_API_KEY"):
        return ("OK", "WANDB_API_KEY set")
    if os.environ.get("WANDB_MODE") == "offline":
        return ("OK", "WANDB_MODE=offline (no API key needed)")
    return ("WARN", "WANDB_API_KEY not set; online runs will prompt")


def _check_data_shards(cfg: RunConfig) -> CheckResult:
    root = cfg.data.root
    if not root.exists():
        return ("FAIL", f"{root}: does not exist")
    # mirror loader.py's glob: substring match, no .npy suffix required
    train = sorted(root.glob(f"*{cfg.data.train_split}*"))
    val = sorted(root.glob(f"*{cfg.data.val_split}*"))
    if not train:
        return ("FAIL", f"{root}: no '*{cfg.data.train_split}*' shards")
    if not val:
        return ("FAIL", f"{root}: no '*{cfg.data.val_split}*' shards")
    return ("OK", f"{len(train)} train + {len(val)} val shards in {root}")


def _check_checkpoint_dir(cfg: RunConfig) -> CheckResult:
    d = cfg.checkpoint.dir
    d.mkdir(parents=True, exist_ok=True)
    if not os.access(d, os.W_OK):
        return ("FAIL", f"{d}: not writable")
    free_gb = shutil.disk_usage(d).free / (1024**3)

    # Estimate checkpoint size from the GPT-2 parameter formula. AdamW saves
    # the params + (m, v) running stats in fp32, so plan ~16 bytes per param
    # (param + m + v + slack for buffers / serialization overhead).
    m = cfg.model
    n_params = (
        m.vocab_size * m.n_embed  # wte (tied with lm_head)
        + m.block_size * m.n_embed  # wpe
        + m.n_layer
        * (
            12 * m.n_embed * m.n_embed  # 4 attn + 8 mlp matrices
            + 13 * m.n_embed
        )  # LN params + biases (approx)
        + 2 * m.n_embed  # final LN
    )
    per_ckpt_gb = n_params * 16 / (1024**3)
    needed_gb = per_ckpt_gb * (cfg.checkpoint.keep_last_n + 1)  # +1 for best.pt

    if free_gb < needed_gb:
        return (
            "FAIL",
            f"{d}: {free_gb:.1f} GB free, need ~{needed_gb:.1f} GB "
            f"({cfg.checkpoint.keep_last_n + 1} x {per_ckpt_gb:.1f} GB)",
        )
    if free_gb < needed_gb * 1.5:
        return ("WARN", f"{d}: {free_gb:.1f} GB free, ~{needed_gb:.1f} GB planned; tight headroom")
    return ("OK", f"{d} writable, {free_gb:.1f} GB free, ~{needed_gb:.1f} GB planned")


def _check_world_size_divisibility(cfg: RunConfig) -> CheckResult:
    """When WORLD_SIZE is set, total_batch_size must divide cleanly into B*T*world_size."""
    world_str = os.environ.get("WORLD_SIZE")
    if world_str is None:
        return ("OK", "WORLD_SIZE not set; skipped")
    try:
        world = int(world_str)
    except ValueError:
        return ("FAIL", f"WORLD_SIZE={world_str!r} is not an integer")
    tokens_per_step = cfg.data.batch_size * cfg.model.block_size * world
    if cfg.total_batch_size % tokens_per_step != 0:
        return (
            "FAIL",
            f"total_batch_size ({cfg.total_batch_size}) is not divisible by "
            f"B*T*WORLD_SIZE ({cfg.data.batch_size}*{cfg.model.block_size}*{world} "
            f"= {tokens_per_step}); training will reject this at startup",
        )
    grad_accum = cfg.total_batch_size // tokens_per_step
    return ("OK", f"grad_accum = {grad_accum} at WORLD_SIZE={world}")


def _check_wandb_auth(cfg: RunConfig) -> CheckResult:
    """Verify WANDB_API_KEY is actually accepted by the wandb server."""
    if not cfg.log.wandb:
        return ("OK", "wandb disabled in config")
    if os.environ.get("WANDB_MODE") == "offline":
        return ("OK", "WANDB_MODE=offline; no remote auth needed")
    if not os.environ.get("WANDB_API_KEY"):
        return ("FAIL", "WANDB_API_KEY not set but cfg.log.wandb=true")
    try:
        import wandb
    except ImportError as e:
        return ("FAIL", f"wandb not importable: {e}")
    try:
        viewer = wandb.Api(timeout=10).viewer
        if viewer is None:
            return ("FAIL", "WANDB_API_KEY rejected (viewer is None)")
        username = getattr(viewer, "username", None) or "unknown"
        return ("OK", f"authenticated as {username}")
    except Exception as e:
        return ("FAIL", f"wandb auth probe failed: {type(e).__name__}: {e}")


_ENV_CHECKS: list[tuple[str, Check]] = [
    ("python", _check_python),
    ("torch", _check_torch),
    ("cuda", _check_cuda),
    ("gpu", _check_gpu),
    ("bf16", _check_bf16),
    ("visible_devices", _check_visible_devices),
    ("ddp_env", _check_ddp_env),
    ("git", _check_git),
    ("wandb", _check_wandb),
]


def _format_line(name: str, status: Status, message: str, name_width: int) -> str:
    badge = typer.style(f"[{status:<4}]", fg=_STATUS_COLOR[status], bold=True)
    return f"{badge} {name:<{name_width}} {message}"


def run_doctor(config_path: Path | None = None) -> int:
    """Run all checks, print results, return 0 if no FAIL else 1"""
    checks: list[tuple[str, Check]] = list(_ENV_CHECKS)
    if config_path is not None:
        try:
            cfg = load_config(config_path, overrides=[])
        except Exception as e:
            badge = typer.style("[FAIL]", fg=typer.colors.RED, bold=True)
            typer.echo(f"{badge} config              {config_path}: {e}")
            return 1
        checks.append(("data", lambda: _check_data_shards(cfg)))
        checks.append(("checkpoint_dir", lambda: _check_checkpoint_dir(cfg)))
        checks.append(("world_size", lambda: _check_world_size_divisibility(cfg)))
        checks.append(("wandb_auth", lambda: _check_wandb_auth(cfg)))

    name_width = max(len(name) for name, _ in checks)
    counts: dict[Status, int] = {"OK": 0, "WARN": 0, "FAIL": 0}
    for name, check in checks:
        try:
            status, message = check()
        except Exception as e:
            status, message = "FAIL", f"check raised: {type(e).__name__}: {e}"
        counts[status] += 1
        typer.echo(_format_line(name, status, message, name_width))

    typer.echo("")
    typer.echo(f"Summary: {counts['OK']} OK, {counts['WARN']} WARN, {counts['FAIL']} FAIL")
    return 0 if counts["FAIL"] == 0 else 1
