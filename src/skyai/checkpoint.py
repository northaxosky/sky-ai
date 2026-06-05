"""Checkpoint save/load with sidecar manifests for artifact lineage"""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from skyai.config.schema import RunConfig
from skyai.log import get_logger

logger = get_logger(__name__)

_BUNDLE_GLOB = "step_*.pt"
_BEST_BUNDLE_NAME = "best.pt"
_BEST_MANIFEST_NAME = "best.json"
_LATEST_NAME = "latest.json"


class _StateDictful(Protocol):
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


@dataclass
class CheckpointBundle:
    step: int
    model_state: dict[str, Any]
    optim_state: dict[str, Any]
    data_loader_state: dict[str, Any]
    rng_state: dict[str, Any]
    wandb_run_id: str | None
    config: RunConfig
    manifest: dict[str, Any]


def save_checkpoint(
    dir: str | Path,
    step: int,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: _StateDictful,
    config: RunConfig,
    metrics: dict[str, float],
    wandb_run_id: str | None = None,
    rank: int = 0,
    keep_last_n: int = 3,
    best_metric: str | None = "val_loss",
    best_direction: str = "min",
) -> Path | None:
    """ "Automatically write step_n.pt + .json, update latest.json & best.*"""
    # Barrier after rank 0 finishes writing so non-zero ranks can't race ahead
    # and hit the next NCCL collective mid-write (timeout risk for large XL
    # checkpoints).
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    try:
        if rank != 0:
            return None

        root = Path(dir)
        root.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "data_loader": data_loader.state_dict(),
            "rng": _rng_state(),
            "step": step,
            "wandb_run_id": wandb_run_id,
        }
        manifest = _build_manifest(step=step, config=config, metrics=metrics)

        bundle_path = root / f"step_{step:08d}.pt"
        manifest_path = root / f"step_{step:08d}.json"

        _atomic_save_torch(bundle, bundle_path)
        _atomic_write_json(manifest, manifest_path)
        _atomic_write_json({"step": step}, root / _LATEST_NAME)

        logger.info("Saved checkpoint step=%d -> %s", step, bundle_path)

        _maybe_update_best(
            root=root,
            bundle_path=bundle_path,
            manifest=manifest,
            metrics=metrics,
            best_metric=best_metric,
            best_direction=best_direction,
        )
        _rotate(root, keep_last_n)
        return bundle_path
    finally:
        if is_distributed:
            torch.distributed.barrier()


def load_checkpoint(path: str | Path) -> CheckpointBundle:
    """Load a checkpoint pair"""
    p = Path(path)
    if p.is_dir():
        latest = p / _LATEST_NAME
        if not latest.is_file():
            raise FileNotFoundError(f"No {_LATEST_NAME} in {p}, pass an explicit checkpoint path")
        p = latest

    if p.name == _LATEST_NAME:
        pointer = json.loads(p.read_text())
        step = pointer["step"]
        bundle_path = p.parent / f"step_{step:08d}.pt"
        manifest_path = p.parent / f"step_{step:08d}.json"
    elif p.name == _BEST_MANIFEST_NAME:
        bundle_path = p.parent / _BEST_BUNDLE_NAME
        manifest_path = p
    elif p.suffix == ".pt":
        bundle_path = p
        manifest_path = p.with_suffix(".json")
    elif p.suffix == ".json":
        manifest_path = p
        bundle_path = p.with_suffix(".pt")
    else:
        raise ValueError(f"Cannot resolve checkpoint from {p}")

    if not bundle_path.is_file():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)

    try:
        cfg = RunConfig.model_validate(manifest["config"])
    except Exception as e:
        raise RuntimeError(
            f"Manifest config in {manifest_path} failed schema validation: {e}"
        ) from e

    return CheckpointBundle(
        step=bundle["step"],
        model_state=bundle["model"],
        optim_state=bundle["optimizer"],
        data_loader_state=bundle["data_loader"],
        rng_state=bundle["rng"],
        wandb_run_id=bundle.get("wandb_run_id"),
        config=cfg,
        manifest=manifest,
    )


def list_checkpoints(dir: str | Path) -> list[Path]:
    """Sorted list of step_*.pt paths"""
    root = Path(dir)
    if not root.is_dir():
        return []
    return sorted(root.glob(_BUNDLE_GLOB), key=lambda p: int(p.stem.split("_")[1]))


def latest_checkpoint(dir: str | Path) -> Path | None:
    """Path to the most recent step_*.pt"""
    paths = list_checkpoints(dir)
    return paths[-1] if paths else None


def restore_rng(state: dict[str, Any]) -> None:
    """Re-apply a captured RNG state to torch, cuda, python random, and numpy"""
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda_all" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_all"])
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])


def _rng_state() -> dict[str, Any]:
    s: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        s["cuda_all"] = torch.cuda.get_rng_state_all()
    return s


def _build_manifest(*, step: int, config: RunConfig, metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "step": step,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "host": platform.node(),
        "torch_version": torch.__version__,
        "metrics": dict(metrics),
        "config": config.model_dump(mode="json"),
    }


def _atomic_save_torch(obj: Any, path: Path) -> None:
    tmp = path.parent / (path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_json(obj: Any, path: Path) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.parent / (dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _maybe_update_best(
    *,
    root: Path,
    bundle_path: Path,
    manifest: dict[str, Any],
    metrics: dict[str, float],
    best_metric: str | None,
    best_direction: str,
) -> None:
    if best_metric is None:
        return
    if best_metric not in metrics:
        logger.warning(f"{best_metric=} not in {list(metrics)=}, skipping best.pt update")
        return
    if best_direction not in ("min", "max"):
        raise ValueError(f"best_direction must be 'min' or 'max' got {best_direction!r}")

    incoming = metrics[best_metric]
    best_manifest_path = root / _BEST_MANIFEST_NAME
    is_better = True
    if best_manifest_path.is_file():
        try:
            prior = json.loads(best_manifest_path.read_text())
            prior_value = prior.get("metrics", {}).get(best_metric)
            if prior_value is not None:
                is_better = (
                    incoming < prior_value if best_direction == "min" else incoming > prior_value
                )
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read existing {best_manifest_path} ({e}); overwriting")

    if not is_better:
        return

    _atomic_copy(bundle_path, root / _BEST_BUNDLE_NAME)

    best_manifest = dict(manifest)
    best_manifest["best_for_step"] = manifest["step"]
    best_manifest["best_metric"] = best_metric
    best_manifest["best_direction"] = best_direction
    _atomic_write_json(best_manifest, best_manifest_path)

    logger.info(
        f"Updated best.pt: {best_metric}={incoming} at step={manifest['step']} ({best_direction})"
    )


def _rotate(root: Path, keep_last_n: int) -> None:
    paths = list_checkpoints(root)
    if len(paths) <= keep_last_n:
        return

    for p in paths[: len(paths) - keep_last_n]:
        manifest = p.with_suffix(".json")
        try:
            p.unlink()
        except OSError:
            logger.warning("Could not delete {p} ({e})")
        try:
            if manifest.is_file():
                manifest.unlink()
        except OSError:
            logger.warning("Could not delete {manifest} ({e})")
        logger.info("Rotated out {p.name}")


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        return out.returncode != 0
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False
