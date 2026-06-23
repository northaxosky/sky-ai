"""Wandb run wrapper that is a no-op when disabled"""

from __future__ import annotations

from typing import Any

import wandb

from harness.config.schema import LogConfig
from harness.log import get_logger

logger = get_logger(__name__)


class WandbLogger:
    """Wraps wandb so the training loop calls one API"""

    def __init__(
        self,
        cfg: LogConfig,
        *,
        rank: int = 0,
        resume_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._enabled = cfg.wandb and rank == 0
        self.run_id: str | None = None
        if not self._enabled:
            logger.info(f"wandb disabled ({cfg.wandb=}, {rank=})")
            return

        if resume_id is not None:
            # Resume mode: fail loudly if wandb can't find the prior run instead
            # of silently starting a fresh one (which would split metrics across
            # two runs after a checkpoint restore).
            run_id = resume_id
            resume_mode = "must"
        else:
            run_id = wandb.util.generate_id()  # pyright: ignore
            resume_mode = "allow"
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            id=run_id,
            resume=resume_mode,
            config=config,
        )
        self.run_id = run_id
        logger.info(f"wandb initialized (project={cfg.wandb_project}, {run_id=})")

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        """Forward a metrics dict to wandb.log; no-op when disabled"""
        if not self._enabled:
            return
        wandb.log(metrics, step=step)

    def update_config(self, extra: dict[str, Any]) -> None:
        """Add or update fields in the wandb run config"""
        if not self._enabled:
            return
        wandb.config.update(extra, allow_val_change=True)

    def finish(self) -> None:
        """Flush and close the wandb run"""
        if not self._enabled:
            return
        wandb.finish()
        self._enabled = False

    def __enter__(self) -> WandbLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.finish()
