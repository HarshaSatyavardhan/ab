"""Logging helpers — wraps Python logging + optional W&B."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any


def get_logger(name: str = "abllava", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(h)
    logger.propagate = False
    return logger


class WandbLogger:
    """Tiny wrapper so train scripts don't crash if wandb is missing/disabled."""

    def __init__(self, enabled: bool, **init_kwargs: Any) -> None:
        self.enabled = enabled
        self.run = None
        if not enabled:
            return
        try:
            import wandb

            mode = init_kwargs.pop("mode", os.environ.get("WANDB_MODE", "online"))
            self.run = wandb.init(mode=mode, **init_kwargs)
            self._wandb = wandb
        except Exception as e:  # noqa: BLE001
            get_logger().warning("wandb disabled (%s)", e)
            self.enabled = False

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self.run is None:
            return
        self._wandb.log(data, step=step)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self._wandb.finish()
