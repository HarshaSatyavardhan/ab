"""AbLLaVA training package.

Public API:
    - ``train(cfg)``: main Hydra-driven training entry.
    - ``Stage``: enum for the two-stage curriculum.
    - ``build_optimizer``, ``build_scheduler``: optimizer / scheduler builders.
    - ``cdr_infill_collate``: helper that produces a CDR-infill flavored batch.
"""

from __future__ import annotations

from enum import Enum

from src.training.losses import (
    cdr_infill_collate,
    causal_lm_labels,
    make_cdr_infill_batch,
)
from src.training.loop import (
    TrainState,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train,
)
from src.training.optim import build_optimizer, build_scheduler


class Stage(str, Enum):
    A = "stage_a"
    B = "stage_b"


__all__ = [
    "Stage",
    "TrainState",
    "build_optimizer",
    "build_scheduler",
    "causal_lm_labels",
    "cdr_infill_collate",
    "evaluate",
    "load_checkpoint",
    "make_cdr_infill_batch",
    "save_checkpoint",
    "train",
]
