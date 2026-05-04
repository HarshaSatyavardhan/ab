"""Optimizer / scheduler construction for AbLLaVA training.

We build AdamW with parameter groups so the projector can have its own
learning-rate multiplier (used in Stage B where LoRA + projector co-train).
Weight decay is only applied to 2D weight matrices (bias / norm parameters
are excluded) following the modern LLM convention.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _is_decay_param(name: str, p: torch.Tensor) -> bool:
    """Return True if the parameter should receive weight decay."""
    if p.ndim < 2:
        return False
    lname = name.lower()
    if "bias" in lname:
        return False
    if "norm" in lname or "layernorm" in lname or "ln" in lname.split("."):
        return False
    return True


def _is_projector_param(name: str) -> bool:
    return "projector" in name.lower()


def build_optimizer(
    model: nn.Module,
    cfg: DictConfig,
    projector_lr_multiplier: float | None = None,
) -> torch.optim.Optimizer:
    """Construct an AdamW optimizer with projector / non-projector groups.

    Args:
        model: the (possibly LoRA-wrapped) model.
        cfg: the ``cfg.training`` (or ``cfg``) DictConfig containing an
             ``optimizer`` block with ``lr``, ``betas``, ``weight_decay``.
        projector_lr_multiplier: if provided, apply ``lr * mult`` to params
             whose qualified name contains "projector".  If ``None``, falls
             back to ``cfg.optimizer.projector_lr_multiplier`` if present,
             else uses 1.0 (single group).
    """
    opt_cfg = cfg.optimizer if "optimizer" in cfg else cfg
    base_lr: float = float(opt_cfg.lr)
    betas = tuple(opt_cfg.get("betas", (0.9, 0.95)))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    eps = float(opt_cfg.get("eps", 1e-8))

    if projector_lr_multiplier is None:
        projector_lr_multiplier = float(opt_cfg.get("projector_lr_multiplier", 1.0))

    proj_decay: list[torch.Tensor] = []
    proj_nodecay: list[torch.Tensor] = []
    other_decay: list[torch.Tensor] = []
    other_nodecay: list[torch.Tensor] = []

    n_trainable = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_trainable += p.numel()
        if _is_projector_param(name):
            (proj_decay if _is_decay_param(name, p) else proj_nodecay).append(p)
        else:
            (other_decay if _is_decay_param(name, p) else other_nodecay).append(p)

    proj_lr = base_lr * float(projector_lr_multiplier)
    groups: list[dict] = []
    if other_decay:
        groups.append({"params": other_decay, "lr": base_lr, "weight_decay": weight_decay, "name": "other_decay"})
    if other_nodecay:
        groups.append({"params": other_nodecay, "lr": base_lr, "weight_decay": 0.0, "name": "other_nodecay"})
    if proj_decay:
        groups.append({"params": proj_decay, "lr": proj_lr, "weight_decay": weight_decay, "name": "projector_decay"})
    if proj_nodecay:
        groups.append({"params": proj_nodecay, "lr": proj_lr, "weight_decay": 0.0, "name": "projector_nodecay"})

    if not groups:
        # Fallback: at least one (empty) group avoids a torch error during
        # smoke tests where nothing is trainable.
        groups = [{"params": [p for p in model.parameters()], "lr": base_lr, "weight_decay": 0.0}]

    logger.info(
        "AdamW: %d trainable params, base_lr=%g, proj_lr=%g, wd=%g, betas=%s",
        n_trainable,
        base_lr,
        proj_lr,
        weight_decay,
        betas,
    )

    optimizer = torch.optim.AdamW(groups, lr=base_lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup → cosine decay LambdaLR.

    Reads ``cfg.scheduler.warmup_steps`` and ``cfg.scheduler.min_lr_ratio``.
    """
    sched_cfg = cfg.scheduler if "scheduler" in cfg else cfg
    warmup = int(sched_cfg.get("warmup_steps", 0))
    min_ratio = float(sched_cfg.get("min_lr_ratio", 0.0))
    total_steps = max(int(total_steps), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / float(max(warmup, 1))
        progress = (step - warmup) / max(total_steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)
