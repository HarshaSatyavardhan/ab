"""Stage B: enable LoRA on the decoder and continue training the projector.

Optionally warm-starts from a Stage-A checkpoint via ``cfg.training.init_from``.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict

from src.training.loop import train
from src.utils.logging import get_logger

logger = get_logger(__name__)


def stage_b(cfg: DictConfig) -> Path:
    OmegaConf.set_struct(cfg, False)
    with open_dict(cfg):
        if "lora" not in cfg.training:
            cfg.training.lora = OmegaConf.create({"enabled": True})
        cfg.training.lora.enabled = True
    assert bool(cfg.training.lora.enabled), "Stage B requires LoRA to be enabled."
    init_from = cfg.training.get("init_from", None)
    if init_from:
        logger.info("Stage B: warm-starting from %s", init_from)
    else:
        logger.info("Stage B: training from scratch (no init_from).")
    return train(cfg)
