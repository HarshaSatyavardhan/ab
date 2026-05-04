"""Stage A: train projector + (optional) attention pooler with the decoder
fully frozen and LoRA disabled.  Thin wrapper around ``train``."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict

from src.training.loop import train
from src.utils.logging import get_logger

logger = get_logger(__name__)


def stage_a(cfg: DictConfig) -> Path:
    """Run Stage-A training.

    Forces ``cfg.training.lora.enabled = False`` and asserts the decoder is
    frozen by the loop's ``_freeze_decoder_keep_projector`` helper.
    """
    OmegaConf.set_struct(cfg, False)
    with open_dict(cfg):
        if "lora" not in cfg.training:
            cfg.training.lora = OmegaConf.create({"enabled": False})
        cfg.training.lora.enabled = False
        cfg.training.freeze_decoder = True
    logger.info("Stage A: LoRA disabled, decoder frozen, projector trainable.")
    return train(cfg)
