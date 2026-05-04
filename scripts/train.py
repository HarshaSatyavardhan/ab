"""Hydra entry point for AbLLaVA training.

Usage:
    python scripts/train.py training=stage_a
    python scripts/train.py training=stage_b training.init_from=outputs/.../final.pt
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from src.training import train

    train(cfg)


if __name__ == "__main__":
    main()
