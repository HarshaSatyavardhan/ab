"""Hydra entry-point for the AbLLaVA evaluation suite.

Usage::

    python scripts/eval.py eval.checkpoint_path=path/to/ckpt.pt
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from src.eval import evaluate_checkpoint

    evaluate_checkpoint(cfg)


if __name__ == "__main__":
    main()
