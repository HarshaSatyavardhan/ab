"""Checkpoint loading and FASTA writing helpers for inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from src.utils.logging import get_logger
from src.utils.tokenizer import AntibodyTokenizer

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from omegaconf import DictConfig
    from src.models import AbLLaVA

logger = get_logger(__name__)


def load_model(
    checkpoint_path: Path,
    cfg: "DictConfig | None" = None,
    device: str = "cuda",
) -> tuple["AbLLaVA", AntibodyTokenizer]:
    """Load an AbLLaVA checkpoint.

    The checkpoint is expected to be a dict with keys:
        - ``state_dict``: the model ``state_dict``.
        - ``cfg``: the OmegaConf training config used to build the model.
        - ``state``: a :class:`TrainState` (unused here, kept for completeness).

    If ``cfg`` is ``None`` the saved one is used. The model is built via
    :func:`src.models.build_abllava` with the appropriate sub-configs and
    moved to ``device`` in eval mode.
    """
    from src.models import build_abllava  # local import: parallel agent may build later

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint from %s", checkpoint_path)
    ckpt: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise KeyError("Checkpoint is missing 'state_dict' key.")

    if cfg is None:
        if "cfg" not in ckpt:
            raise KeyError("Checkpoint has no 'cfg' and no override was supplied.")
        saved_cfg = ckpt["cfg"]
        try:
            from omegaconf import DictConfig as _DictConfig, OmegaConf
        except ImportError as exc:  # pragma: no cover - omegaconf optional at import time
            raise ImportError(
                "omegaconf is required to load a checkpoint without an explicit cfg override."
            ) from exc
        cfg = saved_cfg if isinstance(saved_cfg, _DictConfig) else OmegaConf.create(saved_cfg)

    model = build_abllava(cfg.model, cfg.projector, cfg.pooling, cfg.training.lora)

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        logger.warning("Missing keys when loading state_dict: %d (e.g. %s)", len(missing), missing[:3])
    if unexpected:
        logger.warning(
            "Unexpected keys when loading state_dict: %d (e.g. %s)", len(unexpected), unexpected[:3]
        )

    model.to(device)
    model.eval()

    tokenizer = AntibodyTokenizer()
    return model, tokenizer


def write_fasta(records: list[dict], out_path: Path) -> None:
    """Write paired Fv records to a FASTA file.

    Each record must have ``id``, ``heavy``, and ``light`` keys. Two FASTA
    entries are written per record (``{id}_H`` and ``{id}_L``) with no line
    wrapping.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for rec in records:
        rid = str(rec["id"])
        heavy = str(rec.get("heavy", ""))
        light = str(rec.get("light", ""))
        lines.append(f">{rid}_H")
        lines.append(heavy)
        lines.append(f">{rid}_L")
        lines.append(light)

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    logger.info("Wrote %d records (%d FASTA entries) to %s", len(records), 2 * len(records), out_path)
