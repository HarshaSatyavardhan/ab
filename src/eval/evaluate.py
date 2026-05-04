"""Top-level eval driver. Hydra-config in, JSON report out.

This module ties together :mod:`src.eval.sequence`, :mod:`src.eval.naturalness`,
:mod:`src.eval.structure`, and :mod:`src.eval.developability` into a single
``evaluate_checkpoint`` entry point used by ``scripts/eval.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.eval.developability import compute_developability
from src.eval.naturalness import compute_naturalness
from src.eval.sequence import (
    aa_kl,
    compute_diversity,
    compute_perplexity,
    compute_recovery,
)
from src.eval.structure import StructureScorer, compute_structure_metrics
from src.utils.logging import get_logger
from src.utils.tokenizer import AntibodyTokenizer

logger = get_logger(__name__)


def _enabled(cfg: DictConfig, key: str) -> bool:
    eval_cfg = cfg.get("eval", cfg)
    metrics = eval_cfg.get("metrics", []) if hasattr(eval_cfg, "get") else []
    try:
        return key in list(metrics)
    except Exception:  # noqa: BLE001
        return False


def _build_loader(cfg: DictConfig) -> Any | None:
    """Best-effort dataloader construction. Returns ``None`` on failure."""
    try:
        from src.data import build_dataloader  # type: ignore

        return build_dataloader(cfg, split="val")
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not build eval dataloader: %s", e)
        return None


def _sample_sequences(
    cfg: DictConfig,
    model: Any,
    tokenizer: AntibodyTokenizer,
    n_samples: int,
    device: str,
) -> list[tuple[str, str]]:
    """Generate ``n_samples`` paired Fv strings via the inference API."""
    try:
        from src.inference import generate_from_embedding  # type: ignore
    except Exception as e:  # noqa: BLE001
        logger.warning("Inference API unavailable: %s", e)
        return []
    loader = _build_loader(cfg)
    if loader is None:
        return []
    pairs: list[tuple[str, str]] = []
    try:
        import torch  # local

        with torch.no_grad():
            for batch in loader:
                if len(pairs) >= n_samples:
                    break
                ids = generate_from_embedding(
                    model=model,
                    tokenizer=tokenizer,
                    struct_emb=batch["struct_emb"].to(device),
                    cdr_spans=batch["cdr_spans"].to(device),
                    pad_mask=batch["pad_mask"].to(device),
                    plddt=batch.get("plddt", None),
                )
                if isinstance(ids, torch.Tensor):
                    ids_list = ids.tolist()
                else:
                    ids_list = list(ids)
                for row in ids_list:
                    h, l = tokenizer.split_pair_ids(list(row))
                    pairs.append((h, l))
                    if len(pairs) >= n_samples:
                        break
    except Exception as e:  # noqa: BLE001
        logger.warning("Sampling loop failed: %s", e)
    return pairs[:n_samples]


def evaluate_checkpoint(cfg: DictConfig) -> dict:
    """Run the full eval suite on a checkpoint.

    Reads ``cfg.eval.checkpoint_path``, builds model + dataloader, samples
    ``cfg.eval.n_samples`` Fv pairs, then runs each enabled metric category
    and writes a JSON report to ``cfg.eval.output_path``.
    """
    eval_cfg: DictConfig = cfg.eval if "eval" in cfg else cfg  # type: ignore[assignment]
    ckpt = eval_cfg.get("checkpoint_path", None)
    n_samples = int(eval_cfg.get("n_samples", 0))
    device = str(cfg.get("device", "cuda")) if hasattr(cfg, "get") else "cuda"
    output_path = Path(str(eval_cfg.get("output_path", "outputs/eval_report.json")))

    logger.info("Evaluating checkpoint=%s n_samples=%d", ckpt, n_samples)

    tokenizer = AntibodyTokenizer()
    model: Any = None
    if ckpt:
        try:
            from src.inference import load_model  # type: ignore

            model, tokenizer = load_model(ckpt, device=device)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not load checkpoint %s: %s", ckpt, e)

    loader = _build_loader(cfg)
    enabled_metrics = list(eval_cfg.get("metrics", []))

    report: dict[str, Any] = {
        "checkpoint": str(ckpt) if ckpt is not None else None,
        "n_samples": n_samples,
    }

    # ---- perplexity / recovery (need model + loader) -----------------------
    if model is not None and loader is not None and "perplexity" in enabled_metrics:
        try:
            report["perplexity"] = compute_perplexity(model, loader, device)
        except Exception as e:  # noqa: BLE001
            logger.warning("perplexity failed: %s", e)
            report["perplexity"] = {
                "overall": float("nan"),
                "cdr_h3": float("nan"),
                "cdr_other": float("nan"),
                "fwk": float("nan"),
            }

    if model is not None and loader is not None and "recovery" in enabled_metrics:
        try:
            report["recovery"] = compute_recovery(model, loader, tokenizer, device)
        except Exception as e:  # noqa: BLE001
            logger.warning("recovery failed: %s", e)
            report["recovery"] = {}

    # ---- sample sequences for downstream metrics ---------------------------
    sequences: list[tuple[str, str]] = []
    if model is not None and n_samples > 0:
        sequences = _sample_sequences(cfg, model, tokenizer, n_samples, device)
        logger.info("Sampled %d Fv pairs", len(sequences))

    if "diversity" in enabled_metrics:
        flat = [f"{h}/{l}" for h, l in sequences]
        div = compute_diversity(flat) if flat else {
            "mean_levenshtein": float("nan"),
            "mean_blosum62": float("nan"),
        }
        div["aa_kl"] = aa_kl([s for pair in sequences for s in pair])
        report["diversity"] = div

    if "naturalness" in enabled_metrics and sequences:
        backends = list(eval_cfg.get("naturalness_models", []))
        report["naturalness"] = compute_naturalness(sequences, backends, device)
    elif "naturalness" in enabled_metrics:
        report["naturalness"] = {
            b: float("nan") for b in eval_cfg.get("naturalness_models", [])
        }

    if "structure" in enabled_metrics and sequences:
        scorer = StructureScorer(
            folder=str(eval_cfg.get("refold", {}).get("model", "abodybuilder2")),
            device=device,
        )
        workdir = output_path.parent / "refold"
        report["structure"] = compute_structure_metrics(sequences, scorer, workdir)
    elif "structure" in enabled_metrics:
        report["structure"] = {
            "mean_plddt": float("nan"),
            "frac_plddt_70": float("nan"),
            "self_consistency_rmsd": None,
        }

    if "developability" in enabled_metrics:
        if sequences:
            report["developability"] = compute_developability(sequences)
        else:
            report["developability"] = {
                "liabilities": {},
                "oasis": None,
                "tap": None,
            }

    # ---- write report ------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(_jsonable(report), fh, indent=2, sort_keys=True, default=str)
    logger.info("Wrote eval report to %s", output_path)
    # Also stash the resolved config for reproducibility.
    try:
        cfg_path = output_path.with_suffix(".config.yaml")
        with open(cfg_path, "w") as fh:
            fh.write(OmegaConf.to_yaml(cfg))
    except Exception:  # noqa: BLE001
        pass
    return report


def _jsonable(obj: Any) -> Any:
    """Recursively coerce numpy/torch scalars for JSON serialization."""
    try:
        import numpy as np  # local
    except Exception:  # noqa: BLE001
        np = None  # type: ignore[assignment]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
