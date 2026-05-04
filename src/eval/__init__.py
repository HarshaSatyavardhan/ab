"""Evaluation suite for AbLLaVA.

Public API re-exports the top-level entry-point and the per-category metric
functions. Heavy or optional dependencies are imported lazily inside the
submodules so this package can be imported in lightweight environments
(e.g., test collection, docs).
"""

from __future__ import annotations

from src.eval.developability import (
    LiabilityScanner,
    compute_developability,
)
from src.eval.evaluate import evaluate_checkpoint
from src.eval.naturalness import compute_naturalness
from src.eval.sequence import (
    compute_diversity,
    compute_perplexity,
    compute_recovery,
)
from src.eval.structure import compute_structure_metrics

__all__ = [
    "LiabilityScanner",
    "compute_developability",
    "compute_diversity",
    "compute_naturalness",
    "compute_perplexity",
    "compute_recovery",
    "compute_structure_metrics",
    "evaluate_checkpoint",
]
