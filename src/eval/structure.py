"""Structure-level evaluation metrics.

* ``StructureScorer`` — refold generated sequences with ABodyBuilder2
  (``ImmuneBuilder``) and report pLDDT statistics.
* ``compute_structure_metrics`` — aggregate mean pLDDT, fraction of residues
  with pLDDT >= 70, and self-consistency RMSD (TODO at v0).
* ``kabsch_rmsd`` — superposition-aware RMSD between two coordinate arrays.

Heavy deps (ImmuneBuilder, AbMPNN) are lazy-imported.

Install:

* ``pip install ImmuneBuilder``
* AbMPNN: ``pip install git+https://github.com/Graylab/IgFold-MPNN`` (or
  whatever the upstream package is at runtime).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Kabsch RMSD
# ---------------------------------------------------------------------------


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD between point clouds ``P`` and ``Q`` after optimal superposition.

    Uses the Kabsch algorithm. ``P`` and ``Q`` must have the same shape
    ``(N, 3)``.
    """
    if P.shape != Q.shape:
        raise ValueError(f"shape mismatch: {P.shape} vs {Q.shape}")
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"expected (N, 3); got {P.shape}")
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    Pr = Pc @ R.T
    diff = Pr - Qc
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))


# ---------------------------------------------------------------------------
# Structure scorer
# ---------------------------------------------------------------------------


class StructureScorer:
    """Refold generated paired Fv sequences and report pLDDT.

    Self-consistency (sample → fold → AbMPNN-redesign → fold → RMSD) is left
    as a v0 TODO; the corresponding hook in ``compute_structure_metrics``
    returns ``None`` when no AbMPNN backend is plugged in.
    """

    def __init__(self, folder: str = "abodybuilder2", device: str = "cuda") -> None:
        self.folder = folder
        self.device = device
        self._predictor: Any | None = None
        self._unavailable = False

    def _ensure_predictor(self) -> Any | None:
        if self._unavailable:
            return None
        if self._predictor is not None:
            return self._predictor
        try:
            if self.folder == "abodybuilder2":
                from ImmuneBuilder import ABodyBuilder2  # type: ignore

                self._predictor = ABodyBuilder2()
            else:
                raise ValueError(f"unknown folder: {self.folder}")
        except Exception as e:  # noqa: BLE001
            logger.warning("StructureScorer unavailable (%s): %s", self.folder, e)
            self._unavailable = True
            return None
        return self._predictor

    def fold_pair(self, heavy: str, light: str, out_dir: Path) -> dict:
        """Fold a (heavy, light) pair. Returns pdb path + pLDDT array."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        predictor = self._ensure_predictor()
        if predictor is None:
            return {
                "pdb_path": None,
                "plddt": np.array([], dtype=np.float32),
                "mean_plddt": float("nan"),
            }
        try:
            ab = predictor.predict({"H": heavy, "L": light})
            pdb_path = out_dir / "refold.pdb"
            ab.save(str(pdb_path))
            plddt = self._extract_plddt(ab, pdb_path)
            mean = float(np.nanmean(plddt)) if plddt.size else float("nan")
            return {
                "pdb_path": str(pdb_path),
                "plddt": plddt,
                "mean_plddt": mean,
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("fold_pair failed: %s", e)
            return {
                "pdb_path": None,
                "plddt": np.array([], dtype=np.float32),
                "mean_plddt": float("nan"),
            }

    @staticmethod
    def _extract_plddt(ab: Any, pdb_path: Path) -> np.ndarray:
        # ImmuneBuilder writes pLDDT into the B-factor column. Try reading
        # back per-residue (CA) values from the PDB.
        try:
            vals: list[float] = []
            with open(pdb_path, "r") as fh:
                for line in fh:
                    if line.startswith("ATOM") and line[12:16].strip() == "CA":
                        vals.append(float(line[60:66]))
            if vals:
                return np.asarray(vals, dtype=np.float32)
        except Exception:  # noqa: BLE001
            pass
        # Fallback: attribute access.
        for attr in ("plddt", "confidence", "pLDDT"):
            v = getattr(ab, attr, None)
            if v is not None:
                return np.asarray(v, dtype=np.float32)
        return np.array([], dtype=np.float32)


# ---------------------------------------------------------------------------
# Top-level metric
# ---------------------------------------------------------------------------


def compute_structure_metrics(
    sequences: list[tuple[str, str]],
    scorer: StructureScorer,
    workdir: Path,
) -> dict:
    """Aggregate per-sample pLDDT statistics across ``sequences``."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    means: list[float] = []
    frac_70: list[float] = []
    for i, (h, l) in enumerate(sequences):
        out = scorer.fold_pair(h, l, workdir / f"sample_{i:05d}")
        plddt: np.ndarray = out["plddt"]
        if plddt.size == 0:
            continue
        means.append(float(np.nanmean(plddt)))
        frac_70.append(float(np.nanmean((plddt >= 70.0).astype(np.float32))))

    mean_plddt = float(np.nanmean(means)) if means else float("nan")
    frac = float(np.nanmean(frac_70)) if frac_70 else float("nan")

    # Self-consistency: sample -> fold -> AbMPNN redesign -> fold -> RMSD.
    # Not wired up at v0 (no AbMPNN backend bundled); return None and warn.
    sc_rmsd: float | None = None
    logger.warning(
        "self-consistency RMSD not implemented at v0 — returning None"
    )

    return {
        "mean_plddt": mean_plddt,
        "frac_plddt_70": frac,
        "self_consistency_rmsd": sc_rmsd,
    }
