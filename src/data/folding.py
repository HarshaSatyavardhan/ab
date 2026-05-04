"""Structure prediction wrapper around ImmuneBuilder / ABodyBuilder2.

Lazy-imports :mod:`ImmuneBuilder` so this module is importable without the
optional dependency. Writes a PDB and returns the per-residue pLDDT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

_IB_IMPORT_ERROR = (
    "ImmuneBuilder is not installed. Install the antibody extras: "
    "`pip install -e .[antibody]` or `pip install ImmuneBuilder`."
)


def _try_import_abb2() -> Any:
    try:
        from ImmuneBuilder import ABodyBuilder2  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        raise ImportError(_IB_IMPORT_ERROR) from e
    return ABodyBuilder2


class StructurePredictor:
    """Predict an antibody Fv structure from heavy/light AA sequences."""

    def __init__(self, model: str = "abodybuilder2", device: str = "cuda") -> None:
        self.model_name = model.lower()
        self.device = device
        self._predictor: Any | None = None
        if self.model_name not in {"abodybuilder2", "abb2"}:
            raise ValueError(
                f"Unsupported folding model {model!r}; expected 'abodybuilder2'."
            )

    def _ensure_predictor(self) -> Any:
        if self._predictor is None:
            cls = _try_import_abb2()
            try:
                self._predictor = cls(numbering_scheme="imgt")
            except TypeError:
                self._predictor = cls()
        return self._predictor

    def fold_pair(self, heavy: str, light: str, out_pdb: str | Path) -> dict:
        """Fold a heavy/light Fv pair and write a PDB to *out_pdb*.

        Returns a dict with keys ``pdb_path`` (Path), ``plddt`` (np.ndarray
        of shape ``(N,)``), and ``mean_plddt`` (float).
        """
        out_path = Path(out_pdb)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        predictor = self._ensure_predictor()
        sequences = {"H": heavy, "L": light}
        antibody = predictor.predict(sequences)
        try:
            antibody.save(str(out_path))
        except Exception:  # noqa: BLE001
            # Newer ImmuneBuilder versions may use save_pdb
            antibody.save_pdb(str(out_path))

        plddt = self._extract_plddt(antibody)
        mean_plddt = float(np.nanmean(plddt)) if plddt.size else float("nan")
        return {
            "pdb_path": out_path,
            "plddt": plddt,
            "mean_plddt": mean_plddt,
        }

    @staticmethod
    def _extract_plddt(antibody: Any) -> np.ndarray:
        """Extract a 1D float pLDDT array from an ImmuneBuilder antibody object."""
        for attr in ("plddt", "pLDDT", "pred_lddt", "lddt"):
            if hasattr(antibody, attr):
                vals = getattr(antibody, attr)
                arr = np.asarray(vals, dtype=np.float32)
                return arr.reshape(-1)
        # Fallback: per-atom error -> per-residue mean
        if hasattr(antibody, "error_estimates"):
            arr = np.asarray(antibody.error_estimates, dtype=np.float32)
            return arr.reshape(-1)
        logger.warning("Could not extract pLDDT from antibody object; returning empty.")
        return np.zeros(0, dtype=np.float32)
