"""On-disk cache for precomputed structure-encoder embeddings.

Each entry is stored as a compressed ``.npz`` file at
``<cache_dir>/<id>.npz`` with keys:

    - ``emb``   : ``np.ndarray[N, embedding_dim] float32``
    - ``plddt`` : ``np.ndarray[N] float32`` (optional)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """File-backed cache of per-residue antibody embeddings."""

    def __init__(self, cache_dir: str | Path, embedding_dim: int) -> None:
        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim: int = int(embedding_dim)

    # ------------------------------------------------------------------
    def _path(self, id_: str) -> Path:
        return self.cache_dir / f"{id_}.npz"

    def has(self, id_: str) -> bool:
        return self._path(id_).is_file()

    def save(
        self,
        id_: str,
        emb: np.ndarray,
        plddt: np.ndarray | None = None,
    ) -> None:
        """Save *emb* (and optionally *plddt*) for *id_*."""
        emb = np.ascontiguousarray(np.asarray(emb, dtype=np.float32))
        if emb.ndim != 2 or emb.shape[1] != self.embedding_dim:
            raise ValueError(
                f"emb must have shape (N, {self.embedding_dim}); got {emb.shape}"
            )
        path = self._path(id_)
        if plddt is None:
            np.savez_compressed(path, emb=emb)
        else:
            plddt_arr = np.ascontiguousarray(np.asarray(plddt, dtype=np.float32)).reshape(-1)
            if plddt_arr.shape[0] != emb.shape[0]:
                raise ValueError(
                    f"plddt length {plddt_arr.shape[0]} does not match emb length "
                    f"{emb.shape[0]}"
                )
            np.savez_compressed(path, emb=emb, plddt=plddt_arr)

    def load(self, id_: str) -> tuple[np.ndarray, np.ndarray | None]:
        """Return ``(emb, plddt | None)`` for *id_*."""
        path = self._path(id_)
        if not path.is_file():
            raise KeyError(f"No cached embedding for id={id_!r} at {path}")
        with np.load(path) as f:
            emb = np.asarray(f["emb"], dtype=np.float32)
            plddt = np.asarray(f["plddt"], dtype=np.float32) if "plddt" in f.files else None
        return emb, plddt
