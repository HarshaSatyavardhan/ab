"""Frozen structure / sequence encoder wrappers.

The training path uses precomputed embeddings via ``CachedEmbeddingEncoder``;
the live wrappers (``AntiFoldEncoder``, ``IgFoldEncoder``, ``ESM2Encoder``) are
only meaningful at inference time and lazy-import their backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# cached embeddings
# ---------------------------------------------------------------------------

class CachedEmbeddingEncoder(nn.Module):
    """Loads precomputed embeddings from disk.

    Expects each id to correspond to a file ``<cache_dir>/<id>.npz`` with keys
    ``struct_emb`` ``(N, d)``, ``pad_mask`` ``(N,)`` (bool / int), and ``plddt``
    ``(N,)``. The forward pass loads the requested ids, right-pads to the batch
    max-N, and stacks into ``(B, N_max, d)``.
    """

    def __init__(self, cache_dir: str | Path, embedding_dim: int) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.embedding_dim = int(embedding_dim)

    def _load(self, id_: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path = self.cache_dir / f"{id_}.npz"
        if not path.exists():
            raise FileNotFoundError(f"cached embedding not found: {path}")
        data = np.load(path)
        emb = np.asarray(data["struct_emb"], dtype=np.float32)
        mask = np.asarray(data["pad_mask"]).astype(bool)
        plddt = np.asarray(data["plddt"], dtype=np.float32)
        return emb, mask, plddt

    @torch.no_grad()
    def forward(self, ids: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loaded = [self._load(i) for i in ids]
        n_max = max(e.shape[0] for e, _, _ in loaded)
        d = self.embedding_dim
        b = len(loaded)
        emb = torch.zeros(b, n_max, d, dtype=torch.float32)
        pad = torch.zeros(b, n_max, dtype=torch.bool)
        plddt = torch.full((b, n_max), float("nan"), dtype=torch.float32)
        for i, (e, m, p) in enumerate(loaded):
            n = e.shape[0]
            emb[i, :n] = torch.from_numpy(e)
            pad[i, :n] = torch.from_numpy(m)
            plddt[i, :n] = torch.from_numpy(p)
        return emb, pad, plddt


# ---------------------------------------------------------------------------
# live wrappers (frozen)
# ---------------------------------------------------------------------------

class _FrozenBase(nn.Module):
    """Mixin: freezes parameters at the end of ``__init__``."""

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()


class AntiFoldEncoder(_FrozenBase):
    """Wraps the AntiFold structure encoder (lazy import).

    The real implementation depends on the upstream antifold package. At forward
    time we accept a list of PDB paths and return per-residue embeddings.
    """

    embedding_dim: int = 512

    def __init__(self, checkpoint_path: str | None = None, device: str = "cuda") -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device
        try:
            import antifold  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "AntiFoldEncoder requires the `antifold` package. Install it via the "
                "`antibody` extra or use CachedEmbeddingEncoder for training."
            ) from e
        # Concrete construction is deferred to user code; we just freeze whatever
        # parameters land here.
        self._freeze()

    @torch.no_grad()
    def forward(
        self, pdb_paths: Sequence[str | Path]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # pragma: no cover - lazy
        raise NotImplementedError(
            "AntiFoldEncoder.forward is a stub — wire in the real antifold runner here."
        )


class IgFoldEncoder(_FrozenBase):
    """Hooks the last node-features layer of IgFold's trunk (lazy import)."""

    embedding_dim: int = 512

    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.device = device
        try:
            import igfold  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "IgFoldEncoder requires the `igfold` package. Install it via the "
                "`antibody` extra or use CachedEmbeddingEncoder for training."
            ) from e
        self._freeze()

    @torch.no_grad()
    def forward(
        self, pdb_paths: Sequence[str | Path]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # pragma: no cover - lazy
        raise NotImplementedError(
            "IgFoldEncoder.forward is a stub — wire in the real igfold trunk hook here."
        )


class ESM2Encoder(_FrozenBase):
    """Wraps a HuggingFace ESM-2 model. Per-residue embeddings, frozen."""

    embedding_dim: int = 1280

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", device: str = "cuda") -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "ESM2Encoder requires the `transformers` package."
            ) from e
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self._freeze()

    @torch.no_grad()
    def forward(
        self, sequences: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        toks = self.tokenizer(list(sequences), return_tensors="pt", padding=True)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        out = self.model(**toks)
        emb = out.last_hidden_state  # (B, L, d)
        mask = toks["attention_mask"].to(torch.bool)
        # Drop the leading <cls> and trailing <eos> from ESM tokenization (per residue).
        emb = emb[:, 1:-1]
        mask = mask[:, 1:-1]
        plddt = torch.full(mask.shape, float("nan"), dtype=torch.float32, device=emb.device)
        return emb, mask, plddt


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

_ENCODERS: dict[str, type] = {
    "cached": CachedEmbeddingEncoder,
    "antifold": AntiFoldEncoder,
    "igfold": IgFoldEncoder,
    "esm2": ESM2Encoder,
}


def build_encoder(name: str, **kwargs) -> nn.Module:
    if name not in _ENCODERS:
        raise KeyError(f"unknown encoder '{name}'. options: {list(_ENCODERS)}")
    return _ENCODERS[name](**kwargs)
