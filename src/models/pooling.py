"""CDR / Fv pooling functions for AbLLaVA.

Each pooler maps a per-residue structure embedding ``(B, N, d)`` plus
``cdr_spans`` ``(B, 6, 2)`` and ``pad_mask`` ``(B, N)`` to a small set of
"prefix" tokens ``(B, K, d)`` consumed by the projector.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _valid_span_mask(start: int, end: int, n: int, pad_mask_row: torch.Tensor) -> torch.Tensor:
    """Return a bool mask of length n, True for valid residues inside [start, end)."""
    s = max(0, int(start))
    e = min(int(n), int(end))
    if e <= s:
        return torch.zeros(n, dtype=torch.bool, device=pad_mask_row.device)
    m = torch.zeros(n, dtype=torch.bool, device=pad_mask_row.device)
    m[s:e] = True
    return m & pad_mask_row.bool()


def _safe_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of x[mask] over the residue dimension; zeros if mask is all False.

    x: (N, d), mask: (N,) bool. Returns (d,).
    """
    if not mask.any():
        return torch.zeros(x.size(-1), dtype=x.dtype, device=x.device)
    return x[mask].mean(dim=0)


# ---------------------------------------------------------------------------
# function poolers
# ---------------------------------------------------------------------------

def cdr_mean_pool(
    struct_emb: torch.Tensor,
    cdr_spans: torch.Tensor,
    pad_mask: torch.Tensor,
    plddt: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-aware per-CDR mean pool.

    Args:
        struct_emb: (B, N, d) per-residue embeddings.
        cdr_spans:  (B, 6, 2) [start, end) per-CDR spans.
        pad_mask:   (B, N) bool, True for valid residue.
        plddt:      unused, present for API uniformity.

    Returns:
        (B, 6, d) tensor of means; zeros for any empty (after mask) CDR.
    """
    b, n, d = struct_emb.shape
    out = torch.zeros(b, 6, d, dtype=struct_emb.dtype, device=struct_emb.device)
    for i in range(b):
        for j in range(6):
            start, end = int(cdr_spans[i, j, 0]), int(cdr_spans[i, j, 1])
            m = _valid_span_mask(start, end, n, pad_mask[i])
            out[i, j] = _safe_mean(struct_emb[i], m)
    return out


def cdr_segmented_pool(
    struct_emb: torch.Tensor,
    cdr_spans: torch.Tensor,
    pad_mask: torch.Tensor,
    plddt: torch.Tensor | None = None,
    n_segments: int = 3,
) -> torch.Tensor:
    """Per-CDR torso/apex/torso style pool.

    Splits each (masked) CDR into ``n_segments`` contiguous chunks of as-equal-as-possible
    length and means each chunk. Returns ``(B, 6 * n_segments, d)``. For loops with fewer
    valid residues than ``n_segments``, replicate the single mean across all segments. Empty
    spans yield zeros.
    """
    b, n, d = struct_emb.shape
    out = torch.zeros(b, 6 * n_segments, d, dtype=struct_emb.dtype, device=struct_emb.device)
    for i in range(b):
        for j in range(6):
            start, end = int(cdr_spans[i, j, 0]), int(cdr_spans[i, j, 1])
            m = _valid_span_mask(start, end, n, pad_mask[i])
            idx = m.nonzero(as_tuple=False).flatten()
            base = j * n_segments
            if idx.numel() == 0:
                continue  # leave zeros
            if idx.numel() < n_segments:
                v = struct_emb[i, idx].mean(dim=0)
                for k in range(n_segments):
                    out[i, base + k] = v
                continue
            # roughly equal contiguous chunks
            chunks = torch.chunk(idx, n_segments)
            for k, ch in enumerate(chunks):
                if ch.numel() == 0:
                    continue
                out[i, base + k] = struct_emb[i, ch].mean(dim=0)
    return out


def cdr_plddt_weighted_pool(
    struct_emb: torch.Tensor,
    cdr_spans: torch.Tensor,
    pad_mask: torch.Tensor,
    plddt: torch.Tensor,
) -> torch.Tensor:
    """pLDDT-weighted per-CDR mean.

    Each in-span residue contributes ``plddt_i / sum(plddt)``. plddt is clamped to
    ``>= 1e-3`` first; ``NaN`` plddt is replaced by ``1e-3``. Empty spans yield zeros.
    Returns ``(B, 6, d)``.
    """
    if plddt is None:
        raise ValueError("cdr_plddt_weighted_pool requires plddt")
    b, n, d = struct_emb.shape
    p = torch.nan_to_num(plddt, nan=1e-3).clamp_min(1e-3)
    out = torch.zeros(b, 6, d, dtype=struct_emb.dtype, device=struct_emb.device)
    for i in range(b):
        for j in range(6):
            start, end = int(cdr_spans[i, j, 0]), int(cdr_spans[i, j, 1])
            m = _valid_span_mask(start, end, n, pad_mask[i])
            if not m.any():
                continue
            w = p[i, m].to(struct_emb.dtype)
            w = w / w.sum().clamp_min(1e-12)
            out[i, j] = (struct_emb[i, m] * w.unsqueeze(-1)).sum(dim=0)
    return out


def per_residue_pool(
    struct_emb: torch.Tensor,
    cdr_spans: torch.Tensor,
    pad_mask: torch.Tensor,
    plddt: torch.Tensor | None = None,
) -> torch.Tensor:
    """No-op pool. Returns the full per-residue tensor ``(B, N, d)``.

    The downstream attention layers must respect ``pad_mask`` separately. We keep
    behaviour simple and just zero out the masked positions so a downstream
    consumer that ignores the mask still gets sane values.
    """
    return struct_emb * pad_mask.to(struct_emb.dtype).unsqueeze(-1)


def fv_mean_pool(
    struct_emb: torch.Tensor,
    cdr_spans: torch.Tensor,
    pad_mask: torch.Tensor,
    plddt: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean over all valid residues. Returns ``(B, 1, d)``."""
    b, n, d = struct_emb.shape
    out = torch.zeros(b, 1, d, dtype=struct_emb.dtype, device=struct_emb.device)
    for i in range(b):
        m = pad_mask[i].bool()
        if m.any():
            out[i, 0] = struct_emb[i, m].mean(dim=0)
    return out


# ---------------------------------------------------------------------------
# attention pool (learnable)
# ---------------------------------------------------------------------------

class CDRAttentionPool(nn.Module):
    """Learnable per-CDR attention pool.

    For each of the ``n_cdrs`` CDRs we keep a learnable query vector of dim ``d``.
    Keys are produced by a single linear projection of the input embeddings; values
    are the embeddings themselves. Attention is computed only over residues belonging
    to that CDR's span (other positions, including pads, get a ``-inf`` mask).
    Output shape: ``(B, n_cdrs, d)``.

    Param count: ``d^2 + n_cdrs * d`` (roughly).
    """

    def __init__(self, d: int, n_cdrs: int = 6) -> None:
        super().__init__()
        self.d = d
        self.n_cdrs = n_cdrs
        self.queries = nn.Parameter(torch.randn(n_cdrs, d) * 0.02)
        self.k_proj = nn.Linear(d, d, bias=False)

    def forward(
        self,
        struct_emb: torch.Tensor,
        cdr_spans: torch.Tensor,
        pad_mask: torch.Tensor,
        plddt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, d = struct_emb.shape
        device = struct_emb.device
        out = torch.zeros(b, self.n_cdrs, d, dtype=struct_emb.dtype, device=device)
        keys = self.k_proj(struct_emb)  # (B, N, d)
        scale = d ** -0.5
        for i in range(b):
            for j in range(self.n_cdrs):
                start, end = int(cdr_spans[i, j, 0]), int(cdr_spans[i, j, 1])
                m = _valid_span_mask(start, end, n, pad_mask[i])
                if not m.any():
                    continue  # zero output
                q = self.queries[j]  # (d,)
                k = keys[i, m]  # (M, d)
                v = struct_emb[i, m]  # (M, d)
                scores = (k @ q) * scale  # (M,)
                attn = torch.softmax(scores, dim=0)
                out[i, j] = (attn.unsqueeze(-1) * v).sum(dim=0)
        return out


# ---------------------------------------------------------------------------
# registry / factory
# ---------------------------------------------------------------------------

POOLERS: dict[str, Callable | type] = {
    "cdr_mean": cdr_mean_pool,
    "cdr_attention": CDRAttentionPool,
    "cdr_segmented": cdr_segmented_pool,
    "cdr_plddt": cdr_plddt_weighted_pool,
    "per_residue": per_residue_pool,
    "fv_mean": fv_mean_pool,
}


def build_pooler(name: str, d: int | None = None, **kwargs):
    """Construct a pooler by name.

    Returns either a plain function (for the stateless poolers) or an ``nn.Module``
    (for ``cdr_attention``). For ``cdr_attention``, ``d`` is required.
    """
    if name not in POOLERS:
        raise KeyError(f"unknown pooler '{name}'. options: {list(POOLERS)}")
    obj = POOLERS[name]
    if isinstance(obj, type):
        if name == "cdr_attention":
            if d is None:
                raise ValueError("cdr_attention pool requires d (embedding dim)")
            return obj(d=d, **kwargs)
        return obj(**kwargs)
    return obj
