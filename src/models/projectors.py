"""Projector modules mapping pooled structure tokens into the decoder input space.

Three flavours: a plain MLP (per-token), a BLIP-2 style Q-Former, and a
Perceiver-Resampler. All map ``(B, K_in, d_in)`` to ``(B, K_out, d_out)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLPProjector(nn.Module):
    """2-layer GELU MLP applied per-token.

    Output shape mirrors the input token count: ``(B, K, d_in) -> (B, K, d_out)``.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_hidden = d_hidden if d_hidden is not None else 4 * d_out
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


# ---------------------------------------------------------------------------
# Q-Former (BLIP-2 style)
# ---------------------------------------------------------------------------

class _QFormerBlock(nn.Module):
    def __init__(self, d_q: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q1 = nn.LayerNorm(d_q)
        self.norm_kv = nn.LayerNorm(d_q)
        self.cross = nn.MultiheadAttention(d_q, n_heads, dropout=dropout, batch_first=True)
        self.norm_q2 = nn.LayerNorm(d_q)
        self.self_attn = nn.MultiheadAttention(d_q, n_heads, dropout=dropout, batch_first=True)
        self.norm_q3 = nn.LayerNorm(d_q)
        self.ff = nn.Sequential(
            nn.Linear(d_q, 4 * d_q),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_q, d_q),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor | None) -> torch.Tensor:
        # cross-attn (pre-norm)
        qn = self.norm_q1(q)
        kvn = self.norm_kv(kv)
        c, _ = self.cross(qn, kvn, kvn, key_padding_mask=kv_mask, need_weights=False)
        q = q + c
        # self-attn
        qn = self.norm_q2(q)
        s, _ = self.self_attn(qn, qn, qn, need_weights=False)
        q = q + s
        # ff
        q = q + self.ff(self.norm_q3(q))
        return q


class QFormerProjector(nn.Module):
    """BLIP-2 style projector: ``n_queries`` learnable tokens cross-attend to the input.

    ``forward(x)``: ``(B, N, d_in) -> (B, n_queries, d_out)``. An optional ``key_padding_mask``
    of shape ``(B, N)`` (True = pad) can be supplied via ``forward(x, kv_mask=...)``.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_q: int = 768,
        n_queries: int = 32,
        n_layers: int = 6,
        n_heads: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_q = d_q
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(n_queries, d_q) * 0.02)
        self.kv_proj = nn.Linear(d_in, d_q)
        self.blocks = nn.ModuleList([
            _QFormerBlock(d_q, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_q)
        self.out_proj = nn.Linear(d_q, d_out)
        nn.init.normal_(self.kv_proj.weight, std=0.02)
        nn.init.zeros_(self.kv_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, kv_mask: torch.Tensor | None = None) -> torch.Tensor:
        b = x.size(0)
        kv = self.kv_proj(x)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        for blk in self.blocks:
            q = blk(q, kv, kv_mask)
        return self.out_proj(self.norm_out(q))


# ---------------------------------------------------------------------------
# Perceiver-Resampler
# ---------------------------------------------------------------------------

class _PerceiverBlock(nn.Module):
    def __init__(self, d_lat: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_lat)
        self.norm_kv = nn.LayerNorm(d_lat)
        self.cross = nn.MultiheadAttention(d_lat, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_lat)
        self.ff = nn.Sequential(
            nn.Linear(d_lat, 4 * d_lat),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_lat, d_lat),
            nn.Dropout(dropout),
        )

    def forward(self, lat: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor | None) -> torch.Tensor:
        # Perceiver-Resampler: latents and KV are concatenated on the K side
        latn = self.norm_q(lat)
        kvn = self.norm_kv(kv)
        kv_cat = torch.cat([kvn, latn], dim=1)
        if kv_mask is not None:
            extra = torch.zeros(lat.size(0), lat.size(1), dtype=torch.bool, device=lat.device)
            kv_mask_cat = torch.cat([kv_mask, extra], dim=1)
        else:
            kv_mask_cat = None
        c, _ = self.cross(latn, kv_cat, kv_cat, key_padding_mask=kv_mask_cat, need_weights=False)
        lat = lat + c
        lat = lat + self.ff(self.norm_ff(lat))
        return lat


class PerceiverProjector(nn.Module):
    """Perceiver-Resampler: ``n_latents`` learnable latents attend to the input + themselves.

    ``forward(x)``: ``(B, N, d_in) -> (B, n_latents, d_out)``.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_lat: int = 1024,
        n_latents: int = 32,
        n_layers: int = 1,
        n_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_lat = d_lat
        self.n_latents = n_latents
        self.latents = nn.Parameter(torch.randn(n_latents, d_lat) * 0.02)
        self.kv_proj = nn.Linear(d_in, d_lat)
        self.blocks = nn.ModuleList([
            _PerceiverBlock(d_lat, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_lat)
        self.out_proj = nn.Linear(d_lat, d_out)
        nn.init.normal_(self.kv_proj.weight, std=0.02)
        nn.init.zeros_(self.kv_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, kv_mask: torch.Tensor | None = None) -> torch.Tensor:
        b = x.size(0)
        kv = self.kv_proj(x)
        lat = self.latents.unsqueeze(0).expand(b, -1, -1)
        for blk in self.blocks:
            lat = blk(lat, kv, kv_mask)
        return self.out_proj(self.norm_out(lat))


# ---------------------------------------------------------------------------
# registry / factory
# ---------------------------------------------------------------------------

PROJECTORS: dict[str, type] = {
    "mlp": MLPProjector,
    "qformer": QFormerProjector,
    "perceiver": PerceiverProjector,
}


def build_projector(name: str, **kwargs) -> nn.Module:
    if name not in PROJECTORS:
        raise KeyError(f"unknown projector '{name}'. options: {list(PROJECTORS)}")
    return PROJECTORS[name](**kwargs)
