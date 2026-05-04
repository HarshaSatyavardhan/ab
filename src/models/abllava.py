"""AbLLaVA: assembles encoder embeddings -> pooler -> projector -> decoder.

The encoder is upstream of this module (embeddings come in via ``batch``); this
class is responsible only for pooling, projecting, and threading prefix tokens
into the decoder.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.utils.logging import get_logger
from src.utils.tokenizer import AntibodyTokenizer

from src.models.decoder import AntibodyDecoder
from src.models.lora import apply_lora
from src.models.pooling import build_pooler
from src.models.projectors import build_projector

logger = get_logger(__name__)


class AbLLaVA(nn.Module):
    """End-to-end paired-Fv inverse-folding model.

    Args:
        decoder: an ``AntibodyDecoder`` (optionally PEFT-wrapped).
        projector: ``(B, K, d_e) -> (B, K', d_l)`` mapping from pooled struct
            embeddings to decoder hidden size.
        pooler: either a callable function or an ``nn.Module``.
        pooler_kind: ``"function"`` or ``"module"``. If ``"module"``, the pooler is
            registered as a sub-module so its parameters train with the rest.
    """

    def __init__(
        self,
        decoder: nn.Module,
        projector: nn.Module,
        pooler,
        pooler_kind: str = "function",
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.projector = projector
        self.pooler_kind = pooler_kind
        if pooler_kind == "module":
            if not isinstance(pooler, nn.Module):
                raise ValueError("pooler_kind='module' requires an nn.Module pooler")
            self.pooler_module = pooler
            self._pooler_fn = None
        elif pooler_kind == "function":
            self.pooler_module = None
            self._pooler_fn = pooler
        else:
            raise ValueError(f"unknown pooler_kind: {pooler_kind}")

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _hidden_size(self) -> int:
        # Works for plain decoder and for peft-wrapped decoder
        dec = self._raw_decoder()
        return dec.hidden_size

    def _raw_decoder(self) -> AntibodyDecoder:
        dec = self.decoder
        # peft wraps in PeftModel(.base_model.model)
        for attr in ("base_model",):
            if hasattr(dec, attr):
                inner = getattr(dec, attr)
                if hasattr(inner, "model"):
                    inner = inner.model
                if hasattr(inner, "embed_tokens") or hasattr(inner, "tok_emb"):
                    return inner  # type: ignore[return-value]
        return dec  # type: ignore[return-value]

    def _embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        dec = self._raw_decoder()
        return dec.embed_tokens(ids)

    def pool_and_project(
        self,
        struct_emb: torch.Tensor,
        cdr_spans: torch.Tensor,
        pad_mask: torch.Tensor,
        plddt: torch.Tensor,
    ) -> torch.Tensor:
        """Apply pooler then projector. Returns prefix tokens ``(B, K, d_l)``."""
        if self.pooler_kind == "module":
            pooled = self.pooler_module(struct_emb, cdr_spans, pad_mask, plddt)
        else:
            pooled = self._pooler_fn(struct_emb, cdr_spans, pad_mask, plddt)
        prefix = self.projector(pooled)
        return prefix

    # ------------------------------------------------------------------
    # forward / generate
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | None]:
        """Compute LM loss given a collated batch.

        Required keys: ``struct_emb``, ``cdr_spans``, ``pad_mask``, ``plddt``,
        ``seq_ids``, ``seq_pad_mask``.
        """
        struct_emb = batch["struct_emb"]
        cdr_spans = batch["cdr_spans"]
        pad_mask = batch["pad_mask"]
        plddt = batch["plddt"]
        seq_ids = batch["seq_ids"]
        seq_pad_mask = batch["seq_pad_mask"]

        prefix = self.pool_and_project(struct_emb, cdr_spans, pad_mask, plddt)
        tok_emb = self._embed_tokens(seq_ids)

        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)

        b, k, _ = prefix.shape
        prefix_attn = torch.ones(b, k, dtype=seq_pad_mask.dtype, device=seq_pad_mask.device)
        attention_mask = torch.cat([prefix_attn, seq_pad_mask.to(prefix_attn.dtype)], dim=1)

        # build labels: -100 for prefix and for pad positions in seq.
        prefix_labels = torch.full((b, k), -100, dtype=torch.long, device=seq_ids.device)
        seq_labels = seq_ids.clone()
        seq_labels[seq_pad_mask == 0] = -100
        labels = torch.cat([prefix_labels, seq_labels], dim=1)

        out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out

    @torch.no_grad()
    def generate(
        self,
        struct_emb: torch.Tensor,
        cdr_spans: torch.Tensor,
        pad_mask: torch.Tensor,
        plddt: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
        **gen_kwargs,
    ) -> torch.Tensor:
        prefix = self.pool_and_project(struct_emb, cdr_spans, pad_mask, plddt)
        dec = self._raw_decoder()
        return dec.generate(
            prefix_embeds=prefix,
            bos_id=bos_id,
            eos_id=eos_id,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )


# ---------------------------------------------------------------------------
# build_abllava
# ---------------------------------------------------------------------------

def _to_dict(cfg) -> dict:
    """Best-effort coercion of OmegaConf DictConfig (or plain dict) to plain dict."""
    if cfg is None:
        return {}
    try:
        from omegaconf import OmegaConf, DictConfig  # type: ignore

        if isinstance(cfg, DictConfig):
            return dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    except ImportError:
        pass
    if isinstance(cfg, dict):
        return dict(cfg)
    raise TypeError(f"cannot coerce {type(cfg)} to dict")


def build_decoder(cfg) -> AntibodyDecoder:
    """Instantiate an ``AntibodyDecoder`` from a config dict.

    Vocab size is taken from the tokenizer if not given.
    """
    cfg = _to_dict(cfg)
    cfg.pop("name", None)
    if "vocab_size" not in cfg:
        cfg["vocab_size"] = AntibodyTokenizer().vocab_size
    return AntibodyDecoder(**cfg)


def build_abllava(decoder_cfg, projector_cfg, pooling_cfg, lora_cfg=None) -> AbLLaVA:
    """Factory that assembles an :class:`AbLLaVA` from configs.

    Each config may be a plain dict or an OmegaConf ``DictConfig`` carrying a
    ``name`` field plus kwargs.
    """
    decoder = build_decoder(decoder_cfg)

    proj_d = _to_dict(projector_cfg)
    proj_name = proj_d.pop("name")
    projector = build_projector(proj_name, **proj_d)

    pool_d = _to_dict(pooling_cfg)
    pool_name = pool_d.pop("name")
    pool_d.setdefault("d", decoder.hidden_size if pool_name == "cdr_attention" else None)
    if pool_d.get("d") is None:
        pool_d.pop("d", None)
    pooler = build_pooler(pool_name, **pool_d)
    pooler_kind = "module" if isinstance(pooler, nn.Module) else "function"

    if lora_cfg is not None:
        lora_d = _to_dict(lora_cfg)
        if lora_d.pop("enabled", True):
            decoder = apply_lora(decoder, **lora_d)

    return AbLLaVA(decoder=decoder, projector=projector, pooler=pooler, pooler_kind=pooler_kind)
