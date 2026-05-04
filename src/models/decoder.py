"""From-scratch GPT-style antibody decoder.

- RoPE rotary positional embeddings.
- RMSNorm pre-norm.
- SwiGLU FFN.
- Standard causal self-attention via ``F.scaled_dot_product_attention`` with
  ``is_causal=True``.

At the default config (``d_model=1024, n_layers=16, n_heads=16, d_ff=4096``) the
parameter count lands near 150M with weight tying.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def precompute_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) tables for rotary embeddings.

    Returns two tensors of shape ``(seq_len, head_dim/2)`` each.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (seq_len, half)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to ``x`` of shape ``(B, H, L, D)``.

    ``cos`` and ``sin`` are of shape ``(L, D/2)``.
    """
    b, h, l, d = x.shape
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos[:l].view(1, 1, l, half).to(dtype=x.dtype)
    sin = sin[:l].view(1, 1, l, half).to(dtype=x.dtype)
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.cat([rx1, rx2], dim=-1)


# ---------------------------------------------------------------------------
# Norms / FFN
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast to float32 for numerical stability
        f = x.float()
        rms = f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (f * rms).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    """Gated FFN: ``x, g = chunk(W1(x))``; ``SiLU(g) * x``; ``W2(...)``."""

    def __init__(self, d: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_in = nn.Linear(d, 2 * d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.w_in.weight, std=0.02)
        nn.init.normal_(self.w_out.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, g = self.w_in(x).chunk(2, dim=-1)
        return self.drop(self.w_out(F.silu(g) * a))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention.

    Exposes ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj`` for LoRA targeting.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        rope_base: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.normal_(lin.weight, std=0.02)
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        cos, sin = precompute_rope_cache(max_seq_len, self.head_dim, rope_base, torch.device("cpu"), torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, l, _ = x.shape
        q = self.q_proj(x).view(b, l, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        k = self.k_proj(x).view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, l, self.n_heads, self.head_dim).transpose(1, 2)

        cos = self.rope_cos.to(device=x.device)
        sin = self.rope_sin.to(device=x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # We rely on the SDPA causal mask. Padding positions are simply ignored at
        # the loss level via labels=-100; if attention_mask is supplied, we build an
        # additive bias that adds the causal mask combined with key-padding masking.
        if attention_mask is not None:
            # attention_mask: (B, L) with 1 = valid, 0 = pad
            key_pad = attention_mask.to(torch.bool)  # (B, L)
            # build additive mask (B, 1, L, L)
            causal = torch.ones(l, l, dtype=torch.bool, device=x.device).tril()
            allow = causal.unsqueeze(0) & key_pad.unsqueeze(1)  # (B, L, L)
            attn_bias = torch.zeros(b, 1, l, l, dtype=q.dtype, device=x.device)
            attn_bias.masked_fill_(~allow.unsqueeze(1), float("-inf"))
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        out = out.transpose(1, 2).contiguous().view(b, l, self.d_model)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Block / model
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_base: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, rope_base, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), attention_mask=attention_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class AntibodyDecoder(nn.Module):
    """GPT-style causal decoder.

    Notes:
        * ``forward`` accepts EITHER ``input_ids`` OR ``inputs_embeds``, never both.
        * ``attention_mask`` is ``(B, L)``, 1 = valid, 0 = pad. When provided, an
          explicit additive attention bias is built; otherwise the cheap SDPA
          causal-only path is used.
        * Loss masking is via ``labels=-100`` (standard ``CrossEntropyLoss`` with
          ``ignore_index=-100``).
        * ``generate`` re-encodes the full prefix each step (no KV cache for v0).
          This is fine for the ``L <= 512`` we use in training/inference.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 1024,
        n_layers: int = 16,
        n_heads: int = 16,
        d_ff: int = 4096,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
        tie_word_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, max_seq_len, rope_base, dropout)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        if tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # optional torch.compile via env var
        if os.environ.get("ABLLAVA_COMPILE", "0") == "1":
            try:
                self.blocks = nn.ModuleList([torch.compile(b) for b in self.blocks])
            except Exception as e:  # noqa: BLE001
                logger.warning("torch.compile failed: %s", e)

    @property
    def hidden_size(self) -> int:
        return self.d_model

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must provide input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        x = self.drop(inputs_embeds)
        for blk in self.blocks:
            x = blk(x, attention_mask=attention_mask)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # standard next-token prediction shift
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prefix_embeds: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int | None = None,
        do_sample: bool = True,
        pad_id: int = 0,
        sep_id: int | None = None,
    ) -> torch.Tensor:
        """Sample / greedy decode starting from a soft prefix.

        Appends the BOS token's embedding to ``prefix_embeds`` and decodes step by
        step, re-encoding the full sequence each step (no KV cache for v0).
        Returns generated token ids only (excludes the prefix), shape ``(B, T)``
        with ``T <= max_new_tokens``. EOS terminates a sample; subsequent positions
        are filled with ``pad_id``.
        """
        was_training = self.training
        self.eval()
        try:
            b = prefix_embeds.size(0)
            device = prefix_embeds.device
            bos_emb = self.tok_emb(torch.full((b, 1), bos_id, dtype=torch.long, device=device))
            cur = torch.cat([prefix_embeds, bos_emb], dim=1)  # (B, K+1, d)

            generated = torch.full((b, max_new_tokens), pad_id, dtype=torch.long, device=device)
            done = torch.zeros(b, dtype=torch.bool, device=device)

            for t in range(max_new_tokens):
                if cur.size(1) > self.max_seq_len:
                    logger.warning("generate sequence exceeded max_seq_len; truncating")
                    cur = cur[:, -self.max_seq_len:]
                out = self.forward(inputs_embeds=cur)
                logits = out["logits"][:, -1, :]
                next_ids = self._sample_next(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                )
                next_ids = torch.where(done, torch.full_like(next_ids, pad_id), next_ids)
                generated[:, t] = next_ids
                done = done | (next_ids == eos_id)
                if done.all():
                    break
                next_emb = self.tok_emb(next_ids).unsqueeze(1)
                cur = torch.cat([cur, next_emb], dim=1)
            return generated
        finally:
            if was_training:
                self.train()

    @staticmethod
    def _sample_next(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int | None,
        do_sample: bool,
    ) -> torch.Tensor:
        if not do_sample:
            return logits.argmax(dim=-1)
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            kth = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(-1, sorted_idx, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
