"""Loss/labeling helpers for AbLLaVA training.

The decoder is trained with a causal-LM objective on the paired-sequence
``seq_ids``.  In Stage B we additionally inject a CDR-infill objective:
with probability ``mask_prob`` we mask one CDR span (sampled uniformly over
H1..L3) and only score the masked positions in the loss.
"""

from __future__ import annotations

import random
from typing import Any

import torch

from src.utils.tokenizer import (
    BOS_ID,
    EOS_ID,
    MASK_ID,
    PAD_ID,
    SEP_ID,
    AntibodyTokenizer,
)


def causal_lm_labels(seq_ids: torch.Tensor, pad_id: int = PAD_ID) -> torch.Tensor:
    """Return labels for a causal-LM loss: pad positions become -100."""
    labels = seq_ids.clone()
    labels[labels == pad_id] = -100
    return labels


def _heavy_light_token_offsets(ids_row: torch.Tensor) -> tuple[int, int, int]:
    """Locate heavy / light token start indices in a single ``seq_ids`` row.

    Returns ``(heavy_start, sep_idx, eos_or_end)`` such that:
        - heavy AAs occupy ``[heavy_start, sep_idx)``
        - light AAs occupy ``[sep_idx + 1, eos_or_end)``
    The heavy_start is the index right after the BOS (if any), else 0.
    """
    seq = ids_row.tolist()
    heavy_start = 1 if len(seq) > 0 and seq[0] == BOS_ID else 0

    try:
        sep_idx = seq.index(SEP_ID)
    except ValueError:
        sep_idx = len(seq)

    # Find EOS strictly after sep_idx; default to end of sequence.
    eos_or_end = len(seq)
    for j in range(sep_idx + 1, len(seq)):
        if seq[j] == EOS_ID or seq[j] == PAD_ID:
            eos_or_end = j
            break

    return heavy_start, sep_idx, eos_or_end


def make_cdr_infill_batch(
    batch: dict[str, Any],
    tokenizer: AntibodyTokenizer,
    mask_prob: float = 0.5,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Build a CDR-infill batch in token-space.

    For each row, with probability ``mask_prob`` we sample one CDR span
    (uniformly over the 6 spans in ``cdr_spans``), translate its
    ``struct_emb``-space indices into ``seq_ids``-space positions, replace
    those positions with ``[MASK]`` tokens and create a labels tensor where
    only those positions contribute (others = -100).

    For a non-infill row we fall back to a standard causal-LM label (pad ->
    -100).  The resulting batch dict is a shallow copy with ``seq_ids`` and
    ``labels`` tensors replaced.
    """
    if rng is None:
        rng = random.Random()

    seq_ids: torch.Tensor = batch["seq_ids"].clone()
    cdr_spans: torch.Tensor = batch.get("cdr_spans")
    pad_mask: torch.Tensor | None = batch.get("pad_mask")

    B, L = seq_ids.shape
    labels = torch.full_like(seq_ids, fill_value=-100)
    mask_id = int(getattr(tokenizer, "mask_id", MASK_ID))
    pad_id = int(getattr(tokenizer, "pad_id", PAD_ID))

    # Compute, per row, the count of valid heavy and light residues in the
    # struct_emb space.  We need this so we can map a struct-space CDR
    # interval onto AA positions within the heavy or light chain.
    heavy_struct_lens: list[int] = []
    if pad_mask is not None and cdr_spans is not None:
        # The struct_emb is [heavy_residues, light_residues] concatenated.
        # The boundary between heavy and light is taken from the last H-CDR
        # (index 2 = H3) end position when available; otherwise we fall back
        # on the heavy decoded sequence length.
        for b in range(B):
            heavy_len_in_struct = 0
            if cdr_spans is not None and cdr_spans.shape[1] >= 6:
                # H3 end is the rightmost heavy CDR boundary.
                heavy_len_in_struct = int(cdr_spans[b, 2, 1].item())
            heavy_struct_lens.append(heavy_len_in_struct)
    else:
        heavy_struct_lens = [0] * B

    for b in range(B):
        row_labels = seq_ids[b].clone()
        row_labels[row_labels == pad_id] = -100

        do_infill = (
            cdr_spans is not None
            and rng.random() < float(mask_prob)
        )

        if do_infill:
            heavy_start, sep_idx, end_idx = _heavy_light_token_offsets(seq_ids[b])
            light_start = sep_idx + 1
            heavy_aa_count = max(sep_idx - heavy_start, 0)
            light_aa_count = max(end_idx - light_start, 0)
            heavy_struct_len = heavy_struct_lens[b] if b < len(heavy_struct_lens) else heavy_aa_count

            cdr_idx = rng.randrange(0, 6)
            s = int(cdr_spans[b, cdr_idx, 0].item())
            e = int(cdr_spans[b, cdr_idx, 1].item())
            if e <= s:
                # invalid / empty span — fall back to plain CE.
                pass
            else:
                if cdr_idx < 3:
                    # Heavy CDR — struct positions are within the heavy chain.
                    s_aa = max(s, 0)
                    e_aa = min(e, heavy_aa_count)
                    tok_start = heavy_start + s_aa
                    tok_end = heavy_start + e_aa
                    tok_end = min(tok_end, sep_idx)
                else:
                    # Light CDR — struct positions are heavy_struct_len + offset
                    # into the light chain, but we only know light positions
                    # relative to the light chain.  Convert by subtracting
                    # heavy_struct_len.
                    s_aa = max(s - heavy_struct_len, 0)
                    e_aa = max(e - heavy_struct_len, 0)
                    e_aa = min(e_aa, light_aa_count)
                    tok_start = light_start + s_aa
                    tok_end = light_start + e_aa
                    tok_end = min(tok_end, end_idx)

                if tok_end > tok_start:
                    row_labels = torch.full_like(seq_ids[b], fill_value=-100)
                    row_labels[tok_start:tok_end] = seq_ids[b, tok_start:tok_end]
                    seq_ids[b, tok_start:tok_end] = mask_id

        labels[b] = row_labels

    new_batch = dict(batch)
    new_batch["seq_ids"] = seq_ids
    new_batch["labels"] = labels
    return new_batch


def cdr_infill_collate(
    samples: list[dict[str, Any]],
    base_collate,
    tokenizer: AntibodyTokenizer,
    mask_prob: float = 0.5,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Wrap an existing ``collate_fn`` with on-the-fly CDR-infill masking."""
    batch = base_collate(samples)
    return make_cdr_infill_batch(batch, tokenizer, mask_prob=mask_prob, rng=rng)
