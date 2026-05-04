"""Sequence-level evaluation metrics for AbLLaVA.

Provides:
    * ``compute_perplexity`` — overall and per-region (CDR-H3, other CDRs, FWK).
    * ``compute_recovery`` — CDR-infill amino-acid recovery (AAR).
    * ``levenshtein`` — pure-Python edit distance.
    * ``compute_diversity`` — random-pair Levenshtein + BLOSUM62 score.
    * ``aa_kl`` — KL divergence between observed AA composition and uniform.

All heavy ops avoid prints and use ``get_logger``.
"""

from __future__ import annotations

import math
import random
from typing import Iterable

import torch
import torch.nn.functional as F

from src.utils.logging import get_logger
from src.utils.tokenizer import AntibodyTokenizer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# BLOSUM62 (hand-coded fallback so we don't hard-depend on biopython internals)
# ---------------------------------------------------------------------------

_BLOSUM62_AAS = "ARNDCQEGHILKMFPSTWYV"
# Standard BLOSUM62 matrix in row order matching _BLOSUM62_AAS.
_BLOSUM62_MATRIX: list[list[int]] = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
]


def _blosum62_dict() -> dict[tuple[str, str], int]:
    try:
        from Bio.Align import substitution_matrices  # type: ignore

        m = substitution_matrices.load("BLOSUM62")
        d: dict[tuple[str, str], int] = {}
        for a in m.alphabet:
            for b in m.alphabet:
                d[(a, b)] = int(m[a, b])
        return d
    except Exception:  # noqa: BLE001
        d = {}
        for i, a in enumerate(_BLOSUM62_AAS):
            for j, b in enumerate(_BLOSUM62_AAS):
                d[(a, b)] = _BLOSUM62_MATRIX[i][j]
        return d


_BLOSUM62 = _blosum62_dict()


def _blosum62_score(a: str, b: str) -> float:
    """Sum of BLOSUM62 substitution scores aligned position-wise.

    Sequences of unequal length are aligned at the start; the shorter
    length is used.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    s = 0
    for i in range(n):
        ca = a[i].upper()
        cb = b[i].upper()
        s += _BLOSUM62.get((ca, cb), 0)
    return float(s) / n


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


def _region_masks(
    seq_ids: torch.Tensor,
    cdr_spans: torch.Tensor,
    seq_pad_mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Build (B, L) boolean masks for ``cdr_h3``, ``cdr_other``, ``fwk``.

    ``cdr_spans`` is given in **structure** coordinates (per CONTRACT.md). The
    sequence and structure indices may not align exactly because of BOS/SEP/EOS
    tokens. We approximate the partition by mapping cdr_spans into the
    decoder's sequence indexing assuming token layout
    ``[BOS] heavy [SEP] light [EOS]`` and that struct embeddings concatenate
    heavy then light residues in the same order. Heavy CDR H1/H2/H3 occupy
    positions ``span + 1`` (offset by BOS); light CDR L1/L2/L3 occupy
    positions ``span + 2 + n_heavy`` accounting for BOS+SEP. Because we don't
    know ``n_heavy`` reliably here, we treat the spans as 1-indexed within the
    valid (non-pad) region — sufficient for partitioned NLL averaging.
    """
    B, L = seq_ids.shape
    device = seq_ids.device
    h3_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    other_cdr = torch.zeros(B, L, dtype=torch.bool, device=device)
    for b in range(B):
        spans = cdr_spans[b].tolist()
        # H1, H2, H3, L1, L2, L3
        for idx, (s, e) in enumerate(spans):
            if e <= s:
                continue
            # Add 1 for BOS offset; clamp into valid range.
            s_t = max(1, int(s) + 1)
            e_t = min(L, int(e) + 1)
            if e_t <= s_t:
                continue
            if idx == 2:
                h3_mask[b, s_t:e_t] = True
            else:
                other_cdr[b, s_t:e_t] = True
    valid = (
        seq_pad_mask.bool()
        if seq_pad_mask is not None
        else torch.ones_like(seq_ids, dtype=torch.bool)
    )
    fwk_mask = valid & ~h3_mask & ~other_cdr
    return {
        "cdr_h3": h3_mask & valid,
        "cdr_other": other_cdr & valid,
        "fwk": fwk_mask,
        "overall": valid,
    }


@torch.no_grad()
def compute_perplexity(model, loader, device) -> dict:
    """Compute per-token perplexity overall and per region.

    Region partition uses ``cdr_spans`` from each batch. Returns a dict with
    keys ``overall``, ``cdr_h3``, ``cdr_other``, ``fwk``.
    """
    model.eval()
    region_keys = ["overall", "cdr_h3", "cdr_other", "fwk"]
    nll_sum: dict[str, float] = {k: 0.0 for k in region_keys}
    tok_count: dict[str, int] = {k: 0 for k in region_keys}

    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        out = model(batch)
        logits: torch.Tensor = out["logits"]  # (B, L_pref+L, V) or (B, L, V)
        seq_ids: torch.Tensor = batch["seq_ids"]
        B, L = seq_ids.shape
        # Align logits to seq_ids by taking the trailing L positions.
        if logits.size(1) > L:
            logits = logits[:, -L:, :]

        # Shifted CE: predict token at position t from logits at t-1.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = seq_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(B, L - 1)

        seq_pad_mask = batch.get("seq_pad_mask")
        cdr_spans: torch.Tensor = batch["cdr_spans"]
        masks = _region_masks(seq_ids, cdr_spans, seq_pad_mask)
        # Use targets (positions 1..L-1) for region attribution.
        for k, m in masks.items():
            m_shift = m[:, 1:]
            n = int(m_shift.sum().item())
            if n == 0:
                continue
            nll_sum[k] += float((loss * m_shift.float()).sum().item())
            tok_count[k] += n

    out: dict[str, float] = {}
    for k in region_keys:
        if tok_count[k] == 0:
            out[k] = float("nan")
        else:
            out[k] = math.exp(nll_sum[k] / tok_count[k])
    return out


# ---------------------------------------------------------------------------
# CDR recovery (AAR)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_recovery(
    model,
    loader,
    tokenizer: AntibodyTokenizer,
    device,
    cdrs: tuple[str, ...] | list[str] = ("H1", "H2", "H3", "L1", "L2", "L3"),
) -> dict:
    """For each sample, mask each CDR and predict argmax tokens.

    Reports per-CDR amino-acid recovery and a ``mean`` across the requested
    CDRs.
    """
    model.eval()
    cdr_index = {"H1": 0, "H2": 1, "H3": 2, "L1": 3, "L2": 4, "L3": 5}
    correct: dict[str, int] = {c: 0 for c in cdrs}
    total: dict[str, int] = {c: 0 for c in cdrs}

    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        seq_ids: torch.Tensor = batch["seq_ids"]
        cdr_spans: torch.Tensor = batch["cdr_spans"]
        B, L = seq_ids.shape

        for cdr in cdrs:
            idx = cdr_index[cdr]
            masked_ids = seq_ids.clone()
            # Build (B, L) mask of CDR positions (offset by BOS).
            target_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
            for b in range(B):
                s, e = cdr_spans[b, idx].tolist()
                if e <= s:
                    continue
                s_t = max(1, int(s) + 1)
                e_t = min(L, int(e) + 1)
                if e_t <= s_t:
                    continue
                target_mask[b, s_t:e_t] = True
            if target_mask.sum() == 0:
                continue
            masked_ids[target_mask] = tokenizer.mask_id
            masked_batch = dict(batch)
            masked_batch["seq_ids"] = masked_ids
            out = model(masked_batch)
            logits: torch.Tensor = out["logits"]
            if logits.size(1) > L:
                logits = logits[:, -L:, :]
            preds = logits.argmax(dim=-1)
            ok = (preds == seq_ids) & target_mask
            correct[cdr] += int(ok.sum().item())
            total[cdr] += int(target_mask.sum().item())

    result: dict[str, float] = {}
    vals: list[float] = []
    for c in cdrs:
        if total[c] == 0:
            result[c] = float("nan")
        else:
            v = correct[c] / total[c]
            result[c] = v
            vals.append(v)
    result["mean"] = float(sum(vals) / len(vals)) if vals else float("nan")
    return result


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    """Classic dynamic-programming Levenshtein edit distance."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, cb in enumerate(b, 1):
        cur = [j] + [0] * len(a)
        for i, ca in enumerate(a, 1):
            cost = 0 if ca == cb else 1
            cur[i] = min(
                cur[i - 1] + 1,
                prev[i] + 1,
                prev[i - 1] + cost,
            )
        prev = cur
    return prev[-1]


def compute_diversity(
    sequences: list[str],
    pairs: int = 1000,
    seed: int = 42,
) -> dict:
    """Random-pair sampling diversity.

    Returns ``{'mean_levenshtein': float, 'mean_blosum62': float}``. If fewer
    than two sequences are provided, both values are ``NaN``.
    """
    if len(sequences) < 2:
        return {"mean_levenshtein": float("nan"), "mean_blosum62": float("nan")}
    rng = random.Random(seed)
    n = len(sequences)
    lev_sum = 0.0
    blo_sum = 0.0
    count = 0
    for _ in range(pairs):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        a, b = sequences[i], sequences[j]
        lev_sum += levenshtein(a, b)
        blo_sum += _blosum62_score(a, b)
        count += 1
    if count == 0:
        return {"mean_levenshtein": float("nan"), "mean_blosum62": float("nan")}
    return {
        "mean_levenshtein": lev_sum / count,
        "mean_blosum62": blo_sum / count,
    }


def aa_kl(seqs: Iterable[str]) -> float:
    """KL divergence of observed amino-acid distribution from uniform-over-20."""
    aas = "ACDEFGHIKLMNPQRSTVWY"
    counts: dict[str, int] = {a: 0 for a in aas}
    total = 0
    for s in seqs:
        for c in s.upper():
            if c in counts:
                counts[c] += 1
                total += 1
    if total == 0:
        return float("nan")
    uniform = 1.0 / 20.0
    kl = 0.0
    for a in aas:
        p = counts[a] / total
        if p == 0:
            continue
        kl += p * math.log(p / uniform)
    return kl
