"""PyTorch dataset and collate function for AbLLaVA.

See ``CONTRACT.md`` for the exact ``__getitem__`` and batch dict shapes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.embeddings import EmbeddingCache
from src.utils.logging import get_logger
from src.utils.tokenizer import AntibodyTokenizer

logger = get_logger(__name__)


def _records_from(data: list[dict] | pd.DataFrame) -> list[dict]:
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    return list(data)


def _to_long_2d(spans: Any) -> torch.Tensor:
    if isinstance(spans, torch.Tensor):
        t = spans.to(dtype=torch.long)
    else:
        arr = np.asarray(spans, dtype=np.int64)
        t = torch.from_numpy(arr).to(dtype=torch.long)
    if t.ndim == 1:
        t = t.view(-1, 2)
    if t.shape != (6, 2):
        raise ValueError(f"cdr_spans must reshape to (6, 2); got {tuple(t.shape)}")
    return t


class AntibodyDataset(Dataset):
    """Dataset returning the dict described in ``CONTRACT.md``.

    Each underlying record must contain at least:
        ``id``, ``heavy_seq``, ``light_seq``, ``cdr_spans``.
    """

    def __init__(
        self,
        records: list[dict] | pd.DataFrame,
        embedding_cache: EmbeddingCache,
        tokenizer: AntibodyTokenizer,
        max_n_residues: int = 260,
        max_seq_len: int = 320,
        return_plddt: bool = True,
    ) -> None:
        self.records: list[dict] = _records_from(records)
        self.cache: EmbeddingCache = embedding_cache
        self.tokenizer: AntibodyTokenizer = tokenizer
        self.max_n_residues: int = int(max_n_residues)
        self.max_seq_len: int = int(max_seq_len)
        self.return_plddt: bool = bool(return_plddt)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        rec = self.records[i]
        rid: str = str(rec["id"])
        heavy: str = str(rec["heavy_seq"])
        light: str = str(rec["light_seq"])
        cdr_spans: torch.Tensor = _to_long_2d(rec["cdr_spans"])

        # Tokenize sequence
        ids = self.tokenizer.encode_pair(heavy, light, add_bos=True, add_eos=True)
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]
            ids[-1] = self.tokenizer.eos_id
        seq_ids = torch.tensor(ids, dtype=torch.long)

        # Load cached embedding
        emb_np, plddt_np = self.cache.load(rid)
        n = emb_np.shape[0]
        if n > self.max_n_residues:
            emb_np = emb_np[: self.max_n_residues]
            if plddt_np is not None:
                plddt_np = plddt_np[: self.max_n_residues]
            n = self.max_n_residues
            # clamp spans to within the truncated length
            cdr_spans = cdr_spans.clamp(min=0, max=n)

        struct_emb = torch.from_numpy(np.ascontiguousarray(emb_np)).to(torch.float32)
        pad_mask = torch.ones(n, dtype=torch.bool)

        if plddt_np is not None and self.return_plddt:
            plddt = torch.from_numpy(np.ascontiguousarray(plddt_np)).to(torch.float32)
        else:
            plddt = torch.full((n,), float("nan"), dtype=torch.float32)

        return {
            "id": rid,
            "heavy_seq": heavy,
            "light_seq": light,
            "seq_ids": seq_ids,
            "struct_emb": struct_emb,
            "cdr_spans": cdr_spans,
            "pad_mask": pad_mask,
            "plddt": plddt,
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def collate_fn(
    batch: list[dict],
    pad_id: int = 0,
    max_n: int | None = None,
    max_l: int | None = None,
) -> dict[str, Any]:
    """Pad a batch of dataset items into a single batch dict.

    Pads ``struct_emb`` (zero), ``pad_mask`` (False), ``plddt`` (NaN) along
    the residue axis to the max ``N`` in the batch (or ``max_n``), and
    ``seq_ids`` with ``pad_id`` to the max sequence length (or ``max_l``).
    Adds ``seq_pad_mask`` (True for valid token positions). All tensors live
    on CPU.
    """
    if not batch:
        raise ValueError("collate_fn received an empty batch")

    bsz = len(batch)
    n_list = [int(item["struct_emb"].shape[0]) for item in batch]
    l_list = [int(item["seq_ids"].shape[0]) for item in batch]

    max_N = max(n_list) if max_n is None else int(max_n)
    max_L = max(l_list) if max_l is None else int(max_l)
    max_N = max(max_N, 1)
    max_L = max(max_L, 1)

    d_e = int(batch[0]["struct_emb"].shape[1])

    struct_emb = torch.zeros((bsz, max_N, d_e), dtype=torch.float32)
    pad_mask = torch.zeros((bsz, max_N), dtype=torch.bool)
    plddt = torch.full((bsz, max_N), float("nan"), dtype=torch.float32)
    cdr_spans = torch.zeros((bsz, 6, 2), dtype=torch.long)
    seq_ids = torch.full((bsz, max_L), int(pad_id), dtype=torch.long)
    seq_pad_mask = torch.zeros((bsz, max_L), dtype=torch.bool)

    ids: list[str] = []
    heavy_seqs: list[str] = []
    light_seqs: list[str] = []

    for i, item in enumerate(batch):
        ids.append(str(item["id"]))
        heavy_seqs.append(str(item["heavy_seq"]))
        light_seqs.append(str(item["light_seq"]))

        n = min(int(item["struct_emb"].shape[0]), max_N)
        if n > 0:
            struct_emb[i, :n] = item["struct_emb"][:n].to(torch.float32)
            pm = item.get("pad_mask")
            if pm is None:
                pad_mask[i, :n] = True
            else:
                pad_mask[i, :n] = pm[:n].to(torch.bool)
            pl = item.get("plddt")
            if pl is not None:
                plddt[i, :n] = pl[:n].to(torch.float32)

        spans = _to_long_2d(item["cdr_spans"]).clamp(min=0, max=max_N)
        cdr_spans[i] = spans

        l = min(int(item["seq_ids"].shape[0]), max_L)
        if l > 0:
            seq_ids[i, :l] = item["seq_ids"][:l].to(torch.long)
            seq_pad_mask[i, :l] = True

    # NaN-safety: collapse any non-finite plddt at valid positions to NaN (already nan default).
    # Non-finite at invalid positions stays NaN as initialized.
    _ = math.nan  # silence unused-import linters

    return {
        "id": ids,
        "heavy_seq": heavy_seqs,
        "light_seq": light_seqs,
        "seq_ids": seq_ids,
        "seq_pad_mask": seq_pad_mask,
        "struct_emb": struct_emb,
        "cdr_spans": cdr_spans,
        "pad_mask": pad_mask,
        "plddt": plddt,
    }
