"""Amino-acid level tokenizer for paired antibody decoder.

Vocab (25 tokens):
    0: [PAD]   1: [BOS]   2: [EOS]   3: [SEP]   4: [MASK]
    5..24: 20 amino acids in canonical order
"""

from __future__ import annotations

from typing import Iterable

import torch

SPECIAL_TOKENS: dict[str, int] = {
    "[PAD]": 0,
    "[BOS]": 1,
    "[EOS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
}

AA_VOCAB: list[str] = list("ACDEFGHIKLMNPQRSTVWY")

PAD_ID = SPECIAL_TOKENS["[PAD]"]
BOS_ID = SPECIAL_TOKENS["[BOS]"]
EOS_ID = SPECIAL_TOKENS["[EOS]"]
SEP_ID = SPECIAL_TOKENS["[SEP]"]
MASK_ID = SPECIAL_TOKENS["[MASK]"]


class AntibodyTokenizer:
    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = dict(SPECIAL_TOKENS)
        for i, aa in enumerate(AA_VOCAB):
            self.token_to_id[aa] = len(SPECIAL_TOKENS) + i
        self.id_to_token: dict[int, str] = {i: t for t, i in self.token_to_id.items()}
        self.vocab_size: int = len(self.token_to_id)
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.sep_id = SEP_ID
        self.mask_id = MASK_ID

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, seq: str) -> list[int]:
        return [self.token_to_id[c] for c in seq.upper() if c in self.token_to_id]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i not in self.id_to_token:
                continue
            tok = self.id_to_token[i]
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)

    def encode_pair(
        self,
        heavy: str,
        light: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.encode(heavy))
        ids.append(self.sep_id)
        ids.extend(self.encode(light))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def pad_batch(
        self,
        sequences: list[list[int]],
        max_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pads a batch of token id lists. Returns (input_ids, attention_mask)."""
        if max_len is None:
            max_len = max(len(s) for s in sequences)
        input_ids = torch.full((len(sequences), max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for i, s in enumerate(sequences):
            n = min(len(s), max_len)
            input_ids[i, :n] = torch.tensor(s[:n], dtype=torch.long)
            attention_mask[i, :n] = 1
        return input_ids, attention_mask

    def split_pair_ids(self, ids: list[int]) -> tuple[str, str]:
        """Decode a paired token sequence back to (heavy, light)."""
        if self.sep_id in ids:
            sep = ids.index(self.sep_id)
            heavy = self.decode(ids[:sep])
            light = self.decode(ids[sep + 1 :])
        else:
            heavy = self.decode(ids)
            light = ""
        return heavy, light
