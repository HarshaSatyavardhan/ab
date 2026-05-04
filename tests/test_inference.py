"""Tests for the inference pipeline.

These tests avoid depending on ``src.models`` being fully built (a parallel
agent fills that in). Instead we provide a tiny ``DummyAbLLaVA`` that mimics
the small interface :func:`generate_from_embedding` requires.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from src.inference.generate import GenerationConfig, generate_from_embedding
from src.inference.io import write_fasta
from src.utils.tokenizer import AntibodyTokenizer


class DummyAbLLaVA(torch.nn.Module):
    """Tiny stand-in implementing the ``AbLLaVA.generate`` contract.

    Produces a deterministic-ish heavy/light pair token stream so we can
    assert post-processing works.
    """

    def __init__(self, tokenizer: AntibodyTokenizer, heavy_len: int = 12, light_len: int = 10) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.heavy_len = heavy_len
        self.light_len = light_len
        # Make ``next(model.parameters())`` work for device introspection.
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def generate(
        self,
        struct_emb: torch.Tensor,
        cdr_spans: torch.Tensor,
        pad_mask: torch.Tensor,
        plddt: torch.Tensor | None,
        *,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int | None = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        b = struct_emb.shape[0]
        # AA token ids start at 5 (after the 5 special tokens).
        aa_ids = torch.arange(5, 25)
        heavy = aa_ids[: self.heavy_len].tolist()
        light = aa_ids[: self.light_len].tolist()
        sep = self.tokenizer.sep_id
        seq = [bos_id] + heavy + [sep] + light + [eos_id]
        # Pad to ``max_new_tokens`` so shape is consistent.
        pad_id = self.tokenizer.pad_id
        if len(seq) < max_new_tokens:
            seq = seq + [pad_id] * (max_new_tokens - len(seq))
        else:
            seq = seq[:max_new_tokens]
        out = torch.tensor(seq, dtype=torch.long).unsqueeze(0).repeat(b, 1)
        return out


def _make_inputs(d: int = 8, n: int = 16):
    struct_emb = torch.randn(1, n, d)
    cdr_spans = torch.tensor(
        [[1, 4], [5, 8], [9, 11], [12, 13], [13, 14], [14, 15]], dtype=torch.long
    ).unsqueeze(0)
    pad_mask = torch.ones(1, n, dtype=torch.bool)
    plddt = torch.full((1, n), 80.0)
    return struct_emb, cdr_spans, pad_mask, plddt


def test_generate_from_embedding_decodes_to_valid_aa() -> None:
    tokenizer = AntibodyTokenizer()
    model = DummyAbLLaVA(tokenizer)
    struct_emb, cdr_spans, pad_mask, plddt = _make_inputs()
    cfg = GenerationConfig(n_samples=4, max_new_tokens=64, seed=0)

    records = generate_from_embedding(model, tokenizer, struct_emb, cdr_spans, pad_mask, plddt, cfg)

    assert len(records) == 4
    aa_alphabet = set("ACDEFGHIKLMNPQRSTVWY")
    special_markers = {"[", "]"}
    for rec in records:
        assert set(rec.keys()) >= {"id", "heavy", "light", "tokens"}
        assert rec["heavy"], "heavy chain should be non-empty"
        assert rec["light"], "light chain should be non-empty"
        for ch in rec["heavy"] + rec["light"]:
            assert ch in aa_alphabet, f"unexpected residue char: {ch!r}"
        # No special tokens leaked in textual form.
        for marker in special_markers:
            assert marker not in rec["heavy"]
            assert marker not in rec["light"]


def test_generate_from_embedding_no_sep_treats_all_as_heavy() -> None:
    tokenizer = AntibodyTokenizer()

    class NoSepModel(DummyAbLLaVA):
        @torch.no_grad()
        def generate(self, struct_emb, cdr_spans, pad_mask, plddt, *, bos_id, eos_id,
                     max_new_tokens, **kw):  # type: ignore[override]
            b = struct_emb.shape[0]
            aa = list(range(5, 15))
            seq = [bos_id] + aa + [eos_id]
            pad_id = self.tokenizer.pad_id
            seq = seq + [pad_id] * max(0, max_new_tokens - len(seq))
            seq = seq[:max_new_tokens]
            return torch.tensor(seq, dtype=torch.long).unsqueeze(0).repeat(b, 1)

    model = NoSepModel(tokenizer)
    struct_emb, cdr_spans, pad_mask, plddt = _make_inputs()
    cfg = GenerationConfig(n_samples=2, max_new_tokens=32, seed=1)

    records = generate_from_embedding(model, tokenizer, struct_emb, cdr_spans, pad_mask, plddt, cfg)
    for rec in records:
        assert rec["heavy"]
        assert rec["light"] == ""


def test_write_fasta_roundtrip(tmp_path: Path) -> None:
    records = [
        {"id": "abc_0001", "heavy": "QVQLVESGGG", "light": "DIQMTQSPSS"},
        {"id": "abc_0002", "heavy": "EVQLVESGGG", "light": "DIVMTQSPDS"},
    ]
    out = tmp_path / "samples.fasta"
    write_fasta(records, out)

    content = out.read_text().strip().splitlines()
    assert content[0] == ">abc_0001_H"
    assert content[1] == "QVQLVESGGG"
    assert content[2] == ">abc_0001_L"
    assert content[3] == "DIQMTQSPSS"
    assert content[4] == ">abc_0002_H"
    assert content[5] == "EVQLVESGGG"
    assert content[6] == ">abc_0002_L"
    assert content[7] == "DIVMTQSPDS"

    # Round-trip parse: reconstruct {id: (heavy, light)} from the file.
    parsed: dict[str, dict[str, str]] = {}
    cur_id, cur_kind = None, None
    for line in content:
        if line.startswith(">"):
            tag = line[1:]
            cur_id, cur_kind = tag.rsplit("_", 1)
            parsed.setdefault(cur_id, {})
        else:
            assert cur_id is not None and cur_kind is not None
            parsed[cur_id][cur_kind] = line
    assert parsed["abc_0001"] == {"H": "QVQLVESGGG", "L": "DIQMTQSPSS"}
    assert parsed["abc_0002"] == {"H": "EVQLVESGGG", "L": "DIVMTQSPDS"}


def test_write_fasta_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "deep" / "out.fasta"
    write_fasta([{"id": "x", "heavy": "AC", "light": "DE"}], out)
    assert out.is_file()


def test_generate_from_pdb_skipped_without_encoder() -> None:
    if importlib.util.find_spec("antifold") is None:
        pytest.importorskip("antifold")
    # If antifold is somehow available we still skip — exercising the real
    # encoder requires a real PDB which we do not ship with the test suite.
    pytest.skip("PDB integration test requires a real encoder + fixture PDB.")
