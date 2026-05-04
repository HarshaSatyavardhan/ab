"""Unit tests for the AbLLaVA data pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import AntibodyDataset, collate_fn
from src.data.embeddings import EmbeddingCache
from src.data.filter import OASFilter
from src.data.splits import cluster_split, concat_cdr_string
from src.utils.tokenizer import AntibodyTokenizer


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------
def _make_good_row(heavy_len: int = 120, light_len: int = 110, h3_len: int = 12) -> dict:
    h = "ACDEFGHIKL" * (heavy_len // 10 + 1)
    l = "MNPQRSTVWY" * (light_len // 10 + 1)
    h3 = "A" * h3_len
    return {
        "sequence_alignment_aa_heavy": h[:heavy_len],
        "sequence_alignment_aa_light": l[:light_len],
        "cdr3_aa_heavy": h3,
        "productive_heavy": True,
        "productive_light": True,
        "vj_in_frame_heavy": True,
        "vj_in_frame_light": True,
        "stop_codon_heavy": False,
        "stop_codon_light": False,
        "ANARCI_status_heavy": "",
        "ANARCI_status_light": "",
        "species": "human",
    }


# ---------------------------------------------------------------------------
# OASFilter tests
# ---------------------------------------------------------------------------
class TestOASFilter:
    def setup_method(self) -> None:
        self.f = OASFilter()

    def test_passes_good_row(self) -> None:
        assert self.f.filter_row(_make_good_row()) is True

    def test_fails_non_productive(self) -> None:
        r = _make_good_row()
        r["productive_heavy"] = False
        assert self.f.filter_row(r) is False

    def test_fails_out_of_frame(self) -> None:
        r = _make_good_row()
        r["vj_in_frame_light"] = False
        assert self.f.filter_row(r) is False

    def test_fails_stop_codon(self) -> None:
        r = _make_good_row()
        r["stop_codon_heavy"] = True
        assert self.f.filter_row(r) is False

    def test_fails_anarci_unusual(self) -> None:
        r = _make_good_row()
        r["ANARCI_status_heavy"] = "Unusual residue at position 25"
        assert self.f.filter_row(r) is False

    def test_fails_anarci_indel(self) -> None:
        r = _make_good_row()
        r["ANARCI_status_light"] = "Indel detected"
        assert self.f.filter_row(r) is False

    def test_fails_too_short(self) -> None:
        r = _make_good_row(heavy_len=50)
        assert self.f.filter_row(r) is False

    def test_fails_too_long(self) -> None:
        r = _make_good_row(heavy_len=200)
        assert self.f.filter_row(r) is False

    def test_fails_h3_too_short(self) -> None:
        r = _make_good_row(h3_len=2)
        assert self.f.filter_row(r) is False

    def test_fails_h3_too_long(self) -> None:
        r = _make_good_row(h3_len=40)
        assert self.f.filter_row(r) is False

    def test_fails_unusual_residue(self) -> None:
        r = _make_good_row()
        seq = ("ACDEFGHIKLMNPQRSTVWY" * 6)[:120]
        # inject an X near the middle so it survives any slicing
        r["sequence_alignment_aa_heavy"] = seq[:60] + "X" + seq[61:]
        assert self.f.filter_row(r) is False

    def test_fails_wrong_species(self) -> None:
        r = _make_good_row()
        r["species"] = "mouse"
        assert self.f.filter_row(r) is False

    def test_anarci_optional_when_missing(self) -> None:
        r = _make_good_row()
        r.pop("ANARCI_status_heavy")
        r.pop("ANARCI_status_light")
        assert self.f.filter_row(r) is True


# ---------------------------------------------------------------------------
# EmbeddingCache round-trip
# ---------------------------------------------------------------------------
def test_embedding_cache_roundtrip(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, embedding_dim=16)
    rng = np.random.default_rng(0)
    emb = rng.standard_normal(size=(32, 16)).astype(np.float32)
    plddt = rng.uniform(0, 100, size=(32,)).astype(np.float32)

    assert cache.has("x") is False
    cache.save("x", emb, plddt)
    assert cache.has("x") is True

    emb2, plddt2 = cache.load("x")
    np.testing.assert_array_equal(emb, emb2)
    assert plddt2 is not None
    np.testing.assert_array_equal(plddt, plddt2)


def test_embedding_cache_no_plddt(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, embedding_dim=8)
    emb = np.zeros((4, 8), dtype=np.float32)
    cache.save("y", emb)
    e, p = cache.load("y")
    assert p is None
    np.testing.assert_array_equal(e, emb)


def test_embedding_cache_dim_check(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, embedding_dim=16)
    with pytest.raises(ValueError):
        cache.save("bad", np.zeros((4, 8), dtype=np.float32))


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def test_concat_cdr_string() -> None:
    heavy = "ABCDEFGHIJ"
    light = "KLMNOPQRST"
    spans = torch.tensor(
        [[0, 2], [3, 5], [6, 9], [10, 12], [13, 15], [16, 19]], dtype=torch.long
    )
    s = concat_cdr_string(heavy, light, spans)
    assert s == "AB" + "DE" + "GHI" + "KL" + "NO" + "QRS"


def test_cluster_split_deterministic() -> None:
    records = [
        {"id": f"r{i}", "cdr_concat": f"{'A'*5}{i:03d}"} for i in range(20)
    ]
    out = cluster_split(records, identity=0.9, seed=42)
    assert set(out) == {"train", "val", "test"}
    flat = out["train"] + out["val"] + out["test"]
    assert len(flat) == 20
    assert len(set(flat)) == 20


# ---------------------------------------------------------------------------
# Dataset + collate
# ---------------------------------------------------------------------------
def _build_dataset(tmp_path: Path, dim: int = 8) -> AntibodyDataset:
    cache = EmbeddingCache(tmp_path, embedding_dim=dim)
    rng = np.random.default_rng(0)
    records = []
    for i, n in enumerate([20, 30]):
        rid = f"id{i}"
        emb = rng.standard_normal((n, dim)).astype(np.float32)
        plddt = rng.uniform(0, 100, size=(n,)).astype(np.float32)
        cache.save(rid, emb, plddt)
        records.append(
            {
                "id": rid,
                "heavy_seq": "ACDE" * (n // 8 + 1),
                "light_seq": "FGHI" * (n // 8 + 1),
                "cdr_spans": [[0, 3], [5, 8], [10, 12], [12, 14], [15, 17], [18, 19]],
            }
        )
    tok = AntibodyTokenizer()
    return AntibodyDataset(records, cache, tok, max_n_residues=64, max_seq_len=64)


def test_dataset_getitem(tmp_path: Path) -> None:
    ds = _build_dataset(tmp_path)
    item = ds[0]
    assert item["struct_emb"].dtype == torch.float32
    assert item["struct_emb"].shape[0] == 20
    assert item["pad_mask"].dtype == torch.bool
    assert item["pad_mask"].all()
    assert item["cdr_spans"].dtype == torch.long
    assert item["cdr_spans"].shape == (6, 2)
    assert item["seq_ids"].dtype == torch.long
    assert item["plddt"].shape == (20,)


def test_collate_fn_varying_n(tmp_path: Path) -> None:
    ds = _build_dataset(tmp_path)
    batch = [ds[0], ds[1]]
    out = collate_fn(batch, pad_id=0)

    assert out["struct_emb"].shape == (2, 30, 8)
    assert out["struct_emb"].dtype == torch.float32

    assert out["pad_mask"].dtype == torch.bool
    assert out["pad_mask"][0, :20].all() and not out["pad_mask"][0, 20:].any()
    assert out["pad_mask"][1].all()

    assert out["plddt"].shape == (2, 30)
    # padded positions in row 0 must be NaN
    assert torch.isnan(out["plddt"][0, 20:]).all()
    # valid positions must be finite
    assert torch.isfinite(out["plddt"][0, :20]).all()

    assert out["cdr_spans"].shape == (2, 6, 2)
    assert out["cdr_spans"].dtype == torch.long

    L = out["seq_ids"].shape[1]
    assert out["seq_pad_mask"].shape == (2, L)
    assert out["seq_ids"].dtype == torch.long

    # padding must be the pad_id at masked positions
    pad_positions = ~out["seq_pad_mask"]
    if pad_positions.any():
        assert (out["seq_ids"][pad_positions] == 0).all()

    assert out["id"] == ["id0", "id1"]
    assert isinstance(out["heavy_seq"], list)


# ---------------------------------------------------------------------------
# Optional dependency tests
# ---------------------------------------------------------------------------
def test_anarci_extractor_if_available() -> None:
    pytest.importorskip("anarci")
    from src.data.numbering import CDRSpanExtractor

    ex = CDRSpanExtractor()
    heavy = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYAD"
        "SVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
    )
    light = (
        "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPS"
        "RFSGSGSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
    )
    spans, n = ex.cdrs_for_pair(heavy, light)
    assert spans.shape == (6, 2)
    assert n == len(heavy) + len(light)


def test_abb2_lazy_import() -> None:
    pytest.importorskip("ImmuneBuilder")
    from src.data.folding import StructurePredictor

    sp = StructurePredictor(model="abodybuilder2", device="cpu")
    assert sp.model_name == "abodybuilder2"
