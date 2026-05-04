"""Tests for the evaluation suite.

Heavy / optional dependencies (BioPhi, ablang2, IgLM, ImmuneBuilder) are
guarded with ``pytest.importorskip`` so the test suite remains green on a
minimal install.
"""

from __future__ import annotations

import random

import pytest

from src.eval import LiabilityScanner, compute_diversity
from src.eval.developability import LIABILITY_PATTERNS, compute_developability
from src.eval.sequence import aa_kl, levenshtein
from src.eval.structure import kabsch_rmsd


# ---------------------------------------------------------------------------
# Levenshtein
# ---------------------------------------------------------------------------


def test_levenshtein_basic_cases() -> None:
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("flaw", "lawn") == 2
    assert levenshtein("abc", "") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("a", "b") == 1


# ---------------------------------------------------------------------------
# Liability scanner
# ---------------------------------------------------------------------------


def test_liability_scanner_motifs_full_sequence() -> None:
    scanner = LiabilityScanner(cdr_only=False)
    # Each synthetic seq embeds exactly one occurrence of one motif among
    # otherwise neutral residues. Note RGD trips n_glyc-style is not used;
    # we use simple residue padding to avoid accidental collisions.
    cases: dict[str, str] = {
        "n_glyc": "AAANATAAA",   # NAT matches N[^P][ST]
        "deamid_NG": "AAANGAAA",
        "deamid_NS": "AAANSAAA",  # also matches n_glyc since N S T... no, NS not NxS/T pattern (need 3 chars)
        "isomer_DG": "AAADGAAA",
        "isomer_DS": "AAADSAAA",
        "oxidation_M": "AAAMAAA",
        "oxidation_W": "AAAWAAA",
        "free_cys": "AAACAAA",
        "fragment_DP": "AAADPAAA",
        "integrin_RGD": "AAARGDAAA",
    }
    for motif, seq in cases.items():
        counts = scanner.scan(seq)
        assert counts[motif] >= 1, f"{motif} not detected in {seq}: {counts}"


def test_liability_scanner_cdr_only_filters() -> None:
    scanner = LiabilityScanner(cdr_only=True)
    seq = "AAACAAA"  # 'C' at position 3
    # No CDR spans -> nothing scanned
    counts_empty = scanner.scan(seq, cdr_spans=[])
    assert counts_empty["free_cys"] == 0
    # CDR covering the C
    counts_in = scanner.scan(seq, cdr_spans=[(2, 5)])
    assert counts_in["free_cys"] == 1
    # CDR not covering the C
    counts_out = scanner.scan(seq, cdr_spans=[(0, 2)])
    assert counts_out["free_cys"] == 0


def test_liability_pattern_keys_match_contract() -> None:
    expected = {
        "n_glyc",
        "deamid_NG",
        "deamid_NS",
        "isomer_DG",
        "isomer_DS",
        "oxidation_M",
        "oxidation_W",
        "free_cys",
        "fragment_DP",
        "integrin_RGD",
    }
    assert set(LIABILITY_PATTERNS.keys()) == expected


def test_compute_developability_aggregates() -> None:
    pairs = [("AAACAAA", "AAAMAAA"), ("RGDAAA", "AAANGAAA")]
    out = compute_developability(pairs)
    assert "liabilities" in out
    assert out["liabilities"]["free_cys"] == 1
    assert out["liabilities"]["oxidation_M"] == 1
    assert out["liabilities"]["integrin_RGD"] == 1
    assert out["liabilities"]["deamid_NG"] == 1
    # OASis / TAP not installed in CI -> None
    assert out["oasis"] is None or isinstance(out["oasis"], float)
    assert out["tap"] is None or isinstance(out["tap"], list)


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------


def test_compute_diversity_runs_on_random_strings() -> None:
    rng = random.Random(0)
    aas = "ACDEFGHIKLMNPQRSTVWY"
    seqs = ["".join(rng.choice(aas) for _ in range(100)) for _ in range(10)]
    out = compute_diversity(seqs, pairs=50, seed=0)
    assert "mean_levenshtein" in out
    assert "mean_blosum62" in out
    assert out["mean_levenshtein"] > 0


def test_compute_diversity_handles_too_few_sequences() -> None:
    out = compute_diversity(["AAAA"], pairs=10)
    import math

    assert math.isnan(out["mean_levenshtein"])
    assert math.isnan(out["mean_blosum62"])


def test_aa_kl_uniform_sequence_low_kl() -> None:
    aas = "ACDEFGHIKLMNPQRSTVWY"
    # Even composition -> KL ~ 0
    seq = aas * 100
    kl = aa_kl([seq])
    assert kl < 1e-6
    # Single AA -> KL = log(20)
    import math

    only_a = aa_kl(["A" * 100])
    assert abs(only_a - math.log(20)) < 1e-6


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------


def test_kabsch_rmsd_identity_zero() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    P = rng.standard_normal((20, 3))
    assert kabsch_rmsd(P, P.copy()) < 1e-9


def test_kabsch_rmsd_invariant_under_rotation_and_translation() -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    P = rng.standard_normal((30, 3))
    # Random rotation
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    Q = Q * np.sign(np.linalg.det(Q))
    P_rot = P @ Q.T + rng.standard_normal((1, 3)) * 5.0
    assert kabsch_rmsd(P, P_rot) < 1e-6


# ---------------------------------------------------------------------------
# Optional backend smoke tests
# ---------------------------------------------------------------------------


def test_naturalness_optional_backends_skipped_if_missing() -> None:
    pytest.importorskip("ablang2")
    from src.eval.naturalness import NaturalnessScorer

    scorer = NaturalnessScorer(["ablang2"], device="cpu")
    out = scorer.score("EVQLVESGGGLVQPGGSLRLSCAAS", "DIQMTQSPSSLSASVGDRVTITC")
    assert "ablang2" in out


def test_oasis_skipped_if_missing() -> None:
    pytest.importorskip("biophi")
    from src.eval.developability import biophi_oasis_score

    val = biophi_oasis_score("EVQLVESGGGLVQ", "DIQMTQSPSS")
    assert val is None or isinstance(val, float)


def test_structure_scorer_skipped_if_missing() -> None:
    pytest.importorskip("ImmuneBuilder")
    from src.eval.structure import StructureScorer

    scorer = StructureScorer()
    assert scorer is not None
