"""Cluster-based train/val/test splits for paired antibody records.

Primary path uses MMseqs2 (``mmseqs easy-cluster``) on the concatenated CDR
string per record; falls back to a deterministic Hamming-based greedy
clustering when MMseqs2 is not available (mostly for testing).

MMseqs2 command line documented in :func:`cluster_split`.
"""

from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CDR string helpers
# ---------------------------------------------------------------------------
def concat_cdr_string(heavy: str, light: str, cdr_spans: torch.Tensor) -> str:
    """Concatenate the six CDR strings (H1+H2+H3+L1+L2+L3).

    *cdr_spans* is a ``Tensor[6, 2]`` of inclusive-start / exclusive-end
    indices into ``heavy + light`` (heavy first), as produced by
    :class:`~src.data.numbering.CDRSpanExtractor`.
    """
    if cdr_spans.shape != (6, 2):
        raise ValueError(f"cdr_spans must have shape (6, 2); got {tuple(cdr_spans.shape)}")
    paired = heavy + light
    parts: list[str] = []
    for s, e in cdr_spans.tolist():
        parts.append(paired[int(s) : int(e)])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Greedy Hamming clustering (fallback)
# ---------------------------------------------------------------------------
def _hamming_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    same = sum(1 for i in range(n) if a[i] == b[i])
    # normalize by max length to penalize length mismatch
    return same / max(len(a), len(b))


def _greedy_cluster(
    ids: list[str], seqs: list[str], identity: float
) -> dict[str, str]:
    """Return ``{member_id: representative_id}`` with at most *identity* similarity."""
    rep_seqs: list[tuple[str, str]] = []  # (rep_id, rep_seq)
    membership: dict[str, str] = {}
    for id_, s in zip(ids, seqs):
        assigned: str | None = None
        for rep_id, rep_seq in rep_seqs:
            if _hamming_similarity(s, rep_seq) >= identity:
                assigned = rep_id
                break
        if assigned is None:
            rep_seqs.append((id_, s))
            membership[id_] = id_
        else:
            membership[id_] = assigned
    return membership


# ---------------------------------------------------------------------------
# MMseqs2 clustering
# ---------------------------------------------------------------------------
def _mmseqs_available() -> bool:
    return shutil.which("mmseqs") is not None


def _mmseqs_cluster(
    ids: list[str], seqs: list[str], identity: float
) -> dict[str, str]:
    """Cluster via ``mmseqs easy-cluster`` and return ``{member: representative}``.

    Command line used::

        mmseqs easy-cluster <input.fasta> <out_prefix> <tmp_dir> \
            --min-seq-id <identity> -c 0.8 --cov-mode 0 --threads 1
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fasta = tmp_path / "in.fasta"
        with fasta.open("w") as f:
            for id_, s in zip(ids, seqs):
                if not s:
                    s = "A"  # mmseqs rejects empty sequences
                f.write(f">{id_}\n{s}\n")
        out_prefix = tmp_path / "out"
        tmp_dir = tmp_path / "tmp"
        tmp_dir.mkdir()
        cmd = [
            "mmseqs",
            "easy-cluster",
            str(fasta),
            str(out_prefix),
            str(tmp_dir),
            "--min-seq-id",
            f"{identity}",
            "-c",
            "0.8",
            "--cov-mode",
            "0",
            "--threads",
            "1",
        ]
        logger.info("Running MMseqs2: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True)
        cluster_tsv = Path(f"{out_prefix}_cluster.tsv")
        membership: dict[str, str] = {}
        with cluster_tsv.open() as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                rep, mem = parts[0], parts[1]
                membership[mem] = rep
        return membership


# ---------------------------------------------------------------------------
# Public split function
# ---------------------------------------------------------------------------
def cluster_split(
    records: list[dict],
    identity: float = 0.9,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Cluster *records* by their CDR string and split clusters into train/val/test.

    Each record must contain ``id`` and either:
      - ``cdr_concat`` (str), or
      - ``heavy_seq`` + ``light_seq`` + ``cdr_spans`` (Tensor[6, 2] or
        list-of-list-of-int).

    MMseqs2 is used when available::

        mmseqs easy-cluster <fasta> <out_prefix> <tmp> --min-seq-id <identity> \
            -c 0.8 --cov-mode 0 --threads 1

    Otherwise a deterministic Hamming-similarity greedy fallback is used.
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train+val+test fractions must sum to 1.0; got {total}"
        )

    ids: list[str] = []
    seqs: list[str] = []
    for r in records:
        rid = str(r["id"])
        ids.append(rid)
        if "cdr_concat" in r and r["cdr_concat"]:
            seqs.append(str(r["cdr_concat"]))
            continue
        spans = r.get("cdr_spans")
        if spans is None:
            seqs.append(str(r.get("heavy_seq", "")) + str(r.get("light_seq", "")))
            continue
        if not isinstance(spans, torch.Tensor):
            spans = torch.tensor(spans, dtype=torch.long)
        seqs.append(
            concat_cdr_string(
                str(r.get("heavy_seq", "")),
                str(r.get("light_seq", "")),
                spans,
            )
        )

    if _mmseqs_available():
        try:
            membership = _mmseqs_cluster(ids, seqs, identity)
        except (subprocess.CalledProcessError, OSError) as e:
            logger.warning("MMseqs2 failed (%s); falling back to greedy clustering.", e)
            membership = _greedy_cluster(ids, seqs, identity)
    else:
        logger.info("MMseqs2 not found; using greedy Hamming-similarity clustering.")
        membership = _greedy_cluster(ids, seqs, identity)

    # Group ids by representative (cluster).
    clusters: dict[str, list[str]] = {}
    for member, rep in membership.items():
        clusters.setdefault(rep, []).append(member)

    # Deterministic shuffle of clusters by sorted rep id, then random.
    rng = random.Random(seed)
    rep_list = sorted(clusters.keys())
    rng.shuffle(rep_list)

    n_clusters = len(rep_list)
    n_train = int(round(train_frac * n_clusters))
    n_val = int(round(val_frac * n_clusters))
    # remainder -> test
    n_test = n_clusters - n_train - n_val
    if n_test < 0:
        n_train = max(n_clusters - n_val, 0)
        n_test = 0

    train_reps = rep_list[:n_train]
    val_reps = rep_list[n_train : n_train + n_val]
    test_reps = rep_list[n_train + n_val :]

    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for rep in train_reps:
        out["train"].extend(clusters[rep])
    for rep in val_reps:
        out["val"].extend(clusters[rep])
    for rep in test_reps:
        out["test"].extend(clusters[rep])

    logger.info(
        "cluster_split: %d clusters from %d records -> train=%d val=%d test=%d",
        n_clusters,
        len(ids),
        len(out["train"]),
        len(out["val"]),
        len(out["test"]),
    )
    # sanity: lists are disjoint and exhaustive
    _ = sum(len(v) for v in out.values())
    return out


__all__: Iterable[str] = ("concat_cdr_string", "cluster_split")
