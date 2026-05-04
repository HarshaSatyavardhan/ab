"""Preprocessing driver: filter -> number -> cluster-split -> parquet.

Reads the OAS-derived ``filtered.parquet``, applies :class:`OASFilter` for
extra safety, runs :class:`CDRSpanExtractor` on every paired Fv to produce
six CDR spans, runs :func:`cluster_split` on the CDRs, and writes a single
``dataset.parquet`` with the columns enumerated in
:func:`preprocess_oas`.

Designed to be invoked as a Hydra entry; works on a plain ``DictConfig``-like
mapping with at least ``cfg.data.processed_dir``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.data.filter import OASFilter
from src.data.numbering import CDRSpanExtractor
from src.data.splits import cluster_split, concat_cdr_string
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _cfg_get(cfg: Any, *path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for p in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p, default)
        else:
            cur = getattr(cur, p, default)
    return cur


def preprocess_oas(cfg: Any) -> None:
    """Run the OAS preprocessing pipeline.

    Expects ``cfg.data.processed_dir`` (or dict-equivalent). Reads
    ``<processed_dir>/filtered.parquet``, applies filtering and ANARCI
    numbering, computes a cluster-based train/val/test split, and writes
    ``<processed_dir>/dataset.parquet``.

    Output columns:
        ``id, heavy_seq, light_seq, cdr_spans, v_h, j_h, v_l, j_l, h3_len,
        isotype, split``.
    """
    processed_dir = Path(_cfg_get(cfg, "data", "processed_dir"))
    if not processed_dir:
        raise ValueError("cfg.data.processed_dir is required")
    in_path = processed_dir / "filtered.parquet"
    out_path = processed_dir / "dataset.parquet"
    if not in_path.is_file():
        raise FileNotFoundError(f"Expected {in_path} to exist")

    species = _cfg_get(cfg, "data", "species", default=("human",)) or ("human",)
    min_length = int(_cfg_get(cfg, "data", "min_length", default=100))
    max_length = int(_cfg_get(cfg, "data", "max_length", default=150))
    identity = float(_cfg_get(cfg, "data", "cluster_identity", default=0.9))
    train_frac = float(_cfg_get(cfg, "data", "train_frac", default=0.8))
    val_frac = float(_cfg_get(cfg, "data", "val_frac", default=0.1))
    test_frac = float(_cfg_get(cfg, "data", "test_frac", default=0.1))
    seed = int(_cfg_get(cfg, "seed", default=42))

    logger.info("Reading filtered parquet from %s", in_path)
    df = pd.read_parquet(in_path)

    f = OASFilter(
        species=tuple(species),
        min_length=min_length,
        max_length=max_length,
    )
    df = f.filter_dataframe(df)

    extractor = CDRSpanExtractor(scheme="imgt")

    rows: list[dict] = []
    for rec in tqdm(df.to_dict(orient="records"), desc="numbering"):
        heavy = str(
            rec.get("sequence_alignment_aa_heavy")
            or rec.get("heavy_seq")
            or ""
        ).replace(".", "").replace(" ", "").upper()
        light = str(
            rec.get("sequence_alignment_aa_light")
            or rec.get("light_seq")
            or ""
        ).replace(".", "").replace(" ", "").upper()
        if not heavy or not light:
            continue
        try:
            spans, _paired_len = extractor.cdrs_for_pair(heavy, light)
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping row: ANARCI failed (%s)", e)
            continue

        rid = str(
            rec.get("id")
            or rec.get("sequence_id")
            or rec.get("sequence_id_heavy")
            or f"row{len(rows)}"
        )

        cdr_concat = concat_cdr_string(heavy, light, spans)
        h3_len = int(spans[2, 1].item() - spans[2, 0].item())

        rows.append(
            {
                "id": rid,
                "heavy_seq": heavy,
                "light_seq": light,
                "cdr_spans": spans.flatten().tolist(),
                "cdr_concat": cdr_concat,
                "v_h": rec.get("v_call_heavy") or rec.get("v_h"),
                "j_h": rec.get("j_call_heavy") or rec.get("j_h"),
                "v_l": rec.get("v_call_light") or rec.get("v_l"),
                "j_l": rec.get("j_call_light") or rec.get("j_l"),
                "h3_len": h3_len,
                "isotype": rec.get("isotype_heavy") or rec.get("isotype"),
            }
        )

    if not rows:
        raise RuntimeError("preprocess_oas produced no rows after filtering & numbering")

    # cluster + split
    splits = cluster_split(
        rows,
        identity=identity,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    id_to_split: dict[str, str] = {}
    for name, ids in splits.items():
        for rid in ids:
            id_to_split[rid] = name
    for r in rows:
        r["split"] = id_to_split.get(r["id"], "train")
        r.pop("cdr_concat", None)

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(out_df), out_path)
