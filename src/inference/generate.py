"""Sampling / generation helpers for AbLLaVA.

The main entry points are :func:`generate_from_embedding` (sample from a
pre-computed structure embedding) and :func:`generate_from_pdb` (run the
encoder on a PDB then sample). A small :func:`generate` alias is exposed
for the public API.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from src.utils.logging import get_logger
from src.utils.tokenizer import AntibodyTokenizer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.data import CDRSpanExtractor
    from src.models import AbLLaVA

logger = get_logger(__name__)


@dataclass
class GenerationConfig:
    """Generation hyper-parameters used at inference time."""

    n_samples: int = 100
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int | None = None
    do_sample: bool = True
    max_new_tokens: int = 260
    seed: int | None = None
    cdr_infill: bool = False
    cdr: str = "H3"


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _expand_batch(tensor: torch.Tensor | None, n: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.shape[0] == n:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError(f"Cannot expand tensor of batch size {tensor.shape[0]} to {n}.")
    return tensor.expand(n, *tensor.shape[1:]).contiguous()


def _decode_record(
    tokenizer: AntibodyTokenizer,
    token_ids: torch.Tensor,
    sample_idx: int,
    base_id: str,
) -> dict[str, Any]:
    """Decode a single generated token sequence to a heavy/light pair record."""
    ids = token_ids.tolist()
    # Trim everything after the first EOS — the decoder may keep emitting after.
    if tokenizer.eos_id in ids:
        ids = ids[: ids.index(tokenizer.eos_id)]
    # Drop leading BOS if the model emitted one.
    if ids and ids[0] == tokenizer.bos_id:
        ids = ids[1:]

    heavy, light = tokenizer.split_pair_ids(ids)
    return {
        "id": f"{base_id}_{sample_idx:04d}",
        "heavy": heavy,
        "light": light,
        "tokens": ids,
    }


@torch.no_grad()
def generate_from_embedding(
    model: "AbLLaVA",
    tokenizer: AntibodyTokenizer,
    struct_emb: torch.Tensor,
    cdr_spans: torch.Tensor,
    pad_mask: torch.Tensor,
    plddt: torch.Tensor | None,
    gen_cfg: GenerationConfig,
) -> list[dict]:
    """Sample ``gen_cfg.n_samples`` paired Fv sequences from a structure embedding.

    ``struct_emb`` is assumed to have batch size 1 and is repeated across the
    sample batch dimension. Returns a list of ``{'id', 'heavy', 'light',
    'tokens'}`` dicts.
    """
    if struct_emb.dim() != 3 or struct_emb.shape[0] != 1:
        raise ValueError(
            f"struct_emb must have shape (1, N, d); got {tuple(struct_emb.shape)}."
        )

    _set_seed(gen_cfg.seed)

    if gen_cfg.cdr_infill:
        # v0 limitation: full-Fv resampling regardless of which CDR is requested.
        logger.warning(
            "cdr_infill=True is not yet implemented; falling back to full-Fv sampling for CDR=%s.",
            gen_cfg.cdr,
        )

    n = max(int(gen_cfg.n_samples), 1)
    device = struct_emb.device

    struct_emb_b = _expand_batch(struct_emb, n)
    cdr_spans_b = _expand_batch(cdr_spans, n)
    pad_mask_b = _expand_batch(pad_mask, n)
    plddt_b = _expand_batch(plddt, n) if plddt is not None else None

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": gen_cfg.max_new_tokens,
        "temperature": gen_cfg.temperature,
        "top_p": gen_cfg.top_p,
        "top_k": gen_cfg.top_k,
        "do_sample": gen_cfg.do_sample,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
    }

    logger.info("Generating %d samples (max_new_tokens=%d) on %s", n, gen_cfg.max_new_tokens, device)
    out = model.generate(struct_emb_b, cdr_spans_b, pad_mask_b, plddt_b, **gen_kwargs)

    if not isinstance(out, torch.Tensor):
        raise TypeError(f"model.generate must return a Tensor; got {type(out)!r}.")
    if out.dim() != 2:
        raise ValueError(f"model.generate must return shape (B, L); got {tuple(out.shape)}.")

    base_id = "sample"
    records = [_decode_record(tokenizer, out[i].cpu(), i, base_id) for i in range(out.shape[0])]
    return records


# Public alias: most callers will use ``generate``.
generate = generate_from_embedding


def _read_pdb_paired_sequence(pdb_path: Path) -> tuple[str, str]:
    """Best-effort extraction of (heavy, light) sequences from a PDB file.

    Uses BioPython's ``PDBParser`` and the standard 3-letter -> 1-letter map.
    Chains are sorted alphabetically; the first non-empty chain is treated as
    the heavy chain and the second as the light chain. If no protein chain
    contains residues, both strings are empty.
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import three_to_one
    except Exception as exc:  # pragma: no cover - biopython missing
        raise RuntimeError("biopython is required to read PDB sequences.") from exc

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("ab", str(pdb_path))

    chain_seqs: list[tuple[str, str]] = []
    for model in structure:
        for chain in sorted(model, key=lambda c: c.id):
            seq_chars: list[str] = []
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                try:
                    seq_chars.append(three_to_one(residue.get_resname()))
                except Exception:  # noqa: BLE001 - non-standard residue
                    continue
            if seq_chars:
                chain_seqs.append((chain.id, "".join(seq_chars)))
        break  # first model only

    if not chain_seqs:
        return "", ""
    if len(chain_seqs) == 1:
        return chain_seqs[0][1], ""
    return chain_seqs[0][1], chain_seqs[1][1]


@torch.no_grad()
def generate_from_pdb(
    model: "AbLLaVA",
    tokenizer: AntibodyTokenizer,
    encoder: Any,
    pdb_path: Path,
    cdr_extractor: "CDRSpanExtractor",
    gen_cfg: GenerationConfig,
) -> list[dict]:
    """Run the encoder on a PDB then sample paired Fv sequences.

    The PDB is also read with BioPython to recover the wild-type paired
    sequence so ``cdr_extractor`` can produce CDR spans. If the PDB has no
    protein sequence we raise a :class:`ValueError`.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    heavy, light = _read_pdb_paired_sequence(pdb_path)
    if not heavy and not light:
        raise ValueError(f"PDB {pdb_path} contains no extractable protein sequence.")

    logger.info(
        "Encoding %s (heavy=%d aa, light=%d aa) with %s",
        pdb_path,
        len(heavy),
        len(light),
        type(encoder).__name__,
    )
    encoder_out = encoder({"pdb_path": str(pdb_path), "heavy_seq": heavy, "light_seq": light})
    if isinstance(encoder_out, tuple):
        struct_emb, pad_mask, plddt = encoder_out
    elif isinstance(encoder_out, dict):
        struct_emb = encoder_out["struct_emb"]
        pad_mask = encoder_out["pad_mask"]
        plddt = encoder_out.get("plddt")
    else:
        raise TypeError(
            f"Encoder returned unsupported type {type(encoder_out)!r}; expected tuple or dict."
        )

    if struct_emb.dim() == 2:
        struct_emb = struct_emb.unsqueeze(0)
    if pad_mask.dim() == 1:
        pad_mask = pad_mask.unsqueeze(0)
    if plddt is not None and plddt.dim() == 1:
        plddt = plddt.unsqueeze(0)

    cdr_spans = cdr_extractor(heavy=heavy, light=light)
    if isinstance(cdr_spans, torch.Tensor) and cdr_spans.dim() == 2:
        cdr_spans = cdr_spans.unsqueeze(0)
    elif not isinstance(cdr_spans, torch.Tensor):
        cdr_spans = torch.tensor(cdr_spans, dtype=torch.long).unsqueeze(0)

    device = next(model.parameters()).device
    struct_emb = struct_emb.to(device)
    pad_mask = pad_mask.to(device)
    cdr_spans = cdr_spans.to(device)
    if plddt is not None:
        plddt = plddt.to(device)

    return generate_from_embedding(
        model, tokenizer, struct_emb, cdr_spans, pad_mask, plddt, gen_cfg
    )
