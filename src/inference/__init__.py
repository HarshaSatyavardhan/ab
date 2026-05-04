"""Inference package for AbLLaVA.

Public API:
    - ``load_model``: load a checkpoint and return ``(model, tokenizer)``.
    - ``generate``: alias for :func:`generate_from_embedding`.
    - ``generate_from_embedding``: sample sequences from a pre-computed
      structure embedding.
    - ``generate_from_pdb``: run the encoder on a PDB then sample.
    - ``write_fasta``: write generated paired Fv records to a FASTA file.

Imports are lazy so the package can be imported even when the heavy
:mod:`src.models` / :mod:`src.data` dependencies are not yet built by
parallel agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "GenerationConfig",
    "generate",
    "generate_from_embedding",
    "generate_from_pdb",
    "load_model",
    "write_fasta",
]


def __getattr__(name: str) -> Any:
    if name in {"generate", "generate_from_embedding", "generate_from_pdb", "GenerationConfig"}:
        from src.inference.generate import (  # noqa: WPS433
            GenerationConfig,
            generate,
            generate_from_embedding,
            generate_from_pdb,
        )

        mapping = {
            "GenerationConfig": GenerationConfig,
            "generate": generate,
            "generate_from_embedding": generate_from_embedding,
            "generate_from_pdb": generate_from_pdb,
        }
        return mapping[name]
    if name in {"load_model", "write_fasta"}:
        from src.inference.io import load_model, write_fasta  # noqa: WPS433

        return {"load_model": load_model, "write_fasta": write_fasta}[name]
    raise AttributeError(name)


if TYPE_CHECKING:  # pragma: no cover
    from src.inference.generate import (
        GenerationConfig,
        generate,
        generate_from_embedding,
        generate_from_pdb,
    )
    from src.inference.io import load_model, write_fasta
