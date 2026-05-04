"""Antibody numbering and CDR span extraction via ANARCI.

Lazy-imports :mod:`anarci` so this module can be imported without the
optional dependency. CDR spans are returned as inclusive-start / exclusive-end
indices into the concatenated ``heavy + light`` residue sequence.

Standard IMGT CDR boundaries (residue numbers, inclusive):
    H1: 27-38, H2: 56-65, H3: 105-117
    L1: 27-38, L2: 56-65, L3: 105-117
"""

from __future__ import annotations

from typing import Any

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


# IMGT CDR position ranges, inclusive on both ends per the standard.
_IMGT_CDRS: dict[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = {
    "H": ((27, 38), (56, 65), (105, 117)),
    "L": ((27, 38), (56, 65), (105, 117)),
}


_ANARCI_IMPORT_ERROR = (
    "anarci is not installed. Install the antibody extras: "
    "`pip install -e .[antibody]` or `pip install anarci`."
)


def _try_import_anarci() -> Any:
    try:
        from anarci import anarci as anarci_fn  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        raise ImportError(_ANARCI_IMPORT_ERROR) from e
    return anarci_fn


class CDRSpanExtractor:
    """Extract CDR spans for paired antibody Fv sequences via ANARCI."""

    def __init__(self, scheme: str = "imgt") -> None:
        if scheme.lower() != "imgt":
            raise ValueError(
                f"CDRSpanExtractor currently only supports IMGT, got scheme={scheme!r}"
            )
        self.scheme = scheme.lower()
        self._anarci_fn: Any | None = None

    # ------------------------------------------------------------------
    # ANARCI wrapper
    # ------------------------------------------------------------------
    def _run_anarci(self, seq: str) -> list[tuple[tuple[int, str], str]]:
        """Run ANARCI on a single sequence; return the numbering list-of-tuples.

        Each entry is ``((position:int, insertion:str), residue:str)``. Gap
        positions (``residue == '-'``) are kept so that we can map IMGT
        positions to residue indices.
        """
        if self._anarci_fn is None:
            self._anarci_fn = _try_import_anarci()
        anarci_fn = self._anarci_fn
        result = anarci_fn(
            [("seq", seq)],
            scheme=self.scheme,
            output=False,
            assign_germline=False,
        )
        # anarci returns (numbering, alignment_details, hit_tables) for newer versions
        numbering = result[0]
        if not numbering or numbering[0] is None:
            raise ValueError("ANARCI failed to number sequence")
        # numbering[0] is a list of domains; take the first
        domains = numbering[0]
        if not domains:
            raise ValueError("ANARCI returned no domains for sequence")
        domain_numbering, _start, _end = domains[0]
        return domain_numbering

    @staticmethod
    def _spans_from_numbering(
        domain_numbering: list[tuple[tuple[int, str], str]],
        chain: str,
    ) -> list[tuple[int, int]]:
        """Convert an ANARCI numbering to three (start, end) residue-index spans.

        Indices are exclusive at the end and refer to the position within the
        non-gap residues of the chain.
        """
        ranges = _IMGT_CDRS[chain.upper()]
        # Build map: (pos_in_chain index of non-gap residues) -> imgt_position
        # We walk through the numbering and skip gaps for residue indexing.
        residue_imgt: list[int] = []  # imgt position of each non-gap residue
        for (pos, _ins), aa in domain_numbering:
            if aa == "-" or aa == "" or aa is None:
                continue
            residue_imgt.append(int(pos))

        spans: list[tuple[int, int]] = []
        for lo, hi in ranges:  # inclusive both ends
            start_idx: int | None = None
            end_idx: int | None = None
            for i, p in enumerate(residue_imgt):
                if start_idx is None and p >= lo:
                    start_idx = i
                if p <= hi:
                    end_idx = i
            if start_idx is None or end_idx is None or end_idx < start_idx:
                # No residues in range - emit empty span at the lower-bound position
                # Find first index whose imgt pos >= lo, else len.
                fallback = next(
                    (i for i, p in enumerate(residue_imgt) if p >= lo), len(residue_imgt)
                )
                spans.append((fallback, fallback))
            else:
                spans.append((start_idx, end_idx + 1))  # exclusive end
        return spans

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def cdrs_for_chain(self, seq: str, chain: str) -> list[tuple[int, int]]:
        """Return three CDR ``(start, end)`` spans for *seq* on *chain*.

        *chain* is one of ``"H"`` or ``"L"``. Indices are residue indices into
        *seq* itself, inclusive-start / exclusive-end.
        """
        chain = chain.upper()
        if chain not in {"H", "L"}:
            raise ValueError(f"chain must be 'H' or 'L', got {chain!r}")
        numbering = self._run_anarci(seq)
        return self._spans_from_numbering(numbering, chain)

    def cdrs_for_pair(
        self,
        heavy_seq: str,
        light_seq: str,
    ) -> tuple[torch.Tensor, int]:
        """Return ``(spans, paired_len)`` for a heavy/light Fv pair.

        ``spans`` is a ``Tensor[6, 2]`` of long indices into the concatenated
        ``heavy_seq + light_seq`` (heavy first), in order
        ``H1, H2, H3, L1, L2, L3``. ``paired_len`` is
        ``len(heavy_seq) + len(light_seq)``.
        """
        h_spans = self.cdrs_for_chain(heavy_seq, "H")
        l_spans = self.cdrs_for_chain(light_seq, "L")
        offset = len(heavy_seq)
        out: list[tuple[int, int]] = list(h_spans) + [
            (s + offset, e + offset) for s, e in l_spans
        ]
        if len(out) != 6:
            raise RuntimeError(f"Expected 6 CDR spans, got {len(out)}")
        spans = torch.tensor(out, dtype=torch.long)
        paired_len = len(heavy_seq) + len(light_seq)
        return spans, paired_len
