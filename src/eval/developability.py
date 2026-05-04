"""Developability metrics: liability motifs, OASis humanness, TAP.

The liability scanner is pure-regex and always available. OASis (BioPhi) and
TAP are optional and lazy-imported; missing deps return ``None`` so the suite
still produces a usable report.

Install:

* BioPhi / OASis: ``pip install biophi`` (also requires the OASis db file
  and ANARCI).
* TAP: ``pip install tap-antibody`` or use the upstream Therapeutic Antibody
  Profiler API (see https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/TAP).
"""

from __future__ import annotations

import re
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


LIABILITY_PATTERNS: dict[str, str] = {
    "n_glyc": r"N[^P][ST]",
    "deamid_NG": r"NG",
    "deamid_NS": r"NS",
    "isomer_DG": r"DG",
    "isomer_DS": r"DS",
    "oxidation_M": r"M",
    "oxidation_W": r"W",
    "free_cys": r"C",
    "fragment_DP": r"DP",
    "integrin_RGD": r"RGD",
}


class LiabilityScanner:
    """Counts occurrences of chemical liability motifs in a sequence.

    By default counts only inside CDRs; pass ``cdr_only=False`` to scan the
    entire sequence. Custom motif sets can be supplied via ``patterns``.
    """

    def __init__(
        self,
        patterns: dict[str, str] | None = None,
        cdr_only: bool = True,
    ) -> None:
        self.patterns = dict(patterns) if patterns is not None else dict(LIABILITY_PATTERNS)
        self._compiled: dict[str, re.Pattern[str]] = {
            k: re.compile(v) for k, v in self.patterns.items()
        }
        self.cdr_only = cdr_only

    def _windows(
        self,
        seq: str,
        cdr_spans: list[tuple[int, int]] | None,
    ) -> list[str]:
        if not self.cdr_only or cdr_spans is None:
            return [seq]
        out: list[str] = []
        for s, e in cdr_spans:
            if e <= s:
                continue
            s = max(0, int(s))
            e = min(len(seq), int(e))
            if e > s:
                out.append(seq[s:e])
        return out

    def scan(
        self,
        seq: str,
        cdr_spans: list[tuple[int, int]] | None = None,
    ) -> dict[str, int]:
        """Return ``{motif_name: count}``.

        Uses overlapping matches via ``re.findall`` with a lookahead for
        single-character motifs; multi-character motifs use non-overlapping
        ``findall`` (sufficient for practical liability counts).
        """
        windows = self._windows(seq, cdr_spans)
        counts: dict[str, int] = {k: 0 for k in self._compiled}
        for w in windows:
            for name, pat in self._compiled.items():
                counts[name] += len(pat.findall(w))
        return counts


# ---------------------------------------------------------------------------
# OASis & TAP (optional)
# ---------------------------------------------------------------------------


def biophi_oasis_score(heavy: str, light: str) -> float | None:
    """Run BioPhi's OASis humanness scorer.

    Returns ``None`` if BioPhi is not installed or if scoring fails.
    """
    try:
        from biophi.humanness.methods.oasis import OASisParams, get_oasis_humanness  # type: ignore
    except Exception as e:  # noqa: BLE001
        logger.warning("BioPhi OASis unavailable: %s", e)
        return None
    try:
        params = OASisParams(min_fraction_subjects=0.01)
        h = get_oasis_humanness(heavy, params)
        l = get_oasis_humanness(light, params)
        return float((h.get_oasis_identity() + l.get_oasis_identity()) / 2.0)
    except Exception as e:  # noqa: BLE001
        logger.warning("OASis scoring failed: %s", e)
        return None


def tap_score(heavy: str, light: str) -> dict | None:
    """Run TAP (Therapeutic Antibody Profiler).

    Returns the TAP metric dict, or ``None`` if TAP is unavailable.
    """
    try:
        import tap  # type: ignore
    except Exception as e:  # noqa: BLE001
        logger.warning("TAP unavailable: %s", e)
        return None
    try:
        result: Any = tap.run(heavy=heavy, light=light)
        if isinstance(result, dict):
            return result
        # Best-effort dict conversion.
        return {k: getattr(result, k) for k in dir(result) if not k.startswith("_")}
    except Exception as e:  # noqa: BLE001
        logger.warning("TAP scoring failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def compute_developability(
    sequences: list[tuple[str, str]],
    cdr_spans_list: list[tuple] | None = None,
) -> dict:
    """Aggregate liability counts and (if available) OASis/TAP across samples.

    Liability counts are summed over the heavy and light chains. When
    ``cdr_spans_list`` is provided it must be a list aligned with
    ``sequences``; each entry is a tuple
    ``(heavy_cdr_spans, light_cdr_spans)`` of lists of (start, end). When
    omitted, the scanner falls back to whole-sequence scanning.
    """
    scanner_cdr = LiabilityScanner(cdr_only=True)
    scanner_full = LiabilityScanner(cdr_only=False)

    totals: dict[str, int] = {k: 0 for k in LIABILITY_PATTERNS}
    oasis_vals: list[float] = []
    tap_vals: list[Any] = []

    for i, (h, l) in enumerate(sequences):
        if cdr_spans_list is not None and i < len(cdr_spans_list):
            h_spans, l_spans = cdr_spans_list[i]
            h_counts = scanner_cdr.scan(h, h_spans)
            l_counts = scanner_cdr.scan(l, l_spans)
        else:
            h_counts = scanner_full.scan(h, None)
            l_counts = scanner_full.scan(l, None)
        for k in totals:
            totals[k] += h_counts.get(k, 0) + l_counts.get(k, 0)

        oasis = biophi_oasis_score(h, l)
        if oasis is not None:
            oasis_vals.append(oasis)
        tap = tap_score(h, l)
        if tap is not None:
            tap_vals.append(tap)

    oasis_mean: float | None = (
        float(sum(oasis_vals) / len(oasis_vals)) if oasis_vals else None
    )
    tap_summary: Any | None = tap_vals if tap_vals else None

    return {
        "liabilities": totals,
        "oasis": oasis_mean,
        "tap": tap_summary,
    }
