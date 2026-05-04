"""Row-level OAS filter following the IgBert / IgT5 cleaning recipe.

Filters paired antibody rows from the Observed Antibody Space (OAS) by:
    - productive / vj_in_frame / stop_codon flags,
    - ANARCI_status (drops conserved-cysteine / unusual / indel / shorter-than),
    - heavy & light Fv length bounds,
    - CDR-H3 length bounds,
    - presence of unusual residues (X, *, -),
    - species whitelist.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


_BAD_ANARCI_FLAGS: tuple[str, ...] = (
    "Missing Conserved Cysteine",
    "Unusual residue",
    "Indel",
    "Shorter than",
)

_BAD_AA_CHARS: tuple[str, ...] = ("X", "*", "-")


def _as_bool(val: object) -> bool | None:
    """Coerce OAS truthy/falsy field to a bool, returning None for missing."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        try:
            import math

            if isinstance(val, float) and math.isnan(val):
                return None
        except Exception:  # noqa: BLE001
            pass
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"t", "true", "1", "yes", "y"}:
            return True
        if s in {"f", "false", "0", "no", "n"}:
            return False
        if s in {"", "nan", "none", "null"}:
            return None
    return None


class OASFilter:
    """Filter OAS paired rows by IgBert/IgT5-style criteria.

    The OAS schema uses `_heavy` / `_light` suffixes for chain-specific columns
    (e.g. ``sequence_alignment_aa_heavy``, ``cdr3_aa_heavy``,
    ``ANARCI_status_heavy``). ANARCI status fields are optional.
    """

    def __init__(
        self,
        species: Iterable[str] = ("human",),
        min_length: int = 100,
        max_length: int = 150,
        min_h3_len: int = 3,
        max_h3_len: int = 35,
        drop_unusual: bool = True,
    ) -> None:
        self.species: tuple[str, ...] = tuple(s.lower() for s in species)
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.min_h3_len = int(min_h3_len)
        self.max_h3_len = int(max_h3_len)
        self.drop_unusual = bool(drop_unusual)

    @staticmethod
    def _get(row: dict, *keys: str) -> object:
        for k in keys:
            if k in row and row[k] is not None:
                v = row[k]
                # treat NaN as missing
                try:
                    import math

                    if isinstance(v, float) and math.isnan(v):
                        continue
                except Exception:  # noqa: BLE001
                    pass
                return v
        return None

    def _heavy_seq(self, row: dict) -> str | None:
        v = self._get(
            row,
            "sequence_alignment_aa_heavy",
            "heavy_seq",
            "sequence_aa_heavy",
            "sequence_heavy",
        )
        return str(v) if v is not None else None

    def _light_seq(self, row: dict) -> str | None:
        v = self._get(
            row,
            "sequence_alignment_aa_light",
            "light_seq",
            "sequence_aa_light",
            "sequence_light",
        )
        return str(v) if v is not None else None

    def _h3(self, row: dict) -> str | None:
        v = self._get(row, "cdr3_aa_heavy", "cdrh3", "cdr_h3")
        return str(v) if v is not None else None

    def _flag(self, row: dict, base: str) -> bool | None:
        for suf in ("_heavy", "_light", ""):
            key = f"{base}{suf}"
            if key in row:
                b = _as_bool(row[key])
                if b is None:
                    continue
                return b
        return None

    def _flag_pair(self, row: dict, base: str) -> tuple[bool | None, bool | None]:
        h = _as_bool(row.get(f"{base}_heavy")) if f"{base}_heavy" in row else None
        l = _as_bool(row.get(f"{base}_light")) if f"{base}_light" in row else None
        return h, l

    def _anarci(self, row: dict) -> tuple[str | None, str | None]:
        h = row.get("ANARCI_status_heavy")
        l = row.get("ANARCI_status_light")
        return (
            str(h) if h is not None and not (isinstance(h, float) and h != h) else None,
            str(l) if l is not None and not (isinstance(l, float) and l != l) else None,
        )

    def _species(self, row: dict) -> str | None:
        v = self._get(row, "species", "Species", "species_heavy", "species_light")
        return str(v).lower() if v is not None else None

    def filter_row(self, row: dict) -> bool:
        """Return True iff the row passes all checks."""
        # productive
        h_prod, l_prod = self._flag_pair(row, "productive")
        if h_prod is False or l_prod is False:
            return False

        # vj_in_frame
        h_inf, l_inf = self._flag_pair(row, "vj_in_frame")
        if h_inf is False or l_inf is False:
            return False

        # stop_codon
        h_stop, l_stop = self._flag_pair(row, "stop_codon")
        if h_stop is True or l_stop is True:
            return False

        # ANARCI_status (optional)
        h_anarci, l_anarci = self._anarci(row)
        for status in (h_anarci, l_anarci):
            if status is None:
                continue
            for flag in _BAD_ANARCI_FLAGS:
                if flag in status:
                    return False

        # sequences exist
        heavy = self._heavy_seq(row)
        light = self._light_seq(row)
        if heavy is None or light is None:
            return False
        heavy = heavy.replace(".", "").replace(" ", "").upper()
        light = light.replace(".", "").replace(" ", "").upper()
        if not heavy or not light:
            return False

        # length bounds
        if not (self.min_length <= len(heavy) <= self.max_length):
            return False
        if not (self.min_length <= len(light) <= self.max_length):
            return False

        # CDR-H3 length
        h3 = self._h3(row)
        if h3 is None:
            return False
        h3 = h3.replace(".", "").replace(" ", "").upper()
        if not (self.min_h3_len <= len(h3) <= self.max_h3_len):
            return False

        # unusual residues
        if self.drop_unusual:
            for ch in _BAD_AA_CHARS:
                if ch in heavy or ch in light:
                    return False

        # species
        if self.species:
            sp = self._species(row)
            if sp is None or sp not in self.species:
                return False

        return True

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply :meth:`filter_row` to every row of *df* and return the kept rows."""
        if df.empty:
            return df
        records = df.to_dict(orient="records")
        keep = [self.filter_row(r) for r in records]
        kept = df.loc[keep].reset_index(drop=True)
        logger.info("OASFilter: kept %d / %d rows", len(kept), len(df))
        return kept
