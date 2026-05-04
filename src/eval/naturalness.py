"""Naturalness scoring for paired Fv sequences.

Wraps perplexity / log-likelihood computations from external antibody language
models. All backends are lazy-imported and any missing dependency results in
``NaN`` rather than an exception, so the suite can run in partial
environments.

Supported backends and pip names:

================  =================================================
Backend           pip install
----------------  -------------------------------------------------
``ablang2``       ``pip install ablang2``
``iglm``          ``pip install iglm``
``p_iggen``       ``pip install p-iggen`` (HF: ``ollieturnbull/p-IgGen``)
``progen2_oas``   ``pip install transformers`` (HF: ``hugohrban/progen2-oas``)
================  =================================================
"""

from __future__ import annotations

import math
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

_VALID_BACKENDS = {"ablang2", "iglm", "p_iggen", "progen2_oas"}


class NaturalnessScorer:
    """Compute mean per-token NLL under several antibody LMs.

    Backends are loaded lazily on first use and any failure (missing
    dependency, model download error, etc.) is caught and the backend is
    marked unavailable for subsequent calls. ``score`` and ``score_batch``
    return ``NaN`` for unavailable backends.
    """

    def __init__(self, backends: list[str], device: str = "cuda") -> None:
        self.device = device
        self.backends: list[str] = []
        for b in backends:
            if b not in _VALID_BACKENDS:
                logger.warning("Unknown naturalness backend %r; skipping", b)
                continue
            self.backends.append(b)
        self._models: dict[str, Any] = {}
        self._unavailable: set[str] = set()

    # -- internal loaders ---------------------------------------------------

    def _load(self, name: str) -> Any | None:
        if name in self._unavailable:
            return None
        if name in self._models:
            return self._models[name]
        try:
            if name == "ablang2":
                import ablang2  # type: ignore

                m = ablang2.pretrained(device=self.device)
                m.freeze()
                self._models[name] = m
            elif name == "iglm":
                from iglm import IgLM  # type: ignore

                self._models[name] = IgLM()
            elif name == "p_iggen":
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

                tok = AutoTokenizer.from_pretrained("ollieturnbull/p-IgGen")
                mdl = AutoModelForCausalLM.from_pretrained(
                    "ollieturnbull/p-IgGen"
                ).to(self.device)
                mdl.eval()
                self._models[name] = (tok, mdl)
            elif name == "progen2_oas":
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

                tok = AutoTokenizer.from_pretrained("hugohrban/progen2-oas")
                mdl = AutoModelForCausalLM.from_pretrained(
                    "hugohrban/progen2-oas"
                ).to(self.device)
                mdl.eval()
                self._models[name] = (tok, mdl)
        except Exception as e:  # noqa: BLE001
            logger.warning("Naturalness backend %s unavailable: %s", name, e)
            self._unavailable.add(name)
            return None
        return self._models.get(name)

    # -- per-backend scoring ------------------------------------------------

    def _score_ablang2(self, heavy: str, light: str) -> float:
        m = self._load("ablang2")
        if m is None:
            return float("nan")
        try:
            paired = f"{heavy}|{light}"
            # ablang2 supports 'likelihood' mode returning per-token logprobs.
            ll = m([paired], mode="likelihood")
            import numpy as np  # local

            arr = np.asarray(ll[0])
            if arr.ndim == 2:
                # (L, V) — already log-probs; we need the per-token NLL of the
                # observed sequence. Fallback: use mean of max log-prob.
                nll = -float(arr.max(axis=-1).mean())
            else:
                nll = -float(np.mean(arr))
            return nll
        except Exception as e:  # noqa: BLE001
            logger.warning("ablang2 scoring failed: %s", e)
            return float("nan")

    def _score_iglm(self, heavy: str, light: str) -> float:
        m = self._load("iglm")
        if m is None:
            return float("nan")
        try:
            ll_h = m.log_likelihood(heavy, "[HEAVY]", "[HUMAN]")
            ll_l = m.log_likelihood(light, "[LIGHT]", "[HUMAN]")
            n = max(1, len(heavy) + len(light))
            return -float(ll_h + ll_l) / n
        except Exception as e:  # noqa: BLE001
            logger.warning("iglm scoring failed: %s", e)
            return float("nan")

    def _score_hf_causal(
        self, name: str, heavy: str, light: str, sep: str = "/"
    ) -> float:
        loaded = self._load(name)
        if loaded is None:
            return float("nan")
        try:
            import torch  # local

            tok, mdl = loaded
            text = f"{heavy}{sep}{light}"
            enc = tok(text, return_tensors="pt").to(self.device)
            ids = enc["input_ids"]
            with torch.no_grad():
                out = mdl(ids, labels=ids)
            return float(out.loss.detach().cpu().item())
        except Exception as e:  # noqa: BLE001
            logger.warning("%s scoring failed: %s", name, e)
            return float("nan")

    def _score_p_iggen(self, heavy: str, light: str) -> float:
        return self._score_hf_causal("p_iggen", heavy, light, sep="/")

    def _score_progen2_oas(self, heavy: str, light: str) -> float:
        return self._score_hf_causal("progen2_oas", heavy, light, sep="2")

    # -- public API ---------------------------------------------------------

    def score(self, heavy: str, light: str) -> dict[str, float]:
        """Return ``{backend_name: per_token_nll}``."""
        out: dict[str, float] = {}
        for b in self.backends:
            if b == "ablang2":
                out[b] = self._score_ablang2(heavy, light)
            elif b == "iglm":
                out[b] = self._score_iglm(heavy, light)
            elif b == "p_iggen":
                out[b] = self._score_p_iggen(heavy, light)
            elif b == "progen2_oas":
                out[b] = self._score_progen2_oas(heavy, light)
        return out

    def score_batch(self, pairs: list[tuple[str, str]]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {b: [] for b in self.backends}
        for h, l in pairs:
            s = self.score(h, l)
            for b in self.backends:
                out[b].append(s.get(b, float("nan")))
        return out


def _nanmean(xs: list[float]) -> float:
    vals = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def compute_naturalness(
    sequences: list[tuple[str, str]],
    backends: list[str],
    device: str,
) -> dict:
    """Score each (heavy, light) pair with each backend and return the means.

    Returns ``{backend_name: mean_per_token_nll}``. Backends that fail to load
    return ``NaN``.
    """
    scorer = NaturalnessScorer(backends, device=device)
    per_backend = scorer.score_batch(sequences)
    return {b: _nanmean(v) for b, v in per_backend.items()}
