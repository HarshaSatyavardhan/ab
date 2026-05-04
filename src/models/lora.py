"""LoRA application — thin wrapper over ``peft`` with a hand-rolled fallback.

Primary path uses ``peft.LoraConfig + get_peft_model``; if peft is missing we
fall back to a hand-rolled implementation that wraps every ``nn.Linear`` whose
name contains one of the target substrings.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# fallback hand-rolled LoRA
# ---------------------------------------------------------------------------

class _LoRALinear(nn.Module):
    """Wrap a ``nn.Linear`` with low-rank adapters ``A (r x in) / B (out x r)``.

    The base linear is frozen; only ``A`` and ``B`` train. Bias is left alone (and
    frozen). Output is ``base(x) + scaling * x @ A^T @ B^T``.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        in_f = base.in_features
        out_f = base.out_features
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.drop(x)))


def _hand_rolled_apply_lora(
    decoder: nn.Module,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str],
    bias: str,
) -> nn.Module:
    # Freeze everything first.
    for p in decoder.parameters():
        p.requires_grad_(False)

    targets = tuple(target_modules)

    def _replace(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(t in name for t in targets):
                wrapped = _LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, name, wrapped)
            else:
                _replace(child, full)

    _replace(decoder)

    # Optionally unfreeze biases.
    if bias == "all":
        for p in decoder.parameters():
            if p.dim() == 1:
                p.requires_grad_(True)
    elif bias == "lora_only":
        # only biases on wrapped layers — none for our linears (bias=False) so no-op
        pass

    n_train = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in decoder.parameters())
    logger.info("hand-rolled LoRA applied: trainable=%d / total=%d", n_train, n_total)
    return decoder


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def apply_lora(
    decoder: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    bias: str = "none",
) -> nn.Module:
    """Apply LoRA adapters to ``decoder``.

    Tries ``peft`` first. If unavailable, falls back to a small hand-rolled LoRA
    implementation that wraps every ``nn.Linear`` whose attribute name matches one
    of ``target_modules``.
    """
    target_modules = list(target_modules)
    try:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias=bias,
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(decoder, cfg)
        try:
            peft_model.print_trainable_parameters()
        except Exception:  # noqa: BLE001
            pass
        return peft_model
    except ImportError:
        logger.warning("peft not available — using hand-rolled LoRA")
        return _hand_rolled_apply_lora(decoder, r, alpha, dropout, target_modules, bias)
    except Exception as e:  # noqa: BLE001
        logger.warning("peft.get_peft_model failed (%s) — falling back to hand-rolled LoRA", e)
        return _hand_rolled_apply_lora(decoder, r, alpha, dropout, target_modules, bias)
