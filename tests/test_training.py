"""End-to-end smoke tests for the training package.

These tests build a tiny synthetic model + dataset that obey the contract
specified in ``CONTRACT.md`` (the decoder forward returns ``{"loss", "logits"}``
and the dataset returns the expected dict).  They do *not* depend on the
parallel ``src.models`` / ``src.data`` packages, so the training loop's
public surface (optimizer, scheduler, infill labeler, checkpoint save/load)
can be exercised in isolation.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.training.losses import (
    causal_lm_labels,
    make_cdr_infill_batch,
)
from src.training.loop import (
    TrainState,
    evaluate,
    load_checkpoint,
    save_checkpoint,
)
from src.training.optim import build_optimizer, build_scheduler
from src.utils.tokenizer import AntibodyTokenizer

# ---------------------------------------------------------------------------
# Tiny synthetic model + dataset
# ---------------------------------------------------------------------------


class _TinyDataset(Dataset):
    def __init__(self, n: int = 10, n_residues: int = 16, d_e: int = 8, seed: int = 0):
        rng = random.Random(seed)
        torch.manual_seed(seed)
        tok = AntibodyTokenizer()
        self.samples: list[dict[str, Any]] = []
        for i in range(n):
            heavy_len = rng.randint(6, 8)
            light_len = rng.randint(6, 8)
            heavy = "".join(rng.choices("ACDEFGHIKLMNPQRSTVWY", k=heavy_len))
            light = "".join(rng.choices("ACDEFGHIKLMNPQRSTVWY", k=light_len))
            ids = tok.encode_pair(heavy, light)
            seq_ids = torch.tensor(ids, dtype=torch.long)
            N = heavy_len + light_len
            struct_emb = torch.randn(N, d_e)
            pad_mask = torch.ones(N, dtype=torch.bool)
            plddt = torch.full((N,), 80.0)
            # CDR spans: just three small windows in heavy + three in light.
            spans = []
            for h_start, h_end in [(0, 2), (3, 5), (heavy_len - 2, heavy_len)]:
                spans.append([h_start, h_end])
            for l_start, l_end in [
                (heavy_len, heavy_len + 2),
                (heavy_len + 3, heavy_len + 5),
                (heavy_len + light_len - 2, heavy_len + light_len),
            ]:
                spans.append([l_start, l_end])
            cdr_spans = torch.tensor(spans, dtype=torch.long)
            self.samples.append(
                {
                    "id": str(i),
                    "heavy_seq": heavy,
                    "light_seq": light,
                    "seq_ids": seq_ids,
                    "struct_emb": struct_emb,
                    "cdr_spans": cdr_spans,
                    "pad_mask": pad_mask,
                    "plddt": plddt,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def _tiny_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tok = AntibodyTokenizer()
    B = len(samples)
    max_L = max(s["seq_ids"].numel() for s in samples)
    max_N = max(s["struct_emb"].shape[0] for s in samples)
    d_e = samples[0]["struct_emb"].shape[1]

    seq_ids = torch.full((B, max_L), tok.pad_id, dtype=torch.long)
    seq_pad_mask = torch.zeros((B, max_L), dtype=torch.bool)
    struct_emb = torch.zeros((B, max_N, d_e), dtype=torch.float32)
    pad_mask = torch.zeros((B, max_N), dtype=torch.bool)
    plddt = torch.full((B, max_N), float("nan"), dtype=torch.float32)
    cdr_spans = torch.zeros((B, 6, 2), dtype=torch.long)

    for i, s in enumerate(samples):
        L = s["seq_ids"].numel()
        N = s["struct_emb"].shape[0]
        seq_ids[i, :L] = s["seq_ids"]
        seq_pad_mask[i, :L] = True
        struct_emb[i, :N] = s["struct_emb"]
        pad_mask[i, :N] = s["pad_mask"]
        plddt[i, :N] = s["plddt"]
        cdr_spans[i] = s["cdr_spans"]

    return {
        "seq_ids": seq_ids,
        "seq_pad_mask": seq_pad_mask,
        "struct_emb": struct_emb,
        "pad_mask": pad_mask,
        "plddt": plddt,
        "cdr_spans": cdr_spans,
    }


class _TinyAbLLaVA(nn.Module):
    """A minimal model that consumes the contract batch dict and returns
    ``{"loss", "logits"}`` so the training loop can be smoke-tested without
    the real model in place."""

    def __init__(self, vocab_size: int = 25, d_e: int = 8, d_model: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.projector = nn.Linear(d_e, d_model)  # keyword "projector" matters
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        seq_ids = batch["seq_ids"]
        struct_emb = batch["struct_emb"]
        pad_mask = batch["pad_mask"]
        labels = batch.get("labels")

        prefix = self.projector(struct_emb)  # (B, N, d_model)
        tok = self.embed(seq_ids)  # (B, L, d_model)
        x = torch.cat([prefix, tok], dim=1)
        # Build pad mask matching x: prefix uses pad_mask (True=valid),
        # token side is always valid here for the test.
        valid_prefix = pad_mask
        valid_tokens = torch.ones_like(seq_ids, dtype=torch.bool)
        src_key_padding_mask = ~torch.cat([valid_prefix, valid_tokens], dim=1)
        x = self.layer(x, src_key_padding_mask=src_key_padding_mask)
        token_x = x[:, prefix.shape[1] :, :]
        logits = self.head(token_x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_cfg(out_dir: Path) -> Any:
    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "out_dir": str(out_dir),
            "wandb": {"enabled": False, "project": "abllava"},
            "training": {
                "epochs": 1,
                "grad_accum_steps": 1,
                "micro_batch_size": 2,
                "mixed_precision": "none",
                "gradient_checkpointing": False,
                "clip_grad_norm": 1.0,
                "log_every": 1,
                "eval_every": 0,
                "save_every": 0,
                "max_steps": 5,
                "optimizer": {"lr": 1e-2, "betas": [0.9, 0.95], "weight_decay": 0.0},
                "scheduler": {"warmup_steps": 1, "min_lr_ratio": 0.1},
                "loss": {"infill_prob": 0.0},
                "lora": {"enabled": False},
            },
        }
    )


def test_optimizer_and_scheduler_groups():
    model = _TinyAbLLaVA()
    cfg = _make_cfg(Path("/tmp/abllava_test")).training
    opt = build_optimizer(model, cfg, projector_lr_multiplier=0.5)
    # Two groups expected for projector (decay only, since Linear has bias too).
    proj_groups = [g for g in opt.param_groups if "projector" in g.get("name", "")]
    other_groups = [g for g in opt.param_groups if "projector" not in g.get("name", "")]
    assert len(proj_groups) >= 1
    assert len(other_groups) >= 1
    base_lr = float(cfg.optimizer.lr)
    assert math.isclose(proj_groups[0]["lr"], base_lr * 0.5)

    sched = build_scheduler(opt, cfg, total_steps=10)
    sched.step()  # advance once
    assert sched.get_last_lr()[0] >= 0


def test_causal_lm_labels_masks_pad():
    seq = torch.tensor([[1, 5, 6, 0, 0]])
    labels = causal_lm_labels(seq, pad_id=0)
    assert labels[0, 3].item() == -100
    assert labels[0, 0].item() == 1


def test_make_cdr_infill_batch_changes_seq_ids():
    ds = _TinyDataset(n=4, d_e=8)
    batch = _tiny_collate([ds[i] for i in range(4)])
    tok = AntibodyTokenizer()
    rng = random.Random(123)
    new_batch = make_cdr_infill_batch(batch, tok, mask_prob=1.0, rng=rng)
    assert "labels" in new_batch
    # At least one row should differ from original (mask_prob=1.0).
    assert not torch.equal(new_batch["seq_ids"], batch["seq_ids"])
    # Labels should be -100 except at masked positions; verify ignore_index used.
    assert (new_batch["labels"] == -100).any().item()
    # Where the new seq_ids differs from original, the original token equals the label.
    diff = new_batch["seq_ids"] != batch["seq_ids"]
    if diff.any():
        assert torch.equal(new_batch["labels"][diff], batch["seq_ids"][diff])


def test_train_steps_loss_decreases_and_checkpoint_roundtrip(tmp_path: Path):
    torch.manual_seed(0)
    ds = _TinyDataset(n=16, d_e=8)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=_tiny_collate)
    model = _TinyAbLLaVA()
    cfg = _make_cfg(tmp_path).training
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, total_steps=10)

    losses: list[float] = []
    pad_id = AntibodyTokenizer().pad_id
    it = iter(loader)
    for step in range(5):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch["labels"] = causal_lm_labels(batch["seq_ids"], pad_id=pad_id)
        out = model(batch)
        loss = out["loss"]
        assert torch.isfinite(loss).item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.item()))

    # Roughly decreasing: last < first (small tolerance).
    assert losses[-1] < losses[0] + 1e-3

    # Checkpoint roundtrip.
    state = TrainState(step=5, epoch=0, best_val=losses[-1])
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, opt, sched, state)

    model2 = _TinyAbLLaVA()
    opt2 = build_optimizer(model2, cfg)
    sched2 = build_scheduler(opt2, cfg, total_steps=10)
    state2 = load_checkpoint(ckpt, model2, opt2, sched2, map_location="cpu", strict=True)
    assert state2.step == state.step
    assert math.isclose(state2.best_val, state.best_val, rel_tol=1e-6)
    # Parameter equality
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2)


def test_evaluate_runs(tmp_path: Path):
    ds = _TinyDataset(n=8, d_e=8)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=_tiny_collate)
    model = _TinyAbLLaVA()
    metrics = evaluate(model, loader, torch.device("cpu"), pad_id=0, max_batches=2)
    assert "loss" in metrics and "ppl" in metrics
    assert math.isfinite(metrics["loss"])


def test_training_package_imports():
    import src.training as t

    assert hasattr(t, "train")
    assert hasattr(t, "Stage")
    assert hasattr(t, "build_optimizer")
    assert hasattr(t, "build_scheduler")
    assert hasattr(t, "cdr_infill_collate")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-x", "-q"])
