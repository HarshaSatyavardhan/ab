"""Unit tests for src/models/*."""

from __future__ import annotations

import math

import pytest
import torch

from src.models import (
    AbLLaVA,
    AntibodyDecoder,
    CDRAttentionPool,
    MLPProjector,
    PerceiverProjector,
    QFormerProjector,
    apply_lora,
    build_abllava,
    build_pooler,
    build_projector,
    cdr_mean_pool,
    cdr_plddt_weighted_pool,
    cdr_segmented_pool,
    fv_mean_pool,
    per_residue_pool,
)
from src.utils.tokenizer import AntibodyTokenizer


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

torch.manual_seed(0)


def _make_inputs(b: int = 2, n: int = 32, d: int = 16):
    struct_emb = torch.randn(b, n, d)
    plddt = torch.rand(b, n) * 100
    pad_mask = torch.ones(b, n, dtype=torch.bool)
    # the second sample is half-padded
    pad_mask[1, n // 2:] = False
    cdr_spans = torch.tensor(
        [
            [[0, 4], [6, 9], [11, 14], [16, 18], [20, 23], [25, 28]],
            [[0, 3], [5, 7], [9, 12], [12, 14], [14, 15], [15, 16]],
        ],
        dtype=torch.long,
    )
    return struct_emb, cdr_spans, pad_mask, plddt


# ---------------------------------------------------------------------------
# poolers
# ---------------------------------------------------------------------------

def test_cdr_mean_pool_shape_and_mask():
    struct_emb, cdr_spans, pad_mask, plddt = _make_inputs()
    out = cdr_mean_pool(struct_emb, cdr_spans, pad_mask, plddt)
    assert out.shape == (2, 6, 16)
    # all-pad row -> zeros
    pad_mask2 = torch.zeros_like(pad_mask)
    out2 = cdr_mean_pool(struct_emb, cdr_spans, pad_mask2)
    assert torch.allclose(out2, torch.zeros_like(out2))


def test_cdr_segmented_pool_shape_replicates_short_loops():
    struct_emb, cdr_spans, pad_mask, plddt = _make_inputs()
    out = cdr_segmented_pool(struct_emb, cdr_spans, pad_mask, plddt, n_segments=3)
    assert out.shape == (2, 18, 16)
    # spans [12,14], [14,15], [15,16] in row 1 are short -> single mean replicated.
    base = 3 * 3  # CDR index 3
    for j in (3, 4, 5):
        b = 1
        a = out[b, j * 3 + 0]
        c = out[b, j * 3 + 1]
        d = out[b, j * 3 + 2]
        # short loops with only valid residues should produce equal segments
        if pad_mask[b, cdr_spans[b, j, 0]:cdr_spans[b, j, 1]].any():
            assert torch.allclose(a, c) and torch.allclose(c, d)


def test_cdr_plddt_weighted_pool_handles_nan_and_zero_mask():
    struct_emb, cdr_spans, pad_mask, plddt = _make_inputs()
    plddt = plddt.clone()
    plddt[0, :5] = float("nan")
    out = cdr_plddt_weighted_pool(struct_emb, cdr_spans, pad_mask, plddt)
    assert out.shape == (2, 6, 16)
    assert torch.isfinite(out).all()
    # all-pad -> zeros
    out0 = cdr_plddt_weighted_pool(struct_emb, cdr_spans, torch.zeros_like(pad_mask), plddt)
    assert torch.allclose(out0, torch.zeros_like(out0))


def test_per_residue_pool_zeros_pad():
    struct_emb, cdr_spans, pad_mask, plddt = _make_inputs()
    out = per_residue_pool(struct_emb, cdr_spans, pad_mask, plddt)
    assert out.shape == struct_emb.shape
    # padded positions zeroed
    assert torch.allclose(out[1, 16:], torch.zeros_like(out[1, 16:]))


def test_fv_mean_pool_shape_and_zero_pad():
    struct_emb, cdr_spans, pad_mask, plddt = _make_inputs()
    out = fv_mean_pool(struct_emb, cdr_spans, pad_mask, plddt)
    assert out.shape == (2, 1, 16)
    # all-pad -> zeros
    pad0 = torch.zeros_like(pad_mask)
    out0 = fv_mean_pool(struct_emb, cdr_spans, pad0, plddt)
    assert torch.allclose(out0, torch.zeros_like(out0))


def test_cdr_attention_pool_shape_and_zero_pad():
    struct_emb, cdr_spans, pad_mask, plddt = _make_inputs()
    pool = CDRAttentionPool(d=16, n_cdrs=6)
    out = pool(struct_emb, cdr_spans, pad_mask, plddt)
    assert out.shape == (2, 6, 16)
    pad0 = torch.zeros_like(pad_mask)
    out0 = pool(struct_emb, cdr_spans, pad0, plddt)
    assert torch.allclose(out0, torch.zeros_like(out0))


def test_build_pooler_factory():
    p = build_pooler("cdr_mean")
    assert callable(p)
    pa = build_pooler("cdr_attention", d=16)
    assert isinstance(pa, CDRAttentionPool)
    with pytest.raises(KeyError):
        build_pooler("nope")


# ---------------------------------------------------------------------------
# projectors
# ---------------------------------------------------------------------------

def test_mlp_projector_shape():
    proj = MLPProjector(d_in=16, d_out=32, d_hidden=64)
    x = torch.randn(2, 6, 16)
    y = proj(x)
    assert y.shape == (2, 6, 32)


def test_qformer_projector_shape():
    proj = QFormerProjector(d_in=16, d_out=32, d_q=32, n_queries=4, n_layers=2, n_heads=4)
    x = torch.randn(2, 32, 16)
    y = proj(x)
    assert y.shape == (2, 4, 32)


def test_perceiver_projector_shape():
    proj = PerceiverProjector(d_in=16, d_out=32, d_lat=32, n_latents=4, n_layers=1, n_heads=4)
    x = torch.randn(2, 32, 16)
    y = proj(x)
    assert y.shape == (2, 4, 32)


def test_build_projector_factory():
    p = build_projector("mlp", d_in=8, d_out=16)
    assert isinstance(p, MLPProjector)


# ---------------------------------------------------------------------------
# decoder
# ---------------------------------------------------------------------------

def test_decoder_forward_loss_finite():
    tok = AntibodyTokenizer()
    dec = AntibodyDecoder(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=64,
        dropout=0.0,
    )
    ids = torch.randint(0, tok.vocab_size, (2, 10))
    labels = ids.clone()
    out = dec(input_ids=ids, labels=labels)
    assert out["logits"].shape == (2, 10, tok.vocab_size)
    assert out["loss"] is not None
    assert torch.isfinite(out["loss"])


def test_decoder_inputs_embeds_and_attention_mask():
    tok = AntibodyTokenizer()
    dec = AntibodyDecoder(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=64,
        dropout=0.0,
    )
    ids = torch.randint(0, tok.vocab_size, (2, 8))
    embeds = dec.embed_tokens(ids)
    am = torch.ones(2, 8, dtype=torch.long)
    am[1, 5:] = 0
    out = dec(inputs_embeds=embeds, attention_mask=am, labels=ids)
    assert torch.isfinite(out["loss"])
    with pytest.raises(ValueError):
        dec(input_ids=ids, inputs_embeds=embeds)


def test_decoder_generate():
    tok = AntibodyTokenizer()
    dec = AntibodyDecoder(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        max_seq_len=64,
        dropout=0.0,
    )
    prefix = torch.randn(2, 4, 32)
    gen = dec.generate(
        prefix_embeds=prefix,
        bos_id=tok.bos_id,
        eos_id=tok.eos_id,
        max_new_tokens=6,
        do_sample=False,
        pad_id=tok.pad_id,
    )
    assert gen.shape == (2, 6)
    assert gen.dtype == torch.long


# ---------------------------------------------------------------------------
# AbLLaVA end-to-end
# ---------------------------------------------------------------------------

def _tiny_abllava():
    tok = AntibodyTokenizer()
    decoder_cfg = {
        "name": "decoder",
        "vocab_size": tok.vocab_size,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 64,
        "max_seq_len": 64,
        "dropout": 0.0,
    }
    projector_cfg = {"name": "mlp", "d_in": 16, "d_out": 32, "d_hidden": 32}
    pooling_cfg = {"name": "cdr_mean"}
    return tok, build_abllava(decoder_cfg, projector_cfg, pooling_cfg)


def test_abllava_forward_and_generate():
    tok, model = _tiny_abllava()
    struct_emb, cdr_spans, pad_mask, plddt = _make_inputs(b=2, n=32, d=16)
    seq_ids = torch.randint(5, tok.vocab_size, (2, 8))
    seq_pad_mask = torch.ones(2, 8, dtype=torch.long)
    seq_pad_mask[1, 5:] = 0
    batch = {
        "struct_emb": struct_emb,
        "cdr_spans": cdr_spans,
        "pad_mask": pad_mask,
        "plddt": plddt,
        "seq_ids": seq_ids,
        "seq_pad_mask": seq_pad_mask,
    }
    out = model(batch)
    assert torch.isfinite(out["loss"])
    gen = model.generate(
        struct_emb, cdr_spans, pad_mask, plddt,
        bos_id=tok.bos_id, eos_id=tok.eos_id, max_new_tokens=4,
        do_sample=False, pad_id=tok.pad_id,
    )
    assert gen.shape == (2, 4)


def test_abllava_attention_pool():
    tok = AntibodyTokenizer()
    decoder_cfg = {
        "name": "decoder",
        "vocab_size": tok.vocab_size,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 64,
        "max_seq_len": 64,
        "dropout": 0.0,
    }
    projector_cfg = {"name": "mlp", "d_in": 16, "d_out": 32, "d_hidden": 32}
    pooling_cfg = {"name": "cdr_attention", "d": 16}
    model = build_abllava(decoder_cfg, projector_cfg, pooling_cfg)
    assert model.pooler_kind == "module"
    assert any(p.requires_grad for p in model.pooler_module.parameters())


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

def test_apply_lora_reduces_trainable_params():
    peft = pytest.importorskip("peft")
    tok = AntibodyTokenizer()
    dec = AntibodyDecoder(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        max_seq_len=64,
        dropout=0.0,
    )
    n_full_train = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    wrapped = apply_lora(dec, r=4, alpha=8)
    n_lora_train = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    assert n_lora_train < n_full_train
    # forward still works
    ids = torch.randint(0, tok.vocab_size, (2, 6))
    out = wrapped(input_ids=ids, labels=ids)
    assert torch.isfinite(out["loss"])
