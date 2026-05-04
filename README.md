# AbLLaVA

A frozen-encoder, frozen-decoder approach to paired antibody inverse folding,
with a study of how to pool variable-length CDR loops into prefix tokens for a
generative antibody language model.

## Overview

```
Frozen structure encoder (AntiFold)  ──►  Pooler  ──►  Projector  ──►  Frozen antibody decoder + LoRA
                                            │
                                            └─ {cdr_mean, cdr_attention, cdr_segmented, cdr_plddt,
                                               per_residue, fv_mean}
```

- **Encoder**: AntiFold (default), IgFold, or ESM-2 — frozen, embeddings cached to disk.
- **Pooler**: 6 strategies, exposed as a Hydra config group.
- **Projector**: 2-layer MLP (default), Q-Former, or Perceiver-Resampler.
- **Decoder**: 150 M-param GPT-style with RoPE + RMSNorm + SwiGLU.
- **LoRA** adapters on `q/k/v/o` projections for Stage B fine-tuning.

See `CONTRACT.md` for the interface contract every module respects.

## Layout

```
configs/                  Hydra config groups (data, encoder, pooling, projector, model, training, inference, eval)
src/
├── data/                 OAS filtering, ANARCI numbering, ABB2 folding, cluster splits, embedding cache, Dataset
├── models/               Encoders, pooling, projectors, decoder, LoRA, AbLLaVA assembly
├── training/             Stage A / Stage B loops, optimizer, scheduler, CDR-infill loss
├── inference/            Generate from PDB or precomputed embeddings; FASTA writer
├── eval/                 Perplexity, recovery, diversity, naturalness, structure, developability
└── utils/                Tokenizer (25 tokens) + logging
scripts/                  Hydra entry points: train.py / infer.py / eval.py
tests/                    Unit + smoke tests for every module
```

## Setup

```bash
pip install -e .
# Optional antibody deps (some are gated by license / build tools)
pip install -e ".[antibody]"
```

The core package imports without `anarci`, `ImmuneBuilder`, `ablang2`, `iglm`,
`biophi`, etc. — those are lazy-imported and gracefully degrade.

## Usage

### Training

```bash
# Stage A: train projector only (decoder frozen, no LoRA)
python scripts/train.py training=stage_a

# Stage B: add LoRA r=8 on decoder, mix CE + CDR-infill
python scripts/train.py training=stage_b training.init_from=outputs/stage_a/checkpoints/final.pt
```

Override anything from the CLI:

```bash
python scripts/train.py pooling=cdr_attention projector=qformer encoder=igfold model=decoder_50m
```

### Inference

```bash
python scripts/infer.py \
    inference.checkpoint_path=outputs/stage_b/checkpoints/final.pt \
    inference.target_pdb=trastuzumab.pdb \
    inference.n_samples=100 \
    inference.output_path=outputs/designs.fasta
```

### Evaluation

```bash
python scripts/eval.py eval.checkpoint_path=outputs/stage_b/checkpoints/final.pt
```

Writes a JSON report covering perplexity / recovery / diversity / naturalness /
structure / developability.

## Running the tests

```bash
pytest tests/ -q
```

Tests use synthetic data; no network or GPU required. Optional antibody
backends are skipped via `pytest.importorskip` when not installed.

## Pooling ablations (the headline study)

| Pooler           | K   | Trainable params | Bias                                           |
|------------------|-----|------------------|------------------------------------------------|
| `cdr_mean`       | 6   | 0                | per-CDR average                                |
| `cdr_attention`  | 6   | ≈ d² + 6d        | learned per-CDR query                          |
| `cdr_segmented`  | 18  | 0                | torso / apex / torso decomposition (3 segs)    |
| `cdr_plddt`      | 6   | 0                | folder-confidence-weighted mean                |
| `per_residue`    | N   | 0                | full per-residue prefix                        |
| `fv_mean`        | 1   | 0                | global Fv summary                              |

Switch via `python scripts/train.py pooling=<name>`.

## Status

- All 57 unit tests pass; 6 are skipped (optional antibody backends).
- Hydra entry points are wired but Hydra itself must be installed in your env
  (the sandbox antlr4 build failure does not affect a normal pip install).
