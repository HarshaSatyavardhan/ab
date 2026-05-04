# AbLLaVA implementation contract

This is the **interface contract** every module must respect. Do not change shapes
or names without coordinating across modules.

## Tokenizer (`src/utils/tokenizer.py`) — already implemented

- `AntibodyTokenizer()` with vocab size 25.
- IDs: `[PAD]=0, [BOS]=1, [EOS]=2, [SEP]=3, [MASK]=4`, then 20 AAs (`ACDEFGHIKLMNPQRSTVWY`).
- Methods: `encode(seq)`, `decode(ids)`, `encode_pair(h, l)`, `pad_batch(list[list[int]])`.

## Dataset return value

`AntibodyDataset.__getitem__(i)` returns a dict with these keys:

| Key            | Type         | Shape       | Notes                                                  |
|----------------|--------------|-------------|--------------------------------------------------------|
| `id`           | `str`        | —           | row id                                                 |
| `heavy_seq`    | `str`        | —           | AA string                                              |
| `light_seq`    | `str`        | —           | AA string                                              |
| `seq_ids`      | `Tensor[L]`  | long        | tokenized `[BOS] heavy [SEP] light [EOS]`              |
| `struct_emb`   | `Tensor[N,d_e]` | float32  | per-residue, paired (heavy then light)                 |
| `cdr_spans`    | `Tensor[6,2]` | long       | (start, end) for H1, H2, H3, L1, L2, L3 in `struct_emb` |
| `pad_mask`     | `Tensor[N]`  | bool        | `True` for valid residue, `False` for pad              |
| `plddt`        | `Tensor[N]`  | float32     | per-residue confidence in `[0, 100]`; `nan` if missing |

A `collate_fn` pads `struct_emb`, `pad_mask`, `plddt` to `max_N` and `seq_ids` to
`max_L`, returning a batch dict with the same keys plus `seq_pad_mask`.

## Pooling (`src/models/pooling.py`)

All poolers take `(B, N, d_e) struct_emb`, `(B, 6, 2) cdr_spans`, `(B, N) pad_mask`,
and (optionally) `(B, N) plddt`. They return `(B, K, d_e)`.

| Pooler                    | K    | Params  |
|---------------------------|------|---------|
| `cdr_mean_pool`           | 6    | 0       |
| `cdr_attention_pool`      | 6    | ~d² + 6d|
| `cdr_segmented_pool(n=3)` | 18   | 0       |
| `cdr_plddt_weighted_pool` | 6    | 0       |
| `per_residue_pool`        | N    | 0       |
| `fv_mean_pool`            | 1    | 0       |

Each is registered in a `POOLERS` dict keyed by name (`cdr_mean`, `cdr_attention`,
`cdr_segmented`, `cdr_plddt`, `per_residue`, `fv_mean`).

## Projector (`src/models/projectors.py`)

All projectors map `(B, K_in, d_in) → (B, K_out, d_out)`.

- `MLPProjector(d_in, d_out, d_hidden=4*d_out)` — K_out == K_in
- `QFormerProjector(d_in, d_out, d_q=768, n_queries=32, n_layers=6, n_heads=12)` — K_out == n_queries
- `PerceiverProjector(d_in, d_out, d_lat=1024, n_latents=32, n_layers=1, n_heads=8)` — K_out == n_latents

Registered in `PROJECTORS` dict keyed by name (`mlp`, `qformer`, `perceiver`).

## Encoder (`src/models/encoders.py`)

Frozen wrappers exposing `forward(batch) -> (struct_emb, pad_mask, plddt)`. For
the offline training path, embeddings are precomputed and stored — these classes
exist mostly for inference. Available: `AntiFoldEncoder`, `IgFoldEncoder`,
`ESM2Encoder`. Each has an `embedding_dim` attribute.

For the offline path we ship `CachedEmbeddingEncoder(cache_dir)` which loads `.npy`
files keyed by `id`.

## Decoder (`src/models/decoder.py`)

A from-scratch GPT-style decoder (RoPE, SwiGLU, RMSNorm, FlashAttention if
available) with `~150M` parameters at default config. Supports `inputs_embeds`
to accept projector prefix tokens.

Key signatures:

```python
class AntibodyDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 1024, n_layers: int = 16,
                 n_heads: int = 16, d_ff: int = 4096, max_seq_len: int = 1024,
                 dropout: float = 0.1, rope_base: float = 10000.0): ...

    def forward(self, input_ids: Tensor | None = None,
                inputs_embeds: Tensor | None = None,
                attention_mask: Tensor | None = None,
                labels: Tensor | None = None) -> dict:
        # returns {"loss": Tensor | None, "logits": Tensor[B, L, V]}

    def embed_tokens(self, input_ids: Tensor) -> Tensor: ...

    @torch.no_grad()
    def generate(self, prefix_embeds: Tensor, bos_id: int, eos_id: int,
                 max_new_tokens: int, temperature: float = 1.0,
                 top_p: float = 0.9, top_k: int | None = None,
                 do_sample: bool = True) -> Tensor: ...
```

LoRA application is via PEFT in `apply_lora(decoder, r, alpha, dropout, targets)` —
returns the LoRA-wrapped model.

## Full model (`src/models/abllava.py`)

```python
class AbLLaVA(nn.Module):
    def __init__(self, decoder, projector, pooler_name: str,
                 attention_pooler: nn.Module | None = None): ...

    def forward(self, batch) -> dict:
        # pools struct_emb -> projects -> prepends to decoder.embed_tokens(seq_ids)
        # returns {"loss", "logits"}

    @torch.no_grad()
    def generate(self, struct_emb, cdr_spans, pad_mask, plddt, **gen_kwargs) -> Tensor: ...
```

## Training entry points

Hydra-driven scripts in `scripts/`:

- `python scripts/train.py +stage=stage_a` — projector only, decoder frozen.
- `python scripts/train.py +stage=stage_b` — projector + LoRA on decoder.
- `python scripts/infer.py target=path.pdb` — generate paired Fv from a PDB.
- `python scripts/eval.py` — run full eval suite on a checkpoint.

All use `@hydra.main(config_path="../configs", config_name="config")`.

## Config root (`configs/config.yaml`)

```yaml
defaults:
  - _self_
  - data: poas
  - encoder: antifold
  - pooling: cdr_mean
  - projector: mlp
  - model: decoder_150m
  - training: stage_a
  - inference: default
  - eval: default

seed: 42
device: cuda
out_dir: outputs/${now:%Y-%m-%d_%H-%M-%S}
log_level: INFO
wandb:
  enabled: false
  project: abllava
  entity: null
```

Each module config (e.g. `configs/pooling/cdr_mean.yaml`) carries a `name:` field
plus any kwargs.

## Conventions

- All files start with module docstrings.
- Type-hint Tensors via `torch.Tensor` (no string annotations).
- No print — use `src.utils.logging.get_logger(__name__)`.
- All file paths via `pathlib.Path`.
- Tests under `tests/` mirroring `src/`.

