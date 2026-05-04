"""Public model API for AbLLaVA."""

from src.models.abllava import AbLLaVA, build_abllava, build_decoder
from src.models.decoder import AntibodyDecoder
from src.models.encoders import (
    AntiFoldEncoder,
    CachedEmbeddingEncoder,
    ESM2Encoder,
    IgFoldEncoder,
    build_encoder,
)
from src.models.lora import apply_lora
from src.models.pooling import (
    POOLERS,
    CDRAttentionPool,
    build_pooler,
    cdr_mean_pool,
    cdr_plddt_weighted_pool,
    cdr_segmented_pool,
    fv_mean_pool,
    per_residue_pool,
)
from src.models.projectors import (
    PROJECTORS,
    MLPProjector,
    PerceiverProjector,
    QFormerProjector,
    build_projector,
)

__all__ = [
    # decoder & lora
    "AntibodyDecoder",
    "apply_lora",
    # projectors
    "MLPProjector",
    "QFormerProjector",
    "PerceiverProjector",
    "PROJECTORS",
    # poolers
    "cdr_mean_pool",
    "CDRAttentionPool",
    "cdr_segmented_pool",
    "cdr_plddt_weighted_pool",
    "per_residue_pool",
    "fv_mean_pool",
    "POOLERS",
    # encoders
    "CachedEmbeddingEncoder",
    "AntiFoldEncoder",
    "IgFoldEncoder",
    "ESM2Encoder",
    # full model
    "AbLLaVA",
    # factories
    "build_pooler",
    "build_projector",
    "build_encoder",
    "build_decoder",
    "build_abllava",
]
