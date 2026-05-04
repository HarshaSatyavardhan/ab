"""Hydra entry point for AbLLaVA inference.

Usage::

    python scripts/infer.py inference.checkpoint_path=path/to/ckpt.pt \\
        inference.target_pdb=path/to/structure.pdb \\
        inference.output_path=outputs/generated.fasta
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from src.data import CDRSpanExtractor
    from src.inference import load_model, write_fasta
    from src.inference.generate import GenerationConfig, generate_from_pdb
    from src.models import build_encoder

    model, tok = load_model(Path(cfg.inference.checkpoint_path), cfg=cfg, device=cfg.device)

    encoder_kwargs = {k: v for k, v in cfg.encoder.items() if k != "name"}
    encoder = build_encoder(cfg.encoder.name, **encoder_kwargs)

    extractor = CDRSpanExtractor(scheme=cfg.data.numbering_scheme)

    gen_cfg = GenerationConfig(
        n_samples=cfg.inference.n_samples,
        temperature=cfg.inference.temperature,
        top_p=cfg.inference.top_p,
        top_k=cfg.inference.top_k,
        do_sample=cfg.inference.do_sample,
        max_new_tokens=cfg.inference.max_new_tokens,
        cdr_infill=cfg.inference.cdr_infill.enabled,
        cdr=cfg.inference.cdr_infill.cdr,
    )

    records = generate_from_pdb(
        model, tok, encoder, Path(cfg.inference.target_pdb), extractor, gen_cfg
    )
    write_fasta(records, Path(cfg.inference.output_path))
    logger.info("Inference complete: %d records written.", len(records))


if __name__ == "__main__":
    main()
