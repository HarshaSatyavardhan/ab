"""Generic AbLLaVA training loop used by both Stage A and Stage B.

The loop is intentionally small: it owns the optimizer, scheduler, AMP
autocast context, gradient accumulation, evaluation cadence, and
checkpointing.  All other concerns (model construction, data loading,
LoRA wrapping) live in their respective packages and are wired in via the
Hydra config.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.training.losses import causal_lm_labels, make_cdr_infill_batch
from src.training.optim import build_optimizer, build_scheduler
from src.utils.logging import WandbLogger, get_logger
from src.utils.tokenizer import AntibodyTokenizer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# State + checkpoint helpers
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_val: float = float("inf")
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainState":
        return cls(
            step=int(d.get("step", 0)),
            epoch=int(d.get("epoch", 0)),
            best_val=float(d.get("best_val", float("inf"))),
            extra=dict(d.get("extra", {})),
        )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    state: TrainState,
    cfg: DictConfig | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "state": state.to_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if cfg is not None:
        try:
            payload["cfg"] = OmegaConf.to_container(cfg, resolve=True)
        except Exception:  # noqa: BLE001
            payload["cfg"] = None
    torch.save(payload, path)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> TrainState:
    path = Path(path)
    payload = torch.load(path, map_location=map_location)
    missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
    if missing:
        logger.info("load_checkpoint: %d missing keys", len(missing))
    if unexpected:
        logger.info("load_checkpoint: %d unexpected keys", len(unexpected))
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    state = TrainState.from_dict(payload.get("state", {}))
    logger.info("Loaded checkpoint %s (step=%d)", path, state.step)
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(cfg: DictConfig) -> torch.device:
    requested = str(cfg.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _ensure_labels(batch: dict[str, Any], pad_id: int) -> dict[str, Any]:
    if "labels" in batch and batch["labels"] is not None:
        return batch
    batch = dict(batch)
    batch["labels"] = causal_lm_labels(batch["seq_ids"], pad_id=pad_id)
    return batch


def _autocast_ctx(cfg: DictConfig, device: torch.device):
    mp = str(cfg.training.get("mixed_precision", "none")).lower()
    if mp == "bf16" and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if mp == "fp16" and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)

    class _Null:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    return _Null()


def _maybe_enable_grad_checkpointing(model: nn.Module, enable: bool) -> None:
    if not enable:
        return
    for fn_name in ("gradient_checkpointing_enable", "enable_gradient_checkpointing"):
        fn = getattr(model, fn_name, None)
        if callable(fn):
            try:
                fn()
                logger.info("Enabled gradient checkpointing via %s", fn_name)
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("gradient_checkpointing failed: %s", e)
    # Fallback: try inner decoder
    decoder = getattr(model, "decoder", None)
    if decoder is not None:
        for fn_name in ("gradient_checkpointing_enable", "enable_gradient_checkpointing"):
            fn = getattr(decoder, fn_name, None)
            if callable(fn):
                try:
                    fn()
                    logger.info("Enabled gradient checkpointing on decoder via %s", fn_name)
                    return
                except Exception as e:  # noqa: BLE001
                    logger.warning("decoder gradient_checkpointing failed: %s", e)


def _freeze_decoder_keep_projector(model: nn.Module, lora_enabled: bool) -> None:
    """Stage A: freeze decoder weights, keep projector / pooler trainable.

    Stage B (lora_enabled): we still freeze base decoder weights — the LoRA
    wrapper marks adapter params as trainable on its own.  We additionally
    keep projector / pooler trainable.
    """
    for name, p in model.named_parameters():
        lname = name.lower()
        is_projector = "projector" in lname
        is_pooler = "pooler" in lname or "attention_pool" in lname
        is_lora = "lora" in lname
        if is_projector or is_pooler:
            p.requires_grad = True
            continue
        if lora_enabled and is_lora:
            p.requires_grad = True
            continue
        # Everything else -> frozen.
        p.requires_grad = False


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_id: int = 0,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = _move_batch(batch, device)
            batch = _ensure_labels(batch, pad_id=pad_id)
            out = model(batch)
            loss = out.get("loss")
            if loss is None:
                continue
            total_loss += float(loss.detach().item())
            total_count += 1
    if total_count == 0:
        return {"loss": float("nan"), "ppl": float("nan")}
    mean_loss = total_loss / total_count
    return {"loss": mean_loss, "ppl": math.exp(min(mean_loss, 30.0))}


# ---------------------------------------------------------------------------
# Training entry
# ---------------------------------------------------------------------------


def train(cfg: DictConfig) -> Path:
    """Run a single-stage training job and return the path of the final ckpt."""
    torch.backends.cuda.matmul.allow_tf32 = True

    seed = int(cfg.get("seed", 42))
    _set_seed(seed)

    device = _resolve_device(cfg)
    out_dir = Path(str(cfg.get("out_dir", "outputs/run")))
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AntibodyTokenizer()
    pad_id = tokenizer.pad_id

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    from src.data import AntibodyDataset, collate_fn  # noqa: WPS433

    data_cfg = cfg.data
    train_ds = AntibodyDataset(data_cfg, split="train")
    try:
        val_ds = AntibodyDataset(data_cfg, split="val")
    except Exception as e:  # noqa: BLE001
        logger.warning("No val split available (%s); skipping eval.", e)
        val_ds = None

    micro_bs = int(cfg.training.get("micro_batch_size", data_cfg.get("batch_size", 8)))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False)) and device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=micro_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=micro_bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        if val_ds is not None
        else None
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    from src.models import build_abllava  # noqa: WPS433

    model = build_abllava(
        decoder_cfg=cfg.model,
        projector_cfg=cfg.projector,
        pooling_cfg=cfg.pooling,
        lora_cfg=cfg.training.get("lora", None),
    )
    model.to(device)

    lora_enabled = bool(cfg.training.get("lora", {}).get("enabled", False))
    _freeze_decoder_keep_projector(model, lora_enabled=lora_enabled)
    _maybe_enable_grad_checkpointing(
        model, bool(cfg.training.get("gradient_checkpointing", False))
    )

    # Optionally warm-start from Stage A checkpoint.
    init_from = cfg.training.get("init_from", None)
    if init_from:
        load_checkpoint(Path(str(init_from)), model, map_location=device, strict=False)

    # ------------------------------------------------------------------
    # Optim / sched
    # ------------------------------------------------------------------
    proj_mult = cfg.training.get("optimizer", {}).get("projector_lr_multiplier", None)
    optimizer = build_optimizer(
        model,
        cfg.training,
        projector_lr_multiplier=float(proj_mult) if proj_mult is not None else None,
    )

    epochs = int(cfg.training.get("epochs", 1))
    grad_accum = max(int(cfg.training.get("grad_accum_steps", 1)), 1)
    steps_per_epoch = max(len(train_loader) // grad_accum, 1)
    max_steps_cfg = cfg.training.get("max_steps", None)
    if max_steps_cfg:
        total_steps = int(max_steps_cfg)
    else:
        total_steps = epochs * steps_per_epoch
    total_steps = max(total_steps, 1)

    scheduler = build_scheduler(optimizer, cfg.training, total_steps=total_steps)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    wb = WandbLogger(
        enabled=bool(cfg.get("wandb", {}).get("enabled", False)),
        project=str(cfg.get("wandb", {}).get("project", "abllava")),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    state = TrainState()
    log_every = int(cfg.training.get("log_every", 50))
    eval_every = int(cfg.training.get("eval_every", 0))
    save_every = int(cfg.training.get("save_every", 0))
    clip_norm = float(cfg.training.get("clip_grad_norm", 0.0))
    infill_prob = float(cfg.training.get("loss", {}).get("infill_prob", 0.0))
    infill_rng = random.Random(seed)

    micro_step_in_accum = 0
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_count = 0
    final_ckpt_path: Path | None = None

    logger.info(
        "Starting training: epochs=%d, total_steps=%d, grad_accum=%d, micro_bs=%d, device=%s",
        epochs,
        total_steps,
        grad_accum,
        micro_bs,
        device,
    )

    done = False
    for epoch in range(epochs):
        if done:
            break
        state.epoch = epoch
        for batch in train_loader:
            batch = _move_batch(batch, device)
            if infill_prob > 0.0:
                batch = make_cdr_infill_batch(
                    batch, tokenizer, mask_prob=infill_prob, rng=infill_rng
                )
            else:
                batch = _ensure_labels(batch, pad_id=pad_id)

            model.train()
            with _autocast_ctx(cfg, device):
                out = model(batch)
                loss = out["loss"]
                loss = loss / grad_accum

            loss.backward()
            running_loss += float(loss.detach().item()) * grad_accum
            running_count += 1
            micro_step_in_accum += 1

            if micro_step_in_accum < grad_accum:
                continue

            # Optimizer step boundary.
            if clip_norm and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=clip_norm,
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step_in_accum = 0
            state.step += 1

            # ----- logging -----
            if state.step % max(log_every, 1) == 0 and running_count > 0:
                avg_loss = running_loss / running_count
                lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else 0.0
                ppl = math.exp(min(avg_loss, 30.0))
                logger.info(
                    "step=%d epoch=%d loss=%.4f ppl=%.3f lr=%.3e",
                    state.step,
                    state.epoch,
                    avg_loss,
                    ppl,
                    lr,
                )
                wb.log({"train/loss": avg_loss, "train/ppl": ppl, "train/lr": lr}, step=state.step)
                running_loss = 0.0
                running_count = 0

            # ----- eval -----
            if eval_every > 0 and val_loader is not None and state.step % eval_every == 0:
                metrics = evaluate(model, val_loader, device, pad_id=pad_id)
                logger.info("eval@%d: loss=%.4f ppl=%.3f", state.step, metrics["loss"], metrics["ppl"])
                wb.log({"val/loss": metrics["loss"], "val/ppl": metrics["ppl"]}, step=state.step)
                if metrics["loss"] < state.best_val:
                    state.best_val = metrics["loss"]
                    save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, state, cfg)

            # ----- periodic save -----
            if save_every > 0 and state.step % save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"step_{state.step}.pt", model, optimizer, scheduler, state, cfg
                )

            if state.step >= total_steps:
                done = True
                break

    final_ckpt_path = ckpt_dir / "final.pt"
    save_checkpoint(final_ckpt_path, model, optimizer, scheduler, state, cfg)
    wb.finish()
    return final_ckpt_path
