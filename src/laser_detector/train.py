"""Phase 2 training loop.

Per DESIGN.md §5: AdamW + cosine LR + 1-epoch warmup, focal heatmap loss +
BCE presence loss (no line aux loss yet — that's Phase 3), photometric augs
only, mixed precision (bf16 on Ada Lovelace).

Per-epoch evaluation reuses the existing `eval.evaluate` harness via tiled
inference on the val split. Best checkpoint by val `hit_rate_n3` is saved.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from laser_detector.data import (
    FrameRecord,
    HardNegativeBalancedSampler,
    LaserTileDataset,
    build_records,
)
from laser_detector.eval import PREDICTION_TABLE_SCHEMA, evaluate
from laser_detector.inference import predict_frame
from laser_detector.model import LaserDetector, bce_heatmap_loss, focal_heatmap_loss
from laser_detector.preprocessing.image_loader import ImageLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    # Hard absolute warmup-step count; overrides warmup_epochs when set. Use
    # this when steps_per_epoch is large — `warmup_epochs=1` on the full
    # corpus is 4k+ steps, which leaves the LR sub-1e-5 for ~10% of training.
    warmup_steps: int | None = None
    num_workers: int = 8
    # Per-worker prefetch buffers. With 4 ranks × N workers × prefetch_factor
    # outstanding random reads, the page cache LRU thrashes when the working
    # set sits near RAM capacity. Drop to 1 if the dataset is bigger than RAM.
    prefetch_factor: int = 2
    lambda_heatmap: float = 1.0
    lambda_presence: float = 0.5
    presence_threshold: float = 0.5
    # Heatmap loss formulation. CenterNet's focal loss collapses on this corpus
    # (1-pixel target in 1M-pixel tile → degenerate "predict 0 everywhere"
    # minimum is global). BCE with pos_weight inverts the imbalance directly.
    heatmap_loss: str = "bce"  # "bce" or "focal"
    heatmap_pos_weight: float = 1000.0
    inference_batch_size: int = 8
    seed: int = 0
    use_bf16: bool = True
    # Per DESIGN.md §5.1: how many negatives to re-score for hard-negative
    # mining at the end of each epoch. A random subsample is enough — over
    # multiple epochs all negatives get refreshed without paying a full pass.
    hard_negative_score_sample: int = 1024
    # DESIGN.md §8.1: ms/frame at batch=1 and batch=8, measured once per run.
    latency_benchmark_frames: int = 10
    latency_benchmark_warmup: int = 3
    # Per-epoch val pass uses this many randomly-sampled frames (0 = full val).
    # Tiled inference is ~1s/frame at 4K; 4k val frames × 50 epochs = 60h, so
    # checkpoint selection runs on a stable random subsample and a final full-
    # val pass at the end produces the canonical metrics.
    val_subsample_per_epoch: int = 200
    val_subsample_seed: int = 0


@dataclass
class TrainArtifacts:
    """In-memory state that the trainer hands back to the caller."""

    best_metrics: dict[str, float] = field(default_factory=dict)
    best_epoch: int = -1
    best_checkpoint_path: Path | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    latency_metrics: dict[str, float] = field(default_factory=dict)
    final_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class DDPContext:
    """Process-local view of the distributed group.

    `torchrun` sets RANK / LOCAL_RANK / WORLD_SIZE. Single-GPU runs construct
    a no-op context with rank=0, world_size=1 — same code path either way."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DDPContext:
    """Read torchrun env vars and initialize the NCCL process group.

    Idempotent w.r.t. process startup — call once per process. Returns a
    DDPContext describing this rank.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return DDPContext(
            rank=rank, local_rank=local_rank, world_size=world_size, device=device,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DDPContext(device=device)


def shutdown_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _cosine_with_warmup(optimizer, *, total_steps: int, warmup_steps: int):
    def lr_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_factor)


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int,
    ddp: DDPContext,
) -> dict[str, float]:
    model.train()
    autocast_dtype = torch.bfloat16 if cfg.use_bf16 and device.type == "cuda" else None
    sums = {"loss": 0.0, "loss_heatmap": 0.0, "loss_presence": 0.0}
    n_batches = 0

    iterator = (
        tqdm(loader, desc=f"epoch {epoch} train", leave=False)
        if ddp.is_main else loader
    )
    pbar = iterator if ddp.is_main else None
    for batch in iterator:
        image = batch["image"].to(device, non_blocking=True)
        heatmap_target = batch["heatmap"].to(device, non_blocking=True)
        presence_target = batch["presence"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else _NullCtx()
        )
        with autocast_ctx:
            out = model(image)
        # Forward in bf16 (autocast above), losses in fp32 — bf16 mantissa is
        # too coarse for the log-of-clamped-sigmoid in either focal or BCE.
        heatmap_logits = out["heatmap_logits"].float()
        presence_logits = out["presence_logits"].float()
        if cfg.heatmap_loss == "bce":
            loss_hm = bce_heatmap_loss(
                heatmap_logits, heatmap_target, pos_weight=cfg.heatmap_pos_weight,
            )
        elif cfg.heatmap_loss == "focal":
            loss_hm = focal_heatmap_loss(heatmap_logits, heatmap_target)
        else:
            raise ValueError(f"unknown heatmap_loss: {cfg.heatmap_loss!r}")
        loss_pres = torch.nn.functional.binary_cross_entropy_with_logits(
            presence_logits, presence_target
        )
        loss = cfg.lambda_heatmap * loss_hm + cfg.lambda_presence * loss_pres

        loss.backward()
        optimizer.step()
        scheduler.step()

        sums["loss"] += float(loss.item())
        sums["loss_heatmap"] += float(loss_hm.item())
        sums["loss_presence"] += float(loss_pres.item())
        n_batches += 1
        if pbar is not None and n_batches % 20 == 0:
            pbar.set_postfix(
                loss=f"{sums['loss'] / n_batches:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

    means = {k: v / max(n_batches, 1) for k, v in sums.items()}
    # Average across ranks so the logged number reflects global behavior, not
    # rank 0's slice. Each rank saw a different subset of batches.
    if ddp.is_distributed:
        t = torch.tensor([means["loss"], means["loss_heatmap"], means["loss_presence"]],
                         device=ddp.device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        means["loss"], means["loss_heatmap"], means["loss_presence"] = (
            float(t[0]), float(t[1]), float(t[2]),
        )
    return means


def _build_epoch_val_subsample(
    *,
    val_records: list[FrameRecord],
    frames: pl.DataFrame,
    n: int,
    seed: int,
) -> tuple[list[FrameRecord], pl.DataFrame]:
    """Pick a stratified (positive/negative) subsample of val_records and
    return it together with a frames-table restricted to those image_ids.

    Stratify by is_positive so the rare negative class isn't lost — at 4% in
    v1, an unstratified random sample of 200 frames typically draws 0 or 1
    negatives, which would make the per-epoch presence-AUROC undefined.

    `n <= 0` → no subsampling: return the originals.
    """
    if n <= 0 or n >= len(val_records):
        return list(val_records), frames
    rng = np.random.default_rng(seed)
    pos_records = [r for r in val_records if r.label_xy is not None]
    neg_records = [r for r in val_records if r.label_xy is None]
    pos_frac = len(pos_records) / max(len(val_records), 1)
    n_pos = max(min(int(round(n * pos_frac)), len(pos_records)), 1) if pos_records else 0
    n_neg = max(min(n - n_pos, len(neg_records)), 0)
    if neg_records and n_neg == 0:
        n_neg, n_pos = 1, n_pos - 1
    pos_idx = rng.choice(len(pos_records), size=n_pos, replace=False) if n_pos else np.empty(0, dtype=int)
    neg_idx = rng.choice(len(neg_records), size=n_neg, replace=False) if n_neg else np.empty(0, dtype=int)
    sampled = [pos_records[i] for i in pos_idx] + [neg_records[i] for i in neg_idx]
    rng.shuffle(sampled)
    sampled_ids = [r.image_id for r in sampled]
    return sampled, frames.filter(pl.col("image_id").is_in(sampled_ids))


@torch.inference_mode()
def _score_negatives_for_mining(
    *,
    model: torch.nn.Module,
    dataset: LaserTileDataset,
    sampler: HardNegativeBalancedSampler,
    device: torch.device,
    n_to_score: int,
    batch_size: int,
    autocast_dtype: torch.dtype | None,
    ddp: DDPContext | None = None,
) -> int:
    """Re-score a random sample of negative frames by max heatmap response.

    Per DESIGN.md §5.1, "hard" negatives are those with high heatmap response
    in the previous epoch. We sample `n_to_score` of them, run a single random
    crop through the model, and write the max sigmoid response back into the
    sampler's weight table. Negatives we don't sample this epoch keep their
    previous score — over a few epochs the whole pool gets refreshed.

    Returns the number of negatives actually scored (0 if none exist).
    """
    if not sampler.has_negatives() or n_to_score <= 0:
        return 0

    # Score on rank 0 only, then broadcast updated weights so every rank
    # uses the same `neg_scores` for the next epoch's shuffle weighting.
    is_distributed = ddp is not None and ddp.is_distributed
    if is_distributed and not ddp.is_main:
        # Non-zero ranks wait for the updated scores via broadcast below.
        scores_tensor = torch.from_numpy(sampler.neg_scores).to(ddp.device)
        dist.broadcast(scores_tensor, src=0)
        sampler.neg_scores = scores_tensor.detach().cpu().numpy().astype(np.float64)
        return 0

    chosen_record_indices = sampler.sample_neg_record_indices(n_to_score)
    if chosen_record_indices.size == 0:
        if is_distributed:
            scores_tensor = torch.from_numpy(sampler.neg_scores).to(ddp.device)
            dist.broadcast(scores_tensor, src=0)
        return 0

    inference_model = (
        model.module if isinstance(model, DistributedDataParallel) else model
    )
    was_training = inference_model.training
    inference_model.eval()

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None and device.type == "cuda"
        else _NullCtx()
    )

    images: list[torch.Tensor] = []
    rec_idxs: list[int] = []

    def _flush():
        if not images:
            return
        x = torch.stack(images).to(device, non_blocking=True)
        with autocast_ctx:
            out = inference_model(x)
        max_per_sample = (
            torch.sigmoid(out["heatmap_logits"]).float()
            .view(x.shape[0], -1)
            .max(dim=1)
            .values
            .cpu()
            .numpy()
        )
        for r, s in zip(rec_idxs, max_per_sample):
            sampler.set_neg_score(int(r), float(s))
        images.clear()
        rec_idxs.clear()

    for rec_idx in chosen_record_indices:
        sample = dataset[int(rec_idx)]
        images.append(sample["image"])
        rec_idxs.append(int(rec_idx))
        if len(images) >= batch_size:
            _flush()
    _flush()

    if was_training:
        inference_model.train()

    # Broadcast updated weights so non-zero ranks (which returned early above)
    # get the new sampling distribution for the next epoch.
    if is_distributed:
        scores_tensor = torch.from_numpy(sampler.neg_scores).to(ddp.device)
        dist.broadcast(scores_tensor, src=0)
    return int(chosen_record_indices.size)


@torch.inference_mode()
def _benchmark_latency(
    *,
    model: torch.nn.Module,
    sample_records: list[FrameRecord],
    image_loader: ImageLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Time tiled-inference ms/frame at batch=1 and batch=8 per DESIGN.md §8.1.

    Skipped (returns {}) if we can't decode at least `warmup + measure` frames.
    """
    needed = cfg.latency_benchmark_warmup + cfg.latency_benchmark_frames
    samples: list[tuple[FrameRecord, np.ndarray]] = []
    for rec in sample_records:
        img = image_loader.load(rec.image_path, rec.image_checksum)
        if img is not None:
            samples.append((rec, img))
        if len(samples) >= needed:
            break

    if len(samples) < needed:
        logger.warning(
            "Latency benchmark needs %d images, got %d — skipping", needed, len(samples)
        )
        return {}

    autocast_dtype = torch.bfloat16 if cfg.use_bf16 and device.type == "cuda" else None
    model.eval()
    metrics: dict[str, float] = {}
    for bs in (1, 8):
        for rec, img in samples[: cfg.latency_benchmark_warmup]:
            predict_frame(
                img, model, wavelength=rec.wavelength, device=device,
                batch_size=bs, autocast_dtype=autocast_dtype,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        measured = samples[cfg.latency_benchmark_warmup : needed]
        for rec, img in measured:
            predict_frame(
                img, model, wavelength=rec.wavelength, device=device,
                batch_size=bs, autocast_dtype=autocast_dtype,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        metrics[f"latency_bs{bs}_ms"] = (elapsed / len(measured)) * 1000.0
    return metrics


def _run_val_inference(
    model: torch.nn.Module,
    val_records: list[FrameRecord],
    loader: ImageLoader,
    device: torch.device,
    cfg: TrainConfig,
    ddp: DDPContext,
) -> pl.DataFrame:
    """Run tiled inference over every val frame; return predictions matching
    `PREDICTION_TABLE_SCHEMA`. With DDP active, each rank handles a stride
    of the val list and the rows are gathered to rank 0 (returns an empty
    DataFrame on non-zero ranks — caller must check)."""
    # DDP wraps the model; underlying nn.Module is at .module. Single-GPU
    # path passes the bare model.
    inference_model = model.module if isinstance(model, DistributedDataParallel) else model
    inference_model.eval()
    rows: list[dict] = []
    autocast_dtype = torch.bfloat16 if cfg.use_bf16 else None

    my_records = val_records[ddp.rank :: ddp.world_size]
    iterator = (
        tqdm(my_records, desc="val inference", leave=False)
        if ddp.is_main else my_records
    )
    for rec in iterator:
        image_bgr = loader.load(rec.image_path, rec.image_checksum)
        if image_bgr is None:
            rows.append({
                "image_id": rec.image_id, "pred_x": None, "pred_y": None, "pred_confidence": 0.0,
            })
            continue
        pred = predict_frame(
            image_bgr, inference_model,
            wavelength=rec.wavelength,
            device=device,
            batch_size=cfg.inference_batch_size,
            autocast_dtype=autocast_dtype,
        )
        rows.append({
            "image_id": rec.image_id,
            "pred_x": pred.pred_x,
            "pred_y": pred.pred_y,
            "pred_confidence": pred.pred_confidence,
        })

    if not ddp.is_distributed:
        return pl.DataFrame(rows, schema=PREDICTION_TABLE_SCHEMA)

    # Gather rows from all ranks to rank 0. Non-zero ranks return an empty
    # DataFrame so the trainer skips eval/log on them.
    gathered: list[list[dict] | None] = [None] * ddp.world_size if ddp.is_main else None
    dist.gather_object(rows, gathered, dst=0)
    if ddp.is_main:
        flat = [r for chunk in gathered for r in chunk]
        return pl.DataFrame(flat, schema=PREDICTION_TABLE_SCHEMA)
    return pl.DataFrame([], schema=PREDICTION_TABLE_SCHEMA)


def train(
    *,
    cfg: TrainConfig,
    train_records: list[FrameRecord],
    val_records: list[FrameRecord],
    image_loader: ImageLoader,
    frames: pl.DataFrame,
    splits: pl.DataFrame,
    wavelengths: pl.DataFrame,
    lines: pl.DataFrame,
    checkpoint_dir: Path,
    epoch_callback=None,
    ddp: DDPContext | None = None,
) -> TrainArtifacts:
    """Train the Phase 2 model. Caller is responsible for MLflow setup +
    forwarding `epoch_callback(epoch, metrics, checkpoint_path)` to log per-epoch.

    With `ddp.is_distributed == True`, the trainer expects to run inside a
    `torchrun` group: `cfg.batch_size` is **per rank**, so the effective batch
    is `batch_size * world_size`. Rank 0 owns checkpoint saves, MLflow logging,
    final-val, and latency-benchmark; all ranks participate in train + val
    inference (val records are sharded by rank and gathered to rank 0).
    """
    if ddp is None:
        ddp = DDPContext(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    torch.manual_seed(cfg.seed + ddp.rank)
    np.random.seed(cfg.seed + ddp.rank)
    device = ddp.device
    if ddp.is_main:
        logger.info(
            "Training on %s (world_size=%d, batch_size=%d/rank → global=%d)",
            device, ddp.world_size, cfg.batch_size, cfg.batch_size * ddp.world_size,
        )

    train_ds = LaserTileDataset(
        records=train_records, loader=image_loader, augment=True,
        seed=cfg.seed + ddp.rank,
    )
    # Single-process dataset used for scoring negatives between epochs.
    score_ds = LaserTileDataset(
        records=train_records, loader=image_loader, augment=False,
        seed=cfg.seed + 1 + ddp.rank,
    )
    sampler = HardNegativeBalancedSampler(
        train_records, seed=cfg.seed,
        rank=ddp.rank, world_size=ddp.world_size,
    )
    if ddp.is_main:
        logger.info(
            "Hard-negative sampler: %d positives, %d negatives → %d samples/rank/epoch (world_size=%d)",
            len(sampler.pos_indices), len(sampler.neg_indices),
            len(sampler), ddp.world_size,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sampler=sampler,
        multiprocessing_context="forkserver",
        persistent_workers=True,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=device.type == "cuda",
    )

    model = LaserDetector().to(device)
    if ddp.is_distributed:
        model = DistributedDataParallel(
            model, device_ids=[ddp.local_rank], output_device=ddp.local_rank,
        )
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = (
        cfg.warmup_steps
        if cfg.warmup_steps is not None
        else cfg.warmup_epochs * steps_per_epoch
    )
    if ddp.is_main:
        logger.info(
            "Schedule: %d total steps, %d warmup steps (%.1f%%)",
            cfg.epochs * steps_per_epoch, warmup_steps,
            100.0 * warmup_steps / max(1, cfg.epochs * steps_per_epoch),
        )
    scheduler = _cosine_with_warmup(
        optimizer,
        total_steps=cfg.epochs * steps_per_epoch,
        warmup_steps=warmup_steps,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifacts = TrainArtifacts()
    best_score = -float("inf")

    autocast_dtype = torch.bfloat16 if cfg.use_bf16 and device.type == "cuda" else None

    # Stable per-epoch val subsample so the hit-rate metric used for best-
    # checkpoint selection is comparable across epochs. Stratified by
    # is_positive so the (rare) negatives stay represented.
    epoch_val_records, epoch_val_frames = _build_epoch_val_subsample(
        val_records=val_records, frames=frames,
        n=cfg.val_subsample_per_epoch, seed=cfg.val_subsample_seed,
    )
    if (
        ddp.is_main
        and cfg.val_subsample_per_epoch > 0
        and len(epoch_val_records) < len(val_records)
    ):
        logger.info(
            "Per-epoch val pass uses %d/%d frames; final full-val runs after training",
            len(epoch_val_records), len(val_records),
        )

    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        t0 = time.monotonic()
        train_metrics = _train_one_epoch(
            model, train_loader, optimizer, scheduler, cfg, device, epoch, ddp,
        )
        train_elapsed = time.monotonic() - t0

        t_score = time.monotonic()
        n_scored = _score_negatives_for_mining(
            model=model, dataset=score_ds, sampler=sampler, device=device,
            n_to_score=cfg.hard_negative_score_sample,
            batch_size=cfg.inference_batch_size,
            autocast_dtype=autocast_dtype,
            ddp=ddp,
        )
        score_elapsed = time.monotonic() - t_score

        t1 = time.monotonic()
        predictions = _run_val_inference(
            model, epoch_val_records, image_loader, device, cfg, ddp,
        )
        val_elapsed = time.monotonic() - t1

        if ddp.is_main:
            eval_result = evaluate(
                predictions, frames=epoch_val_frames, splits=splits,
                wavelengths=wavelengths, lines=lines,
                split="val", presence_threshold=cfg.presence_threshold,
            )

            score = float(eval_result.metrics.get("hit_rate_n3", 0.0))
            epoch_metrics = {
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in eval_result.metrics.items()},
                "train_seconds": train_elapsed,
                "val_seconds": val_elapsed,
                "hard_negative_score_seconds": score_elapsed,
                "hard_negative_score_n": float(n_scored),
            }
            artifacts.history.append(epoch_metrics)

            ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            state_to_save = (
                model.module if isinstance(model, DistributedDataParallel) else model
            ).state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": state_to_save,
                    "metrics": epoch_metrics,
                    "cfg": cfg.__dict__,
                },
                ckpt_path,
            )

            improved = score > best_score
            if improved:
                best_score = score
                artifacts.best_epoch = epoch
                artifacts.best_metrics = dict(epoch_metrics)
                artifacts.best_checkpoint_path = ckpt_path

            logger.info(
                "epoch %d: train_loss=%.4f val_hit_rate_n3=%.3f val_auroc=%.3f (%.1fs train + %.1fs val) %s",
                epoch,
                train_metrics["loss"],
                eval_result.metrics.get("hit_rate_n3", float("nan")),
                eval_result.metrics.get("presence_auroc", float("nan")),
                train_elapsed, val_elapsed,
                "[best]" if improved else "",
            )

            if epoch_callback is not None:
                epoch_callback(epoch, epoch_metrics, ckpt_path, improved)

        # All ranks resync before the next epoch's DataLoader iter.
        if ddp.is_distributed:
            dist.barrier()

    # If per-epoch val was subsampled, run canonical metrics on the full val
    # set once at the end. All ranks participate (sharded inference + gather);
    # eval lands on rank 0.
    if cfg.val_subsample_per_epoch > 0 and len(epoch_val_records) < len(val_records):
        if ddp.is_main:
            logger.info("Running final full-val pass on %d frames", len(val_records))
        t_final = time.monotonic()
        final_predictions = _run_val_inference(
            model, val_records, image_loader, device, cfg, ddp,
        )
        if ddp.is_main:
            final_eval = evaluate(
                final_predictions, frames=frames, splits=splits,
                wavelengths=wavelengths, lines=lines,
                split="val", presence_threshold=cfg.presence_threshold,
            )
            final_metrics = {
                **{f"final_val_{k}": v for k, v in final_eval.metrics.items()},
                "final_val_seconds": time.monotonic() - t_final,
            }
            artifacts.final_metrics = final_metrics
            logger.info(
                "Final val: hit_rate_n3=%.3f hit_rate_n4=%.3f auroc=%.3f fpr=%.3f",
                final_eval.metrics.get("hit_rate_n3", float("nan")),
                final_eval.metrics.get("hit_rate_n4", float("nan")),
                final_eval.metrics.get("presence_auroc", float("nan")),
                final_eval.metrics.get("fpr_at_threshold", float("nan")),
            )

    # DESIGN.md §8.1: latency benchmark — runs on rank 0 only since the
    # eventual UUV deployment is single-GPU; per-rank numbers don't matter.
    if ddp.is_main and cfg.latency_benchmark_frames > 0 and val_records:
        bench_model = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )
        latency = _benchmark_latency(
            model=bench_model, sample_records=val_records, image_loader=image_loader,
            device=device, cfg=cfg,
        )
        artifacts.latency_metrics = latency
        if latency:
            logger.info(
                "Inference latency: bs=1 %.1f ms/frame, bs=8 %.1f ms/frame",
                latency.get("latency_bs1_ms", float("nan")),
                latency.get("latency_bs8_ms", float("nan")),
            )

    if ddp.is_distributed:
        dist.barrier()
    return artifacts


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
