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
from dataclasses import dataclass, field, replace
from datetime import timedelta
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
from laser_detector.inference import (
    predict_frame,
    predict_frame_with_cascade,
    rectify_prediction,
    rig_prior_log_mask_batched,
)
from laser_detector.model import (
    LaserDetector,
    bce_heatmap_loss,
    focal_heatmap_loss,
    line_consistency_loss,
)
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
    # DESIGN.md §5.1 L_line aux loss (Phase 3). 0 disables it. Active only on
    # tiles where (a) the dive's line is confident, AND (b) the per-tile
    # presence target is 1 — i.e. the label landed in the crop, so the
    # heatmap soft-argmax has a meaningful relationship to the line.
    lambda_line: float = 0.0
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
    # DESIGN.md §6.2 soft-snap inference. When True, val inference projects
    # the heatmap argmax toward the dive's line on confident-line dives,
    # blended by α = sigmoid(line_conf - τ) * (1 - pred_conf), capped at
    # `inference_soft_snap_alpha_max`. Default off so it's an opt-in
    # comparison rather than silently changing val numbers.
    inference_soft_snap: bool = False
    inference_soft_snap_alpha_max: float = 0.3
    # Per-step metric logging cadence (rank-0 only). 0 disables it; default 50
    # gives ~850 data points across a 10-epoch full-corpus run, which renders
    # as a smooth loss curve in MLflow / TensorBoard. Per-epoch metrics are
    # logged separately via epoch_callback.
    log_every_n_steps: int = 50
    # Early stopping: number of epochs of no `hit_rate_n3` improvement before
    # bailing. 0 disables; the recommended default for 50-epoch runs is ~10
    # given subsample variance can make 3-4-epoch dips look real.
    early_stop_patience: int = 0
    # When True, the dataset uses uint16-compatible photometric augs (drops
    # HueSaturationValue + ImageCompression which are uint8-only). Set this
    # via run_train.py's `--image-pipeline linear`.
    linear_cache: bool = False
    # When True, the per-tile heatmap is multiplied by the static rig-prior
    # mask before argmax. See inference._rig_prior_for_tile.
    inference_rig_prior: bool = False
    # Gaussian floor inside the bbox. 1.0 = pure bbox (Gaussian disabled).
    # None falls back to inference.DEFAULT_RIG_PRIOR_FLOOR.
    inference_rig_prior_floor: float | None = None
    # Gaussian σ override at inference. None → DEFAULT_RIG_PRIOR_SIGMA.
    # Override y_max of the rig-prior bbox (default 2180). Phase 3.1 tightens
    # this to clip wandering catastrophic predictions that land below any
    # legitimate label position; val/test green label y_max is 1552/1624 and
    # red is 1520/1514, so a y_max around 1700 is safe.
    inference_rig_prior_bbox_ymax: int | None = None
    # Phase 3.1d: per-dive line corridor mask. Zeros heatmap pixels farther
    # than `inference_line_mask_corridor_px` from the fitted dive line at
    # argmax time. Active only on frames where line_confidence > 0. Much
    # tighter than the rig bbox (per-dive geometry instead of static bbox);
    # safe widths from data analysis: ±25 px covers test labels p99=8.98,
    # masks distractors at 200+ px (e.g., val:427 cluster).
    inference_line_mask_corridor_px: float | None = None
    inference_rig_prior_sigma_x: float | None = None
    inference_rig_prior_sigma_y: float | None = None
    # When True, the static rig-prior log-mask is added to heatmap logits
    # *during training* before BCE, so the model learns its outputs against
    # an inference-time consistent prior. Pairs with `inference_rig_prior`.
    train_rig_prior: bool = False
    # Floor for the training-time prior. None → DEFAULT_RIG_PRIOR_FLOOR.
    train_rig_prior_floor: float | None = None
    # When True, val/eval inference uses `predict_frame_with_cascade` (Phase 5
    # refinement crop) instead of the single-pass `predict_frame`.
    inference_cascade: bool = False
    # Refinement window size for cascade. None → predict_frame_with_cascade default.
    inference_cascade_refine_window: int | None = None
    # When True, refine the heatmap argmax to sub-pixel via a 3-point parabolic
    # peak fit on the cross neighborhood (see inference._subpixel_refine_peak).
    # Phase 2A; Phase 1A showed ~63% of failures are 3-10 px borderline misses.
    inference_subpixel_refine: bool = False
    # Checkpoint-specific pixel-bias calibration. Subtracted from the final
    # (pred_x, pred_y) before clamping. Defaults to no correction. Originates
    # from the Bayer-excess upsample shift in preprocessing/image_loader.py
    # (np.repeat puts each supercell value at the top-left of the 2×2 block
    # rather than its centroid). Empirical value for the 6-ch run3 ckpt on val
    # is approx (−1.13, −2.07); subtracting that lifts hit_n3 0.526 → 0.797
    # (LOO-validated). Should be 0 for 4-ch JPEG checkpoints.
    inference_pixel_bias_offset_x: float = 0.0
    inference_pixel_bias_offset_y: float = 0.0
    # Issue #9: labels live in rectified (undistorted) pixel space but the
    # image loader consumes raw images, so predictions land in raw pixel space.
    # When True, apply cv2.undistortPoints per prediction using per-rig K + dist
    # loaded from `inference_rig_intrinsics_path` — moves predictions from raw
    # → rectified. Empirically <0.1 pp hit_n3 impact but load-bearing for
    # downstream 3D reconstruction consistency.
    inference_rectify_output: bool = False
    inference_rig_intrinsics_path: str | None = None
    # Number of input channels to LaserDetector. Default 4 (chrom + wavelength).
    # 6 when bayer_excess channels are added.
    in_channels: int = 4
    # Encoder backbone for smp.Unet. Default resnet34 matches run3.
    # Phase 3.2: try HRNet variants ("tu-hrnet_w18", "tu-hrnet_w32") as a
    # higher-capacity / different-inductive-bias alternative.
    encoder_name: str = "resnet34"
    # σ of the Gaussian target heatmap (px). Default 3.0 matches run3/5/6/7.
    # Phase 3.2: a smaller σ (1.5) gives sharper supervision peaks and may
    # tighten the borderline-precision mode (70% of remaining test failures).
    heatmap_sigma_px: float = 3.0
    # Decoder upsample mode in smp.Unet. "nearest" matches smp's own default
    # but introduces an axis-asymmetric argmax-tie bias; "bilinear" removes
    # it. See notes/bias_attribution.md for the synthetic ablation.
    decoder_interpolation: str = "nearest"
    # When True, the dataset loads a parallel Bayer-excess cache and appends
    # (G_excess, R_excess) as channels 5 and 6. Pairs with in_channels=6.
    use_bayer_excess: bool = False
    # When True (in addition to use_bayer_excess), the Bayer-excess cache also
    # includes a third channel G_diff = G1 − G2 (anti-diagonal sub-supercell
    # asymmetry). Pairs with in_channels=7 and the "bayer_excess_diff"
    # cache pipeline. See notes/bias_attribution.md.
    bayer_diff_channel: bool = False
    # When True, the HardNegativeBalancedSampler weights positives inversely
    # to their wavelength group size so the rarer green frames get oversampled.
    # Addresses the dive-averaged green deficit seen in the failure audit on
    # epoch_021 of the sensor 6-ch run3.
    wavelength_balance: bool = False


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
            # Default NCCL collective timeout is 10 min; raise to 60 min so val
            # passes on the I/O-bound 6-ch linear+bayer cache don't trip the
            # heartbeat watchdog on stragglers.
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=60),
            )
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


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return `epoch_NNN.pt` with the highest NNN, or None if none exist.

    Phase 2 saves checkpoints as `epoch_{epoch:03d}.pt`; this gives us a
    deterministic "latest" for `--resume auto` without needing a separate
    pointer file."""
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("epoch_*.pt"))
    return candidates[-1] if candidates else None


def _save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    sampler: HardNegativeBalancedSampler,
    epoch: int,
    global_step: int,
    best_score: float,
    best_epoch: int,
    patience_counter: int,
    metrics: dict[str, float],
    cfg: TrainConfig,
) -> None:
    """Save full training state for resume.

    Captured: model + optimizer + scheduler + sampler weights + RNG state +
    early-stopping counters + per-epoch metrics + the cfg used for this run.
    `model.module.state_dict()` if the model is DDP-wrapped — saving the
    wrapper would make the checkpoint load-only-under-DDP, which breaks
    `eval_checkpoint.py` and any future single-GPU resume.
    """
    # CUDA RNG state intentionally not saved: with DDP, each rank should
    # evolve its own GPU RNG independently, and saving rank-0's view +
    # restoring it on every rank would destroy that. Torch CPU RNG + numpy
    # RNG cover the correctness-critical determinism (sampler, augmentations).
    underlying = model.module if isinstance(model, DistributedDataParallel) else model
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
        "model_state_dict": underlying.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "sampler_state": {
            "neg_scores": sampler.neg_scores,
            "epoch": sampler._epoch,
            "score_rng_state": sampler._score_rng.bit_generator.state,
        },
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "metrics": metrics,
        "cfg": cfg.__dict__,
    }
    # Atomic write: save to <path>.tmp then rename. A kill mid-write leaves
    # a .tmp file (cleaned up by callers / safely ignored by load) but never
    # corrupts the eventual checkpoint. Without this, an interrupted save
    # produces a truncated .pt that breaks `torch.load` on resume.
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.rename(path)


def _load_checkpoint_for_resume(
    *,
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    sampler: HardNegativeBalancedSampler,
    device: torch.device,
) -> dict:
    """Restore everything `_save_checkpoint` wrote. Returns a dict with the
    counters the trainer needs (`start_epoch`, `global_step`, `best_score`,
    `best_epoch`, `patience_counter`)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    underlying = model.module if isinstance(model, DistributedDataParallel) else model
    underlying.load_state_dict(ckpt["model_state_dict"])

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if "sampler_state" in ckpt:
        s = ckpt["sampler_state"]
        sampler.neg_scores = np.asarray(s["neg_scores"], dtype=np.float64)
        sampler._epoch = int(s["epoch"])
        sampler._score_rng.bit_generator.state = s["score_rng_state"]

    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    if "numpy_rng_state" in ckpt:
        np.random.set_state(ckpt["numpy_rng_state"])

    return {
        "start_epoch": int(ckpt["epoch"]) + 1,
        "global_step": int(ckpt.get("global_step", 0)),
        "best_score": float(ckpt.get("best_score", -float("inf"))),
        "best_epoch": int(ckpt.get("best_epoch", -1)),
        "patience_counter": int(ckpt.get("patience_counter", 0)),
    }


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
    *,
    global_step_start: int = 0,
    step_callback=None,
) -> tuple[dict[str, float], int]:
    model.train()
    autocast_dtype = torch.bfloat16 if cfg.use_bf16 and device.type == "cuda" else None
    sums = {"loss": 0.0, "loss_heatmap": 0.0, "loss_presence": 0.0, "loss_line": 0.0}
    n_batches = 0
    global_step = int(global_step_start)

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
        if cfg.train_rig_prior:
            crop_offset = batch["crop_offset"].to(device, non_blocking=True)
            tile_size = heatmap_logits.shape[-1]
            log_mask_kwargs: dict = {}
            if cfg.train_rig_prior_floor is not None:
                log_mask_kwargs["floor"] = cfg.train_rig_prior_floor
            log_mask = rig_prior_log_mask_batched(
                crop_offset, tile_size, **log_mask_kwargs,
            )  # [B, tile, tile]
            # heatmap_logits is [B, 1, tile, tile]; broadcast log_mask over channel.
            heatmap_logits = heatmap_logits + log_mask.unsqueeze(1)
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
        if cfg.lambda_line > 0.0:
            crop_offset = batch["crop_offset"].to(device, non_blocking=True)
            line_abc_t = batch["line_abc"].to(device, non_blocking=True)
            line_conf_t = batch["line_confidence"].to(device, non_blocking=True)
            is_line_conf_t = batch["is_line_confident"].to(device, non_blocking=True)
            # Only frames whose dive has a confident line AND whose label is in
            # the crop contribute. The presence target == 1 ↔ label-in-crop.
            valid_mask = is_line_conf_t & (presence_target > 0.5)
            loss_line = line_consistency_loss(
                heatmap_logits, crop_offset, line_abc_t, line_conf_t, valid_mask,
            )
        else:
            loss_line = heatmap_logits.new_zeros(())
        loss = (
            cfg.lambda_heatmap * loss_hm
            + cfg.lambda_presence * loss_pres
            + cfg.lambda_line * loss_line
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        loss_hm_val = float(loss_hm.item())
        loss_pres_val = float(loss_pres.item())
        loss_line_val = float(loss_line.item())
        sums["loss"] += loss_val
        sums["loss_heatmap"] += loss_hm_val
        sums["loss_presence"] += loss_pres_val
        sums["loss_line"] += loss_line_val
        n_batches += 1
        global_step += 1
        if pbar is not None and n_batches % 20 == 0:
            pbar.set_postfix(
                loss=f"{sums['loss'] / n_batches:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
        if (
            step_callback is not None
            and ddp.is_main
            and cfg.log_every_n_steps > 0
            and global_step % cfg.log_every_n_steps == 0
        ):
            step_callback(global_step, {
                "step_loss": loss_val,
                "step_loss_heatmap": loss_hm_val,
                "step_loss_presence": loss_pres_val,
                "step_loss_line": loss_line_val,
                "lr": float(scheduler.get_last_lr()[0]),
            })

    means = {k: v / max(n_batches, 1) for k, v in sums.items()}
    # Average across ranks so the logged number reflects global behavior, not
    # rank 0's slice. Each rank saw a different subset of batches.
    if ddp.is_distributed:
        t = torch.tensor(
            [means["loss"], means["loss_heatmap"], means["loss_presence"], means["loss_line"]],
            device=ddp.device,
        )
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        (means["loss"], means["loss_heatmap"],
         means["loss_presence"], means["loss_line"]) = (
            float(t[0]), float(t[1]), float(t[2]), float(t[3]),
        )
    return means, global_step


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
    bayer_excess_loader: ImageLoader | None = None,
) -> dict[str, float]:
    """Time tiled-inference ms/frame at batch=1 and batch=8 per DESIGN.md §8.1.

    Skipped (returns {}) if we can't decode at least `warmup + measure` frames.
    """
    needed = cfg.latency_benchmark_warmup + cfg.latency_benchmark_frames
    # Each sample carries its optional Bayer-excess tile so a 6-channel model
    # (cfg.use_bayer_excess) is fed the same inputs here as in val inference;
    # without it predict_frame builds a 4-channel tile and conv1 mismatches.
    samples: list[tuple[FrameRecord, np.ndarray, np.ndarray | None]] = []
    for rec in sample_records:
        img = image_loader.load(rec.image_path, rec.image_checksum)
        if img is None:
            continue
        bayer_img = (
            bayer_excess_loader.load(rec.image_path, rec.image_checksum)
            if (cfg.use_bayer_excess and bayer_excess_loader is not None)
            else None
        )
        samples.append((rec, img, bayer_img))
        if len(samples) >= needed:
            break

    if len(samples) < needed:
        logger.warning(
            "Latency benchmark needs %d images, got %d — skipping", needed, len(samples)
        )
        return {}

    autocast_dtype = torch.bfloat16 if cfg.use_bf16 and device.type == "cuda" else None
    model.eval()

    def _predict(rec, img, bayer_img, bs):
        bayer_kwargs = (
            {"bayer_excess_image": bayer_img} if bayer_img is not None else {}
        )
        predict_frame(
            img, model, wavelength=rec.wavelength, device=device,
            batch_size=bs, autocast_dtype=autocast_dtype,
            **bayer_kwargs,
        )

    metrics: dict[str, float] = {}
    for bs in (1, 8):
        for rec, img, bayer_img in samples[: cfg.latency_benchmark_warmup]:
            _predict(rec, img, bayer_img, bs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        measured = samples[cfg.latency_benchmark_warmup : needed]
        for rec, img, bayer_img in measured:
            _predict(rec, img, bayer_img, bs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        metrics[f"latency_bs{bs}_ms"] = (elapsed / len(measured)) * 1000.0
    return metrics


def _load_rig_intrinsics(path: str | None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load per-rig camera K + dist coefficients from a parquet written by
    scripts/ingest_camera_intrinsics.py. Returns {} on missing path so the
    caller can noop rectification cleanly. Schema:
    rig_id (int), fx/fy/cx/cy (float), dist (list[float])."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning('rig intrinsics path %s does not exist; rectification will noop', p)
        return {}
    df = pl.read_parquet(p)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for r in df.iter_rows(named=True):
        K = np.array([
            [r['fx'], 0.0, r['cx']],
            [0.0, r['fy'], r['cy']],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        dist = np.array(r['dist'], dtype=np.float32)
        out[int(r['rig_id'])] = (K, dist)
    return out


def _run_val_inference(
    model: torch.nn.Module,
    val_records: list[FrameRecord],
    loader: ImageLoader,
    device: torch.device,
    cfg: TrainConfig,
    ddp: DDPContext,
    bayer_excess_loader: ImageLoader | None = None,
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

    # Issue #9: per-rig intrinsics for output rectification. Loaded once per
    # rank; noop dict if the feature is off or the parquet is missing.
    rig_intrinsics: dict[int, tuple[np.ndarray, np.ndarray]] = (
        _load_rig_intrinsics(cfg.inference_rig_intrinsics_path)
        if cfg.inference_rectify_output else {}
    )
    if cfg.inference_rectify_output and not rig_intrinsics and ddp.is_main:
        logger.warning('inference_rectify_output=True but no rig intrinsics loaded; rectification will noop')

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
        snap_line_abc = (
            rec.line_abc
            if (cfg.inference_soft_snap and rec.is_line_confident and rec.line_abc is not None)
            else None
        )
        snap_line_conf = rec.line_confidence if snap_line_abc is not None else 0.0
        rig_prior_kwargs: dict = {"rig_prior": cfg.inference_rig_prior}
        if cfg.inference_rig_prior_floor is not None:
            rig_prior_kwargs["rig_prior_floor"] = cfg.inference_rig_prior_floor
        if cfg.inference_rig_prior_sigma_x is not None and cfg.inference_rig_prior_sigma_y is not None:
            rig_prior_kwargs["rig_prior_sigma"] = (
                cfg.inference_rig_prior_sigma_x,
                cfg.inference_rig_prior_sigma_y,
            )
        if cfg.inference_rig_prior_bbox_ymax is not None:
            from laser_detector.inference import DEFAULT_RIG_PRIOR_BBOX
            x0, y0, x1, _y1 = DEFAULT_RIG_PRIOR_BBOX
            rig_prior_kwargs["rig_prior_bbox"] = (x0, y0, x1, cfg.inference_rig_prior_bbox_ymax)
        predict_fn = (
            predict_frame_with_cascade if cfg.inference_cascade else predict_frame
        )
        cascade_kwargs: dict = {}
        if cfg.inference_cascade and cfg.inference_cascade_refine_window is not None:
            cascade_kwargs["refine_window"] = cfg.inference_cascade_refine_window
        bayer_image = (
            bayer_excess_loader.load(rec.image_path, rec.image_checksum)
            if (cfg.use_bayer_excess and bayer_excess_loader is not None)
            else None
        )
        bayer_kwargs: dict = {}
        if bayer_image is not None:
            bayer_kwargs["bayer_excess_image"] = bayer_image
        pred = predict_fn(
            image_bgr, inference_model,
            wavelength=rec.wavelength,
            device=device,
            batch_size=cfg.inference_batch_size,
            autocast_dtype=autocast_dtype,
            line_abc=snap_line_abc,
            line_confidence=snap_line_conf,
            alpha_max=cfg.inference_soft_snap_alpha_max,
            subpixel_refine=cfg.inference_subpixel_refine,
            line_mask_corridor_px=cfg.inference_line_mask_corridor_px,
            **rig_prior_kwargs,
            **cascade_kwargs,
            **bayer_kwargs,
        )
        if pred.pred_x is not None and (
            cfg.inference_pixel_bias_offset_x != 0.0
            or cfg.inference_pixel_bias_offset_y != 0.0
        ):
            h, w = image_bgr.shape[:2]
            px = pred.pred_x - cfg.inference_pixel_bias_offset_x
            py = pred.pred_y - cfg.inference_pixel_bias_offset_y
            px = max(0.0, min(px, float(w - 1)))
            py = max(0.0, min(py, float(h - 1)))
            pred = replace(pred, pred_x=px, pred_y=py)
        # Issue #9: rectify raw-pixel-space predictions into rectified space
        # (where labels live). Order matters: rectify AFTER bias offset — the
        # offset was calibrated against label residuals in raw space, so it
        # cancels the raw-space bias first; rectification then does the final
        # coord-frame conversion. Skip on out-of-bbox or missing intrinsics.
        if (
            pred.pred_x is not None
            and cfg.inference_rectify_output
            and rec.rig_id is not None
            and rec.rig_id in rig_intrinsics
        ):
            K, dist = rig_intrinsics[rec.rig_id]
            rx, ry = rectify_prediction(pred.pred_x, pred.pred_y, K, dist)
            h, w = image_bgr.shape[:2]
            rx = max(0.0, min(rx, float(w - 1)))
            ry = max(0.0, min(ry, float(h - 1)))
            pred = replace(pred, pred_x=rx, pred_y=ry)
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
    bayer_excess_loader: ImageLoader | None = None,
    epoch_callback=None,
    step_callback=None,
    ddp: DDPContext | None = None,
    resume_from: Path | None = None,
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
        linear_cache=cfg.linear_cache,
        bayer_excess_loader=bayer_excess_loader if cfg.use_bayer_excess else None,
        heatmap_sigma_px=cfg.heatmap_sigma_px,
        seed=cfg.seed + ddp.rank,
    )
    # Single-process dataset used for scoring negatives between epochs.
    score_ds = LaserTileDataset(
        records=train_records, loader=image_loader, augment=False,
        linear_cache=cfg.linear_cache,
        bayer_excess_loader=bayer_excess_loader if cfg.use_bayer_excess else None,
        heatmap_sigma_px=cfg.heatmap_sigma_px,
        seed=cfg.seed + 1 + ddp.rank,
    )
    sampler = HardNegativeBalancedSampler(
        train_records, seed=cfg.seed,
        rank=ddp.rank, world_size=ddp.world_size,
        wavelength_balance=cfg.wavelength_balance,
    )
    if ddp.is_main:
        logger.info(
            "Hard-negative sampler: %d positives, %d negatives → %d samples/rank/epoch (world_size=%d)",
            len(sampler.pos_indices), len(sampler.neg_indices),
            len(sampler), ddp.world_size,
        )
        if cfg.wavelength_balance:
            # Tally positives by wavelength group so the log shows what the
            # inverse-frequency reweighting is actually compensating for.
            wl_counts: dict[object, int] = {}
            for i in sampler.pos_indices:
                w = train_records[int(i)].wavelength
                wl_counts[w] = wl_counts.get(w, 0) + 1
            logger.info(
                "Wavelength-balanced sampling on; positive counts by wavelength: %s",
                {str(k): v for k, v in sorted(wl_counts.items(), key=lambda kv: -kv[1])},
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

    model = LaserDetector(
        encoder_name=cfg.encoder_name,
        in_channels=cfg.in_channels,
        decoder_interpolation=cfg.decoder_interpolation,
    ).to(device)
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
    start_epoch = 0
    global_step = 0
    patience_counter = 0

    if resume_from is not None:
        if ddp.is_main:
            logger.info("Resuming from %s", resume_from)
        restored = _load_checkpoint_for_resume(
            ckpt_path=resume_from, model=model, optimizer=optimizer,
            scheduler=scheduler, sampler=sampler, device=device,
        )
        start_epoch = restored["start_epoch"]
        global_step = restored["global_step"]
        best_score = restored["best_score"]
        artifacts.best_epoch = restored["best_epoch"]
        patience_counter = restored["patience_counter"]
        if ddp.is_main:
            logger.info(
                "Resumed: start_epoch=%d global_step=%d best_score=%.4f best_epoch=%d patience=%d",
                start_epoch, global_step, best_score, artifacts.best_epoch, patience_counter,
            )
        if start_epoch >= cfg.epochs:
            if ddp.is_main:
                logger.warning(
                    "Resumed checkpoint is at epoch %d but cfg.epochs=%d — nothing to do",
                    start_epoch - 1, cfg.epochs,
                )

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

    for epoch in range(start_epoch, cfg.epochs):
        sampler.set_epoch(epoch)
        t0 = time.monotonic()
        train_metrics, global_step = _train_one_epoch(
            model, train_loader, optimizer, scheduler, cfg, device, epoch, ddp,
            global_step_start=global_step, step_callback=step_callback,
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

        # Safety net: save a "pre-val" checkpoint so a val-pass deadlock (e.g.
        # NCCL collective timeout on I/O-bound stragglers) does not lose this
        # epoch's training progress. Overwritten by the full save after val.
        if ddp.is_main:
            _save_checkpoint(
                path=checkpoint_dir / f"epoch_{epoch:03d}.pt",
                model=model, optimizer=optimizer, scheduler=scheduler,
                sampler=sampler,
                epoch=epoch, global_step=global_step,
                best_score=best_score, best_epoch=artifacts.best_epoch,
                patience_counter=patience_counter,
                metrics={**{f"train_{k}": v for k, v in train_metrics.items()},
                         "train_seconds": train_elapsed,
                         "hard_negative_score_seconds": score_elapsed,
                         "hard_negative_score_n": float(n_scored)},
                cfg=cfg,
            )

        t1 = time.monotonic()
        predictions = _run_val_inference(
            model, epoch_val_records, image_loader, device, cfg, ddp,
            bayer_excess_loader=bayer_excess_loader,
        )
        val_elapsed = time.monotonic() - t1

        # Early-stopping decision is computed on rank 0; broadcast to others
        # so all ranks exit the loop together (otherwise non-rank-0 would hang
        # on the next sampler.set_epoch + DataLoader iter).
        stop_signal = torch.zeros(1, device=ddp.device, dtype=torch.int32)

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

            improved = score > best_score
            if improved:
                best_score = score
                patience_counter = 0
                artifacts.best_epoch = epoch
                artifacts.best_metrics = dict(epoch_metrics)
                # best_checkpoint_path set after the save below.
            else:
                patience_counter += 1

            ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            _save_checkpoint(
                path=ckpt_path,
                model=model, optimizer=optimizer, scheduler=scheduler,
                sampler=sampler,
                epoch=epoch, global_step=global_step,
                best_score=best_score, best_epoch=artifacts.best_epoch,
                patience_counter=patience_counter,
                metrics=epoch_metrics, cfg=cfg,
            )
            if improved:
                artifacts.best_checkpoint_path = ckpt_path

            logger.info(
                "epoch %d: train_loss=%.4f val_hit_rate_n3=%.3f val_auroc=%.3f (%.1fs train + %.1fs val) %s",
                epoch,
                train_metrics["loss"],
                eval_result.metrics.get("hit_rate_n3", float("nan")),
                eval_result.metrics.get("presence_auroc", float("nan")),
                train_elapsed, val_elapsed,
                "[best]" if improved else f"[no-improve {patience_counter}/{cfg.early_stop_patience}]"
                if cfg.early_stop_patience > 0 else "",
            )

            if epoch_callback is not None:
                epoch_callback(epoch, epoch_metrics, ckpt_path, improved)

            if cfg.early_stop_patience > 0 and patience_counter >= cfg.early_stop_patience:
                logger.info(
                    "Early stopping triggered: %d epochs without improvement (best=%d, score=%.4f)",
                    patience_counter, artifacts.best_epoch, best_score,
                )
                stop_signal[0] = 1

        # All ranks resync before the next epoch's DataLoader iter.
        if ddp.is_distributed:
            dist.broadcast(stop_signal, src=0)
            dist.barrier()
        if int(stop_signal.item()) == 1:
            break

    # If per-epoch val was subsampled, run canonical metrics on the full val
    # set once at the end. All ranks participate (sharded inference + gather);
    # eval lands on rank 0.
    if cfg.val_subsample_per_epoch > 0 and len(epoch_val_records) < len(val_records):
        if ddp.is_main:
            logger.info("Running final full-val pass on %d frames", len(val_records))
        t_final = time.monotonic()
        final_predictions = _run_val_inference(
            model, val_records, image_loader, device, cfg, ddp,
            bayer_excess_loader=bayer_excess_loader,
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
            device=device, cfg=cfg, bayer_excess_loader=bayer_excess_loader,
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
