"""CLI to run Phase 2 supervised training + log to MLflow.

Loads the Phase 0 artifacts (frames, splits, wavelengths, lines), builds
per-frame `FrameRecord` lists for the train and val splits, sets up MLflow,
and runs `train.train()` with an `epoch_callback` that logs per-epoch metrics
and snapshots the best checkpoint as a run artifact.

Usage (single GPU):
    uv run python scripts/run_train.py
    uv run python scripts/run_train.py --epochs 5 --batch-size 8 --max-train-dives 4 --max-val-dives 2

Usage (multi-GPU DDP):
    uv run torchrun --standalone --nproc_per_node=4 scripts/run_train.py --epochs 10 --batch-size 16

Under torchrun the per-rank batch size is `--batch-size`, so the global batch
becomes `batch_size * world_size`. MLflow logging, checkpoints, and the
final-val pass run on rank 0; all ranks participate in train + val inference.

Smoke-run flags (`--max-*-dives`, `--max-*-frames`) restrict the dataset
without changing behavior, so a quick end-to-end sanity check costs minutes
rather than hours. The same `--no-mlflow` flag as the baseline runner is
useful when iterating locally.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import polars as pl

from laser_detector.data import build_records, load_orf_flip
from laser_detector.preprocessing.config import load_config
from laser_detector.preprocessing.image_loader import (
    CachingImageLoader,
    LocalFilesystemImageLoader,
    make_cached_image_loader,
)
from laser_detector.tracking import setup_mlflow
from laser_detector.train import (
    TrainConfig,
    find_latest_checkpoint,
    init_distributed,
    shutdown_distributed,
    train,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 2 detector + log to MLflow")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader per-worker prefetch buffer. Drop to 1 when the cache "
        "working set is close to RAM capacity to reduce page-cache thrashing.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Linear LR warmup steps. Overrides warmup_epochs. "
        "Default 1000 — fits the full-corpus run; smoke runs can leave it as-is "
        "(it caps at the schedule total).",
    )
    parser.add_argument(
        "--max-train-dives",
        type=int,
        default=0,
        help="Cap on train-split dives for smoke runs. 0 = no cap.",
    )
    parser.add_argument(
        "--max-val-dives",
        type=int,
        default=0,
        help="Cap on val-split dives for smoke runs. 0 = no cap.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("data/phase2/checkpoints"),
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow setup and just print per-epoch metrics.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--heatmap-loss",
        choices=("bce", "focal"),
        default="bce",
        help="Heatmap loss formulation. BCE+pos_weight is default after the "
        "2026-05-03 focal collapse; focal is kept for ablation.",
    )
    parser.add_argument(
        "--heatmap-pos-weight",
        type=float,
        default=1000.0,
        help="pos_weight for BCE heatmap loss. Counterweights the "
        "1-pos-pixel-vs-1M-neg-pixel imbalance per tile.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without val hit_rate_n3 improvement. "
        "0 disables. Recommended ≥10 for 50-epoch runs (subsample variance).",
    )
    parser.add_argument(
        "--lambda-line",
        type=float,
        default=0.0,
        help="Weight for the L_line aux loss (DESIGN.md §5.1). 0 disables; "
        "Phase 3 default per DESIGN is 0.1. Active only on tiles where the "
        "dive's line is confident AND the label is in the crop.",
    )
    parser.add_argument(
        "--soft-snap-inference",
        action="store_true",
        help="Apply DESIGN.md §6.2 soft-snap-to-line at inference time. "
        "Pulls the heatmap argmax toward the dive line (confident dives "
        "only); blend = sigmoid(line_conf - τ) * (1 - pred_conf), capped "
        "at --soft-snap-alpha-max. Affects per-epoch val + final-val.",
    )
    parser.add_argument(
        "--soft-snap-alpha-max",
        type=float,
        default=0.3,
        help="Maximum blend weight α for soft-snap. DESIGN guidance: ≤ 0.3.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint. Path to a .pt file, or 'auto' "
        "to pick the highest-numbered epoch_NNN.pt in --checkpoint-dir.",
    )
    parser.add_argument(
        "--image-pipeline",
        choices=("jpeg", "linear", "linear_npy"),
        default="jpeg",
        help="`jpeg`: legacy uint8 cache via fishsense-core (default). "
        "`linear`: rawpy-direct uint16 PNG cache, no CLAHE — see "
        "notes/state.md 'Audit findings'. `linear_npy`: same source, .npy "
        "cache for faster dataloader.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override the cache directory. Defaults to config.cache_dir for "
        "the JPEG pipeline; <cache_dir>_linear/ for the linear pipeline.",
    )
    parser.add_argument(
        "--train-rig-prior",
        action="store_true",
        help="Add the static rig-prior log-mask to heatmap logits during "
        "training (before BCE). Bakes the rig geometry into the trained "
        "weights so train and inference share the same prior assumption.",
    )
    parser.add_argument(
        "--rig-prior-floor",
        type=float,
        default=None,
        help="Override the rig-prior Gaussian floor for both training and "
        "inference. 1.0 = pure bbox (no Gaussian bias).",
    )
    parser.add_argument(
        "--inference-rig-prior",
        action="store_true",
        help="Apply the rig-prior multiplicatively to heatmap probs at "
        "inference. Use alongside --train-rig-prior or alone.",
    )
    parser.add_argument(
        "--bayer-excess",
        action="store_true",
        help="Add Bayer-derived (G_excess, R_excess) channels as inputs 5-6. "
        "Requires the prewarmed bayer-excess cache (see prewarm_bayer_excess_cache.py). "
        "Sets in_channels=6 on the model and loads the parallel cache for "
        "training/eval.",
    )
    parser.add_argument(
        "--bayer-excess-cache-dir",
        type=Path,
        default=None,
        help="Override the bayer-excess cache dir. Default: <cache_dir>_bayer_excess.",
    )
    parser.add_argument(
        "--wavelength-balance",
        action="store_true",
        help="Inverse-frequency weight positive sampling by wavelength group "
        "(red/green/None). Compensates for the ~4x red:green imbalance in v1 "
        "and is the natural follow-up to the run3 audit, which showed a ~6 pp "
        "dive-averaged green deficit on epoch_021.",
    )
    parser.add_argument(
        "--decoder-interpolation",
        choices=("nearest", "bilinear", "bicubic"),
        default="nearest",
        help="smp.Unet decoder upsample mode. Default 'nearest' matches smp's "
        "own default but introduces an axis-asymmetric argmax-tie bias; "
        "'bilinear' removes it. See notes/bias_attribution.md.",
    )
    parser.add_argument(
        "--bayer-diff-channel",
        action="store_true",
        help="Add a third Bayer-excess channel G_diff = G1 − G2 (anti-diagonal "
        "sub-supercell asymmetry). Requires --bayer-excess and a prewarmed "
        "'bayer_excess_diff' cache. Sets in_channels=7. See notes/bias_attribution.md.",
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        default="resnet34",
        help="smp.Unet encoder backbone. Default 'resnet34' matches run3. "
        "Try 'tu-hrnet_w18'/'tu-hrnet_w32'/'tu-hrnet_w48' for HRNet variants "
        "(Phase 3.2).",
    )
    parser.add_argument(
        "--heatmap-sigma-px",
        type=float,
        default=3.0,
        help="σ of the Gaussian target heatmap (px). Default 3.0 matches "
        "run3-7. Smaller values (e.g. 1.5) give sharper supervision peaks "
        "for sub-pixel-aware training.",
    )
    return parser.parse_args(argv)


def _split_records(
    *,
    frames: pl.DataFrame,
    splits: pl.DataFrame,
    wavelengths: pl.DataFrame,
    lines: pl.DataFrame | None,
    split: str,
    max_dives: int,
    orf_flip: pl.DataFrame | None = None,
):
    dive_ids = (
        splits.filter(pl.col("split") == split)["dive_id"].unique().to_list()
    )
    if max_dives > 0:
        dive_ids = dive_ids[:max_dives]
    split_frames = frames.filter(pl.col("dive_id").is_in(dive_ids))
    return (
        build_records(split_frames, wavelengths, lines, orf_flip=orf_flip),
        dive_ids,
        split_frames.height,
    )


def _filter_loadable(
    records: list,
    image_loader: CachingImageLoader,
) -> tuple[list, int]:
    """Drop records whose JPEG cache file is missing.

    The 230 prewarm-failed frames in the 2026-05-04 corpus (NAS-path issues
    on dives 237/219/249) are clustered in record order; the dataset's
    8-consecutive-failure retry budget can be exhausted by a single block of
    them, crashing training at a random epoch. Filtering up-front is faster,
    more deterministic, and free of restart cost.
    """
    n_before = len(records)
    keep = [r for r in records if image_loader.cache_path(r.image_checksum).exists()]
    return keep, n_before - len(keep)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ddp = init_distributed()
    # Rank 0 logs at INFO; non-zero ranks at WARNING so the console isn't
    # flooded with `world_size` copies of every progress message.
    logging.basicConfig(
        level=logging.INFO if ddp.is_main else logging.WARNING,
        format=f"%(asctime)s [r{ddp.rank}] [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config()

    if config.image_root is None:
        logging.error(
            "No image root configured. Training needs image bytes; "
            "set `images.root` in settings.local.toml."
        )
        return 2

    cache_dir = args.cache_dir or (
        Path(f"{config.cache_dir}_linear_npy") if args.image_pipeline == "linear_npy"
        else Path(f"{config.cache_dir}_linear") if args.image_pipeline == "linear"
        else config.cache_dir
    )
    image_loader = make_cached_image_loader(
        config.image_root, cache_dir,
        pipeline=args.image_pipeline,
        jpeg_quality=config.cache_jpeg_quality,
    )
    bayer_excess_loader = None
    if args.bayer_excess:
        bayer_pipeline = "bayer_excess_diff" if args.bayer_diff_channel else "bayer_excess"
        bayer_cache_dir = args.bayer_excess_cache_dir or Path(
            f"{config.cache_dir}_{bayer_pipeline}"
        )
        bayer_excess_loader = make_cached_image_loader(
            config.image_root, bayer_cache_dir,
            pipeline=bayer_pipeline,
        )
    if ddp.is_main:
        logging.info(
            "Image pipeline: %s (cache=%s)%s",
            args.image_pipeline, cache_dir,
            f"; bayer_excess cache={bayer_cache_dir}" if args.bayer_excess else "",
        )

    frames = pl.read_parquet(config.data_dir / "frames.parquet")
    splits = pl.read_parquet(config.data_dir / "dive_splits.parquet")
    wavelengths = pl.read_parquet(config.data_dir / "dive_wavelengths.parquet")
    lines = pl.read_parquet(config.data_dir / "dive_lines.parquet")

    # Drop upstream-superseded frames before *both* training and eval — those
    # labels were flagged as outliers and would unfairly count against the
    # model in the val/full-val pass. `build_records` does the same filter
    # internally; we mirror it here so `evaluate()`'s frames-table is consistent.
    if "superseded" in frames.columns:
        n_before = frames.height
        frames = frames.filter(~pl.col("superseded"))
        if ddp.is_main and frames.height < n_before:
            logging.info(
                "Dropped %d superseded frames (kept %d / %d)",
                n_before - frames.height, frames.height, n_before,
            )

    orf_flip = load_orf_flip(config.data_dir)
    if ddp.is_main and orf_flip is not None:
        logging.info("Loaded orf_flip parquet: %d rows", orf_flip.height)
    train_records, train_dives, n_train = _split_records(
        frames=frames, splits=splits, wavelengths=wavelengths, lines=lines,
        split="train", max_dives=args.max_train_dives,
        orf_flip=orf_flip,
    )
    val_records, val_dives, n_val = _split_records(
        frames=frames, splits=splits, wavelengths=wavelengths, lines=lines,
        split="val", max_dives=args.max_val_dives,
        orf_flip=orf_flip,
    )
    train_records, n_train_dropped = _filter_loadable(train_records, image_loader)
    val_records, n_val_dropped = _filter_loadable(val_records, image_loader)
    if ddp.is_main:
        logging.info(
            "Train: %d dives, %d frames (dropped %d unloadable) | Val: %d dives, %d frames (dropped %d unloadable)",
            len(train_dives), len(train_records), n_train_dropped,
            len(val_dives), len(val_records), n_val_dropped,
        )

    # When --max-*-dives caps the records, restrict the splits frame the
    # trainer hands to evaluate() so it doesn't score against the full val
    # set (every uninferenced dive would count as a miss).
    sampled_dives = set(train_dives) | set(val_dives)
    eval_splits = splits.filter(pl.col("dive_id").is_in(list(sampled_dives)))

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        heatmap_loss=args.heatmap_loss,
        heatmap_pos_weight=args.heatmap_pos_weight,
        early_stop_patience=args.early_stop_patience,
        lambda_line=args.lambda_line,
        inference_soft_snap=args.soft_snap_inference,
        inference_soft_snap_alpha_max=args.soft_snap_alpha_max,
        linear_cache=args.image_pipeline.startswith("linear"),
        train_rig_prior=args.train_rig_prior,
        train_rig_prior_floor=args.rig_prior_floor,
        inference_rig_prior=args.inference_rig_prior,
        inference_rig_prior_floor=args.rig_prior_floor,
        use_bayer_excess=args.bayer_excess,
        bayer_diff_channel=args.bayer_diff_channel,
        in_channels=(
            7 if (args.bayer_excess and args.bayer_diff_channel)
            else 6 if args.bayer_excess
            else 4
        ),
        wavelength_balance=args.wavelength_balance,
        decoder_interpolation=args.decoder_interpolation,
        encoder_name=args.encoder_name,
        heatmap_sigma_px=args.heatmap_sigma_px,
    )

    resume_from: Path | None = None
    if args.resume:
        if args.resume == "auto":
            resume_from = find_latest_checkpoint(args.checkpoint_dir)
            if resume_from is None and ddp.is_main:
                logging.info(
                    "--resume auto: no checkpoint found in %s; starting from scratch",
                    args.checkpoint_dir,
                )
        else:
            resume_from = Path(args.resume)
            if not resume_from.exists():
                if ddp.is_main:
                    logging.error("--resume path does not exist: %s", resume_from)
                shutdown_distributed()
                return 2

    if args.no_mlflow:
        train(
            cfg=cfg,
            train_records=train_records,
            val_records=val_records,
            image_loader=image_loader,
            bayer_excess_loader=bayer_excess_loader,
            frames=frames, splits=eval_splits, wavelengths=wavelengths, lines=lines,
            checkpoint_dir=args.checkpoint_dir,
            ddp=ddp,
            resume_from=resume_from,
        )
        shutdown_distributed()
        return 0

    # Only rank 0 talks to MLflow. Other ranks run training without a run
    # context — the trainer's epoch_callback hands metrics to whichever
    # rank wraps the call, and the non-zero ranks pass `None` callback.
    if not ddp.is_main:
        train(
            cfg=cfg,
            train_records=train_records,
            val_records=val_records,
            image_loader=image_loader,
            bayer_excess_loader=bayer_excess_loader,
            frames=frames, splits=eval_splits, wavelengths=wavelengths, lines=lines,
            checkpoint_dir=args.checkpoint_dir,
            ddp=ddp,
            resume_from=resume_from,
        )
        shutdown_distributed()
        return 0

    setup_mlflow(config)
    with mlflow.start_run(run_name="phase2_train") as run:
        mlflow.set_tag("phase", "phase2_train")
        mlflow.set_tag("detector", "resnet34_unet_heatmap")
        mlflow.set_tag("world_size", str(ddp.world_size))
        mlflow.log_params(
            {
                **cfg.__dict__,
                "world_size": ddp.world_size,
                "global_batch_size": cfg.batch_size * ddp.world_size,
                "n_train_dives": len(train_dives),
                "n_train_frames": n_train,
                "n_val_dives": len(val_dives),
                "n_val_frames": n_val,
                "max_train_dives": args.max_train_dives,
                "max_val_dives": args.max_val_dives,
            }
        )

        def _on_epoch(epoch, metrics, ckpt_path, improved):
            # Per-epoch metrics are stepped by epoch index; per-step metrics
            # below use the global batch count, so they live in separate
            # series and don't collide.
            mlflow.log_metrics(metrics, step=epoch)
            if improved:
                mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints/best")

        def _on_step(global_step, step_metrics):
            mlflow.log_metrics(step_metrics, step=global_step)

        artifacts = train(
            cfg=cfg,
            train_records=train_records,
            val_records=val_records,
            image_loader=image_loader,
            bayer_excess_loader=bayer_excess_loader,
            frames=frames, splits=eval_splits, wavelengths=wavelengths, lines=lines,
            checkpoint_dir=args.checkpoint_dir,
            epoch_callback=_on_epoch,
            step_callback=_on_step,
            ddp=ddp,
            resume_from=resume_from,
        )

        if artifacts.best_checkpoint_path is not None:
            mlflow.log_metrics(
                {f"best_{k}": v for k, v in artifacts.best_metrics.items() if isinstance(v, (int, float))},
            )
        if artifacts.latency_metrics:
            mlflow.log_metrics(artifacts.latency_metrics)
        if artifacts.final_metrics:
            mlflow.log_metrics(
                {k: v for k, v in artifacts.final_metrics.items() if isinstance(v, (int, float))}
            )
        logging.info(
            "Training done. Run %s at %s/#/experiments/%s/runs/%s",
            run.info.run_id,
            config.mlflow_tracking_uri,
            run.info.experiment_id,
            run.info.run_id,
        )

    shutdown_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
