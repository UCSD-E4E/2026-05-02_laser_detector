"""Evaluate a saved checkpoint on a split.

Loads `epoch_NNN.pt`, rebuilds the model, runs the same tiled val inference
that the trainer uses (DDP-sharded across `torchrun` ranks), then runs the
standard eval harness on rank 0. Logs to MLflow under `phase2_eval_only`.

Usage:
    uv run torchrun --standalone --nproc_per_node=4 scripts/eval_checkpoint.py \
        --checkpoint data/phase2/checkpoints_bce/epoch_004.pt --split val
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import polars as pl
import torch

from laser_detector.data import build_records, load_orf_flip
from laser_detector.eval import evaluate
from laser_detector.model import LaserDetector
from laser_detector.preprocessing.config import load_config
from laser_detector.preprocessing.image_loader import (
    CachingImageLoader,
    LocalFilesystemImageLoader,
    make_cached_image_loader,
)
from laser_detector.tracking import setup_mlflow
from laser_detector.train import (
    TrainConfig,
    _run_val_inference,
    init_distributed,
    shutdown_distributed,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval a saved checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "test"), default="val",
    )
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--soft-snap-inference",
        action="store_true",
        help="Apply DESIGN.md §6.2 soft-snap-to-line at inference time.",
    )
    parser.add_argument(
        "--soft-snap-alpha-max", type=float, default=0.3,
    )
    parser.add_argument(
        "--rig-prior", action="store_true",
        help="Multiply heatmap by static rig-prior mask (bbox + Gaussian) "
        "centered on the empirical laser-position distribution. Hard-zeros "
        "predictions outside the bbox.",
    )
    parser.add_argument(
        "--rig-prior-floor", type=float, default=None,
        help="Override the Gaussian floor inside the bbox. 1.0 = pure bbox, "
        "no Gaussian bias. Default: inference module's DEFAULT_RIG_PRIOR_FLOOR.",
    )
    parser.add_argument("--rig-prior-sigma-x", type=float, default=None)
    parser.add_argument("--rig-prior-sigma-y", type=float, default=None)
    parser.add_argument(
        "--cascade", action="store_true",
        help="Use predict_frame_with_cascade (Phase 5 refinement crop) "
        "instead of single-pass predict_frame.",
    )
    parser.add_argument(
        "--cascade-refine-window", type=int, default=None,
        help="Cascade refinement window size. Default: predict_frame_with_cascade default.",
    )
    parser.add_argument(
        "--pixel-bias-offset", type=float, nargs=2, default=None, metavar=("DX", "DY"),
        help="Subtract (dx, dy) px from each final prediction. Calibrates out "
        "the constant per-checkpoint bias from the Bayer-excess upsample shift "
        "(image_loader.py:148-150). Use e.g. -1.13 -2.07 for the 6-ch run3 ckpt.",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Skip MLflow logging; just print metrics.",
    )
    parser.add_argument(
        "--image-pipeline",
        choices=("jpeg", "linear", "linear_npy"),
        default="jpeg",
        help="`jpeg`/`linear`/`linear_npy` — must match the pipeline the checkpoint was trained on.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Override cache directory.",
    )
    parser.add_argument(
        "--bayer-excess-cache-dir", type=Path, default=None,
        help="Override the bayer_excess cache dir for 6-ch checkpoints. "
        "Default: <cache_dir>_bayer_excess. Use the centered cache "
        "(data/image_cache_bayer_excess_centered) when evaluating models "
        "trained against it (i.e., run4+).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ddp = init_distributed()
    logging.basicConfig(
        level=logging.INFO if ddp.is_main else logging.WARNING,
        format=f"%(asctime)s [r{ddp.rank}] [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config()

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

    frames = pl.read_parquet(config.data_dir / "frames.parquet")
    splits = pl.read_parquet(config.data_dir / "dive_splits.parquet")
    wavelengths = pl.read_parquet(config.data_dir / "dive_wavelengths.parquet")
    lines = pl.read_parquet(config.data_dir / "dive_lines.parquet")
    if "superseded" in frames.columns:
        frames = frames.filter(~pl.col("superseded"))

    split_dive_ids = (
        splits.filter(pl.col("split") == args.split)["dive_id"].unique().to_list()
    )
    split_frames = frames.filter(pl.col("dive_id").is_in(split_dive_ids))
    orf_flip = load_orf_flip(config.data_dir)
    records = build_records(split_frames, wavelengths, lines, orf_flip=orf_flip)
    if ddp.is_main:
        logging.info(
            "Eval split=%s: %d dives, %d frames, ckpt=%s",
            args.split, len(split_dive_ids), len(records), args.checkpoint,
        )

    # Load checkpoint, recover the saved TrainConfig FIRST, then build a model
    # with the matching in_channels — so 4-ch JPEG and 6-ch sensor+Bayer
    # checkpoints both load.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in ckpt["cfg"].items() if k in TrainConfig.__dataclass_fields__})
    model = LaserDetector(
        in_channels=cfg.in_channels,
        decoder_interpolation=cfg.decoder_interpolation,
    ).to(ddp.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if ddp.is_main:
        logging.info(
            "Loaded checkpoint (epoch=%d, train_loss=%.4f, in_channels=%d, bayer=%s)",
            ckpt.get("epoch", -1),
            ckpt.get("metrics", {}).get("train_loss", float("nan")),
            cfg.in_channels, cfg.use_bayer_excess,
        )

    # Parallel Bayer-excess cache loader when the checkpoint is 6-ch.
    bayer_excess_loader = None
    if cfg.use_bayer_excess:
        bayer_cache_dir = args.bayer_excess_cache_dir or Path(
            f"{config.cache_dir}_bayer_excess"
        )
        if ddp.is_main:
            logging.info("bayer_excess cache: %s", bayer_cache_dir)
        bayer_excess_loader = make_cached_image_loader(
            config.image_root, bayer_cache_dir, pipeline="bayer_excess",
        )

    # Override saved-cfg's inference settings with the CLI flags so this
    # script can A/B the same checkpoint with and without the snap/prior.
    cfg.inference_soft_snap = args.soft_snap_inference
    cfg.inference_soft_snap_alpha_max = args.soft_snap_alpha_max
    cfg.inference_rig_prior = args.rig_prior
    cfg.inference_rig_prior_floor = args.rig_prior_floor
    cfg.inference_rig_prior_sigma_x = args.rig_prior_sigma_x
    cfg.inference_rig_prior_sigma_y = args.rig_prior_sigma_y
    cfg.inference_cascade = args.cascade
    cfg.inference_cascade_refine_window = args.cascade_refine_window
    if args.pixel_bias_offset is not None:
        cfg.inference_pixel_bias_offset_x = args.pixel_bias_offset[0]
        cfg.inference_pixel_bias_offset_y = args.pixel_bias_offset[1]
    predictions = _run_val_inference(
        model, records, image_loader, ddp.device, cfg, ddp,
        bayer_excess_loader=bayer_excess_loader,
    )

    if not ddp.is_main:
        shutdown_distributed()
        return 0

    eval_result = evaluate(
        predictions, frames=frames, splits=splits,
        wavelengths=wavelengths, lines=lines,
        split=args.split, presence_threshold=args.presence_threshold,
    )

    print()
    print(f"=== eval results (checkpoint={args.checkpoint.name}, split={args.split}) ===")
    for k in (
        "n_frames", "n_positive", "n_negative",
        "hit_rate_n3", "hit_rate_n4",
        "mean_pixel_error", "median_pixel_error",
        "presence_auroc", "fpr_at_threshold", "recall_at_threshold",
        "fraction_localized",
    ):
        if k in eval_result.metrics:
            print(f"  {k:25s} {eval_result.metrics[k]:.4f}")

    if not args.no_mlflow:
        setup_mlflow(config)
        with mlflow.start_run(run_name=f"phase2_eval_{args.checkpoint.stem}"):
            mlflow.set_tag("phase", "phase2_eval_only")
            mlflow.set_tag("checkpoint", str(args.checkpoint))
            mlflow.set_tag("split", args.split)
            mlflow.log_params({
                "checkpoint": str(args.checkpoint),
                "split": args.split,
                "presence_threshold": args.presence_threshold,
                "ckpt_epoch": ckpt.get("epoch", -1),
            })
            mlflow.log_metrics(eval_result.metrics)

    shutdown_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
