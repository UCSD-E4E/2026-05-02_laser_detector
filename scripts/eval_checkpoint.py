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

from laser_detector.data import build_records
from laser_detector.eval import evaluate
from laser_detector.model import LaserDetector
from laser_detector.preprocessing.config import load_config
from laser_detector.preprocessing.image_loader import (
    CachingImageLoader,
    LocalFilesystemImageLoader,
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
        "--no-mlflow", action="store_true",
        help="Skip MLflow logging; just print metrics.",
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

    inner = LocalFilesystemImageLoader(config.image_root)
    image_loader = CachingImageLoader(
        inner=inner, cache_dir=config.cache_dir, jpeg_quality=config.cache_jpeg_quality,
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
    records = build_records(split_frames, wavelengths, lines)
    if ddp.is_main:
        logging.info(
            "Eval split=%s: %d dives, %d frames, ckpt=%s",
            args.split, len(split_dive_ids), len(records), args.checkpoint,
        )

    # Load checkpoint into a fresh model. The state_dict was saved unwrapped
    # (rank-0 saves `model.module.state_dict()` when DDP-wrapped, per train.py).
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = LaserDetector().to(ddp.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if ddp.is_main:
        logging.info(
            "Loaded checkpoint (epoch=%d, train_loss=%.4f)",
            ckpt.get("epoch", -1), ckpt.get("metrics", {}).get("train_loss", float("nan")),
        )

    cfg = TrainConfig(**{k: v for k, v in ckpt["cfg"].items() if k in TrainConfig.__dataclass_fields__})
    predictions = _run_val_inference(
        model, records, image_loader, ddp.device, cfg, ddp,
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
