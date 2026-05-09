"""Inference-time sweep over rig-prior (floor, σ_x, σ_y) on a single checkpoint.

Loads the checkpoint + val records once, then re-runs `_run_val_inference`
for each config in the grid and prints a comparison table. Results land in
`<out-dir>/rig_prior_sweep.parquet` for ad-hoc analysis.

Usage:
    uv run torchrun --standalone --nproc_per_node=4 scripts/sweep_rig_prior.py \\
        --checkpoint data/phase2/checkpoints_linear_npy_50e/epoch_006.pt \\
        --image-pipeline linear_npy
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch

from laser_detector.data import build_records, load_orf_flip
from laser_detector.eval import evaluate
from laser_detector.model import LaserDetector
from laser_detector.preprocessing.config import load_config
from laser_detector.preprocessing.image_loader import make_cached_image_loader
from laser_detector.train import (
    TrainConfig,
    _run_val_inference,
    init_distributed,
    shutdown_distributed,
)


@dataclass(frozen=True)
class SweepConfig:
    floor: float
    sigma_x: float | None
    sigma_y: float | None
    label: str


SWEEP_CONFIGS: tuple[SweepConfig, ...] = (
    SweepConfig(1.0, None, None, "pure_bbox"),
    SweepConfig(0.1, 200.0, 300.0, "narrow_low_floor"),
    SweepConfig(0.5, 200.0, 300.0, "narrow_mid_floor"),
    SweepConfig(0.5, 600.0, 900.0, "wide_mid_floor"),
    SweepConfig(0.7, 600.0, 900.0, "wide_high_floor"),
    SweepConfig(0.3, 1000.0, 1500.0, "very_wide_low_floor"),
    SweepConfig(0.5, 1000.0, 1500.0, "very_wide_mid_floor"),
    SweepConfig(0.7, 1000.0, 1500.0, "very_wide_high_floor"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep rig-prior (floor, σ_x, σ_y)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument(
        "--image-pipeline",
        choices=("jpeg", "linear", "linear_npy"),
        default="linear_npy",
    )
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/audit"),
        help="Where to write rig_prior_sweep.parquet.",
    )
    p.add_argument(
        "--soft-snap-inference",
        action="store_true",
        help="Apply DESIGN.md §6.2 soft-snap-to-line at inference time.",
    )
    return p.parse_args(argv)


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
            "Sweep on %d frames across %d dives (ckpt=%s, %d configs)",
            len(records), len(split_dive_ids), args.checkpoint, len(SWEEP_CONFIGS),
        )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = LaserDetector().to(ddp.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    base_cfg = TrainConfig(**{
        k: v for k, v in ckpt["cfg"].items() if k in TrainConfig.__dataclass_fields__
    })
    base_cfg.inference_soft_snap = args.soft_snap_inference
    base_cfg.inference_rig_prior = True

    rows: list[dict] = []
    for sweep in SWEEP_CONFIGS:
        cfg = TrainConfig(**{**base_cfg.__dict__})
        cfg.inference_rig_prior_floor = sweep.floor
        cfg.inference_rig_prior_sigma_x = sweep.sigma_x
        cfg.inference_rig_prior_sigma_y = sweep.sigma_y
        if ddp.is_main:
            logging.info(
                "[%s] floor=%.2f σ=(%s, %s)",
                sweep.label, sweep.floor,
                f"{sweep.sigma_x:.0f}" if sweep.sigma_x is not None else "default",
                f"{sweep.sigma_y:.0f}" if sweep.sigma_y is not None else "default",
            )
        predictions = _run_val_inference(
            model, records, image_loader, ddp.device, cfg, ddp,
        )
        if not ddp.is_main:
            continue
        result = evaluate(
            predictions, frames=frames, splits=splits,
            wavelengths=wavelengths, lines=lines,
            split=args.split,
        )
        rows.append({
            "label": sweep.label,
            "floor": sweep.floor,
            "sigma_x": sweep.sigma_x,
            "sigma_y": sweep.sigma_y,
            "hit_rate_n3": result.metrics.get("hit_rate_n3"),
            "hit_rate_n4": result.metrics.get("hit_rate_n4"),
            "presence_auroc": result.metrics.get("presence_auroc"),
            "fpr_at_threshold": result.metrics.get("fpr_at_threshold"),
            "mean_pixel_error": result.metrics.get("mean_pixel_error"),
            "median_pixel_error": result.metrics.get("median_pixel_error"),
        })

    if not ddp.is_main:
        shutdown_distributed()
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "rig_prior_sweep.parquet"
    df = pl.DataFrame(rows)
    df.write_parquet(out_path)

    print()
    print(f"=== rig-prior sweep on {args.checkpoint.name} ===")
    with pl.Config(tbl_rows=20, tbl_cols=10, fmt_float="full"):
        print(df.sort("hit_rate_n3", descending=True))
    print(f"\nSaved {out_path}")

    shutdown_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
