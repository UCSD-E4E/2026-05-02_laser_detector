"""CLI to run the Phase 1 classical-CV baseline + log results to MLflow.

Loads the Phase 0 artifacts, restricts to the requested split, runs the
baseline detector on every frame in the split (parallelized), writes a
predictions parquet under `data/phase1/baseline/predictions_<split>.parquet`,
evaluates against ground truth via `eval.evaluate`, and logs metrics +
artifacts to MLflow.

Usage:
    uv run python scripts/run_baseline.py --split val
    uv run python scripts/run_baseline.py --split test
    uv run python scripts/run_baseline.py --split val --max-dives 5  # smoke

Env-var overrides for ad-hoc threshold tuning (default values match the
DESIGN.md/audit-derived constants in `baseline.py`):
    LASER_BASELINE__SAT_MIN, LASER_BASELINE__VAL_MIN, etc.

(Threshold sweeps are Phase 4. For now the baseline is a single fixed
config — establishing the floor that a learned model has to beat.)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import polars as pl

from laser_detector import baseline
from laser_detector.eval import evaluate
from laser_detector.preprocessing.config import load_config
from laser_detector.preprocessing.image_loader import (
    CachingImageLoader,
    LocalFilesystemImageLoader,
)
from laser_detector.tracking import setup_mlflow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 baseline detector + eval")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="Which dive split to evaluate on. Default val.",
    )
    parser.add_argument(
        "--max-dives",
        type=int,
        default=0,
        help="Cap on dives in the split for smoke runs. 0 = no cap.",
    )
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold above which a frame is called positive.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow setup + logging (just write the parquet and print metrics).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)
    config = load_config()

    if config.image_root is None:
        logging.error(
            "No image root configured. The baseline detector needs image bytes; "
            "set `images.root` in settings.local.toml."
        )
        return 2

    inner = LocalFilesystemImageLoader(config.image_root)
    loader = CachingImageLoader(
        inner=inner,
        cache_dir=config.cache_dir,
        jpeg_quality=config.cache_jpeg_quality,
    )

    frames = pl.read_parquet(config.data_dir / "frames.parquet")
    splits = pl.read_parquet(config.data_dir / "dive_splits.parquet")
    wavelengths = pl.read_parquet(config.data_dir / "dive_wavelengths.parquet")
    lines = pl.read_parquet(config.data_dir / "dive_lines.parquet")

    split_dive_ids = (
        splits.filter(pl.col("split") == args.split)["dive_id"].unique().to_list()
    )
    if args.max_dives > 0:
        split_dive_ids = split_dive_ids[: args.max_dives]
        logging.info("Capped to %d dives for smoke run", args.max_dives)

    eval_frames = frames.filter(pl.col("dive_id").is_in(split_dive_ids))
    logging.info(
        "Running baseline on split=%s: %d dives, %d frames (%d positive)",
        args.split,
        len(split_dive_ids),
        eval_frames.height,
        eval_frames.filter(pl.col("is_positive")).height,
    )

    predictions = baseline.run_baseline(
        eval_frames,
        wavelengths=wavelengths,
        lines=lines,
        loader=loader,
        n_workers=config.image_workers,
    )

    pred_dir = Path("data/phase1/baseline")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"predictions_{args.split}.parquet"
    predictions.write_parquet(pred_path)
    logging.info("Wrote predictions to %s", pred_path)

    res = evaluate(
        predictions,
        frames=frames,
        splits=splits,
        wavelengths=wavelengths,
        lines=lines,
        split=args.split,
        presence_threshold=args.presence_threshold,
    )

    print()
    print(f"=== eval results (split={args.split}) ===")
    for k in (
        "n_frames",
        "n_positive",
        "n_negative",
        "hit_rate_n3",
        "hit_rate_n4",
        "mean_pixel_error",
        "median_pixel_error",
        "presence_auroc",
        "fpr_at_threshold",
        "recall_at_threshold",
        "fraction_localized",
    ):
        if k in res.metrics:
            print(f"  {k:25s} {res.metrics[k]:.4f}")

    if args.no_mlflow:
        return 0

    setup_mlflow(config)
    with mlflow.start_run(run_name=f"phase1_baseline_{args.split}") as run:
        mlflow.set_tag("phase", "phase1_baseline")
        mlflow.set_tag("split", args.split)
        mlflow.set_tag("detector", "classical_cv_color_line")
        mlflow.log_params(
            {
                "presence_threshold": args.presence_threshold,
                "sat_min": baseline.SAT_MIN,
                "val_min": baseline.VAL_MIN,
                "min_blob_area": baseline.MIN_BLOB_AREA,
                "max_blob_area": baseline.MAX_BLOB_AREA,
                "line_proximity_sigma_px": baseline.LINE_PROXIMITY_SIGMA_PX,
                "n_dives": len(split_dive_ids),
                "max_dives_cap": args.max_dives,
            }
        )
        mlflow.log_metrics(res.metrics)
        mlflow.log_artifact(str(pred_path))
        logging.info(
            "Logged run %s to %s/#/experiments/%s/runs/%s",
            run.info.run_id,
            config.mlflow_tracking_uri,
            run.info.experiment_id,
            run.info.run_id,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
