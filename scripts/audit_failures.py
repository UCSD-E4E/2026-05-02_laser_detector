"""Per-dive failure audit on a checkpoint.

Two modes:

1. Inference + audit (requires torchrun, runs the same tiled val pass as
   `eval_checkpoint.py`, then aggregates):

       uv run torchrun --standalone --nproc_per_node=4 scripts/audit_failures.py \
           --checkpoint data/phase2/checkpoints_bce_clean_50e/epoch_007.pt \
           --soft-snap-inference

2. Audit-only on the saved per-frame parquet (no GPU needed):

       uv run python scripts/audit_failures.py --from-cache

Outputs (under `data/audit/<checkpoint-stem>/`):

- `predictions_with_meta.parquet` — joined per-frame table from `evaluate()`
- `per_dive_metrics.parquet`      — per-dive aggregates, sorted worst-first
- `wavelength_x_lineq.parquet`    — wavelength × line-confidence-quartile crosstab
- `plots/<dive_id>.png`           — ground-truth (red) + predictions (cyan)
                                    overlaid on a representative frame, for the
                                    N worst dives.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from laser_detector.data import build_records
from laser_detector.eval import evaluate
from laser_detector.model import LaserDetector
from laser_detector.preprocessing.config import load_config
from laser_detector.preprocessing.image_loader import (
    CachingImageLoader,
    LocalFilesystemImageLoader,
    make_cached_image_loader,
)
from laser_detector.train import (
    TrainConfig,
    _run_val_inference,
    init_distributed,
    shutdown_distributed,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-dive failure audit on a checkpoint")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--presence-threshold", type=float, default=0.5)
    p.add_argument("--soft-snap-inference", action="store_true")
    p.add_argument("--soft-snap-alpha-max", type=float, default=0.3)
    p.add_argument(
        "--rig-prior", action="store_true",
        help="Apply static rig-prior bbox+Gaussian mask to heatmap at argmax time.",
    )
    p.add_argument(
        "--rig-prior-floor", type=float, default=None,
        help="Override the Gaussian floor inside the bbox. 1.0 = pure bbox.",
    )
    p.add_argument("--rig-prior-sigma-x", type=float, default=None)
    p.add_argument("--rig-prior-sigma-y", type=float, default=None)
    p.add_argument(
        "--cascade", action="store_true",
        help="Use predict_frame_with_cascade refinement at inference.",
    )
    p.add_argument("--cascade-refine-window", type=int, default=None)
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Default: data/audit/<checkpoint-stem>",
    )
    p.add_argument("--n-worst-dives", type=int, default=8)
    p.add_argument(
        "--plot-all-dives", action="store_true",
        help="Plot every dive in the split, not just the top --n-worst-dives.",
    )
    p.add_argument(
        "--from-cache", action="store_true",
        help="Load predictions_with_meta.parquet from --out-dir; skip inference. "
        "Audit-only path that doesn't need a GPU.",
    )
    p.add_argument(
        "--image-pipeline",
        choices=("jpeg", "linear", "linear_npy"),
        default="jpeg",
        help="`jpeg`/`linear`/`linear_npy` — must match the pipeline the checkpoint was trained on.",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Override cache directory.",
    )
    return p.parse_args(argv)


def _add_err_column(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        ((pl.col("pred_x") - pl.col("label_x")) ** 2
         + (pl.col("pred_y") - pl.col("label_y")) ** 2).sqrt().alias("err"),
    )


def per_dive_metrics(per_frame: pl.DataFrame) -> pl.DataFrame:
    pos = _add_err_column(per_frame.filter(pl.col("is_positive")))
    return (
        pos.group_by("dive_id")
        .agg(
            n_positive=pl.len(),
            mean_err=pl.col("err").mean(),
            median_err=pl.col("err").median(),
            hit_rate_n3=(pl.col("err") <= 3.0).cast(pl.Float64).mean(),
            hit_rate_n4=(pl.col("err") <= 4.0).cast(pl.Float64).mean(),
            wavelength=pl.col("wavelength").first(),
            line_confidence=pl.col("line_confidence").first(),
        )
        .sort("mean_err", descending=True)
    )


def wavelength_lineq_crosstab(per_frame: pl.DataFrame) -> pl.DataFrame:
    confidences = per_frame["line_confidence"].drop_nulls().to_numpy()
    if confidences.size < 4:
        return pl.DataFrame()
    edges = np.quantile(confidences, [0.25, 0.5, 0.75])

    binned = per_frame.with_columns(
        pl.when(pl.col("line_confidence").is_null()).then(pl.lit("none"))
        .when(pl.col("line_confidence") <= edges[0]).then(pl.lit("q1"))
        .when(pl.col("line_confidence") <= edges[1]).then(pl.lit("q2"))
        .when(pl.col("line_confidence") <= edges[2]).then(pl.lit("q3"))
        .otherwise(pl.lit("q4"))
        .alias("line_q"),
    )
    pos = _add_err_column(binned.filter(pl.col("is_positive")))
    return (
        pos.group_by(["wavelength", "line_q"])
        .agg(
            n=pl.len(),
            hit_rate_n3=(pl.col("err") <= 3.0).cast(pl.Float64).mean(),
            mean_err=pl.col("err").mean(),
        )
        .sort(["wavelength", "line_q"])
    )


def plot_summary(per_frame: pl.DataFrame, per_dive: pl.DataFrame, out_path: Path) -> None:
    """Two-panel diagnostic chart: per-dive scatter + per-frame error histogram."""
    pos = _add_err_column(per_frame.filter(pl.col("is_positive")))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: median_err vs mean_err per dive (log–log), color=wavelength, size=n_pos.
    for wl, color in (("red", "tab:red"), ("green", "tab:green")):
        sub = per_dive.filter(pl.col("wavelength") == wl)
        if sub.is_empty():
            continue
        ax1.scatter(
            sub["median_err"].to_numpy(),
            sub["mean_err"].to_numpy(),
            s=np.clip(sub["n_positive"].to_numpy() * 1.5, 20, 400),
            c=color, alpha=0.55, edgecolors="black", linewidths=0.5,
            label=f"{wl} (n_dives={sub.height})",
        )
    lim = max(per_dive["median_err"].max(), per_dive["mean_err"].max(), 10) * 1.1
    ax1.plot([0.5, lim], [0.5, lim], "k--", alpha=0.3, label="median = mean")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("median pixel error (per dive)")
    ax1.set_ylabel("mean pixel error (per dive)")
    ax1.set_title("Per-dive: outlier-driven dives sit far above y=x")
    ax1.legend(loc="lower right")
    ax1.grid(True, which="both", alpha=0.2)

    # Panel 2: per-frame error histogram, separated by wavelength (log x).
    bins = np.logspace(-1, np.log10(max(pos["err"].max(), 10)), 40)
    for wl, color in (("red", "tab:red"), ("green", "tab:green")):
        errs = pos.filter(pl.col("wavelength") == wl)["err"].to_numpy()
        if errs.size:
            ax2.hist(errs, bins=bins, alpha=0.55, color=color,
                     label=f"{wl} (n={errs.size})", edgecolor="black", linewidth=0.3)
    ax2.axvline(3, color="gray", linestyle="--", alpha=0.6, label="hit_n3 boundary")
    ax2.axvline(4, color="gray", linestyle=":", alpha=0.6, label="hit_n4 boundary")
    ax2.set_xscale("log")
    ax2.set_xlabel("per-frame pixel error")
    ax2.set_ylabel("frame count")
    ax2.set_title("Per-frame error: green tails much higher than red")
    ax2.legend(loc="upper right")
    ax2.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_worst_dives(
    per_frame: pl.DataFrame,
    per_dive: pl.DataFrame,
    image_loader,
    out_dir: Path,
    n: int,
    plot_all: bool = False,
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    worst = per_dive["dive_id"].to_list() if plot_all else per_dive.head(n)["dive_id"].to_list()
    for dive_id in worst:
        rows = per_frame.filter(
            (pl.col("dive_id") == dive_id) & pl.col("is_positive")
        )
        if rows.height == 0:
            continue

        first = rows.row(0, named=True)
        cache_path = image_loader.cache_path(first["image_checksum"])
        if not cache_path.exists():
            logging.warning("dive %s: no cached JPEG, skipping plot", dive_id)
            continue
        if cache_path.suffix == ".npy":
            img_bgr = np.load(cache_path)
        else:
            img_bgr = cv2.imread(str(cache_path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            logging.warning("dive %s: failed to load cache file, skipping", dive_id)
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Linear cache stores uint16; matplotlib needs [0,1] floats or uint8.
        if img_rgb.dtype == np.uint16:
            img_rgb = (img_rgb.astype(np.float32) / 65535.0).clip(0, 1)

        labels_xy = rows.select("label_x", "label_y").to_numpy()
        preds_xy = rows.select("pred_x", "pred_y").to_numpy()
        valid_pred = ~np.isnan(preds_xy).any(axis=1)

        dive_row = per_dive.filter(pl.col("dive_id") == dive_id).row(0, named=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_rgb)
        ax.scatter(labels_xy[:, 0], labels_xy[:, 1],
                   c="red", s=12, label=f"label (n={rows.height})")
        if valid_pred.any():
            ax.scatter(preds_xy[valid_pred, 0], preds_xy[valid_pred, 1],
                       c="cyan", s=12, label=f"pred (n={int(valid_pred.sum())})")
        line_conf = dive_row["line_confidence"]
        line_conf_str = f"{line_conf:.1f}" if line_conf is not None else "n/a"
        ax.set_title(
            f"dive {dive_id} ({dive_row['wavelength']}, line_conf={line_conf_str}) — "
            f"mean_err={dive_row['mean_err']:.0f}px hit_n3={dive_row['hit_rate_n3']:.2f}"
        )
        ax.legend(loc="upper right")
        ax.axis("off")

        out_path = plot_dir / f"{dive_id}.png"
        fig.savefig(out_path, dpi=72, bbox_inches="tight")
        plt.close(fig)
        logging.info("wrote %s", out_path)


def run_inference(
    args: argparse.Namespace, config, frames, splits, wavelengths, lines,
) -> pl.DataFrame | None:
    """rank0 returns the per_frame DataFrame; other ranks return None."""
    ddp = init_distributed()
    logging.basicConfig(
        level=logging.INFO if ddp.is_main else logging.WARNING,
        format=f"%(asctime)s [r{ddp.rank}] [%(levelname)s] %(name)s: %(message)s",
    )

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

    split_dive_ids = (
        splits.filter(pl.col("split") == args.split)["dive_id"].unique().to_list()
    )
    split_frames = frames.filter(pl.col("dive_id").is_in(split_dive_ids))
    records = build_records(split_frames, wavelengths, lines)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = LaserDetector().to(ddp.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cfg = TrainConfig(**{
        k: v for k, v in ckpt["cfg"].items() if k in TrainConfig.__dataclass_fields__
    })
    cfg.inference_soft_snap = args.soft_snap_inference
    cfg.inference_soft_snap_alpha_max = args.soft_snap_alpha_max
    cfg.inference_rig_prior = args.rig_prior
    cfg.inference_rig_prior_floor = args.rig_prior_floor
    cfg.inference_rig_prior_sigma_x = args.rig_prior_sigma_x
    cfg.inference_rig_prior_sigma_y = args.rig_prior_sigma_y
    cfg.inference_cascade = args.cascade
    cfg.inference_cascade_refine_window = args.cascade_refine_window

    predictions = _run_val_inference(
        model, records, image_loader, ddp.device, cfg, ddp,
    )

    if not ddp.is_main:
        shutdown_distributed()
        return None

    eval_result = evaluate(
        predictions, frames=frames, splits=splits,
        wavelengths=wavelengths, lines=lines,
        split=args.split, presence_threshold=args.presence_threshold,
    )
    shutdown_distributed()
    return eval_result.per_frame


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out_dir is None:
        if args.checkpoint is None and not args.from_cache:
            print("--checkpoint required unless --from-cache", file=sys.stderr)
            return 2
        if args.checkpoint is not None:
            args.out_dir = Path("data/audit") / args.checkpoint.stem
        else:
            print("--out-dir required when using --from-cache", file=sys.stderr)
            return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "predictions_with_meta.parquet"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config()
    frames = pl.read_parquet(config.data_dir / "frames.parquet")
    splits = pl.read_parquet(config.data_dir / "dive_splits.parquet")
    wavelengths = pl.read_parquet(config.data_dir / "dive_wavelengths.parquet")
    lines = pl.read_parquet(config.data_dir / "dive_lines.parquet")
    if "superseded" in frames.columns:
        frames = frames.filter(~pl.col("superseded"))

    if args.from_cache:
        if not cache_path.exists():
            print(f"ERROR: --from-cache but {cache_path} does not exist", file=sys.stderr)
            return 1
        per_frame = pl.read_parquet(cache_path)
        logging.info("Loaded %d rows from %s", per_frame.height, cache_path)
    else:
        per_frame = run_inference(args, config, frames, splits, wavelengths, lines)
        if per_frame is None:
            return 0  # non-rank-0
        per_frame.write_parquet(cache_path)
        logging.info("Saved %d rows to %s", per_frame.height, cache_path)

    # Per-dive metrics
    per_dive = per_dive_metrics(per_frame)
    per_dive.write_parquet(args.out_dir / "per_dive_metrics.parquet")
    print()
    print(f"=== {args.n_worst_dives} worst dives by mean_pixel_error ===")
    print(per_dive.head(args.n_worst_dives))

    # Wavelength × line_q crosstab
    crosstab = wavelength_lineq_crosstab(per_frame)
    if not crosstab.is_empty():
        crosstab.write_parquet(args.out_dir / "wavelength_x_lineq.parquet")
        print()
        print("=== wavelength × line_confidence quartile ===")
        with pl.Config(tbl_rows=20, tbl_cols=10):
            print(crosstab)

    # Plots — build a loader matching the pipeline used at inference.
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
    plot_worst_dives(
        per_frame, per_dive, image_loader,
        args.out_dir, n=args.n_worst_dives,
        plot_all=args.plot_all_dives,
    )
    plot_summary(per_frame, per_dive, args.out_dir / "summary.png")
    logging.info("wrote %s", args.out_dir / "summary.png")

    print()
    print(f"Outputs in {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
