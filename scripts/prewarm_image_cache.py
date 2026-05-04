"""Pre-warm the image cache for one or more dive splits.

ORF decode is the bottleneck for any image-bound job (Phase 1 baseline run,
Phase 2 training). Run this once per split so subsequent jobs read the cached
JPEGs instead of re-decoding 25 MB ORFs.

Already-cached frames are detected by `loader.cache_path(checksum).exists()`
and skipped without decoding — re-running this script after an interruption
only pays for the frames that didn't finish.

Usage:
    uv run python scripts/prewarm_image_cache.py --splits train test
    uv run python scripts/prewarm_image_cache.py --splits train --max-frames 100
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm

from laser_detector.preprocessing.config import load_config
from laser_detector.preprocessing.image_loader import (
    CachingImageLoader,
    LocalFilesystemImageLoader,
)

_WORKER_LOADER: CachingImageLoader | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-warm the image cache")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "test"),
        help="Which dive splits to warm. Default: train + test (val is warm from Phase 1).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Cap on frames per run for smoke tests. 0 = no cap.",
    )
    return parser.parse_args(argv)


def _init_worker(loader: CachingImageLoader) -> None:
    global _WORKER_LOADER
    _WORKER_LOADER = loader


def _warm_one(row: dict) -> str:
    """Warm a single frame. Returns 'warm' / 'decoded' / 'failed'."""
    assert _WORKER_LOADER is not None
    loader = _WORKER_LOADER
    if loader.cache_path(row["image_checksum"]).exists():
        return "warm"
    image = loader.load(row["image_path"], row["image_checksum"])
    return "decoded" if image is not None else "failed"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)
    config = load_config()

    if config.image_root is None:
        logging.error(
            "No image root configured. Set `images.root` in settings.local.toml."
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

    target_dive_ids = (
        splits.filter(pl.col("split").is_in(list(args.splits)))["dive_id"]
        .unique()
        .to_list()
    )
    target_frames = frames.filter(pl.col("dive_id").is_in(target_dive_ids))
    if args.max_frames > 0:
        target_frames = target_frames.head(args.max_frames)

    rows = list(target_frames.iter_rows(named=True))
    logging.info(
        "Pre-warming %d frames across splits=%s (%d dives), %d workers",
        len(rows),
        list(args.splits),
        len(target_dive_ids),
        config.image_workers,
    )

    counts: dict[str, int] = {"warm": 0, "decoded": 0, "failed": 0}
    start = time.monotonic()
    mp_context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(
        max_workers=config.image_workers,
        mp_context=mp_context,
        initializer=_init_worker,
        initargs=(loader,),
    ) as ex:
        for status in tqdm(
            ex.map(_warm_one, rows, chunksize=4),
            total=len(rows),
            desc="prewarm",
        ):
            counts[status] += 1

    elapsed = time.monotonic() - start
    logging.info(
        "Done in %.1fs: warm=%d decoded=%d failed=%d (%.2f frames/s overall)",
        elapsed,
        counts["warm"],
        counts["decoded"],
        counts["failed"],
        len(rows) / elapsed if elapsed > 0 else 0.0,
    )
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
