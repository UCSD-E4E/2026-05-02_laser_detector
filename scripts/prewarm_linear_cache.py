"""Pre-warm the *linear* (16-bit, no-CLAHE) image cache for one or more splits.

Companion to `prewarm_image_cache.py`. Same parallel pattern but uses the
rawpy-direct linear loader (`LocalFilesystemLinearRawImageLoader`) wrapped
in `CachingLinearImageLoader`, which writes 16-bit PNGs.

The linear cache lives at `<cache_dir>_linear/` by default so the existing
JPEG cache is untouched. Old training scripts continue to work; new linear
runs point at this cache via `--cache-dir`.

Usage:
    uv run python scripts/prewarm_linear_cache.py --splits train val test
    uv run python scripts/prewarm_linear_cache.py --splits val --max-frames 50
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
    CachingLinearImageLoader,
    LocalFilesystemLinearRawImageLoader,
)

_WORKER_LOADER: CachingLinearImageLoader | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-warm the linear (16-bit) image cache")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Cap on frames per run for smoke tests. 0 = no cap.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override cache dir. Default: <config.cache_dir>_linear",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=6,
        help="PNG compression level [0-9]; higher = smaller files, slower writes.",
    )
    return parser.parse_args(argv)


def _init_worker(loader: CachingLinearImageLoader) -> None:
    global _WORKER_LOADER
    _WORKER_LOADER = loader


def _warm_one(row: dict) -> str:
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

    cache_dir = args.cache_dir or Path(f"{config.cache_dir}_linear")
    inner = LocalFilesystemLinearRawImageLoader(config.image_root)
    loader = CachingLinearImageLoader(
        inner=inner,
        cache_dir=cache_dir,
        png_compression=args.png_compression,
    )

    frames = pl.read_parquet(config.data_dir / "frames.parquet")
    splits = pl.read_parquet(config.data_dir / "dive_splits.parquet")
    if "superseded" in frames.columns:
        frames = frames.filter(~pl.col("superseded"))

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
        "Pre-warming linear cache: %d frames, %d dives, splits=%s, cache=%s, %d workers",
        len(rows),
        len(target_dive_ids),
        list(args.splits),
        cache_dir,
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
            desc="prewarm-linear",
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
