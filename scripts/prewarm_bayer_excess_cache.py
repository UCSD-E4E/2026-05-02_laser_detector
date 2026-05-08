"""Pre-warm the Bayer-excess cache: per-frame (G_excess, R_excess) channels.

Reads ORF mosaics directly via rawpy, computes wavelength-discriminative
features at the photosite level (see `_decode_raw_bayer_excess`), and writes
uncompressed .npy files keyed by checksum. The cache lives at
`<config.cache_dir>_bayer_excess/` by default and is meant to be loaded
*alongside* the existing linear_npy cache, not as a replacement.

Usage:
    uv run python scripts/prewarm_bayer_excess_cache.py --splits train val test
    uv run python scripts/prewarm_bayer_excess_cache.py --splits val --max-frames 50
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
    CachingLinearNpyImageLoader,
    LocalFilesystemBayerExcessLoader,
)

_WORKER_LOADER: CachingLinearNpyImageLoader | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-warm the Bayer-excess cache")
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
        help="Override cache dir. Default: <config.cache_dir>_bayer_excess",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Worker count. Lower = gentler on system; we hit thrashing at "
        "16 with the data processor running so 4 is the safe default.",
    )
    return parser.parse_args(argv)


def _init_worker(loader: CachingLinearNpyImageLoader) -> None:
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

    cache_dir = args.cache_dir or Path(f"{config.cache_dir}_bayer_excess")
    inner = LocalFilesystemBayerExcessLoader(config.image_root)
    loader = CachingLinearNpyImageLoader(inner=inner, cache_dir=cache_dir)

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
        "Pre-warming Bayer-excess cache: %d frames, %d dives, splits=%s, "
        "cache=%s, %d workers",
        len(rows), len(target_dive_ids), list(args.splits),
        cache_dir, args.workers,
    )

    counts: dict[str, int] = {"warm": 0, "decoded": 0, "failed": 0}
    start = time.monotonic()
    mp_context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp_context,
        initializer=_init_worker, initargs=(loader,),
    ) as ex:
        for status in tqdm(
            ex.map(_warm_one, rows, chunksize=4),
            total=len(rows), desc="prewarm-bayer",
        ):
            counts[status] += 1

    elapsed = time.monotonic() - start
    logging.info(
        "Done in %.1fs: warm=%d decoded=%d failed=%d (%.2f frames/s overall)",
        elapsed, counts["warm"], counts["decoded"], counts["failed"],
        len(rows) / elapsed if elapsed > 0 else 0.0,
    )
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
