"""Extract per-frame EXIF flip values from ORF files into a parquet sidecar.

The cache pipelines now produce images in sensor coordinates (no EXIF
rotation). Labels in `frames.parquet` were collected against the rotated
(world-coordinate) versions, so we need each frame's flip to inverse-rotate
its label into sensor space at training time.

This script reads each ORF's `raw.sizes.flip` (EXIF orientation) — a fast
metadata-only lookup, no demosaic. Output: `data/orf_flip.parquet` with
columns (image_checksum, flip).

Usage:
    uv run python scripts/extract_orf_flip.py
    uv run python scripts/extract_orf_flip.py --workers 8
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ORF EXIF flip per frame")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output parquet. Default: <data_dir>/orf_flip.parquet")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args(argv)


_WORKER_ROOT: Path | None = None


def _init_worker(image_root_str: str) -> None:
    global _WORKER_ROOT
    _WORKER_ROOT = Path(image_root_str)


def _read_flip(row: dict) -> tuple[str, int | None]:
    """Return (image_checksum, flip) tuple. flip=None on failure."""
    import rawpy  # noqa: PLC0415

    assert _WORKER_ROOT is not None
    path = Path(row["image_path"])
    if not path.is_absolute():
        path = _WORKER_ROOT / path
    cs = row["image_checksum"]
    if not path.exists():
        return cs, None
    try:
        with rawpy.imread(str(path)) as raw:
            return cs, int(raw.sizes.flip)
    except Exception:
        return cs, None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)
    config = load_config()
    if config.image_root is None:
        logging.error("No image root configured. Set `images.root` in settings.local.toml.")
        return 2

    out_path = args.out or (config.data_dir / "orf_flip.parquet")

    frames = pl.read_parquet(config.data_dir / "frames.parquet")
    # One row per unique checksum (multiple frames can share a file? unlikely
    # but defensive). Sort for stable iteration.
    unique = frames.unique(subset=["image_checksum"]).sort("image_checksum")
    if args.max_frames > 0:
        unique = unique.head(args.max_frames)
    rows = list(unique.iter_rows(named=True))

    logging.info(
        "Extracting EXIF flip from %d ORFs (workers=%d) → %s",
        len(rows), args.workers, out_path,
    )
    results: list[tuple[str, int | None]] = []
    start = time.monotonic()
    mp_context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp_context,
        initializer=_init_worker, initargs=(str(config.image_root),),
    ) as ex:
        for r in tqdm(
            ex.map(_read_flip, rows, chunksize=8),
            total=len(rows), desc="extract-flip",
        ):
            results.append(r)
    elapsed = time.monotonic() - start

    df = pl.DataFrame(
        {
            "image_checksum": [r[0] for r in results],
            "flip": [r[1] for r in results],
        }
    )
    n_failed = df.filter(pl.col("flip").is_null()).height
    df.write_parquet(out_path)
    logging.info(
        "Done in %.1fs: %d rows, %d unreadable. Wrote %s",
        elapsed, df.height, n_failed, out_path,
    )
    print(df["flip"].value_counts().sort("flip"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
