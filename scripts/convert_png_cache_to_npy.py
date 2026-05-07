"""Convert the existing 16-bit PNG linear cache to uncompressed `.npy`.

PNG decode at 4K-16-bit costs ~1-2 s/frame on Python; numpy `.npy` decode is
essentially memcpy. Switching the on-disk format trades modest disk overhead
(~20% larger files) for ~5-10x faster dataloader throughput during training.

This is a local-only operation — no NAS reads. Reads PNGs from the existing
`<cache>` directory, writes `.npy` files to `<out_dir>` with the same
checksum-based fanout layout. Safe to interrupt and restart (re-skips
existing outputs).

Usage:
    uv run python scripts/convert_png_cache_to_npy.py
    uv run python scripts/convert_png_cache_to_npy.py --in-dir data/image_cache_linear --out-dir data/image_cache_linear_npy --workers 16
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from laser_detector.preprocessing.config import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PNG linear cache → numpy .npy for faster decode"
    )
    parser.add_argument(
        "--in-dir", type=Path, default=None,
        help="Source PNG cache. Default: <config.cache_dir>_linear",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Destination NPY cache. Default: <config.cache_dir>_linear_npy",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-files", type=int, default=0)
    return parser.parse_args(argv)


def _convert_one(args: tuple[Path, Path]) -> str:
    in_path, out_path = args
    if out_path.exists():
        return "skip"
    img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return "fail"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, img, allow_pickle=False)
    tmp.rename(out_path)
    return "done"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)
    config = load_config()

    in_dir = args.in_dir or Path(f"{config.cache_dir}_linear")
    out_dir = args.out_dir or Path(f"{config.cache_dir}_linear_npy")
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(in_dir.rglob("*.png"))
    if args.max_files > 0:
        pngs = pngs[: args.max_files]
    work = []
    for p in pngs:
        rel = p.relative_to(in_dir)
        out_path = (out_dir / rel).with_suffix(".npy")
        work.append((p, out_path))

    logging.info(
        "Converting %d PNG → NPY  (in=%s out=%s workers=%d)",
        len(work), in_dir, out_dir, args.workers,
    )

    counts: dict[str, int] = {"done": 0, "skip": 0, "fail": 0}
    start = time.monotonic()
    mp_context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp_context,
    ) as ex:
        for status in tqdm(
            ex.map(_convert_one, work, chunksize=8),
            total=len(work), desc="convert",
        ):
            counts[status] += 1

    elapsed = time.monotonic() - start
    logging.info(
        "Done in %.1fs: done=%d skip=%d fail=%d (%.1f frames/s)",
        elapsed, counts["done"], counts["skip"], counts["fail"],
        len(work) / elapsed if elapsed > 0 else 0.0,
    )
    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
