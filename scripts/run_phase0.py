"""CLI entry point for the Phase 0 preprocessing pipeline.

Usage:
    uv run python scripts/run_phase0.py
    uv run python scripts/run_phase0.py --force
    uv run python scripts/run_phase0.py --image-root /path/to/images
    uv run python scripts/run_phase0.py --no-cache  # skip JPEG cache layer

Setup before first run:
    - Copy `.secrets.toml.example` → `.secrets.toml`, fill in api credentials.
    - Copy `settings.local.toml.example` → `settings.local.toml`, set
      `images.root` to your local mount. (Or pass --image-root every run.)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from laser_detector.preprocessing.config import load_config
from laser_detector.preprocessing.image_loader import (
    CachingImageLoader,
    LocalFilesystemImageLoader,
    NullImageLoader,
)
from laser_detector.preprocessing.pipeline import run_phase0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 0 preprocessing")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute every Phase 0 step even if cached parquet files exist.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Filesystem root under which Image.path resolves. Overrides "
            "`images.root` from settings.local.toml. Required (here or in "
            "settings) for wavelength clustering and laser-size audit. ORF "
            "files decode via fishsense-core."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the JPEG cache layer; every load re-decodes from source.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)
    config = load_config()

    # CLI overrides config; either may be None.
    image_root = args.image_root if args.image_root is not None else config.image_root

    if image_root is None:
        image_loader = NullImageLoader()
        logging.warning(
            "No image root configured (set images.root in settings.local.toml "
            "or pass --image-root). Wavelength clustering and laser-size audit "
            "will produce empty results; the frame table, line fits, and "
            "splits still run."
        )
    else:
        inner = LocalFilesystemImageLoader(image_root)
        if args.no_cache:
            image_loader = inner
            logging.info("Loading images from %s (cache disabled)", image_root)
        else:
            image_loader = CachingImageLoader(
                inner=inner,
                cache_dir=config.cache_dir,
                jpeg_quality=config.cache_jpeg_quality,
            )
            logging.info(
                "Loading images from %s; JPEG cache at %s (quality=%d)",
                image_root,
                config.cache_dir,
                config.cache_jpeg_quality,
            )

    artifacts = run_phase0(config, image_loader=image_loader, force=args.force)
    print()
    print("Phase 0 artifacts written to:", config.data_dir)
    for name, df in artifacts.items():
        print(f"  {name:20s} rows={df.height}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
