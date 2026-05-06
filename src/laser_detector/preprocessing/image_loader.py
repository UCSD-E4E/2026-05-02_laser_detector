"""Pluggable image-loading backends with caching.

The fishsense SDK exposes `Image.path` and `Image.checksum` but does not provide
an image-bytes endpoint. How bytes flow into this pipeline depends on the
deployment: a mounted filesystem, an S3 bucket, an internal HTTP service. We
accept any object satisfying the `ImageLoader` protocol.

Decoder choice is also pluggable. ORF (Olympus RAW) files must go through
`fishsense_core.image.raw_image.RawImage` — it applies the project-standard
pipeline (rawpy + auto-gamma + CLAHE) so what this detector sees matches what
every other fishsense model sees. Using a different decoder would silently
shift the input distribution.

Caching: ORF decode is slow (hundreds of ms each). `CachingImageLoader` wraps
any inner loader, writes a JPEG keyed by checksum on first load, and reads
back on every subsequent load. The cache lives under `data/image_cache/` by
default and is gitignored.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Olympus RAW. Add other RAW extensions here if rigs change.
RAW_EXTENSIONS = {".orf"}


class ImageLoader(Protocol):
    """Returns an HxWx3 uint8 BGR (OpenCV convention) array, or None if unavailable."""

    def load(self, image_path: str, checksum: str) -> np.ndarray | None:
        ...


def _decode_with_fishsense_core(path: Path) -> np.ndarray | None:
    """Decode a RAW file with fishsense-core's project-standard pipeline."""
    # Imported lazily so that workflows which never touch RAW don't pay the
    # ~200 MB import cost.
    from fishsense_core.image.raw_image import RawImage  # noqa: PLC0415

    try:
        return RawImage(path).data
    except Exception as exc:
        logger.warning("fishsense-core failed to decode %s: %s", path, exc)
        return None


def _decode_raw_linear(path: Path) -> np.ndarray | None:
    """Decode a RAW file with rawpy directly: linear, 16-bit, no CLAHE.

    Returns uint16 BGR (downstream OpenCV convention). Skips fishsense-core
    on purpose: the project-standard pipeline applies CLAHE, which saturates
    bright laser blobs across all channels and destroys wavelength selectivity
    (see notes/state.md "Audit findings"). For laser detection we want the raw
    sensor data while keeping demosaicing + camera white balance.
    """
    import rawpy  # noqa: PLC0415 — lazy, big native library

    try:
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                output_bps=16,
                gamma=(1, 1),                 # linear (no gamma correction)
                no_auto_bright=True,           # don't auto-stretch the histogram
                use_camera_wb=True,            # apply camera white balance
                output_color=rawpy.ColorSpace.sRGB,
            )
    except Exception as exc:
        logger.warning("rawpy linear decode failed for %s: %s", path, exc)
        return None
    # rawpy returns RGB; downstream expects BGR.
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class LocalFilesystemImageLoader:
    """Loads images from the local filesystem under a configurable root.

    Treats `image.path` as relative to `root`. If `image.path` is absolute,
    `root` is ignored. RAW files (extension in `RAW_EXTENSIONS`) decode through
    fishsense-core; everything else uses `cv2.imread`.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def load(self, image_path: str, checksum: str) -> np.ndarray | None:
        path = Path(image_path)
        if not path.is_absolute():
            path = self.root / path
        if not path.exists():
            return None
        if path.suffix.lower() in RAW_EXTENSIONS:
            return _decode_with_fishsense_core(path)
        return cv2.imread(str(path), cv2.IMREAD_COLOR)


class CachingImageLoader:
    """Decorator that caches decoded images as JPEGs keyed by checksum.

    On `load()`:
        1. Compute a cache path from `checksum` (with 2-level fanout to avoid
           millions of files in one directory).
        2. If the cached JPEG exists, decode and return it.
        3. Otherwise, call the inner loader. If it returns an image, write a
           JPEG to the cache path, then return the image.

    Cache writes are atomic (temp file + rename) so a crashed process can't
    leave a partial JPEG that later loads as garbage.
    """

    def __init__(
        self,
        inner: ImageLoader,
        cache_dir: Path | str,
        jpeg_quality: int = 95,
    ):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.jpeg_quality = int(jpeg_quality)

    def cache_path(self, checksum: str) -> Path:
        """Return the cache file path for a given checksum."""
        if not checksum:
            raise ValueError("checksum must be non-empty")
        # 2-level fanout. Even if the dataset grows 10x, dirs stay small.
        return self.cache_dir / checksum[:2] / checksum[2:4] / f"{checksum}.jpg"

    def load(self, image_path: str, checksum: str) -> np.ndarray | None:
        cache_path = self.cache_path(checksum)
        if cache_path.exists():
            cached = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)
            if cached is not None:
                return cached
            logger.warning(
                "Cache file unreadable, re-decoding: %s", cache_path
            )

        image = self.inner.load(image_path, checksum)
        if image is None:
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            logger.warning("Failed to encode JPEG cache for %s", checksum)
            return image
        # Atomic write: bytes → temp file → rename. Filename suffix is irrelevant
        # since we did the encoding ourselves.
        tmp_path = cache_path.with_name(cache_path.name + ".tmp")
        tmp_path.write_bytes(encoded.tobytes())
        tmp_path.rename(cache_path)
        return image


class LocalFilesystemLinearRawImageLoader:
    """Like LocalFilesystemImageLoader, but uses `_decode_raw_linear` for ORF.

    Returns uint16 BGR for RAW inputs (linear, no CLAHE). Non-RAW files load
    via `cv2.imread(IMREAD_UNCHANGED)` so 16-bit TIFFs/PNGs preserve their bit
    depth.

    This loader DELIBERATELY DEVIATES from the CLAUDE.md guidance to use
    fishsense-core for ORF decode. Other fishsense models should keep using
    the project-standard pipeline; only the laser detector's input pipeline
    skips CLAHE — see notes/state.md.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def load(self, image_path: str, checksum: str) -> np.ndarray | None:
        path = Path(image_path)
        if not path.is_absolute():
            path = self.root / path
        if not path.exists():
            return None
        if path.suffix.lower() in RAW_EXTENSIONS:
            return _decode_raw_linear(path)
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


class CachingLinearImageLoader:
    """Decorator that caches uint16 BGR images as 16-bit PNGs keyed by checksum.

    Mirrors `CachingImageLoader`'s API but the cache format is lossless 16-bit
    PNG (~3-5x compression on linear sensor data, ~10-15 MB per 4K frame).
    """

    def __init__(
        self,
        inner: ImageLoader,
        cache_dir: Path | str,
        png_compression: int = 6,
    ):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not 0 <= int(png_compression) <= 9:
            raise ValueError("png_compression must be in [0, 9]")
        self.png_compression = int(png_compression)

    def cache_path(self, checksum: str) -> Path:
        if not checksum:
            raise ValueError("checksum must be non-empty")
        return self.cache_dir / checksum[:2] / checksum[2:4] / f"{checksum}.png"

    def load(self, image_path: str, checksum: str) -> np.ndarray | None:
        cache_path = self.cache_path(checksum)
        if cache_path.exists():
            cached = cv2.imread(str(cache_path), cv2.IMREAD_UNCHANGED)
            if cached is not None:
                return cached
            logger.warning(
                "Cache file unreadable, re-decoding: %s", cache_path
            )

        image = self.inner.load(image_path, checksum)
        if image is None:
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(
            ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression]
        )
        if not ok:
            logger.warning("Failed to encode PNG cache for %s", checksum)
            return image
        tmp_path = cache_path.with_name(cache_path.name + ".tmp")
        tmp_path.write_bytes(encoded.tobytes())
        tmp_path.rename(cache_path)
        return image


def make_cached_image_loader(
    image_root: Path | str,
    cache_dir: Path | str,
    *,
    pipeline: str = "jpeg",
    jpeg_quality: int = 95,
    png_compression: int = 6,
):
    """Build a checksum-keyed image loader chain.

    `pipeline` selects:
    - "jpeg"   — current default. uint8 BGR via fishsense-core (rawpy + auto-gamma
                 + CLAHE), cached as JPEGs. Matches the rest of the fishsense ecosystem.
    - "linear" — laser-detector-specific. uint16 BGR via rawpy direct (linear,
                 no CLAHE), cached as 16-bit PNGs. See notes/state.md "Audit findings"
                 for why we deviate.
    """
    if pipeline == "linear":
        inner = LocalFilesystemLinearRawImageLoader(image_root)
        return CachingLinearImageLoader(
            inner=inner, cache_dir=cache_dir, png_compression=png_compression,
        )
    if pipeline == "jpeg":
        inner = LocalFilesystemImageLoader(image_root)
        return CachingImageLoader(
            inner=inner, cache_dir=cache_dir, jpeg_quality=jpeg_quality,
        )
    raise ValueError(f"unknown pipeline: {pipeline!r}")


class NullImageLoader:
    """A loader that always returns None.

    Use when running parts of the pipeline that don't need image bytes. Steps
    that do will log a warning and skip those frames.
    """

    def load(self, image_path: str, checksum: str) -> np.ndarray | None:
        return None
