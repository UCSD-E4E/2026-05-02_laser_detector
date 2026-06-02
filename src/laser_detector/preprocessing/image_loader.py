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


def _apply_rawpy_flip(arr: np.ndarray, flip: int) -> np.ndarray:
    """Apply rawpy/libraw EXIF rotation flag to a HxW or HxWxC array.

    flip values per libraw: 0 no-op, 3 = 180°, 5 = 90° CCW, 6 = 90° CW.
    `rawpy.postprocess` applies this automatically; our direct mosaic
    extraction must replicate it to keep labels aligned across pipelines.
    """
    if flip == 0:
        return arr
    if flip == 3:
        return np.rot90(arr, k=2)
    if flip == 5:
        return np.rot90(arr, k=1)
    if flip == 6:
        return np.rot90(arr, k=3)
    logger.warning("unknown rawpy flip value %s; not rotating", flip)
    return arr


def _decode_raw_bayer_excess(path: Path) -> np.ndarray | None:
    """Decode an ORF and compute (G_excess, R_excess) channels at full resolution.

    Each photosite in the Bayer mosaic sees only a single wavelength band. A
    green laser saturates the G photosites first while leaving R/B headroom;
    a red laser saturates R first. Computing per-2x2-cell:

        G_avg    = (G1 + G2) / 2
        G_excess = max(0, G_avg - max(R, B))     # "this 2x2 cell is green-bright"
        R_excess = max(0, R     - max(G_avg, B)) # "this 2x2 cell is red-bright"

    gives wavelength-discriminative features that survive demosaicing +
    chromaticity normalization (which both wash this signal out). Black levels
    are subtracted before the comparison.

    Returns uint16 `[H, W, 2]` with channels (G_excess, R_excess) at the
    original frame resolution. The half-res excess values are tiled 2x2 to
    full res for direct compatibility with the existing tile pipeline.
    """
    import rawpy  # noqa: PLC0415

    try:
        with rawpy.imread(str(path)) as raw:
            mosaic = raw.raw_image_visible.copy()
            pattern = np.asarray(raw.raw_pattern)  # 2x2 indices
            color_desc = raw.color_desc.decode()    # e.g. "RGBG"
            black = list(raw.black_level_per_channel)  # 4 ints
            flip = int(raw.sizes.flip)
    except Exception as exc:
        logger.warning("rawpy bayer decode failed for %s: %s", path, exc)
        return None

    if mosaic.ndim != 2:
        logger.warning("unexpected raw_image_visible shape %s for %s", mosaic.shape, path)
        return None
    H, W = mosaic.shape

    # Map each 2x2 offset to its color name ('R'/'G'/'B').
    color_at: dict[tuple[int, int], str] = {}
    for di in range(2):
        for dj in range(2):
            color_at[(di, dj)] = color_desc[int(pattern[di, dj])]

    r_offsets = [(di, dj) for (di, dj), c in color_at.items() if c == "R"]
    g_offsets = [(di, dj) for (di, dj), c in color_at.items() if c == "G"]
    b_offsets = [(di, dj) for (di, dj), c in color_at.items() if c == "B"]
    if len(r_offsets) != 1 or len(g_offsets) != 2 or len(b_offsets) != 1:
        logger.warning(
            "unexpected Bayer pattern (R=%s G=%s B=%s) for %s",
            len(r_offsets), len(g_offsets), len(b_offsets), path,
        )
        return None

    def _slice_at(off: tuple[int, int]) -> np.ndarray:
        return mosaic[off[0]::2, off[1]::2]

    # Black-level-correct each photosite plane with its per-channel offset.
    # rawpy's black_level_per_channel ordering matches the pattern indices.
    def _bl(off: tuple[int, int]) -> int:
        idx = int(pattern[off[0], off[1]])
        return int(black[idx]) if idx < len(black) else 0

    r_arr = np.maximum(_slice_at(r_offsets[0]).astype(np.int32) - _bl(r_offsets[0]), 0)
    b_arr = np.maximum(_slice_at(b_offsets[0]).astype(np.int32) - _bl(b_offsets[0]), 0)
    g1 = np.maximum(_slice_at(g_offsets[0]).astype(np.int32) - _bl(g_offsets[0]), 0)
    g2 = np.maximum(_slice_at(g_offsets[1]).astype(np.int32) - _bl(g_offsets[1]), 0)
    g_avg = (g1 + g2) // 2

    rb_max = np.maximum(r_arr, b_arr)
    gb_max = np.maximum(g_avg, b_arr)
    g_excess = np.maximum(g_avg - rb_max, 0).astype(np.uint16)
    r_excess = np.maximum(r_arr - gb_max, 0).astype(np.uint16)

    # Half-resolution → upsample 2× to match demosaiced cache shape, using
    # centered bilinear interpolation. Each supercell at half-res index (i, j)
    # represents the 2×2 photosite region whose centroid sits at full-res
    # (2i+0.5, 2j+0.5); cv2.resize INTER_LINEAR places the supercell value at
    # that centroid (its pixel-coord convention is `src = (dst+0.5) * scale −
    # 0.5`, centroid-aligned for scale=0.5). The previous `np.repeat(..., 2)`
    # variant placed the supercell value at the block top-left (2i, 2j)
    # instead — a +0.5 px shift — and caused a constant ~(−1.1, −2.1) px
    # label-bias on the 6-ch sensor model (see DESIGN.md §10 Risks).
    half = np.stack([g_excess, r_excess], axis=2)  # [H/2, W/2, 2]
    half_h, half_w = half.shape[:2]
    full = cv2.resize(
        half, (half_w * 2, half_h * 2), interpolation=cv2.INTER_LINEAR
    )

    # Trim/pad to the demosaiced visible size if odd-by-one.
    full = full[:H, :W]

    # Both caches now operate in sensor coordinates (no rotation), so we don't
    # apply the EXIF flip here. Labels get inverse-rotated into sensor coords
    # via the per-frame `flip` lookup in build_records. `flip` is captured for
    # callers that want to record per-frame rotation alongside the cache.
    _ = flip  # explicit no-op to keep callers stable
    return full


def _decode_raw_linear(path: Path) -> np.ndarray | None:
    """Decode a RAW file with rawpy directly: linear, 16-bit, no CLAHE.

    Returns uint16 BGR (downstream OpenCV convention) in **sensor coordinates**
    (no EXIF rotation). The rig is a body-frame property — the laser sits at a
    fixed location in the camera's coordinate frame. Operating in sensor
    coordinates lets a single rig prior bbox/Gaussian apply uniformly across
    all frames regardless of camera-body rotation.

    Skips fishsense-core on purpose: the project-standard pipeline applies
    CLAHE, which saturates bright laser blobs across all channels and destroys
    wavelength selectivity. For laser detection we want raw sensor data with
    demosaicing + camera white balance, but no rotation, no CLAHE.
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
                user_flip=0,                   # disable EXIF rotation (sensor coords)
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


class LocalFilesystemBayerExcessLoader:
    """Like LocalFilesystemImageLoader but computes Bayer-derived (G_excess,
    R_excess) channels via `_decode_raw_bayer_excess`. Returns uint16 [H, W, 2].

    Used as the inner of a CachingLinearNpyImageLoader. Non-RAW files load
    nothing (returns None) — the Bayer features only make sense for RAW.
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
            return _decode_raw_bayer_excess(path)
        return None


class CachingLinearNpyImageLoader:
    """Decorator that caches uint16 BGR images as uncompressed `.npy` files.

    Tradeoff vs `CachingLinearImageLoader` (16-bit PNG):
    - Files are larger (~72 MB uncompressed vs ~60 MB PNG) — modest.
    - Decode is essentially memcpy via np.load → 5–10x faster than PNG decode,
      which makes training I/O-bound rather than CPU-bound on the dataloader.

    `mmap_mode='r'` is used at read time so only the actual bytes that downstream
    code touches get paged in. With cropping in the dataset, that's typically
    one tile (~6 MB) instead of the full 72 MB image.
    """

    def __init__(self, inner: ImageLoader, cache_dir: Path | str):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, checksum: str) -> Path:
        if not checksum:
            raise ValueError("checksum must be non-empty")
        return self.cache_dir / checksum[:2] / checksum[2:4] / f"{checksum}.npy"

    def load(self, image_path: str, checksum: str) -> np.ndarray | None:
        cache_path = self.cache_path(checksum)
        if cache_path.exists():
            try:
                # mmap_mode='r' avoids loading the whole array into RAM up front.
                # Downstream code that crops a tile pages in only those bytes.
                cached = np.load(cache_path, mmap_mode="r")
                # Return a writable copy so callers can crop / augment freely
                # without backing-file constraints. Even with the copy, this is
                # significantly faster than PNG decode.
                return np.array(cached)
            except Exception as exc:
                logger.warning(
                    "Cache file unreadable, re-decoding: %s (%s)", cache_path, exc
                )

        image = self.inner.load(image_path, checksum)
        if image is None:
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(cache_path.name + ".tmp")
        # np.save would append ".npy" if given a Path; pass an open handle to
        # disable that behavior and atomic-rename ourselves.
        with open(tmp_path, "wb") as f:
            np.save(f, image, allow_pickle=False)
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
    - "jpeg"       — current default. uint8 BGR via fishsense-core (rawpy + auto-gamma
                     + CLAHE), cached as JPEGs. Matches the rest of the fishsense ecosystem.
    - "linear"     — laser-detector-specific. uint16 BGR via rawpy direct (linear,
                     no CLAHE), cached as 16-bit PNGs. See notes/state.md "Audit findings".
    - "linear_npy" — same source as "linear" but cached as uncompressed `.npy`. Larger
                     on disk (~72 MB/file vs 60 MB) but ~10x faster decode at training
                     time, which keeps the dataloader off the bottleneck.
    """
    if pipeline == "linear_npy":
        inner = LocalFilesystemLinearRawImageLoader(image_root)
        return CachingLinearNpyImageLoader(inner=inner, cache_dir=cache_dir)
    if pipeline == "bayer_excess":
        inner = LocalFilesystemBayerExcessLoader(image_root)
        return CachingLinearNpyImageLoader(inner=inner, cache_dir=cache_dir)
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
