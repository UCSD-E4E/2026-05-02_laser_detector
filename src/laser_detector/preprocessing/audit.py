"""Laser-size audit: measure on-screen laser blob diameter on a sample of positives.

Confirms (or refutes) the 3–8 px assumption that drives §4.1's tiling decision.
For each sampled positive frame, segment the laser blob locally around the
labeled pixel and compute its pixel diameter.

Outputs per-frame diameter measurements as a parquet table. The Phase 0
deliverable summarizes the distribution (min, p10, median, p90, max).
"""

from __future__ import annotations

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import polars as pl
from tqdm import tqdm

from laser_detector.preprocessing.image_loader import ImageLoader

logger = logging.getLogger(__name__)


PATCH_HALF_SIZE = 31  # 63x63 window around the label is enough at 4K
DEFAULT_N_WORKERS = 32  # ORF decode is CPU-bound; processes get their own GIL.
# A real laser blob on a 4K frame is at most ~20 px in diameter (~314 pixels =
# 7.9% of a 63×63 patch). Anything occupying >25% of the patch area is almost
# certainly water/tile/reflection segmentation, not a laser. Reject those
# rather than returning a contaminated diameter.
MAX_BLOB_AREA_FRAC = 0.25

_WORKER_LOADER: ImageLoader | None = None
_WORKER_WAVELENGTH_BY_DIVE: dict[int, str | None] = {}


def _init_worker(
    loader: ImageLoader, wavelength_by_dive: dict[int, str | None]
) -> None:
    global _WORKER_LOADER, _WORKER_WAVELENGTH_BY_DIVE
    _WORKER_LOADER = loader
    _WORKER_WAVELENGTH_BY_DIVE = wavelength_by_dive


def _process_sample(row: dict) -> dict | None:
    image = _WORKER_LOADER.load(row["image_path"], row["image_checksum"])
    if image is None:
        return None
    measurement = _segment_blob(image, row["label_x"], row["label_y"])
    if measurement is None:
        return None
    diameter, blob_pixels = measurement
    return {
        "dive_id": int(row["dive_id"]),
        "image_id": int(row["image_id"]),
        "wavelength": _WORKER_WAVELENGTH_BY_DIVE.get(int(row["dive_id"])),
        "blob_diameter_px": diameter,
        "blob_pixels": blob_pixels,
    }


AUDIT_TABLE_SCHEMA = {
    "dive_id": pl.Int64,
    "image_id": pl.Int64,
    "wavelength": pl.Utf8,
    "blob_diameter_px": pl.Float64,
    "blob_pixels": pl.Int64,
}


def _segment_blob(
    image: np.ndarray, x: float, y: float, half_size: int = PATCH_HALF_SIZE
) -> tuple[float, int] | None:
    """Segment the laser blob in a window around (x, y).

    Returns (diameter_px, pixel_count) of the connected component **containing
    the labeled pixel**, after a saturation+brightness threshold. Returns None
    if the labeled pixel doesn't sit on a thresholded component, or if the
    component is implausibly large (>25% of the patch area — almost certainly
    water/tile segmentation, not a laser).

    Note: previously fell back to "the largest blob in the patch" when the
    label wasn't on a thresholded pixel. That fallback consistently picked up
    reflections/tiles in saturated underwater scenes, contaminating the upper
    tail of the audit. Returning None instead is honest about which frames the
    audit can measure.
    """
    h, w = image.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return None
    x0, x1 = max(0, cx - half_size), min(w, cx + half_size + 1)
    y0, y1 = max(0, cy - half_size), min(h, cy + half_size + 1)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat > 60) & (val > 200)).astype(np.uint8)
    if mask.sum() == 0:
        return None

    _, components = cv2.connectedComponents(mask)
    label_at_center = int(components[cy - y0, cx - x0])
    if label_at_center == 0:
        return None

    blob_pixels = int((components == label_at_center).sum())
    if blob_pixels == 0:
        return None
    if blob_pixels > MAX_BLOB_AREA_FRAC * mask.size:
        return None
    diameter = float(2.0 * np.sqrt(blob_pixels / np.pi))
    return diameter, blob_pixels


def laser_size_audit(
    frames: pl.DataFrame,
    wavelengths: pl.DataFrame,
    loader: ImageLoader,
    *,
    sample_dives: int = 30,
    samples_per_dive: int = 5,
    rng_seed: int = 0,
    n_workers: int = DEFAULT_N_WORKERS,
) -> pl.DataFrame:
    """Sample positive frames across dives and measure laser blob sizes.

    `n_workers` parallelizes image loads via a thread pool — ORF decode
    dominates and releases the GIL.
    """
    rng = np.random.default_rng(rng_seed)

    positives = frames.filter(pl.col("is_positive"))
    dive_ids = positives["dive_id"].unique().to_list()
    if sample_dives < len(dive_ids):
        dive_ids = list(rng.choice(dive_ids, size=sample_dives, replace=False))

    wavelength_by_dive = {
        int(row["dive_id"]): row["wavelength"]
        for row in wavelengths.iter_rows(named=True)
    }

    sample_rows: list[dict] = []
    for dive_id in dive_ids:
        dive_positives = positives.filter(pl.col("dive_id") == int(dive_id))
        n_take = min(samples_per_dive, dive_positives.height)
        if n_take == 0:
            continue
        sampled = dive_positives.sample(n=n_take, seed=rng_seed)
        sample_rows.extend(sampled.iter_rows(named=True))

    rows: list[dict] = []
    # forkserver — see wavelength.py for the same rationale.
    mp_context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp_context,
        initializer=_init_worker,
        initargs=(loader, wavelength_by_dive),
    ) as ex:
        for result in tqdm(
            ex.map(_process_sample, sample_rows),
            total=len(sample_rows),
            desc="laser-size audit",
        ):
            if result is not None:
                rows.append(result)

    df = pl.DataFrame(rows, schema=AUDIT_TABLE_SCHEMA)
    if df.height > 0:
        diameters = df["blob_diameter_px"].to_numpy()
        logger.info(
            "Laser-size audit: %d measurements; diameter min=%.1f p10=%.1f median=%.1f p90=%.1f max=%.1f px",
            df.height,
            float(np.min(diameters)),
            float(np.percentile(diameters, 10)),
            float(np.median(diameters)),
            float(np.percentile(diameters, 90)),
            float(np.max(diameters)),
        )
    else:
        logger.warning("Laser-size audit produced 0 measurements (no images loaded?)")
    return df
