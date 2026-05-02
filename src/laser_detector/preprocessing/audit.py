"""Laser-size audit: measure on-screen laser blob diameter on a sample of positives.

Confirms (or refutes) the 3–8 px assumption that drives §4.1's tiling decision.
For each sampled positive frame, segment the laser blob locally around the
labeled pixel and compute its pixel diameter.

Outputs per-frame diameter measurements as a parquet table. The Phase 0
deliverable summarizes the distribution (min, p10, median, p90, max).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import polars as pl
from tqdm import tqdm

from laser_detector.preprocessing.image_loader import ImageLoader

logger = logging.getLogger(__name__)


PATCH_HALF_SIZE = 31  # 63x63 window around the label is enough at 4K


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

    Returns (diameter_px, pixel_count). Diameter is the equivalent-circle
    diameter from the blob area. Pixel count is the connected component
    containing the labeled pixel after a saturation+brightness threshold.
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
    # Lasers are bright and saturated against the underwater background. We
    # threshold on (saturation, value) jointly. Tunable; a reasonable default
    # for a first audit.
    mask = ((sat > 60) & (val > 200)).astype(np.uint8)
    if mask.sum() == 0:
        return None

    n_components, components = cv2.connectedComponents(mask)
    label_at_center = int(components[cy - y0, cx - x0])
    if label_at_center == 0:
        # The brightest blob isn't at the labeled pixel; pick the largest blob
        # in the patch instead. Reflects label imprecision but still a valid
        # blob measurement.
        sizes = [(int((components == k).sum()), k) for k in range(1, n_components)]
        if not sizes:
            return None
        _, label_at_center = max(sizes)

    blob_pixels = int((components == label_at_center).sum())
    if blob_pixels == 0:
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
) -> pl.DataFrame:
    """Sample positive frames across dives and measure laser blob sizes."""
    rng = np.random.default_rng(rng_seed)

    positives = frames.filter(pl.col("is_positive"))
    dive_ids = positives["dive_id"].unique().to_list()
    if sample_dives < len(dive_ids):
        dive_ids = list(rng.choice(dive_ids, size=sample_dives, replace=False))

    wavelength_by_dive = {
        int(row["dive_id"]): row["wavelength"]
        for row in wavelengths.iter_rows(named=True)
    }

    rows: list[dict] = []
    for dive_id in tqdm(dive_ids, desc="laser-size audit"):
        dive_positives = positives.filter(pl.col("dive_id") == int(dive_id))
        n_take = min(samples_per_dive, dive_positives.height)
        if n_take == 0:
            continue
        sampled = dive_positives.sample(n=n_take, seed=rng_seed)
        for row in sampled.iter_rows(named=True):
            image = loader.load(row["image_path"], row["image_checksum"])
            if image is None:
                continue
            measurement = _segment_blob(image, row["label_x"], row["label_y"])
            if measurement is None:
                continue
            diameter, blob_pixels = measurement
            rows.append(
                {
                    "dive_id": int(row["dive_id"]),
                    "image_id": int(row["image_id"]),
                    "wavelength": wavelength_by_dive.get(int(row["dive_id"])),
                    "blob_diameter_px": diameter,
                    "blob_pixels": blob_pixels,
                }
            )

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
