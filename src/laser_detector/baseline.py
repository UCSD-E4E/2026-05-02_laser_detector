"""Phase 1 classical CV baseline detector.

Per DESIGN.md §9 Phase 1: "Per dive: compute a wavelength-specific color
mask, find the brightest blob, score by closeness to the dive's line. No
learning. This is the floor."

Per-frame algorithm (`detect_in_frame`):
1. Compute an HSV mask combining wavelength-specific hue range with the
   shared "saturated AND bright" cut from the Phase-0 blob audit.
2. Find connected components and reject any whose area is outside the
   3–1000 px range (≈2–36 px diameter — informed by the Phase-0 audit
   distribution after the segmentation rework).
3. Score each surviving component by `mean_value * line_proximity`, where
   `line_proximity` is a Gaussian on perpendicular distance to the dive's
   line (only applied when the dive's line is confident; otherwise 1.0).
4. Return the highest-scoring component's centroid as `(pred_x, pred_y)`,
   with `pred_confidence = mean_value/255 * line_proximity` clipped to
   [0, 1]. If no components survive, return a no-detection prediction.

Batch driver (`run_baseline`) parallelizes across frames via the same
ProcessPoolExecutor + forkserver pattern as wavelength/audit, and writes a
predictions table matching `eval.PREDICTION_TABLE_SCHEMA`.
"""

from __future__ import annotations

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
import polars as pl
from tqdm import tqdm

from laser_detector.eval import PREDICTION_TABLE_SCHEMA
from laser_detector.preprocessing.image_loader import ImageLoader

logger = logging.getLogger(__name__)


# OpenCV HSV ranges. Hue is 0-179 (not 0-359). Red wraps around, so use two
# bands. Saturation/value cut is shared and matches the Phase-0 audit
# threshold so what we detect lines up with what we measure.
RED_HUE_BANDS: tuple[tuple[int, int], ...] = ((0, 15), (160, 179))
GREEN_HUE_BANDS: tuple[tuple[int, int], ...] = ((35, 95),)
SAT_MIN = 100
VAL_MIN = 200

# Blob area sanity gate (px²). Below 3 ≈ 2 px diameter (label noise);
# above 1000 ≈ 36 px diameter (above any plausible laser blob, per Phase-0
# audit max of 35 px after segmentation rework).
MIN_BLOB_AREA = 3
MAX_BLOB_AREA = 1000

# Gaussian σ on perpendicular line distance for the line-proximity score
# (px). Roughly the upper end of label noise (label_noise_mad p90 ≈ 9 px).
# A laser blob more than ~3σ off the line is almost certainly not the laser.
LINE_PROXIMITY_SIGMA_PX = 9.0

# Pool defaults — match wavelength.py / audit.py.
DEFAULT_N_WORKERS = 32

# Process-local state, set by the worker initializer.
_WORKER_LOADER: ImageLoader | None = None
_WORKER_DIVE_INFO: dict[int, "DiveInfo"] = {}


@dataclass(frozen=True)
class DiveInfo:
    """Per-dive context passed to the per-frame detector."""

    wavelength: str | None  # "red" / "green" / None
    line_a: float | None
    line_b: float | None
    line_c: float | None
    is_line_confident: bool


@dataclass(frozen=True)
class Detection:
    """Per-frame output. None xy means "no laser detected"."""

    pred_x: float | None
    pred_y: float | None
    pred_confidence: float


_NO_DETECTION = Detection(pred_x=None, pred_y=None, pred_confidence=0.0)


def _hue_mask(hsv: np.ndarray, bands: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Union of hue bands, intersected with sat/val cuts."""
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    bright_saturated = (sat >= SAT_MIN) & (val >= VAL_MIN)
    in_hue = np.zeros_like(hue, dtype=bool)
    for lo, hi in bands:
        in_hue |= (hue >= lo) & (hue <= hi)
    return (in_hue & bright_saturated).astype(np.uint8)


def _color_mask(image_bgr: np.ndarray, wavelength: str | None) -> np.ndarray:
    """HSV color mask for the wavelength. Unknown wavelength → union of red+green."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if wavelength == "red":
        return _hue_mask(hsv, RED_HUE_BANDS)
    if wavelength == "green":
        return _hue_mask(hsv, GREEN_HUE_BANDS)
    return _hue_mask(hsv, RED_HUE_BANDS + GREEN_HUE_BANDS)


def detect_in_frame(image_bgr: np.ndarray, dive: DiveInfo) -> Detection:
    """Per-frame baseline detector. See module docstring for algorithm."""
    mask = _color_mask(image_bgr, dive.wavelength)
    if mask.sum() == 0:
        return _NO_DETECTION

    # connectedComponentsWithStats returns: n, label_image, stats, centroids.
    # stats columns: x, y, w, h, area; centroids rows: (cx, cy).
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return _NO_DETECTION

    # Per-component mean V (brightness) — used in scoring.
    val_channel = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]

    best_score = 0.0
    best: Detection = _NO_DETECTION
    for i in range(1, n):  # skip background (label 0)
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
            continue
        component_mask = labels == i
        mean_val = float(val_channel[component_mask].mean())
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])

        if dive.is_line_confident and dive.line_a is not None:
            perp = abs(dive.line_a * cx + dive.line_b * cy + dive.line_c)
            line_proximity = float(
                np.exp(-(perp ** 2) / (2 * LINE_PROXIMITY_SIGMA_PX ** 2))
            )
        else:
            line_proximity = 1.0

        score = (mean_val / 255.0) * line_proximity
        if score > best_score:
            best_score = score
            best = Detection(pred_x=cx, pred_y=cy, pred_confidence=min(score, 1.0))

    return best


def _init_worker(
    loader: ImageLoader, dive_info: dict[int, DiveInfo]
) -> None:
    global _WORKER_LOADER, _WORKER_DIVE_INFO
    _WORKER_LOADER = loader
    _WORKER_DIVE_INFO = dive_info


def _process_frame(row: dict) -> dict:
    """Worker entry point. Loads image, runs detector, returns prediction row."""
    image = _WORKER_LOADER.load(row["image_path"], row["image_checksum"])
    if image is None:
        return {
            "image_id": int(row["image_id"]),
            "pred_x": None,
            "pred_y": None,
            "pred_confidence": 0.0,
        }
    dive = _WORKER_DIVE_INFO.get(int(row["dive_id"]))
    if dive is None:
        # No dive context — fall back to wavelength=None, no line.
        dive = DiveInfo(
            wavelength=None,
            line_a=None,
            line_b=None,
            line_c=None,
            is_line_confident=False,
        )
    detection = detect_in_frame(image, dive)
    return {
        "image_id": int(row["image_id"]),
        "pred_x": detection.pred_x,
        "pred_y": detection.pred_y,
        "pred_confidence": detection.pred_confidence,
    }


def _build_dive_info(
    wavelengths: pl.DataFrame, lines: pl.DataFrame
) -> dict[int, DiveInfo]:
    joined = wavelengths.select("dive_id", "wavelength").join(
        lines.select(
            "dive_id", "line_a", "line_b", "line_c", "is_line_confident"
        ),
        on="dive_id",
        how="left",
    )
    out: dict[int, DiveInfo] = {}
    for row in joined.iter_rows(named=True):
        out[int(row["dive_id"])] = DiveInfo(
            wavelength=row["wavelength"],
            line_a=row["line_a"],
            line_b=row["line_b"],
            line_c=row["line_c"],
            is_line_confident=bool(row["is_line_confident"])
            if row["is_line_confident"] is not None
            else False,
        )
    return out


def run_baseline(
    frames: pl.DataFrame,
    wavelengths: pl.DataFrame,
    lines: pl.DataFrame,
    loader: ImageLoader,
    *,
    n_workers: int = DEFAULT_N_WORKERS,
) -> pl.DataFrame:
    """Run the baseline detector on every frame in `frames`.

    Returns a predictions table matching `eval.PREDICTION_TABLE_SCHEMA`.
    Caller is responsible for filtering `frames` to whatever subset (e.g.
    only val-split dives) before calling.
    """
    dive_info = _build_dive_info(wavelengths, lines)

    sample_rows = list(frames.iter_rows(named=True))
    rows: list[dict] = []
    mp_context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp_context,
        initializer=_init_worker,
        initargs=(loader, dive_info),
    ) as ex:
        for result in tqdm(
            ex.map(_process_frame, sample_rows),
            total=len(sample_rows),
            desc="baseline detector",
        ):
            rows.append(result)

    df = pl.DataFrame(rows, schema=PREDICTION_TABLE_SCHEMA)
    n_detected = df.filter(pl.col("pred_x").is_not_null()).height
    logger.info(
        "Baseline detector: %d frames, %d detections (%.1f%% localized)",
        df.height,
        n_detected,
        100.0 * n_detected / df.height if df.height else 0.0,
    )
    return df
