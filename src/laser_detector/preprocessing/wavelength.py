"""Per-dive wavelength inference (green vs. blue) by clustering label-site colors.

Each dive is single-color but the SDK doesn't record which one. We recover it:
1. For each positive label, sample a small patch around the labeled pixel.
2. Take the brightest-pixel color in the patch (the laser dominates).
3. Average those colors across the dive → one color vector per dive.
4. KMeans cluster dive vectors with k=2 → assign green/blue based on which
   cluster centroid has more green vs. blue energy.

If the SDK's `LaserLabel.label` string already contains color words (e.g.,
"green", "blue") consistently across a dive, that wins — clustering is a
fallback when the field is missing or unstructured.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from tqdm import tqdm

from laser_detector.preprocessing.image_loader import ImageLoader

logger = logging.getLogger(__name__)


PATCH_HALF_SIZE = 5  # 11x11 patch around the labeled pixel
MAX_LABELS_PER_DIVE_FOR_COLOR = 10  # cap sampling for speed

GREEN_PATTERN = re.compile(r"\bgreen\b", re.IGNORECASE)
BLUE_PATTERN = re.compile(r"\bblue\b", re.IGNORECASE)


WAVELENGTH_TABLE_SCHEMA = {
    "dive_id": pl.Int64,
    "wavelength": pl.Utf8,  # "green", "blue", or None if undetermined
    "wavelength_source": pl.Utf8,  # "label_string" or "color_cluster"
    "dive_color_b": pl.Float64,  # mean BGR of label sites (cv2 convention)
    "dive_color_g": pl.Float64,
    "dive_color_r": pl.Float64,
}


@dataclass
class _DiveColor:
    dive_id: int
    color_bgr: np.ndarray  # shape (3,)
    label_string_green: int
    label_string_blue: int


def _wavelength_from_labels(label_strings: list[str | None]) -> str | None:
    """If label strings unambiguously say green or blue across a dive, use that."""
    has_green = any(s and GREEN_PATTERN.search(s) for s in label_strings)
    has_blue = any(s and BLUE_PATTERN.search(s) for s in label_strings)
    if has_green and not has_blue:
        return "green"
    if has_blue and not has_green:
        return "blue"
    return None


def _sample_label_color(
    image: np.ndarray, x: float, y: float, half_size: int = PATCH_HALF_SIZE
) -> np.ndarray | None:
    """Return the BGR color of the brightest pixel in a patch around (x, y)."""
    h, w = image.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return None
    x0 = max(0, cx - half_size)
    x1 = min(w, cx + half_size + 1)
    y0 = max(0, cy - half_size)
    y1 = min(h, cy + half_size + 1)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    # Brightest pixel = best chance of capturing the laser itself rather than
    # the surrounding fish/water. Use V from HSV.
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    flat_idx = int(np.argmax(hsv[:, :, 2]))
    py, px = np.unravel_index(flat_idx, hsv.shape[:2])
    return patch[py, px].astype(np.float64)


def _compute_dive_color(
    dive_id: int,
    dive_frames: pl.DataFrame,
    loader: ImageLoader,
) -> _DiveColor | None:
    """Average the brightest-pixel color across a dive's labeled images."""
    positives = dive_frames.filter(pl.col("is_positive"))
    sample = positives.head(MAX_LABELS_PER_DIVE_FOR_COLOR)
    colors: list[np.ndarray] = []
    for row in sample.iter_rows(named=True):
        image = loader.load(row["image_path"], row["image_checksum"])
        if image is None:
            continue
        color = _sample_label_color(image, row["label_x"], row["label_y"])
        if color is None:
            continue
        colors.append(color)
    if not colors:
        return None

    label_strings = positives["label_string"].to_list()
    return _DiveColor(
        dive_id=dive_id,
        color_bgr=np.mean(np.stack(colors), axis=0),
        label_string_green=sum(
            1 for s in label_strings if s and GREEN_PATTERN.search(s)
        ),
        label_string_blue=sum(
            1 for s in label_strings if s and BLUE_PATTERN.search(s)
        ),
    )


def _assign_clusters(dive_colors: list[_DiveColor]) -> dict[int, str]:
    """KMeans-cluster dive colors and map clusters to "green"/"blue"."""
    if len(dive_colors) < 2:
        return {}
    matrix = np.stack([dc.color_bgr for dc in dive_colors])
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(matrix)
    centers = kmeans.cluster_centers_  # rows are BGR
    # Greener centroid: G - max(B, R) is largest
    green_idx = int(np.argmax(centers[:, 1] - np.maximum(centers[:, 0], centers[:, 2])))
    blue_idx = 1 - green_idx
    cluster_to_color = {green_idx: "green", blue_idx: "blue"}
    return {
        dc.dive_id: cluster_to_color[int(label)]
        for dc, label in zip(dive_colors, kmeans.labels_)
    }


def infer_wavelengths(
    frames: pl.DataFrame,
    loader: ImageLoader,
) -> pl.DataFrame:
    """Per-dive wavelength inference. Returns one row per dive."""
    rows: list[dict] = []
    dive_colors: list[_DiveColor] = []
    label_string_assignments: dict[int, str] = {}

    grouped = list(frames.group_by("dive_id"))
    for (dive_id_t,), group in tqdm(grouped, desc="wavelength: per-dive color"):
        dive_id = int(dive_id_t)
        labels = group.filter(pl.col("is_positive"))["label_string"].to_list()
        from_labels = _wavelength_from_labels(labels)
        if from_labels is not None:
            label_string_assignments[dive_id] = from_labels

        dive_color = _compute_dive_color(dive_id, group, loader)
        if dive_color is not None:
            dive_colors.append(dive_color)

    cluster_assignments = _assign_clusters(dive_colors) if dive_colors else {}
    color_by_id = {dc.dive_id: dc.color_bgr for dc in dive_colors}

    all_dive_ids = sorted({int(d) for d in frames["dive_id"].unique().to_list()})
    for dive_id in all_dive_ids:
        if dive_id in label_string_assignments:
            wavelength = label_string_assignments[dive_id]
            source = "label_string"
        elif dive_id in cluster_assignments:
            wavelength = cluster_assignments[dive_id]
            source = "color_cluster"
        else:
            wavelength = None
            source = None

        color = color_by_id.get(dive_id)
        rows.append(
            {
                "dive_id": dive_id,
                "wavelength": wavelength,
                "wavelength_source": source,
                "dive_color_b": float(color[0]) if color is not None else None,
                "dive_color_g": float(color[1]) if color is not None else None,
                "dive_color_r": float(color[2]) if color is not None else None,
            }
        )

    df = pl.DataFrame(rows, schema=WAVELENGTH_TABLE_SCHEMA)
    n_known = df.filter(pl.col("wavelength").is_not_null()).height
    logger.info(
        "Assigned wavelengths to %d / %d dives (%d via label_string, %d via clustering)",
        n_known,
        df.height,
        df.filter(pl.col("wavelength_source") == "label_string").height,
        df.filter(pl.col("wavelength_source") == "color_cluster").height,
    )
    return df
