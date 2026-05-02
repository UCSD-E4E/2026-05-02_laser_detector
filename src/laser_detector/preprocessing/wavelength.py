"""Per-dive wavelength inference (red vs. green) for the v1 corpus.

Each dive is nominally single-color but the SDK doesn't record which one. The
rig has moved to green-only going forward, but a large red backlog remains, so
both must be first-class. ~42% of dives have *some* labels of the minority
color; in practice those are annotator slips and the majority-color tag is the
right one for the dive.

Resolution order per dive:
1. **Label-string majority.** If `LaserLabel.label` strings carry color words,
   the most common one wins (handles "Red Laser" / "Green Laser" labels and
   the mixed-color case where one color dominates).
2. **Color clustering (fallback).** If no label_string yields a color (or the
   counts are tied), KMeans-cluster per-dive label-site colors with k=2 and
   tag clusters by R−G: the redder centroid → "red", the other → "green".

Blue is recognized in label_strings for forward-compat but the v1 corpus has
no blue dives — clustering is two-way over red/green only.
"""

from __future__ import annotations

import logging
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
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
# ORF decode is CPU-bound enough that threads stall on the GIL inside the
# Python wrapping around rawpy/CLAHE. Each worker process gets its own GIL.
# 32 keeps headroom on the 128-logical-core dev box.
DEFAULT_N_WORKERS = 32

# Process-local loader, set by `_init_worker` so it isn't pickled per call.
_WORKER_LOADER: ImageLoader | None = None


def _init_worker(loader: ImageLoader) -> None:
    global _WORKER_LOADER
    _WORKER_LOADER = loader

RED_PATTERN = re.compile(r"\bred\b", re.IGNORECASE)
GREEN_PATTERN = re.compile(r"\bgreen\b", re.IGNORECASE)
BLUE_PATTERN = re.compile(r"\bblue\b", re.IGNORECASE)
_COLOR_PATTERNS = {"red": RED_PATTERN, "green": GREEN_PATTERN, "blue": BLUE_PATTERN}


WAVELENGTH_TABLE_SCHEMA = {
    "dive_id": pl.Int64,
    "wavelength": pl.Utf8,  # "red", "green", "blue", or None if undetermined
    "wavelength_source": pl.Utf8,  # "label_string" or "color_cluster"
    "dive_color_b": pl.Float64,  # mean BGR of label sites (cv2 convention)
    "dive_color_g": pl.Float64,
    "dive_color_r": pl.Float64,
}


@dataclass
class _DiveColor:
    dive_id: int
    color_bgr: np.ndarray  # shape (3,)


def _wavelength_from_labels(label_strings: list[str | None]) -> str | None:
    """Return the majority color word in this dive's label strings.

    Mixed dives are common (~42% of the v1 corpus): a dive labeled "Red Laser"
    100× plus "Green Laser" 2× is treated as red — the minority is almost
    always an annotator slip. Returns None if no color words appear or if the
    top two are tied (in which case the caller falls back to color clustering).
    """
    counts = {
        color: sum(1 for s in label_strings if s and pattern.search(s))
        for color, pattern in _COLOR_PATTERNS.items()
    }
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    top_color, top_count = ranked[0]
    second_count = ranked[1][1]
    if top_count == 0 or top_count == second_count:
        return None
    return top_color


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
    return _DiveColor(
        dive_id=dive_id,
        color_bgr=np.mean(np.stack(colors), axis=0),
    )


def _assign_clusters(dive_colors: list[_DiveColor]) -> dict[int, str]:
    """KMeans-cluster dive colors and map clusters to "red"/"green".

    Two-way clustering for the v1 corpus. The redder centroid (largest R−G in
    BGR) is "red"; the other is "green". No blue cluster — the v1 corpus has
    no blue dives.
    """
    if len(dive_colors) < 2:
        return {}
    matrix = np.stack([dc.color_bgr for dc in dive_colors])
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(matrix)
    centers = kmeans.cluster_centers_  # rows are BGR
    redness = centers[:, 2] - centers[:, 1]  # R - G
    red_idx = int(np.argmax(redness))
    green_idx = 1 - red_idx
    cluster_to_color = {red_idx: "red", green_idx: "green"}
    return {
        dc.dive_id: cluster_to_color[int(label)]
        for dc, label in zip(dive_colors, kmeans.labels_)
    }


def _process_dive(item: tuple) -> tuple[int, str | None, "_DiveColor | None"]:
    """Worker entry point for the process pool. Reads the loader from
    `_WORKER_LOADER`, set once at worker startup by `_init_worker`."""
    (dive_id_t,), group = item
    dive_id = int(dive_id_t)
    labels = group.filter(pl.col("is_positive"))["label_string"].to_list()
    from_labels = _wavelength_from_labels(labels)
    dive_color = _compute_dive_color(dive_id, group, _WORKER_LOADER)
    return dive_id, from_labels, dive_color


def infer_wavelengths(
    frames: pl.DataFrame,
    loader: ImageLoader,
    *,
    n_workers: int = DEFAULT_N_WORKERS,
) -> pl.DataFrame:
    """Per-dive wavelength inference. Returns one row per dive.

    `n_workers` parallelizes per-dive image loads via a process pool — ORF
    decode is CPU-bound enough that threads stall on the GIL.
    """
    rows: list[dict] = []
    dive_colors: list[_DiveColor] = []
    label_string_assignments: dict[int, str] = {}

    grouped = list(frames.group_by("dive_id"))

    # forkserver instead of fork — polars/numpy/cv2 init thread pools in the
    # parent before this point, and inheriting that pthread state across fork()
    # deadlocks the workers. forkserver forks from a clean intermediate process.
    mp_context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp_context,
        initializer=_init_worker,
        initargs=(loader,),
    ) as ex:
        for dive_id, from_labels, dive_color in tqdm(
            ex.map(_process_dive, grouped),
            total=len(grouped),
            desc="wavelength: per-dive color",
        ):
            if from_labels is not None:
                label_string_assignments[dive_id] = from_labels
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
