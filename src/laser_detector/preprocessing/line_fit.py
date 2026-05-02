"""Per-dive RANSAC line fit on positive labels.

The fixed-rig laser geometry means all positive labels in a dive should be
colinear in image space (modulo label noise). This module fits that line and
computes a confidence score so downstream consumers can decide whether to use
the prior or fall back to the unconstrained model.

Output schema (one row per dive):
    dive_id, n_positives, line_a, line_b, line_c,
    inlier_count, inlier_fraction, residual_std,
    line_confidence, is_line_confident

Line representation: `a*x + b*y + c = 0` with `a^2 + b^2 = 1`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Minimum positive labels required to attempt a line fit. A degenerate fit
# (2 points) always succeeds; we want enough redundancy for RANSAC to mean
# something.
MIN_POINTS_FOR_LINE = 5

# RANSAC inlier tolerance in pixels (perpendicular distance). 4K frames + a
# 3 px laser blob → ~4 px is a generous-but-not-loose tolerance for label
# noise. Tunable; logged so we can revisit.
RANSAC_INLIER_TOL_PX = 4.0

# Max RANSAC iterations.
RANSAC_MAX_ITERS = 200

# Confidence threshold below which we say the line is ambiguous and the prior
# should not be applied. Eigenvalue ratio (along-line spread / perp spread).
LINE_CONFIDENCE_THRESHOLD = 5.0


# MAD → consistent estimator of σ for normally distributed residuals.
MAD_TO_SIGMA = 1.4826

# Floor on the MAD-derived σ used by `flag_label_outliers`. On very small dives
# whose RANSAC inliers happen to be sub-pixel-tight, MAD collapses to ~0 and
# every label gets flagged. Labels can't reasonably be more precise than ~1 px
# at native 4K resolution, so any threshold below this is non-physical.
LABEL_NOISE_MAD_FLOOR_PX = 1.0


@dataclass
class LineFit:
    """A normalized line `a*x + b*y + c = 0` plus quality metrics."""

    a: float
    b: float
    c: float
    n_points: int
    inlier_count: int
    inlier_fraction: float
    residual_std: float  # perp-distance std among inliers, in px (RANSAC tightness)
    label_noise_mad: float  # 1.4826 * MAD over ALL positive labels' perp distance, in px
    line_confidence: float  # along-line spread / perp spread (covariance eigenratio)

    def perpendicular_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perpendicular distance from each (x,y) to this line, in pixels."""
        return np.abs(self.a * x + self.b * y + self.c)


def _fit_line_total_least_squares(xy: np.ndarray) -> tuple[float, float, float]:
    """Fit a 2D line via SVD on centered points (total least squares).

    Returns normalized `(a, b, c)` for `a*x + b*y + c = 0`.
    """
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    # SVD: smallest singular vector is the normal to the best-fit line
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    a, b = float(normal[0]), float(normal[1])
    norm = float(np.hypot(a, b))
    if norm == 0.0:
        return 1.0, 0.0, 0.0
    a, b = a / norm, b / norm
    c = float(-(a * centroid[0] + b * centroid[1]))
    return a, b, c


def _ransac_line(
    xy: np.ndarray,
    tol_px: float,
    max_iters: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, np.ndarray]:
    """RANSAC line fit. Returns (a, b, c, inlier_mask)."""
    n = xy.shape[0]
    best_inliers: np.ndarray | None = None
    best_count = -1

    for _ in range(max_iters):
        idx = rng.choice(n, size=2, replace=False)
        p0, p1 = xy[idx[0]], xy[idx[1]]
        # Line through p0, p1
        dx, dy = p1 - p0
        norm = float(np.hypot(dx, dy))
        if norm == 0.0:
            continue
        # Normal direction
        a, b = -dy / norm, dx / norm
        c = -(a * p0[0] + b * p0[1])
        dist = np.abs(a * xy[:, 0] + b * xy[:, 1] + c)
        inliers = dist < tol_px
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 2:
        # Fall back to a TLS fit on all points
        a, b, c = _fit_line_total_least_squares(xy)
        return a, b, c, np.ones(n, dtype=bool)

    # Refit on inliers via TLS
    a, b, c = _fit_line_total_least_squares(xy[best_inliers])
    # Recompute inliers with the refined line
    dist = np.abs(a * xy[:, 0] + b * xy[:, 1] + c)
    inliers = dist < tol_px
    return a, b, c, inliers


def _line_confidence(xy: np.ndarray, a: float, b: float) -> float:
    """Ratio of along-line variance to perpendicular variance.

    A high ratio means the points are spread out along the line (well-determined
    direction). A low ratio means they cluster, leaving the line direction
    ambiguous.
    """
    centered = xy - xy.mean(axis=0)
    along = np.array([-b, a])  # tangent (perpendicular to normal)
    perp = np.array([a, b])
    var_along = float(np.var(centered @ along))
    var_perp = float(np.var(centered @ perp))
    if var_perp <= 1e-9:
        return float("inf")
    return var_along / var_perp


def fit_dive_line(
    xy: np.ndarray,
    *,
    tol_px: float = RANSAC_INLIER_TOL_PX,
    max_iters: int = RANSAC_MAX_ITERS,
    rng: np.random.Generator | None = None,
) -> LineFit | None:
    """Fit a line to one dive's positive labels."""
    if xy.shape[0] < MIN_POINTS_FOR_LINE:
        return None
    rng = rng or np.random.default_rng(0)
    a, b, c, inliers = _ransac_line(xy, tol_px=tol_px, max_iters=max_iters, rng=rng)
    inlier_xy = xy[inliers]
    n = xy.shape[0]
    inlier_count = int(inliers.sum())

    dist_inliers = np.abs(a * inlier_xy[:, 0] + b * inlier_xy[:, 1] + c)
    residual_std = float(np.std(dist_inliers))
    confidence = _line_confidence(inlier_xy, a, b)

    # MAD on ALL positive labels — the population the outlier flag will be applied
    # against. residual_std is bounded by the RANSAC tolerance and so under-states
    # true label-noise scale; MAD is robust to the gross outliers in the tail.
    dist_all = np.abs(a * xy[:, 0] + b * xy[:, 1] + c)
    label_noise_mad = float(MAD_TO_SIGMA * np.median(np.abs(dist_all - np.median(dist_all))))

    return LineFit(
        a=a,
        b=b,
        c=c,
        n_points=n,
        inlier_count=inlier_count,
        inlier_fraction=inlier_count / n,
        residual_std=residual_std,
        label_noise_mad=label_noise_mad,
        line_confidence=confidence,
    )


LINE_TABLE_SCHEMA = {
    "dive_id": pl.Int64,
    "n_positives": pl.Int64,
    "line_a": pl.Float64,
    "line_b": pl.Float64,
    "line_c": pl.Float64,
    "inlier_count": pl.Int64,
    "inlier_fraction": pl.Float64,
    "residual_std": pl.Float64,
    "label_noise_mad": pl.Float64,
    "line_confidence": pl.Float64,
    "is_line_confident": pl.Boolean,
}


def fit_lines_per_dive(
    frames: pl.DataFrame,
    *,
    seed: int = 0,
    confidence_threshold: float = LINE_CONFIDENCE_THRESHOLD,
) -> pl.DataFrame:
    """Fit a line per dive and return a one-row-per-dive table.

    Dives with fewer than `MIN_POINTS_FOR_LINE` positives, or where RANSAC fails,
    appear with all line fields null and `is_line_confident=False`.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    positives = frames.filter(pl.col("is_positive"))
    for dive_id, group in positives.group_by("dive_id"):
        dive_id_int = int(dive_id[0])
        xy = np.column_stack(
            [
                group["label_x"].to_numpy(),
                group["label_y"].to_numpy(),
            ]
        )
        fit = fit_dive_line(xy, rng=rng)
        if fit is None:
            rows.append(
                {
                    "dive_id": dive_id_int,
                    "n_positives": int(xy.shape[0]),
                    "line_a": None,
                    "line_b": None,
                    "line_c": None,
                    "inlier_count": None,
                    "inlier_fraction": None,
                    "residual_std": None,
                    "label_noise_mad": None,
                    "line_confidence": None,
                    "is_line_confident": False,
                }
            )
            continue
        rows.append(
            {
                "dive_id": dive_id_int,
                "n_positives": fit.n_points,
                "line_a": fit.a,
                "line_b": fit.b,
                "line_c": fit.c,
                "inlier_count": fit.inlier_count,
                "inlier_fraction": fit.inlier_fraction,
                "residual_std": fit.residual_std,
                "label_noise_mad": fit.label_noise_mad,
                "line_confidence": fit.line_confidence,
                "is_line_confident": fit.line_confidence >= confidence_threshold,
            }
        )

    df = pl.DataFrame(rows, schema=LINE_TABLE_SCHEMA)
    n_confident = df.filter(pl.col("is_line_confident")).height
    logger.info(
        "Fit %d dive lines (%d confident at threshold %.1f)",
        df.height,
        n_confident,
        confidence_threshold,
    )
    return df


def flag_label_outliers(
    frames: pl.DataFrame,
    line_table: pl.DataFrame,
    *,
    sigma: float = 3.0,
    mad_floor_px: float = LABEL_NOISE_MAD_FLOOR_PX,
) -> pl.DataFrame:
    """Add a `label_is_outlier` column to the frame table.

    For each positive label, compute perpendicular distance to its dive's line
    and flag rows where distance > sigma * max(label_noise_mad, mad_floor_px),
    only for dives with a confident line. `label_noise_mad` is a robust
    estimate of the population σ over all positive labels; using it instead of
    `residual_std` avoids the inlier-conditioning that would otherwise re-flag
    the RANSAC outliers as "≥3σ" outliers regardless of true label-noise
    scale. The floor handles small-N dives where MAD collapses to sub-pixel
    values and would otherwise flag every label.
    """
    # Join line params onto frames
    joined = frames.join(
        line_table.select(
            "dive_id",
            "line_a",
            "line_b",
            "line_c",
            "label_noise_mad",
            "is_line_confident",
        ),
        on="dive_id",
        how="left",
    )

    # Perpendicular distance for positives where we have a confident line
    perp_dist = (
        pl.col("line_a") * pl.col("label_x")
        + pl.col("line_b") * pl.col("label_y")
        + pl.col("line_c")
    ).abs()

    effective_mad = pl.max_horizontal(
        pl.col("label_noise_mad"), pl.lit(mad_floor_px, dtype=pl.Float64)
    )
    outlier_threshold = sigma * effective_mad
    flagged = joined.with_columns(
        pl.when(
            pl.col("is_positive")
            & pl.col("is_line_confident")
            & pl.col("label_noise_mad").is_not_null()
        )
        .then(perp_dist > outlier_threshold)
        .otherwise(False)
        .alias("label_is_outlier"),
        perp_dist.alias("perp_distance_to_line"),
    )

    n_outliers = flagged.filter(pl.col("label_is_outlier")).height
    logger.info(
        "Flagged %d outlier labels (>%.1fσ; σ=max(MAD*1.4826, %.1f px))",
        n_outliers,
        sigma,
        mad_floor_px,
    )
    return flagged.drop(
        ["line_a", "line_b", "line_c", "label_noise_mad", "is_line_confident"]
    )
