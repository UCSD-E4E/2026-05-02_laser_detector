"""Evaluation harness for laser-detector predictions.

Consumes a predictions table (one row per frame, with predicted xy and
confidence) plus the Phase 0 artifacts (frames, wavelengths, lines, splits)
and produces the metrics defined in DESIGN.md §7:

- **Hit rate** at fixed-px tolerance (N=3 strict, N=4 lenient by default).
- **Mean pixel error** on positive frames.
- **Presence AUROC** discriminating positive vs negative frames.
- **FPR @ presence threshold**.

Sliced by wavelength and `line_confidence` quartile per DESIGN.md §7.2.

The blob-tolerance hit rate (the primary metric in §7.1) requires loading
images and is intentionally NOT included here — it's implemented separately
in the audit module's `_segment_blob`. Add a blob-tolerance pass via a
follow-up if/when fixed-tolerance numbers stop being informative.

The same harness runs against any detector that produces the predictions
schema below — the Phase 1 classical-CV baseline and the Phase 2+ learned
models share this code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# Per-frame prediction. `pred_x` / `pred_y` are null when the detector says
# "no laser"; `pred_confidence` is the detector's presence-score (higher =
# more confident a laser is present), used for AUROC + presence-threshold
# sweep. Confidence must be defined (use 0.0 for explicit no-detect).
PREDICTION_TABLE_SCHEMA = {
    "image_id": pl.Int64,
    "pred_x": pl.Float64,
    "pred_y": pl.Float64,
    "pred_confidence": pl.Float64,
}

# DESIGN.md §7.1 fixed-tolerance fallback: 3 px = "worst-case blob radius at
# z=6 m, no divergence" (strict); 4 px = typical with divergence (lenient).
DEFAULT_TOLERANCES_PX: tuple[int, ...] = (3, 4)


@dataclass(frozen=True)
class EvalResult:
    """Flat metrics dict suitable for `mlflow.log_metrics`, plus the joined
    per-frame frame to support ad-hoc slicing.
    """

    metrics: dict[str, float]
    per_frame: pl.DataFrame


def _hit_at_tolerance(
    gt_xy: np.ndarray, pred_xy: np.ndarray, max_px: float
) -> np.ndarray:
    """Vectorized: True iff Euclidean distance(gt, pred) <= max_px.

    Rows where either gt or pred is NaN are False (a missing prediction on a
    positive frame is a miss).
    """
    valid = ~(
        np.isnan(gt_xy).any(axis=1) | np.isnan(pred_xy).any(axis=1)
    )
    dist = np.full(gt_xy.shape[0], np.inf)
    if valid.any():
        dx = pred_xy[valid, 0] - gt_xy[valid, 0]
        dy = pred_xy[valid, 1] - gt_xy[valid, 1]
        dist[valid] = np.hypot(dx, dy)
    return dist <= max_px


def _pixel_errors(gt_xy: np.ndarray, pred_xy: np.ndarray) -> np.ndarray:
    """Per-row Euclidean distance; NaN where either side is missing."""
    dx = pred_xy[:, 0] - gt_xy[:, 0]
    dy = pred_xy[:, 1] - gt_xy[:, 1]
    return np.hypot(dx, dy)


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """sklearn's roc_auc_score raises if either class is empty; we'd rather
    return None and let the caller skip the metric than crash on a slice
    that happens to be all-positive or all-negative.
    """
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _aggregate(
    joined: pl.DataFrame,
    *,
    presence_threshold: float,
    tolerances_px: tuple[int, ...],
) -> dict[str, float]:
    """Compute the metric dict over a (possibly sliced) joined table."""
    n = joined.height
    if n == 0:
        return {"n_frames": 0.0}

    is_positive = joined["is_positive"].to_numpy()
    n_pos = int(is_positive.sum())
    n_neg = int((~is_positive).sum())

    metrics: dict[str, float] = {
        "n_frames": float(n),
        "n_positive": float(n_pos),
        "n_negative": float(n_neg),
    }

    # Presence: AUROC + FPR at threshold. Confidence is required (no nulls);
    # callers that want "no detection" should report 0.0 confidence.
    confidence = joined["pred_confidence"].fill_null(0.0).to_numpy()
    auroc = _safe_auroc(is_positive.astype(int), confidence)
    if auroc is not None:
        metrics["presence_auroc"] = auroc

    presence_pred = confidence >= presence_threshold
    if n_neg > 0:
        metrics["fpr_at_threshold"] = float(presence_pred[~is_positive].mean())
    if n_pos > 0:
        metrics["recall_at_threshold"] = float(presence_pred[is_positive].mean())

    # Localization metrics on positive frames only.
    if n_pos == 0:
        return metrics

    pos_mask = is_positive
    gt_xy = np.column_stack(
        [
            joined["label_x"].to_numpy()[pos_mask],
            joined["label_y"].to_numpy()[pos_mask],
        ]
    )
    pred_xy = np.column_stack(
        [
            joined["pred_x"].to_numpy()[pos_mask],
            joined["pred_y"].to_numpy()[pos_mask],
        ]
    )

    for tol_px in tolerances_px:
        hits = _hit_at_tolerance(gt_xy, pred_xy, tol_px)
        metrics[f"hit_rate_n{tol_px}"] = float(hits.mean())

    # Mean pixel error over positives that *got a prediction*. Frames where
    # the detector said "no laser" don't have a meaningful pixel error.
    pred_mask = ~np.isnan(pred_xy).any(axis=1)
    metrics["fraction_localized"] = float(pred_mask.mean())
    if pred_mask.any():
        errs = _pixel_errors(gt_xy[pred_mask], pred_xy[pred_mask])
        metrics["mean_pixel_error"] = float(errs.mean())
        metrics["median_pixel_error"] = float(np.median(errs))

    return metrics


def evaluate(
    predictions: pl.DataFrame,
    frames: pl.DataFrame,
    splits: pl.DataFrame,
    wavelengths: pl.DataFrame,
    lines: pl.DataFrame,
    *,
    split: str = "val",
    presence_threshold: float = 0.5,
    tolerances_px: tuple[int, ...] = DEFAULT_TOLERANCES_PX,
) -> EvalResult:
    """Compute eval metrics on `split` ("train" / "val" / "test").

    Returns an `EvalResult` whose `.metrics` is flat (suitable for
    `mlflow.log_metrics`) with overall metrics and per-slice variants
    prefixed `wavelength_<wl>/...` and `line_q<1-4>/...`.

    Slicing per DESIGN.md §7.2:
    - **wavelength** (red / green) — does the model lean on color?
    - **line_confidence quartile** (q1=lowest, q4=highest) — does it lean on
      the per-dive line prior?
    """
    dive_split = splits.filter(pl.col("split") == split)
    eval_frames = frames.join(dive_split, on="dive_id", how="inner")

    eval_frames = eval_frames.join(
        wavelengths.select("dive_id", "wavelength"), on="dive_id", how="left"
    )
    eval_frames = eval_frames.join(
        lines.select("dive_id", "line_confidence"), on="dive_id", how="left"
    )

    joined = eval_frames.join(predictions, on="image_id", how="left")

    overall = _aggregate(
        joined,
        presence_threshold=presence_threshold,
        tolerances_px=tolerances_px,
    )

    metrics: dict[str, float] = dict(overall)

    # wavelength slices — only emit metrics for present wavelengths
    for wl in (
        joined["wavelength"]
        .drop_nulls()
        .unique()
        .sort()
        .to_list()
    ):
        sliced = joined.filter(pl.col("wavelength") == wl)
        for k, v in _aggregate(
            sliced,
            presence_threshold=presence_threshold,
            tolerances_px=tolerances_px,
        ).items():
            metrics[f"wavelength_{wl}/{k}"] = v

    # line_confidence quartile slices — uses the eval-set distribution as
    # the bin edges so quartiles are meaningful for *this* split.
    confidences = (
        joined["line_confidence"].drop_nulls().to_numpy()
    )
    if confidences.size >= 4:
        edges = np.quantile(confidences, [0.25, 0.5, 0.75])
        for q_idx, (lo_q, hi_q, label) in enumerate(
            [
                (-np.inf, edges[0], "q1"),
                (edges[0], edges[1], "q2"),
                (edges[1], edges[2], "q3"),
                (edges[2], np.inf, "q4"),
            ]
        ):
            sliced = joined.filter(
                pl.col("line_confidence").is_not_null()
                & (pl.col("line_confidence") > lo_q)
                & (pl.col("line_confidence") <= hi_q)
            )
            for k, v in _aggregate(
                sliced,
                presence_threshold=presence_threshold,
                tolerances_px=tolerances_px,
            ).items():
                metrics[f"line_{label}/{k}"] = v

    logger.info(
        "Eval (split=%s, presence_threshold=%.2f): %d frames, %d positives → "
        "hit_rate_n3=%.3f hit_rate_n4=%.3f auroc=%.3f fpr=%.3f",
        split,
        presence_threshold,
        int(overall.get("n_frames", 0)),
        int(overall.get("n_positive", 0)),
        overall.get("hit_rate_n3", float("nan")),
        overall.get("hit_rate_n4", float("nan")),
        overall.get("presence_auroc", float("nan")),
        overall.get("fpr_at_threshold", float("nan")),
    )

    return EvalResult(metrics=metrics, per_frame=joined)
