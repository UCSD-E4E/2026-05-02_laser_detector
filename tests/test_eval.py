"""Synthetic-data tests for the eval harness.

Build a small frames table with known ground truth, hand-craft a predictions
table, and verify the metrics in `evaluate()` match what we computed by hand.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from laser_detector.eval import (
    DEFAULT_TOLERANCES_PX,
    PREDICTION_TABLE_SCHEMA,
    _hit_at_tolerance,
    _safe_auroc,
    evaluate,
)
from laser_detector.preprocessing.ingest import FRAME_TABLE_SCHEMA
from laser_detector.preprocessing.line_fit import LINE_TABLE_SCHEMA
from laser_detector.preprocessing.splits import SPLIT_TABLE_SCHEMA
from laser_detector.preprocessing.wavelength import WAVELENGTH_TABLE_SCHEMA


import numpy as np


def _frame(image_id: int, dive_id: int, label_xy: tuple[float, float] | None) -> dict:
    is_pos = label_xy is not None
    return {
        "dive_id": dive_id,
        "image_id": image_id,
        "rig_id": 1,
        "image_path": f"d{dive_id}/i{image_id}.jpg",
        "image_checksum": f"sum{image_id}",
        "label_x": float(label_xy[0]) if is_pos else None,
        "label_y": float(label_xy[1]) if is_pos else None,
        "is_positive": is_pos,
        "label_string": "Red Laser" if is_pos else None,
        "label_studio_task_id": None,
        "label_studio_project_id": None,
        "superseded": False,
        "completed": True,
    }


def _pred(image_id: int, xy: tuple[float, float] | None, conf: float) -> dict:
    return {
        "image_id": image_id,
        "pred_x": float(xy[0]) if xy is not None else None,
        "pred_y": float(xy[1]) if xy is not None else None,
        "pred_confidence": float(conf),
    }


@pytest.fixture
def synthetic_phase0() -> dict[str, pl.DataFrame]:
    """One val dive with 4 positives + 1 negative; one train dive (ignored)."""
    frames = pl.DataFrame(
        [
            # val dive
            _frame(1, dive_id=10, label_xy=(100.0, 100.0)),
            _frame(2, dive_id=10, label_xy=(200.0, 200.0)),
            _frame(3, dive_id=10, label_xy=(300.0, 300.0)),
            _frame(4, dive_id=10, label_xy=(400.0, 400.0)),
            _frame(5, dive_id=10, label_xy=None),  # negative
            # train dive (should NOT appear in val eval)
            _frame(99, dive_id=20, label_xy=(50.0, 50.0)),
        ],
        schema=FRAME_TABLE_SCHEMA,
    )

    splits = pl.DataFrame(
        [
            {"dive_id": 10, "split": "val"},
            {"dive_id": 20, "split": "train"},
        ],
        schema=SPLIT_TABLE_SCHEMA,
    )

    wavelengths = pl.DataFrame(
        [
            {
                "dive_id": 10,
                "wavelength": "red",
                "wavelength_source": "label_string",
                "dive_color_b": 100.0,
                "dive_color_g": 100.0,
                "dive_color_r": 200.0,
            },
            {
                "dive_id": 20,
                "wavelength": "green",
                "wavelength_source": "label_string",
                "dive_color_b": 100.0,
                "dive_color_g": 200.0,
                "dive_color_r": 100.0,
            },
        ],
        schema=WAVELENGTH_TABLE_SCHEMA,
    )

    lines = pl.DataFrame(
        [
            {
                "dive_id": 10,
                "n_positives": 4,
                "line_a": 1.0,
                "line_b": 0.0,
                "line_c": 0.0,
                "inlier_count": 4,
                "inlier_fraction": 1.0,
                "residual_std": 0.5,
                "label_noise_mad": 0.5,
                "line_confidence": 100.0,
                "is_line_confident": True,
            },
            {
                "dive_id": 20,
                "n_positives": 1,
                "line_a": None,
                "line_b": None,
                "line_c": None,
                "inlier_count": None,
                "inlier_fraction": None,
                "residual_std": None,
                "label_noise_mad": None,
                "line_confidence": None,
                "is_line_confident": False,
            },
        ],
        schema=LINE_TABLE_SCHEMA,
    )

    return {
        "frames": frames,
        "splits": splits,
        "wavelengths": wavelengths,
        "lines": lines,
    }


def test_hit_at_tolerance_basic():
    gt = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]])
    pred = np.array([[100.0, 102.0], [205.0, 200.0], [400.0, 300.0]])
    # distances: 2.0, 5.0, 100.0
    assert _hit_at_tolerance(gt, pred, 3.0).tolist() == [True, False, False]
    assert _hit_at_tolerance(gt, pred, 5.0).tolist() == [True, True, False]


def test_hit_at_tolerance_handles_nan_predictions():
    """Missing prediction (model said 'no laser') is a miss, not a crash."""
    gt = np.array([[100.0, 100.0], [200.0, 200.0]])
    pred = np.array([[np.nan, np.nan], [200.0, 200.0]])
    assert _hit_at_tolerance(gt, pred, 3.0).tolist() == [False, True]


def test_safe_auroc_handles_single_class():
    """Returning None on degenerate slices avoids sklearn raising."""
    assert _safe_auroc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])) is None
    assert _safe_auroc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) is None


def test_safe_auroc_perfect_separation():
    auroc = _safe_auroc(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    )
    assert auroc == pytest.approx(1.0)


def test_evaluate_counts_only_eval_split(synthetic_phase0):
    """Train-dive frames must not leak into val metrics."""
    predictions = pl.DataFrame(
        [
            _pred(1, (100.0, 100.0), conf=0.9),
            _pred(2, (200.0, 200.0), conf=0.9),
            _pred(3, (300.0, 300.0), conf=0.9),
            _pred(4, (400.0, 400.0), conf=0.9),
            _pred(5, None, conf=0.1),
            _pred(99, (50.0, 50.0), conf=0.9),  # train dive
        ],
        schema=PREDICTION_TABLE_SCHEMA,
    )
    res = evaluate(predictions, split="val", **synthetic_phase0)
    assert res.metrics["n_frames"] == 5
    assert res.metrics["n_positive"] == 4
    assert res.metrics["n_negative"] == 1


def test_evaluate_perfect_predictions(synthetic_phase0):
    """All hits, perfect AUROC, FPR=0."""
    predictions = pl.DataFrame(
        [
            _pred(1, (100.0, 100.0), conf=0.95),
            _pred(2, (200.0, 200.0), conf=0.95),
            _pred(3, (300.0, 300.0), conf=0.95),
            _pred(4, (400.0, 400.0), conf=0.95),
            _pred(5, None, conf=0.0),  # correct no-detect
        ],
        schema=PREDICTION_TABLE_SCHEMA,
    )
    res = evaluate(predictions, split="val", **synthetic_phase0)
    assert res.metrics["hit_rate_n3"] == 1.0
    assert res.metrics["hit_rate_n4"] == 1.0
    assert res.metrics["mean_pixel_error"] == pytest.approx(0.0)
    assert res.metrics["presence_auroc"] == pytest.approx(1.0)
    assert res.metrics["fpr_at_threshold"] == 0.0
    assert res.metrics["recall_at_threshold"] == 1.0


def test_evaluate_partial_hits_and_misses(synthetic_phase0):
    """2 hits inside 3px, 1 hit between 3 and 4px, 1 miss; one false positive."""
    predictions = pl.DataFrame(
        [
            _pred(1, (100.0, 102.0), conf=0.9),  # 2 px → hit at both N
            _pred(2, (200.0, 203.5), conf=0.9),  # 3.5 px → miss at N=3, hit at N=4
            _pred(3, (300.0, 300.0), conf=0.9),  # 0 px → hit
            _pred(4, (500.0, 500.0), conf=0.9),  # ~141 px → miss
            _pred(5, (50.0, 50.0), conf=0.9),  # negative frame, predicted with confidence → FP
        ],
        schema=PREDICTION_TABLE_SCHEMA,
    )
    res = evaluate(predictions, split="val", **synthetic_phase0)
    # 2 of 4 positives within 3 px (id=1, id=3); 3 of 4 within 4 px (also id=2)
    assert res.metrics["hit_rate_n3"] == pytest.approx(2 / 4)
    assert res.metrics["hit_rate_n4"] == pytest.approx(3 / 4)
    # mean pixel error over the 4 localized positives = (2 + 3.5 + 0 + sqrt(20000)) / 4
    expected_mpe = (2.0 + 3.5 + 0.0 + math.hypot(100.0, 100.0)) / 4
    assert res.metrics["mean_pixel_error"] == pytest.approx(expected_mpe)
    # 1 negative, predicted with conf=0.9 ≥ 0.5 → FPR = 1.0
    assert res.metrics["fpr_at_threshold"] == 1.0


def test_evaluate_missing_prediction_counts_as_miss(synthetic_phase0):
    """A positive frame with pred_x=null is a miss, not skipped."""
    predictions = pl.DataFrame(
        [
            _pred(1, None, conf=0.0),
            _pred(2, None, conf=0.0),
            _pred(3, (300.0, 300.0), conf=0.9),
            _pred(4, (400.0, 400.0), conf=0.9),
            _pred(5, None, conf=0.0),
        ],
        schema=PREDICTION_TABLE_SCHEMA,
    )
    res = evaluate(predictions, split="val", **synthetic_phase0)
    # 2 hits among 4 positives
    assert res.metrics["hit_rate_n3"] == pytest.approx(2 / 4)
    assert res.metrics["hit_rate_n4"] == pytest.approx(2 / 4)
    # fraction_localized = 2 / 4 (only id=3 and id=4 got predictions)
    assert res.metrics["fraction_localized"] == pytest.approx(2 / 4)


def test_evaluate_emits_wavelength_slice(synthetic_phase0):
    """The val split has only red dives → expect a `wavelength_red/...` slice."""
    predictions = pl.DataFrame(
        [
            _pred(1, (100.0, 100.0), conf=0.9),
            _pred(2, (200.0, 200.0), conf=0.9),
            _pred(3, (300.0, 300.0), conf=0.9),
            _pred(4, (400.0, 400.0), conf=0.9),
            _pred(5, None, conf=0.1),
        ],
        schema=PREDICTION_TABLE_SCHEMA,
    )
    res = evaluate(predictions, split="val", **synthetic_phase0)
    assert "wavelength_red/hit_rate_n3" in res.metrics
    assert res.metrics["wavelength_red/hit_rate_n3"] == 1.0
