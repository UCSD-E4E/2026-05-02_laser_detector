"""Smoke tests for the parts of Phase 0 that don't need API access.

These exercise the logic on synthetic data so we catch regressions in line
fitting and splits without depending on the fishsense API.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from laser_detector.preprocessing.ingest import FRAME_TABLE_SCHEMA
from laser_detector.preprocessing.line_fit import (
    LINE_CONFIDENCE_THRESHOLD,
    fit_dive_line,
    fit_lines_per_dive,
    flag_label_outliers,
)
from laser_detector.preprocessing.splits import make_dive_splits
from laser_detector.preprocessing.wavelength import _wavelength_from_labels


def _make_synthetic_frames(
    n_dives: int,
    points_per_dive: int,
    spread: float,
    seed: int,
) -> pl.DataFrame:
    """Build a frames table where each dive's positives lie on a line."""
    rng = np.random.default_rng(seed)
    rows = []
    for dive_id in range(n_dives):
        # Random line direction; points spread along it
        theta = rng.uniform(0, np.pi)
        direction = np.array([np.cos(theta), np.sin(theta)])
        normal = np.array([-direction[1], direction[0]])
        midpoint = rng.uniform(500, 3500, size=2)
        ts = rng.uniform(-spread, spread, size=points_per_dive)
        # Small perpendicular noise (label imprecision)
        perp_noise = rng.normal(0, 1.0, size=points_per_dive)
        for i, t in enumerate(ts):
            xy = midpoint + t * direction + perp_noise[i] * normal
            rows.append(
                {
                    "dive_id": dive_id,
                    "image_id": dive_id * 10000 + i,
                    "rig_id": 1,
                    "image_path": f"dive{dive_id}/img{i}.jpg",
                    "image_checksum": f"sum{dive_id}_{i}",
                    "label_x": float(xy[0]),
                    "label_y": float(xy[1]),
                    "is_positive": True,
                    "label_string": None,
                    "label_studio_task_id": None,
                    "label_studio_project_id": None,
                    "superseded": False,
                    "completed": True,
                }
            )
    return pl.DataFrame(rows, schema=FRAME_TABLE_SCHEMA)


def test_fit_dive_line_recovers_simple_line():
    rng = np.random.default_rng(0)
    # y = 2x + 100 with small noise
    xs = rng.uniform(0, 1000, size=40)
    ys = 2.0 * xs + 100 + rng.normal(0, 0.5, size=40)
    fit = fit_dive_line(np.column_stack([xs, ys]), rng=rng)
    assert fit is not None
    # Verify points lie within RANSAC tolerance
    dist = fit.perpendicular_distance(xs, ys)
    assert dist.max() < 4.0
    assert fit.line_confidence > LINE_CONFIDENCE_THRESHOLD


def test_fit_lines_per_dive_marks_confidence():
    frames = _make_synthetic_frames(n_dives=5, points_per_dive=30, spread=400, seed=1)
    lines = fit_lines_per_dive(frames, seed=1)
    assert lines.height == 5
    # All synthetic dives have wide spread, so all should be confident.
    assert lines.filter(pl.col("is_line_confident")).height == 5


def test_fit_lines_per_dive_flags_low_confidence_for_clustered_points():
    # Synthetic dive whose points cluster in a tiny region — line direction is
    # ambiguous, confidence should be low.
    frames = _make_synthetic_frames(n_dives=1, points_per_dive=30, spread=2.0, seed=2)
    lines = fit_lines_per_dive(frames, seed=2)
    assert lines.filter(pl.col("is_line_confident")).height == 0


def test_flag_label_outliers_mad_floor_protects_tiny_dives():
    """Small-N dives with sub-pixel-tight inliers used to have every label
    flagged because MAD collapsed to ~0 and `3 * MAD` was below realistic label
    precision. The MAD floor (default 1.0 px) prevents that collapse: a 0.5 px
    near-miss is kept, a 10 px outlier is still flagged."""
    rows = []
    # 6 perfectly-colinear points (MAD = 0)
    for i, t in enumerate(np.linspace(-300, 300, 6)):
        rows.append({
            "dive_id": 0, "image_id": i, "rig_id": 1,
            "image_path": f"d/{i}.jpg", "image_checksum": f"s{i}",
            "label_x": float(1000.0 + t), "label_y": 1000.0,
            "is_positive": True, "label_string": None,
            "label_studio_task_id": None, "label_studio_project_id": None,
            "superseded": False, "completed": True,
        })
    # A near-miss at 0.5 px — well within the floor (3 * 1 = 3 px threshold)
    rows.append({
        "dive_id": 0, "image_id": 100, "rig_id": 1,
        "image_path": "d/near.jpg", "image_checksum": "near",
        "label_x": 1100.0, "label_y": 1000.5,
        "is_positive": True, "label_string": None,
        "label_studio_task_id": None, "label_studio_project_id": None,
        "superseded": False, "completed": True,
    })
    # A clear outlier at 10 px — must still be flagged
    rows.append({
        "dive_id": 0, "image_id": 101, "rig_id": 1,
        "image_path": "d/far.jpg", "image_checksum": "far",
        "label_x": 1200.0, "label_y": 1010.0,
        "is_positive": True, "label_string": None,
        "label_studio_task_id": None, "label_studio_project_id": None,
        "superseded": False, "completed": True,
    })
    frames = pl.DataFrame(rows, schema=FRAME_TABLE_SCHEMA)
    lines = fit_lines_per_dive(frames, seed=0)
    flagged = flag_label_outliers(frames, lines)
    flagged_ids = set(
        flagged.filter(pl.col("label_is_outlier"))["image_checksum"].to_list()
    )
    assert flagged_ids == {"far"}, f"expected only the 10px outlier flagged, got {flagged_ids}"


def test_flag_label_outliers_marks_off_line_points():
    frames = _make_synthetic_frames(n_dives=2, points_per_dive=30, spread=400, seed=3)
    # Inject one obvious outlier per dive
    poison_rows = []
    for dive_id in (0, 1):
        poison_rows.append(
            {
                "dive_id": dive_id,
                "image_id": dive_id * 10000 + 9999,
                "rig_id": 1,
                "image_path": f"dive{dive_id}/poison.jpg",
                "image_checksum": "poison",
                "label_x": 100.0,
                "label_y": 100.0,  # almost certainly far off
                "is_positive": True,
                "label_string": None,
                "label_studio_task_id": None,
                "label_studio_project_id": None,
                "superseded": False,
                "completed": True,
            }
        )
    frames = pl.concat(
        [frames, pl.DataFrame(poison_rows, schema=FRAME_TABLE_SCHEMA)]
    )
    lines = fit_lines_per_dive(frames, seed=3)
    flagged = flag_label_outliers(frames, lines)
    # The poisoned rows are far from the line; expect them flagged.
    poisons = flagged.filter(pl.col("image_checksum") == "poison")
    assert poisons.filter(pl.col("label_is_outlier")).height == 2


def test_make_dive_splits_stratifies_by_wavelength():
    wavelengths = pl.DataFrame(
        {
            "dive_id": list(range(20)),
            "wavelength": ["green"] * 10 + ["blue"] * 10,
            "wavelength_source": ["color_cluster"] * 20,
            "dive_color_b": [0.0] * 20,
            "dive_color_g": [0.0] * 20,
            "dive_color_r": [0.0] * 20,
        }
    )
    splits = make_dive_splits(wavelengths, train_frac=0.8, val_frac=0.1, seed=0)
    assert splits.height == 20
    counts = splits.group_by("split").len().to_dict(as_series=False)
    counts = dict(zip(counts["split"], counts["len"]))
    assert counts.get("train", 0) == 16
    assert counts.get("val", 0) == 2
    assert counts.get("test", 0) == 2

    # Each split contains both wavelengths
    joined = splits.join(wavelengths.select("dive_id", "wavelength"), on="dive_id")
    for split_name in ("train", "val", "test"):
        wls = set(
            joined.filter(pl.col("split") == split_name)["wavelength"].to_list()
        )
        assert wls == {"green", "blue"}


def test_wavelength_from_labels_recognizes_color_words():
    # Single-color cases (the common path on the v1 corpus)
    assert _wavelength_from_labels(["Red Laser", "Red Laser", None]) == "red"
    assert _wavelength_from_labels(["laser-green", "green laser", None]) == "green"
    assert _wavelength_from_labels(["blue laser", "Blue", "blue"]) == "blue"
    # Mixed-color dive: majority wins (annotator-slip pattern is real)
    assert _wavelength_from_labels(["Red Laser"] * 146 + ["Green Laser"] * 2) == "red"
    assert _wavelength_from_labels(["Green Laser"] * 8 + ["Red Laser"] * 210) == "red"
    # Tied majority → caller falls back to clustering
    assert _wavelength_from_labels(["red", "green"]) is None
    # No color words present → caller falls back to clustering
    assert _wavelength_from_labels([None, None, "no laser"]) is None
