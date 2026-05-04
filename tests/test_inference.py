"""Tests for tiled frame-level inference (Phase 2).

Tile-grid math is plain arithmetic; `predict_frame` runs an end-to-end forward
on a small synthetic image with a randomly-initialized model so we can confirm
the pipeline produces a finite (x, y, confidence) without NaNs or shape errors.
"""

from __future__ import annotations

import numpy as np
import torch

from laser_detector.inference import (
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    compute_tile_grid,
    predict_frame,
)
from laser_detector.model import LaserDetector


def test_tile_grid_for_4k_frame():
    grid = compute_tile_grid(2160, 3840)
    # 5 columns × 3 rows = 15 tiles per DESIGN.md §4.1
    assert len(grid.origins) == 15
    # First tile at origin, last tile snaps to the bottom-right edge.
    assert (0, 0) in grid.origins
    assert (3840 - DEFAULT_TILE_SIZE, 2160 - DEFAULT_TILE_SIZE) in grid.origins
    assert grid.original_h == 2160
    assert grid.original_w == 3840


def test_tile_grid_for_image_smaller_than_tile():
    grid = compute_tile_grid(800, 600)
    assert grid.origins == [(0, 0)]
    assert grid.padded_h == DEFAULT_TILE_SIZE
    assert grid.padded_w == DEFAULT_TILE_SIZE


def test_tile_grid_overlap_respects_stride():
    grid = compute_tile_grid(2160, 3840, overlap=DEFAULT_TILE_OVERLAP)
    xs = sorted({x for x, _ in grid.origins})
    stride = DEFAULT_TILE_SIZE - DEFAULT_TILE_OVERLAP
    # Interior tiles step by `stride`; the final one snaps to the edge so the
    # last gap may be smaller (we just check it isn't *larger* than `stride`).
    diffs = np.diff(xs)
    assert (diffs <= stride).all()


def test_predict_frame_returns_finite_xy_in_bounds():
    model = LaserDetector(encoder_weights=None).eval()
    # Small image — 1 tile after reflect-padding. Keeps the test fast (<5s on CPU).
    image = np.random.default_rng(0).integers(0, 255, size=(800, 600, 3), dtype=np.uint8)
    pred = predict_frame(
        image, model,
        wavelength="red",
        device=torch.device("cpu"),
        tile=DEFAULT_TILE_SIZE,
        overlap=DEFAULT_TILE_OVERLAP,
        autocast_dtype=None,
    )
    assert pred.pred_x is not None and pred.pred_y is not None
    # Padded-margin clamp keeps predictions inside the original frame.
    assert 0 <= pred.pred_x <= 599
    assert 0 <= pred.pred_y <= 799
    assert 0.0 <= pred.pred_confidence <= 1.0


def test_predict_frame_presence_threshold_returns_none_xy():
    """A high threshold suppresses the prediction even when the heatmap has a peak."""
    model = LaserDetector(encoder_weights=None).eval()
    image = np.random.default_rng(0).integers(0, 255, size=(800, 600, 3), dtype=np.uint8)
    pred = predict_frame(
        image, model,
        wavelength="green",
        device=torch.device("cpu"),
        autocast_dtype=None,
        presence_threshold=2.0,  # impossible to exceed; sigmoid ≤ 1
    )
    assert pred.pred_x is None
    assert pred.pred_y is None
