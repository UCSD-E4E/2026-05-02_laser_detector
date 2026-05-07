"""Tests for tiled frame-level inference (Phase 2).

Tile-grid math is plain arithmetic; `predict_frame` runs an end-to-end forward
on a small synthetic image with a randomly-initialized model so we can confirm
the pipeline produces a finite (x, y, confidence) without NaNs or shape errors.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from laser_detector.inference import (
    DEFAULT_RIG_PRIOR_BBOX,
    DEFAULT_RIG_PRIOR_CENTER,
    DEFAULT_RIG_PRIOR_FLOOR,
    DEFAULT_RIG_PRIOR_SIGMA,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    _project_point_onto_line,
    _rig_prior_for_tile,
    compute_tile_grid,
    predict_frame,
    predict_frame_with_cascade,
    rig_prior_log_mask,
    rig_prior_log_mask_batched,
    soft_snap_to_line,
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


def test_project_point_onto_horizontal_line():
    # Line y = 100 → 0*x + 1*y - 100 = 0; project (50, 200) → (50, 100).
    px, py = _project_point_onto_line(50.0, 200.0, 0.0, 1.0, -100.0)
    assert px == 50.0
    assert py == 100.0


def test_project_point_already_on_line_is_identity():
    # Line y = x → 1/√2 * x - 1/√2 * y = 0; (3, 3) is on the line.
    s = 1.0 / np.sqrt(2)
    px, py = _project_point_onto_line(3.0, 3.0, s, -s, 0.0)
    assert abs(px - 3.0) < 1e-9
    assert abs(py - 3.0) < 1e-9


def test_soft_snap_high_pred_confidence_skips_snap():
    """High prediction confidence + low line confidence → α ≈ 0, no snap."""
    fx, fy, alpha = soft_snap_to_line(
        50.0, 200.0,
        line_abc=(0.0, 1.0, -100.0),
        line_confidence=2.0,  # below tau=5 → low strength
        pred_confidence=0.95,
        tau_line=5.0, alpha_max=0.3,
    )
    assert alpha < 0.05
    assert abs(fx - 50.0) < 5.0
    assert abs(fy - 200.0) < 5.0


def test_soft_snap_low_pred_high_line_pulls_toward_line():
    """Low pred + high line confidence → significant pull toward line."""
    fx, fy, alpha = soft_snap_to_line(
        50.0, 200.0,
        line_abc=(0.0, 1.0, -100.0),  # y = 100
        line_confidence=20.0,  # well above tau=5 → strength ≈ 1
        pred_confidence=0.1,   # low → (1 - p) ≈ 0.9 → α capped at 0.3
        tau_line=5.0, alpha_max=0.3,
    )
    assert alpha == pytest.approx(0.3, abs=0.01)
    assert fx == 50.0  # x doesn't change for a horizontal line
    # 30% pull from y=200 toward y=100 → expected y = 200 - 0.3*100 = 170
    assert abs(fy - 170.0) < 0.5


def test_soft_snap_alpha_capped_at_alpha_max():
    """Even with maximal line + zero pred confidence, α can't exceed alpha_max."""
    _, _, alpha = soft_snap_to_line(
        0.0, 1000.0,
        line_abc=(0.0, 1.0, 0.0),
        line_confidence=100.0,
        pred_confidence=0.0,
        tau_line=5.0, alpha_max=0.25,
    )
    assert alpha <= 0.25 + 1e-9


def test_predict_frame_soft_snap_pulls_into_line_band():
    """End-to-end: passing line params to predict_frame should pull the
    prediction toward the line when confidence numbers favor a snap."""
    model = LaserDetector(encoder_weights=None).eval()
    image = np.random.default_rng(0).integers(0, 255, size=(800, 600, 3), dtype=np.uint8)
    no_snap = predict_frame(
        image, model,
        wavelength="red",
        device=torch.device("cpu"),
        autocast_dtype=None,
    )
    snapped = predict_frame(
        image, model,
        wavelength="red",
        device=torch.device("cpu"),
        autocast_dtype=None,
        line_abc=(0.0, 1.0, -100.0),  # y = 100 in frame coords
        line_confidence=20.0,
        tau_line=5.0, alpha_max=0.3,
    )
    # Soft-snap shouldn't ever NaN out, and y should move toward 100.
    assert snapped.pred_x is not None and snapped.pred_y is not None
    assert no_snap.pred_x is not None and no_snap.pred_y is not None
    # The snap moves y closer to 100. With low pred_confidence (random init),
    # alpha will be near alpha_max=0.3, so |y_snap - y_no_snap| should be ~30%
    # of the original distance to the line.
    no_snap_dist = abs(no_snap.pred_y - 100.0)
    snap_dist = abs(snapped.pred_y - 100.0)
    assert snap_dist <= no_snap_dist + 1e-6  # never overshoots


def test_cascade_returns_no_detection_when_coarse_does():
    """If coarse pass returns None xy (presence below threshold), cascade does too."""
    model = LaserDetector(encoder_weights=None).eval()
    image = np.random.default_rng(0).integers(0, 255, size=(800, 600, 3), dtype=np.uint8)
    pred = predict_frame_with_cascade(
        image, model,
        wavelength="red",
        device=torch.device("cpu"),
        autocast_dtype=None,
        presence_threshold=2.0,  # impossible to exceed
    )
    assert pred.pred_x is None and pred.pred_y is None


def test_cascade_returns_in_bounds_pred():
    model = LaserDetector(encoder_weights=None).eval()
    image = np.random.default_rng(1).integers(0, 255, size=(800, 600, 3), dtype=np.uint8)
    pred = predict_frame_with_cascade(
        image, model,
        wavelength="green",
        device=torch.device("cpu"),
        autocast_dtype=None,
        refine_window=256,
    )
    assert pred.pred_x is not None and pred.pred_y is not None
    assert 0 <= pred.pred_x <= 599
    assert 0 <= pred.pred_y <= 799


def test_rig_prior_zeroes_outside_bbox():
    """Pixels outside the rig-prior bbox should mask to 0."""
    bbox = (1400, 0, 3000, 2200)
    center = (2000.0, 1300.0)
    sigma = (200.0, 300.0)
    floor = 0.1
    # Tile at top-left corner of the canvas — entirely outside the bbox in x.
    mask = _rig_prior_for_tile(0, 0, 1024, bbox, center, sigma, floor)
    assert mask.shape == (1024, 1024)
    assert (mask == 0.0).all()


def test_rig_prior_peaks_near_center():
    """A tile centered on the prior center should peak at ~1.0 at the center."""
    bbox = DEFAULT_RIG_PRIOR_BBOX
    center = DEFAULT_RIG_PRIOR_CENTER
    sigma = DEFAULT_RIG_PRIOR_SIGMA
    # Place tile so its top-left puts the prior center in its interior.
    cx, cy = center
    tile_origin_x = int(cx - 256)
    tile_origin_y = int(cy - 256)
    mask = _rig_prior_for_tile(
        tile_origin_x, tile_origin_y, 1024, bbox, center, sigma, DEFAULT_RIG_PRIOR_FLOOR,
    )
    # The pixel at (cx, cy) in tile-local coords lives at row=cy-tile_origin_y, col=cx-tile_origin_x.
    local_row = int(cy) - tile_origin_y
    local_col = int(cx) - tile_origin_x
    assert mask[local_row, local_col] == pytest.approx(1.0, abs=1e-3)


def test_rig_prior_floors_inside_bbox():
    """Inside the bbox but far from the Gaussian center, the mask is the floor."""
    bbox = (0, 0, 4000, 2200)  # encompasses the full canvas in x
    center = (0.0, 0.0)         # far from any test point
    sigma = (10.0, 10.0)        # tight Gaussian → exponential ≈ 0 far away
    floor = 0.15
    mask = _rig_prior_for_tile(2000, 1000, 1024, bbox, center, sigma, floor)
    assert mask.min() == pytest.approx(floor, abs=1e-6)
    assert mask.max() <= 1.0


def test_predict_frame_rig_prior_keeps_pred_in_bbox():
    """Argmax should land inside the prior bbox when rig_prior=True."""
    model = LaserDetector(encoder_weights=None).eval()
    image = np.random.default_rng(0).integers(0, 255, size=(2160, 3840, 3), dtype=np.uint8)
    pred = predict_frame(
        image, model,
        wavelength="red",
        device=torch.device("cpu"),
        autocast_dtype=None,
        rig_prior=True,
    )
    assert pred.pred_x is not None and pred.pred_y is not None
    bx0, by0, bx1, by1 = DEFAULT_RIG_PRIOR_BBOX
    assert bx0 <= pred.pred_x < bx1
    assert by0 <= pred.pred_y < by1


def test_rig_prior_log_mask_batched_matches_numpy_version():
    """Batched torch implementation should agree with the numpy single-tile version."""
    crop_offsets = torch.tensor([[1500, 800], [0, 0], [2000, 1300]], dtype=torch.float32)
    tile = 256
    batched = rig_prior_log_mask_batched(crop_offsets, tile)
    assert batched.shape == (3, tile, tile)
    for i, (ox, oy) in enumerate(crop_offsets.tolist()):
        ref = rig_prior_log_mask(int(ox), int(oy), tile)
        np.testing.assert_allclose(batched[i].cpu().numpy(), ref, atol=1e-5)


def test_rig_prior_log_mask_batched_floors_outside_bbox():
    """Outside the bbox the log-mask should hit log(eps) ≈ -9.2."""
    crop_offsets = torch.tensor([[0, 0]], dtype=torch.float32)  # tile entirely outside bbox in x
    tile = 128
    log_mask = rig_prior_log_mask_batched(crop_offsets, tile, eps=1e-4)
    assert log_mask.max() == pytest.approx(np.log(1e-4), abs=1e-5)
