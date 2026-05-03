"""Tests for the Phase 1 baseline detector.

Synthesize tiny BGR images with a known bright spot of a known color and
verify `detect_in_frame` recovers the expected centroid (or returns
no-detection when the spot doesn't match the wavelength).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from laser_detector.baseline import (
    LINE_PROXIMITY_SIGMA_PX,
    Detection,
    DiveInfo,
    detect_in_frame,
)


def _frame_with_dot(
    color_bgr: tuple[int, int, int], xy: tuple[int, int], radius: int = 3
) -> np.ndarray:
    """A 200x200 dark-water-ish background with one bright filled circle."""
    img = np.full((200, 200, 3), (40, 30, 20), dtype=np.uint8)  # dark teal
    cv2.circle(img, xy, radius, color_bgr, thickness=-1)
    return img


def _no_line_dive(wavelength: str | None) -> DiveInfo:
    return DiveInfo(
        wavelength=wavelength,
        line_a=None,
        line_b=None,
        line_c=None,
        is_line_confident=False,
    )


def test_detects_bright_red_dot():
    img = _frame_with_dot(color_bgr=(0, 0, 255), xy=(120, 80))
    det = detect_in_frame(img, _no_line_dive("red"))
    assert det.pred_x == pytest.approx(120.0, abs=1.0)
    assert det.pred_y == pytest.approx(80.0, abs=1.0)
    assert det.pred_confidence > 0.5


def test_detects_bright_green_dot():
    img = _frame_with_dot(color_bgr=(0, 255, 0), xy=(40, 150))
    det = detect_in_frame(img, _no_line_dive("green"))
    assert det.pred_x == pytest.approx(40.0, abs=1.0)
    assert det.pred_y == pytest.approx(150.0, abs=1.0)


def test_wrong_wavelength_returns_no_detection():
    """A red dot in a dive tagged 'green' shouldn't be picked up."""
    img = _frame_with_dot(color_bgr=(0, 0, 255), xy=(100, 100))
    det = detect_in_frame(img, _no_line_dive("green"))
    assert det.pred_x is None
    assert det.pred_y is None
    assert det.pred_confidence == 0.0


def test_unknown_wavelength_accepts_either():
    """Wavelength=None should match red OR green."""
    img = _frame_with_dot(color_bgr=(0, 0, 255), xy=(100, 100))
    det = detect_in_frame(img, _no_line_dive(None))
    assert det.pred_x is not None


def test_dark_image_returns_no_detection():
    img = np.full((200, 200, 3), (40, 30, 20), dtype=np.uint8)
    det = detect_in_frame(img, _no_line_dive("red"))
    assert det.pred_x is None


def test_blob_size_below_floor_is_rejected():
    """A 1-pixel-radius dot has area ~5 px — borderline. A 0-radius has area ~1 → rejected."""
    img = np.full((200, 200, 3), (40, 30, 20), dtype=np.uint8)
    img[100, 100] = (0, 0, 255)  # single pixel
    det = detect_in_frame(img, _no_line_dive("red"))
    assert det.pred_x is None


def test_blob_size_above_ceiling_is_rejected():
    """A 50-radius (area ~7854 px²) blob exceeds the 1000-px max."""
    img = np.full((200, 200, 3), (40, 30, 20), dtype=np.uint8)
    cv2.circle(img, (100, 100), 50, (0, 0, 255), thickness=-1)
    det = detect_in_frame(img, _no_line_dive("red"))
    assert det.pred_x is None


def test_picks_blob_closer_to_line_when_tied_on_brightness():
    """Two equally bright dots; only the one near the line should win."""
    img = np.full((400, 400, 3), (40, 30, 20), dtype=np.uint8)
    cv2.circle(img, (100, 100), 4, (0, 0, 255), thickness=-1)  # off-line
    cv2.circle(img, (100, 300), 4, (0, 0, 255), thickness=-1)  # on the line y=300
    # Line: y = 300 → 0*x + 1*y - 300 = 0
    dive = DiveInfo(
        wavelength="red",
        line_a=0.0,
        line_b=1.0,
        line_c=-300.0,
        is_line_confident=True,
    )
    det = detect_in_frame(img, dive)
    # The on-line dot at (100, 300) should be picked over (100, 100).
    assert det.pred_y == pytest.approx(300.0, abs=1.0)


def test_brightest_wins_when_no_line_constraint():
    """Without a confident line, brightness alone decides."""
    img = np.full((400, 400, 3), (40, 30, 20), dtype=np.uint8)
    cv2.circle(img, (100, 100), 4, (0, 0, 200), thickness=-1)  # dimmer red
    cv2.circle(img, (300, 300), 4, (0, 0, 255), thickness=-1)  # brighter red
    det = detect_in_frame(img, _no_line_dive("red"))
    assert det.pred_x == pytest.approx(300.0, abs=1.0)
    assert det.pred_y == pytest.approx(300.0, abs=1.0)
