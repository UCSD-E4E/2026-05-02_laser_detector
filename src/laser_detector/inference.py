"""Frame-level tiled inference for the Phase 2 detector.

Per DESIGN.md §6: for each 4K frame, run the model on a grid of overlapping
1024×1024 tiles, take the heatmap-max location across tiles as the predicted
xy, and take the max of per-tile presence sigmoids as the frame-level confidence.

Line-aware tile selection (DESIGN.md §4.1) and line-snap refinement (§6.2) are
Phase 3+ — Phase 2 always runs all tiles and reports the raw argmax.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from laser_detector.data import (
    DEFAULT_TILE_SIZE,
    UNKNOWN_WAVELENGTH_CHANNEL,
    WAVELENGTH_CHANNEL,
    _chromaticity_norm,
)

logger = logging.getLogger(__name__)

DEFAULT_TILE_OVERLAP = 256


@dataclass(frozen=True)
class TileGrid:
    """Tile placement for one image. Origins are pixel offsets into the (possibly
    reflect-padded) image."""

    origins: list[tuple[int, int]]  # (x, y) per tile
    padded_h: int
    padded_w: int
    original_h: int
    original_w: int


def compute_tile_grid(
    h: int, w: int, *, tile: int = DEFAULT_TILE_SIZE, overlap: int = DEFAULT_TILE_OVERLAP
) -> TileGrid:
    """Tile (x, y) origins covering an `h`×`w` image with `overlap` between tiles.

    The last tile in each axis snaps to the image edge (so it may overlap the
    previous tile by more than `overlap`). If the image is smaller than `tile`
    in some axis, the grid has one tile and reflect-padding fills it.
    """
    stride = tile - overlap
    if w <= tile:
        xs = [0]
        padded_w = max(w, tile)
    else:
        xs = list(range(0, w - tile, stride)) + [w - tile]
    if h <= tile:
        ys = [0]
        padded_h = max(h, tile)
    else:
        ys = list(range(0, h - tile, stride)) + [h - tile]
    padded_h = max(h, tile)
    padded_w = max(w, tile)
    origins = [(x, y) for y in ys for x in xs]
    return TileGrid(
        origins=origins,
        padded_h=padded_h,
        padded_w=padded_w,
        original_h=h,
        original_w=w,
    )


def _reflect_pad(image_bgr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Pad to (h, w) with reflection if smaller; otherwise return as-is."""
    src_h, src_w = image_bgr.shape[:2]
    pad_h = max(h - src_h, 0)
    pad_w = max(w - src_w, 0)
    if pad_h == 0 and pad_w == 0:
        return image_bgr
    return cv2.copyMakeBorder(
        image_bgr, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101
    )


def _preprocess_tile(
    tile_bgr: np.ndarray, wavelength_value: float
) -> np.ndarray:
    """BGR uint8 → float32 [4, H, W] (chromaticity + wavelength)."""
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    chrom = _chromaticity_norm(rgb)
    h, w = chrom.shape[:2]
    wavelength_channel = np.full((h, w, 1), wavelength_value, dtype=np.float32)
    x = np.concatenate([chrom, wavelength_channel], axis=2)
    return np.transpose(x, (2, 0, 1)).copy()


@dataclass(frozen=True)
class FramePrediction:
    pred_x: float | None
    pred_y: float | None
    pred_confidence: float


@torch.inference_mode()
def predict_frame(
    image_bgr: np.ndarray,
    model: torch.nn.Module,
    *,
    wavelength: str | None,
    device: torch.device,
    tile: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_TILE_OVERLAP,
    batch_size: int = 8,
    presence_threshold: float | None = None,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
) -> FramePrediction:
    """Run tiled inference on a single 4K frame.

    Returns the (pred_x, pred_y) of the heatmap maximum across all tiles, or
    (None, None) if `presence_threshold` is set and the frame-level confidence
    falls below it.

    Frame-level confidence = max(sigmoid(tile_presence_logits)) per DESIGN.md §6.
    """
    grid = compute_tile_grid(*image_bgr.shape[:2], tile=tile, overlap=overlap)
    padded = _reflect_pad(image_bgr, grid.padded_h, grid.padded_w)

    wavelength_value = (
        WAVELENGTH_CHANNEL.get(wavelength, UNKNOWN_WAVELENGTH_CHANNEL)
        if wavelength is not None
        else UNKNOWN_WAVELENGTH_CHANNEL
    )

    tile_arrays = [
        _preprocess_tile(padded[y : y + tile, x : x + tile], wavelength_value)
        for x, y in grid.origins
    ]
    tile_batch = torch.from_numpy(np.stack(tile_arrays))  # [N, 4, H, W]

    best_value = -1.0
    best_xy = (None, None)
    presence_max = 0.0

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None and device.type == "cuda"
        else _NullCtx()
    )

    for chunk_start in range(0, len(tile_arrays), batch_size):
        chunk = tile_batch[chunk_start : chunk_start + batch_size].to(device, non_blocking=True)
        with autocast_ctx:
            out = model(chunk)
        heatmap_probs = torch.sigmoid(out["heatmap_logits"]).float()
        presence_probs = torch.sigmoid(out["presence_logits"]).float()

        flat = heatmap_probs.view(heatmap_probs.shape[0], -1)
        max_vals, max_idx = flat.max(dim=1)

        for i, (mv, mi) in enumerate(zip(max_vals.tolist(), max_idx.tolist())):
            if mv > best_value:
                best_value = mv
                local_y, local_x = divmod(mi, tile)
                ox, oy = grid.origins[chunk_start + i]
                best_xy = (float(local_x + ox), float(local_y + oy))
        presence_max = max(presence_max, float(presence_probs.max().item()))

    if presence_threshold is not None and presence_max < presence_threshold:
        return FramePrediction(pred_x=None, pred_y=None, pred_confidence=presence_max)

    pred_x, pred_y = best_xy
    if pred_x is not None:
        # Don't report predictions inside the reflect-padded margin.
        pred_x = min(pred_x, float(grid.original_w - 1))
        pred_y = min(pred_y, float(grid.original_h - 1))
    return FramePrediction(
        pred_x=pred_x, pred_y=pred_y, pred_confidence=presence_max
    )


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
