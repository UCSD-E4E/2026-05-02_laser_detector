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

# Static rig prior on laser position in original-frame pixels (4K canvas).
# Empirically derived from all positive train labels (median ≈ (1977, 1342),
# tight diagonal stripe around the vanishing point). The hard bbox is a
# generous superset that survives slight rig changes; y_min=0 because the
# laser can reach the top edge at effective distance. Within the bbox, a soft
# Gaussian centered at the vanishing point biases argmax toward where the
# laser is *most likely* to land. Outside the bbox → 0 (hard reject).
DEFAULT_RIG_PRIOR_BBOX: tuple[int, int, int, int] = (1400, 0, 3000, 2200)
DEFAULT_RIG_PRIOR_CENTER: tuple[float, float] = (1977.0, 1342.0)
DEFAULT_RIG_PRIOR_SIGMA: tuple[float, float] = (200.0, 300.0)
# Floor inside the bbox so a strong heatmap response in a low-prior region
# isn't zeroed out — the prior nudges, doesn't overrule.
DEFAULT_RIG_PRIOR_FLOOR: float = 0.1


def _rig_prior_for_tile(
    tile_origin_x: int,
    tile_origin_y: int,
    tile: int,
    bbox: tuple[int, int, int, int],
    center: tuple[float, float],
    sigma: tuple[float, float],
    floor: float,
) -> np.ndarray:
    """Build a `[tile, tile]` float32 mask in [0, 1] for a single tile.

    Pixels outside the global bbox → 0 (hard reject).
    Pixels inside → `max(floor, exp(-((dx/σx)² + (dy/σy)²)/2))` (soft Gaussian,
    bounded below so a confident heatmap pixel inside the bbox is never erased).
    """
    ys, xs = np.mgrid[
        tile_origin_y : tile_origin_y + tile,
        tile_origin_x : tile_origin_x + tile,
    ].astype(np.float32)

    bx0, by0, bx1, by1 = bbox
    in_bbox = (xs >= bx0) & (xs < bx1) & (ys >= by0) & (ys < by1)

    cx, cy = center
    sx, sy = sigma
    gauss = np.exp(-0.5 * (((xs - cx) / sx) ** 2 + ((ys - cy) / sy) ** 2))
    soft = np.maximum(gauss, floor).astype(np.float32)

    mask = np.where(in_bbox, soft, 0.0).astype(np.float32)
    return mask


def rig_prior_log_mask(
    tile_origin_x: int,
    tile_origin_y: int,
    tile: int,
    bbox: tuple[int, int, int, int] = DEFAULT_RIG_PRIOR_BBOX,
    center: tuple[float, float] = DEFAULT_RIG_PRIOR_CENTER,
    sigma: tuple[float, float] = DEFAULT_RIG_PRIOR_SIGMA,
    floor: float = DEFAULT_RIG_PRIOR_FLOOR,
    eps: float = 1e-4,
) -> np.ndarray:
    """`[tile, tile]` float32 log-mask suitable for adding to heatmap logits.

    Returns `log(max(rig_prior_mask, eps))`. Outside the bbox the mask is 0,
    so the log floors at `log(eps) ≈ -9.2` — a strong but bounded suppression
    that keeps BCE-with-logits stable. Inside the bbox the log is in
    `[log(floor), 0]` and gently biases logits toward the high-density region.
    """
    mask = _rig_prior_for_tile(
        tile_origin_x, tile_origin_y, tile, bbox, center, sigma, floor,
    )
    return np.log(np.maximum(mask, eps)).astype(np.float32)


def rig_prior_log_mask_batched(
    crop_offsets: "torch.Tensor",
    tile: int,
    bbox: tuple[int, int, int, int] = DEFAULT_RIG_PRIOR_BBOX,
    center: tuple[float, float] = DEFAULT_RIG_PRIOR_CENTER,
    sigma: tuple[float, float] = DEFAULT_RIG_PRIOR_SIGMA,
    floor: float = DEFAULT_RIG_PRIOR_FLOOR,
    eps: float = 1e-4,
) -> "torch.Tensor":
    """Batched torch version: `[B, tile, tile]` log-mask given `[B, 2]` offsets.

    Used in training to add a static rig prior to the heatmap logits before
    BCE. Computed on whatever device `crop_offsets` lives on, no host sync.
    """
    B = crop_offsets.shape[0]
    device = crop_offsets.device
    dtype = torch.float32

    cx, cy = center
    sx, sy = sigma
    bx0, by0, bx1, by1 = bbox

    # Pixel coords inside each tile, in frame space.
    grid = torch.arange(tile, device=device, dtype=dtype)  # [tile]
    ox = crop_offsets[:, 0].to(dtype).unsqueeze(1) + grid.unsqueeze(0)  # [B, tile]
    oy = crop_offsets[:, 1].to(dtype).unsqueeze(1) + grid.unsqueeze(0)  # [B, tile]

    in_bbox_x = (ox >= bx0) & (ox < bx1)  # [B, tile]
    in_bbox_y = (oy >= by0) & (oy < by1)
    in_bbox = in_bbox_y.unsqueeze(2) & in_bbox_x.unsqueeze(1)  # [B, tile, tile]

    dx = (ox - cx) / sx  # [B, tile]
    dy = (oy - cy) / sy
    quad = 0.5 * (dy.unsqueeze(2) ** 2 + dx.unsqueeze(1) ** 2)  # [B, tile, tile]
    gauss = torch.exp(-quad)
    soft = torch.maximum(gauss, torch.tensor(floor, device=device, dtype=dtype))

    mask = torch.where(in_bbox, soft, torch.zeros_like(soft))
    return torch.log(torch.clamp(mask, min=eps))


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


def _project_point_onto_line(
    x: float, y: float, a: float, b: float, c: float
) -> tuple[float, float]:
    """Orthogonal projection of (x, y) onto the line `a*x + b*y + c = 0`.

    With (a, b) unit-normalized (`a^2 + b^2 = 1`, as Phase 0 produces),
    the formula simplifies to `p_proj = p - (a*x + b*y + c) * (a, b)`."""
    norm_sq = a * a + b * b
    if norm_sq <= 1e-12:
        return x, y  # degenerate line — return point as-is
    t = (a * x + b * y + c) / norm_sq
    return x - t * a, y - t * b


def soft_snap_to_line(
    x: float, y: float, *,
    line_abc: tuple[float, float, float],
    line_confidence: float,
    pred_confidence: float,
    tau_line: float = 5.0,
    alpha_max: float = 0.3,
) -> tuple[float, float, float]:
    """Project (x, y) toward the dive line per DESIGN.md §6.2:

        final_xy = (1 - α) * argmax + α * project(argmax, line)

    α blends in line proximity smoothly. We want α high when the prediction
    is uncertain and the line is confident; α low when the prediction is
    already confident. Concretely:

        α_raw  = sigmoid(line_confidence - τ_line) * (1 - pred_confidence)
        α      = clip(α_raw, 0, alpha_max)

    The `(1 - pred_confidence)` factor keeps high-confidence predictions
    free to disagree with the line — useful when the line itself is wrong
    on a particular frame. `alpha_max=0.3` caps the snap so even an
    uncertain prediction won't be wholly displaced; matches the "α small,
    ≤ 0.3" guidance in DESIGN.

    Returns `(final_x, final_y, alpha)` so the caller can log α for tuning.
    """
    a, b, c = line_abc
    line_strength = 1.0 / (1.0 + np.exp(-(line_confidence - tau_line)))
    alpha_raw = float(line_strength * (1.0 - pred_confidence))
    alpha = max(0.0, min(alpha_raw, alpha_max))
    if alpha <= 0.0:
        return x, y, 0.0
    proj_x, proj_y = _project_point_onto_line(x, y, a, b, c)
    return (
        (1.0 - alpha) * x + alpha * proj_x,
        (1.0 - alpha) * y + alpha * proj_y,
        alpha,
    )


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
    line_abc: tuple[float, float, float] | None = None,
    line_confidence: float = 0.0,
    tau_line: float = 5.0,
    alpha_max: float = 0.3,
    rig_prior: bool = False,
    rig_prior_bbox: tuple[int, int, int, int] = DEFAULT_RIG_PRIOR_BBOX,
    rig_prior_center: tuple[float, float] = DEFAULT_RIG_PRIOR_CENTER,
    rig_prior_sigma: tuple[float, float] = DEFAULT_RIG_PRIOR_SIGMA,
    rig_prior_floor: float = DEFAULT_RIG_PRIOR_FLOOR,
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

    # Per-tile rig-prior mask (precomputed; same shape as the heatmap output).
    rig_masks: list[torch.Tensor] | None = None
    if rig_prior:
        rig_masks = [
            torch.from_numpy(
                _rig_prior_for_tile(
                    ox, oy, tile,
                    rig_prior_bbox, rig_prior_center, rig_prior_sigma,
                    rig_prior_floor,
                )
            )
            for (ox, oy) in grid.origins
        ]

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

        if rig_masks is not None:
            chunk_masks = torch.stack(
                [rig_masks[chunk_start + i] for i in range(heatmap_probs.shape[0])]
            ).to(heatmap_probs.device)
            # heatmap_probs is [B, 1, H, W]; unsqueeze masks to broadcast cleanly.
            heatmap_probs = heatmap_probs * chunk_masks.unsqueeze(1)

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
        # Optional soft-snap toward the dive's line (DESIGN.md §6.2).
        if line_abc is not None and line_confidence > 0.0:
            pred_x, pred_y, _alpha = soft_snap_to_line(
                pred_x, pred_y,
                line_abc=line_abc,
                line_confidence=line_confidence,
                pred_confidence=presence_max,
                tau_line=tau_line, alpha_max=alpha_max,
            )
            pred_x = max(0.0, min(pred_x, float(grid.original_w - 1)))
            pred_y = max(0.0, min(pred_y, float(grid.original_h - 1)))
    return FramePrediction(
        pred_x=pred_x, pred_y=pred_y, pred_confidence=presence_max
    )


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@torch.inference_mode()
def predict_frame_with_cascade(
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
    line_abc: tuple[float, float, float] | None = None,
    line_confidence: float = 0.0,
    tau_line: float = 5.0,
    alpha_max: float = 0.3,
    refine_window: int = 256,
) -> FramePrediction:
    """Two-pass inference (Phase 5 cascade, DESIGN.md §9 followup).

    Pass 1: same global tiled inference as `predict_frame` to find the coarse
    laser location (or "no detection").
    Pass 2: re-run the model on a single `refine_window`-sized crop centered
    on the pass-1 argmax, take its argmax, and translate back to frame coords.

    The audit on epoch_007 (2026-05-06) showed bimodal per-frame errors:
    most frames are within 1-3 px of the label, the rest are 1000+ px off.
    A meaningful fraction of the "1000+ px" cluster are cases where the
    correct tile won the argmax race but the wrong sub-pixel was selected
    *within* that tile because of a confuser blob nearby. Refining around
    the coarse argmax should rescue those.

    `refine_window` defaults to the tile size; smaller values (e.g. 128)
    focus the refinement tighter at the cost of false-localization risk.
    """
    coarse = predict_frame(
        image_bgr, model,
        wavelength=wavelength, device=device,
        tile=tile, overlap=overlap, batch_size=batch_size,
        presence_threshold=presence_threshold,
        autocast_dtype=autocast_dtype,
        line_abc=None, line_confidence=0.0,  # snap only after refinement
    )
    if coarse.pred_x is None or coarse.pred_y is None:
        return coarse

    h, w = image_bgr.shape[:2]
    half = refine_window // 2
    cx = int(round(coarse.pred_x))
    cy = int(round(coarse.pred_y))

    # Crop window, clamped to image bounds; then reflect-pad to refine_window.
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, x0 + refine_window)
    y1 = min(h, y0 + refine_window)
    x0 = max(0, x1 - refine_window)
    y0 = max(0, y1 - refine_window)
    crop = image_bgr[y0:y1, x0:x1]
    crop = _reflect_pad(crop, refine_window, refine_window)

    wavelength_value = (
        WAVELENGTH_CHANNEL.get(wavelength, UNKNOWN_WAVELENGTH_CHANNEL)
        if wavelength is not None
        else UNKNOWN_WAVELENGTH_CHANNEL
    )
    arr = _preprocess_tile(crop, wavelength_value)
    batch = torch.from_numpy(arr[None]).to(device, non_blocking=True)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None and device.type == "cuda"
        else _NullCtx()
    )
    with autocast_ctx:
        out = model(batch)
    heatmap_probs = torch.sigmoid(out["heatmap_logits"][0]).float()
    presence_prob = float(torch.sigmoid(out["presence_logits"][0]).max().item())

    flat = heatmap_probs.view(-1)
    refined_idx = int(flat.argmax().item())
    refined_value = float(flat.max().item())
    local_y, local_x = divmod(refined_idx, refine_window)
    refined_x = float(x0 + local_x)
    refined_y = float(y0 + local_y)

    # Only accept the refinement if the local heatmap actually has a peak; if
    # presence drops below threshold or the local peak is much weaker than
    # the coarse one, fall back to coarse.
    if presence_threshold is not None and presence_prob < presence_threshold:
        return coarse
    if refined_value < 0.5 * coarse.pred_confidence:
        return coarse

    refined_x = max(0.0, min(refined_x, float(w - 1)))
    refined_y = max(0.0, min(refined_y, float(h - 1)))

    final_conf = max(coarse.pred_confidence, presence_prob)
    if line_abc is not None and line_confidence > 0.0:
        refined_x, refined_y, _alpha = soft_snap_to_line(
            refined_x, refined_y,
            line_abc=line_abc,
            line_confidence=line_confidence,
            pred_confidence=final_conf,
            tau_line=tau_line, alpha_max=alpha_max,
        )
        refined_x = max(0.0, min(refined_x, float(w - 1)))
        refined_y = max(0.0, min(refined_y, float(h - 1)))

    return FramePrediction(
        pred_x=refined_x, pred_y=refined_y, pred_confidence=final_conf,
    )
