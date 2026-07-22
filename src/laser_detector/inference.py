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


def rectify_prediction(
    pred_x: float, pred_y: float,
    K: np.ndarray, dist: np.ndarray,
) -> tuple[float, float]:
    """Convert a prediction from raw pixel space to rectified (undistorted)
    pixel space via `cv2.undistortPoints(..., P=K)`.

    Addresses issue #9: labels are made on rectified images (via
    `RectifiedImage(RawImage(...))` in the labeling UI) but the detector
    processes raw images, so predictions land in raw pixel space. Downstream
    3D reconstruction expects rectified coordinates. This shifts the
    prediction into the rectified frame using the per-rig camera intrinsics.

    Empirical impact per issue #9's own analysis: median 0.02 px, p95 0.23 px,
    p99 1.01 px displacement. Applying this is a small correction; skipping it
    is harmless for aggregate hit_n3 metrics but wrong for downstream 3D.
    """
    pts = np.asarray([[[pred_x, pred_y]]], dtype=np.float32)
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float32).reshape(-1)
    out = cv2.undistortPoints(pts, K, dist, P=K)
    return float(out[0, 0, 0]), float(out[0, 0, 1])

# Static rig prior on laser position in **sensor-coordinate** pixels of the
# Olympus TG-6 (3016×4014). Derived from all positive train labels mapped
# back into sensor coords via the orf_flip parquet: median ≈ (1977, 1343),
# p99 box (1681–2643) × (987–1928), p1 (1).
#
# This was previously derived in world-coordinate space (post EXIF rotation),
# which had to span both landscape and portrait laser positions and so was
# noticeably looser on the y-axis. With sensor coords the rig is body-frame
# stationary and the prior is correspondingly tighter.
#
# Outside the bbox → 0 (hard reject). Inside, a soft Gaussian biases argmax
# toward the dense center, with a floor (default 0.5 from the world-coords
# sweep) so a strong heatmap response in a low-prior region isn't crushed.
DEFAULT_RIG_PRIOR_BBOX: tuple[int, int, int, int] = (1100, 700, 2950, 2180)
DEFAULT_RIG_PRIOR_CENTER: tuple[float, float] = (1977.0, 1343.0)
DEFAULT_RIG_PRIOR_SIGMA: tuple[float, float] = (300.0, 300.0)
DEFAULT_RIG_PRIOR_FLOOR: float = 0.5


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


def _line_mask_for_tile(
    tile_origin_x: int,
    tile_origin_y: int,
    tile: int,
    line_abc: tuple[float, float, float],
    corridor_px: float,
) -> np.ndarray:
    """Build a `[tile, tile]` float32 binary mask for a single tile that is
    1.0 where the pixel is within `corridor_px` of the dive line `a*x + b*y +
    c = 0` and 0.0 otherwise.

    Phase 3.1: per-dive geometric constraint, much tighter than the rig-prior
    bbox. Population val/test label-to-line p99 is ≤ 13 px; a ±25 corridor
    safely includes essentially all real labels while killing distractors
    living far from the line (val:427 distractor cluster is 203 px off-line).

    `line_abc` is `(a, b, c)` from `dive_lines.parquet`; (a, b) is unit-
    normalized in the Phase 0 fit so `|a*x + b*y + c|` is already the
    perpendicular distance. Falls back to the general formula if not unit
    (cheap and bullet-proof).
    """
    a, b, c = line_abc
    norm = float((a * a + b * b) ** 0.5)
    if norm <= 1e-12:
        return np.ones((tile, tile), dtype=np.float32)
    ys, xs = np.mgrid[
        tile_origin_y : tile_origin_y + tile,
        tile_origin_x : tile_origin_x + tile,
    ].astype(np.float32)
    dist = np.abs(a * xs + b * ys + c) / norm
    return (dist <= corridor_px).astype(np.float32)


def _subpixel_refine_peak(
    heatmap_2d: "torch.Tensor | np.ndarray", x: int, y: int
) -> tuple[float, float]:
    """Refine an integer-pixel peak `(x, y)` to sub-pixel via a separable
    parabolic fit on the 3-point cross neighborhood. For each axis:

        delta = 0.5 * (v_minus - v_plus) / (v_minus - 2*v_center + v_plus)

    The denominator is positive when `(x, y)` is a real local maximum; the
    shift is in `(-0.5, 0.5)`. Peaks on the heatmap edge, degenerate fits, or
    shifts that escape `(-0.5, 0.5)` return the original integer (the parabola
    assumption fails — typically because the integer pixel wasn't the true
    local max). Accepts torch.Tensor or np.ndarray. O(5) tensor reads.
    """
    h, w = heatmap_2d.shape[-2:]
    if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1:
        return float(x), float(y)
    if isinstance(heatmap_2d, torch.Tensor):
        def get(i: int, j: int) -> float:
            return float(heatmap_2d[i, j].item())
    else:
        def get(i: int, j: int) -> float:
            return float(heatmap_2d[i, j])
    v_c = get(y, x)
    v_xm, v_xp = get(y, x - 1), get(y, x + 1)
    v_ym, v_yp = get(y - 1, x), get(y + 1, x)
    den_x = v_xm - 2.0 * v_c + v_xp
    den_y = v_ym - 2.0 * v_c + v_yp
    dx = 0.5 * (v_xm - v_xp) / den_x if abs(den_x) > 1e-12 else 0.0
    dy = 0.5 * (v_ym - v_yp) / den_y if abs(den_y) > 1e-12 else 0.0
    if not (-0.5 < dx < 0.5):
        dx = 0.0
    if not (-0.5 < dy < 0.5):
        dy = 0.0
    return float(x) + dx, float(y) + dy


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
    tile_bgr: np.ndarray,
    wavelength_value: float,
    bayer_excess_tile: np.ndarray | None = None,
    bayer_excess_scale: float = 4096.0,
) -> np.ndarray:
    """BGR (uint8/uint16) → float32 [C, H, W] tile input.

    C=4 by default: chromaticity (3) + wavelength (1).
    C=6 when `bayer_excess_tile` is given (uint16 [H, W, 2] of G_excess, R_excess).
    Bayer values are normalized by `bayer_excess_scale` to land in roughly [0, 1].
    """
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    chrom = _chromaticity_norm(rgb)
    h, w = chrom.shape[:2]
    wavelength_channel = np.full((h, w, 1), wavelength_value, dtype=np.float32)
    parts = [chrom, wavelength_channel]
    if bayer_excess_tile is not None:
        parts.append(bayer_excess_tile.astype(np.float32) / bayer_excess_scale)
    x = np.concatenate(parts, axis=2)
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
    bayer_excess_image: np.ndarray | None = None,
    bayer_excess_scale: float = 4096.0,
    subpixel_refine: bool = False,
    line_mask_corridor_px: float | None = None,
    intermediates_out: dict | None = None,
) -> FramePrediction:
    """Run tiled inference on a single 4K frame.

    Returns the (pred_x, pred_y) of the heatmap maximum across all tiles, or
    (None, None) if `presence_threshold` is set and the frame-level confidence
    falls below it.

    Frame-level confidence = max(sigmoid(tile_presence_logits)) per DESIGN.md §6.

    If `intermediates_out` is a mutable dict, key stages are recorded into it
    for external instrumentation (validation-bundle builder, debugging). Keys
    populated: `winning_tile_input` (6-ch preprocessed input for the winning
    tile), `winning_tile_heatmap_logits` (fp32), `winning_tile_origin` (x, y
    frame coords), `winning_tile_idx`, `per_tile_presence` (float32 array),
    `tile_origins` (int64 array), `coarse_local_argmax` (x, y in tile
    coords), `coarse_argmax_pre_subpixel` (x, y in frame coords, integer),
    `coarse_argmax` (x, y after any subpixel refinement).
    """
    grid = compute_tile_grid(*image_bgr.shape[:2], tile=tile, overlap=overlap)
    padded = _reflect_pad(image_bgr, grid.padded_h, grid.padded_w)

    wavelength_value = (
        WAVELENGTH_CHANNEL.get(wavelength, UNKNOWN_WAVELENGTH_CHANNEL)
        if wavelength is not None
        else UNKNOWN_WAVELENGTH_CHANNEL
    )

    bayer_padded: np.ndarray | None = None
    if bayer_excess_image is not None:
        bayer_padded = _reflect_pad(bayer_excess_image, grid.padded_h, grid.padded_w)

    def _maybe_bayer_tile(x: int, y: int) -> np.ndarray | None:
        if bayer_padded is None:
            return None
        return bayer_padded[y : y + tile, x : x + tile]

    tile_arrays = [
        _preprocess_tile(
            padded[y : y + tile, x : x + tile],
            wavelength_value,
            bayer_excess_tile=_maybe_bayer_tile(x, y),
            bayer_excess_scale=bayer_excess_scale,
        )
        for x, y in grid.origins
    ]
    tile_batch = torch.from_numpy(np.stack(tile_arrays))  # [N, C, H, W]

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

    # Per-tile line-corridor mask (when caller provides line_abc + corridor).
    # Multiplies the rig prior; zeros pixels farther than corridor_px from
    # the dive line, which is much tighter than the rig bbox. See
    # _line_mask_for_tile docstring.
    line_masks: list[torch.Tensor] | None = None
    if line_mask_corridor_px is not None and line_abc is not None and line_confidence > 0.0:
        line_masks = [
            torch.from_numpy(
                _line_mask_for_tile(ox, oy, tile, line_abc, line_mask_corridor_px)
            )
            for (ox, oy) in grid.origins
        ]

    best_value = -1.0
    best_xy = (None, None)
    best_local: tuple[int, int] | None = None
    best_origin: tuple[int, int] | None = None
    best_heatmap_2d: torch.Tensor | None = None  # winning tile, for sub-pixel refine
    best_tile_idx = -1
    best_tile_input: np.ndarray | None = None
    presence_max = 0.0
    # Per-tile presence capture for intermediates. Kept out of the hot path
    # unless requested (still cheap — one float per tile).
    per_tile_presence = (
        np.zeros(len(tile_arrays), dtype=np.float32) if intermediates_out is not None else None
    )

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None and device.type == "cuda"
        else _NullCtx()
    )

    for chunk_start in range(0, len(tile_arrays), batch_size):
        chunk = tile_batch[chunk_start : chunk_start + batch_size].to(device, non_blocking=True)
        with autocast_ctx:
            out = model(chunk)
        heatmap_logits = out["heatmap_logits"].float()  # full-precision copy for sub-pixel
        heatmap_probs = torch.sigmoid(heatmap_logits)
        presence_probs = torch.sigmoid(out["presence_logits"]).float()

        if rig_masks is not None:
            chunk_masks = torch.stack(
                [rig_masks[chunk_start + i] for i in range(heatmap_probs.shape[0])]
            ).to(heatmap_probs.device)
            # heatmap_probs is [B, 1, H, W]; unsqueeze masks to broadcast cleanly.
            heatmap_probs = heatmap_probs * chunk_masks.unsqueeze(1)
        if line_masks is not None:
            chunk_line_masks = torch.stack(
                [line_masks[chunk_start + i] for i in range(heatmap_probs.shape[0])]
            ).to(heatmap_probs.device)
            heatmap_probs = heatmap_probs * chunk_line_masks.unsqueeze(1)

        flat = heatmap_probs.view(heatmap_probs.shape[0], -1)
        max_vals, max_idx = flat.max(dim=1)

        for i, (mv, mi) in enumerate(zip(max_vals.tolist(), max_idx.tolist())):
            tile_i = chunk_start + i
            if per_tile_presence is not None:
                per_tile_presence[tile_i] = float(presence_probs[i].max().item())
            if mv > best_value:
                best_value = mv
                local_y, local_x = divmod(mi, tile)
                ox, oy = grid.origins[tile_i]
                best_xy = (float(local_x + ox), float(local_y + oy))
                best_local = (local_x, local_y)
                best_origin = (ox, oy)
                best_tile_idx = tile_i
                if intermediates_out is not None:
                    best_tile_input = tile_arrays[tile_i]
                if subpixel_refine or intermediates_out is not None:
                    # Refine on LOGITS, not probs: under bf16 autocast, sigmoid
                    # saturates to 1.0 at the peak and the 3-point cross can't
                    # distinguish "true peak" from "neighbor +1 step". Logits
                    # keep dynamic range and give the same parabolic peak
                    # (sigmoid is monotonic). Detach to CPU so the GPU tile
                    # tensor can be freed when the next chunk loads.
                    best_heatmap_2d = heatmap_logits[i, 0].detach().cpu()
        presence_max = max(presence_max, float(presence_probs.max().item()))

    if presence_threshold is not None and presence_max < presence_threshold:
        return FramePrediction(pred_x=None, pred_y=None, pred_confidence=presence_max)

    pred_x, pred_y = best_xy
    coarse_pre_subpixel: tuple[float, float] | None = (
        (float(pred_x), float(pred_y)) if pred_x is not None else None
    )
    if pred_x is not None:
        if subpixel_refine and best_heatmap_2d is not None and best_local is not None and best_origin is not None:
            local_x, local_y = best_local
            rx, ry = _subpixel_refine_peak(best_heatmap_2d, local_x, local_y)
            pred_x = float(best_origin[0]) + rx
            pred_y = float(best_origin[1]) + ry
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
    if intermediates_out is not None:
        intermediates_out['tile_origins'] = np.asarray(grid.origins, dtype=np.int64)
        intermediates_out['per_tile_presence'] = per_tile_presence
        intermediates_out['winning_tile_idx'] = best_tile_idx
        intermediates_out['winning_tile_origin'] = (
            np.asarray(best_origin, dtype=np.int64) if best_origin is not None else None
        )
        intermediates_out['winning_tile_input'] = best_tile_input
        intermediates_out['winning_tile_heatmap_logits'] = (
            best_heatmap_2d.numpy().copy() if best_heatmap_2d is not None else None
        )
        intermediates_out['coarse_local_argmax'] = (
            np.asarray(best_local, dtype=np.int64) if best_local is not None else None
        )
        intermediates_out['coarse_argmax_pre_subpixel'] = coarse_pre_subpixel
        intermediates_out['coarse_argmax'] = (
            (float(pred_x), float(pred_y)) if pred_x is not None else None
        )
        intermediates_out['presence_max'] = presence_max
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
    rig_prior: bool = False,
    rig_prior_bbox: tuple[int, int, int, int] = DEFAULT_RIG_PRIOR_BBOX,
    rig_prior_center: tuple[float, float] = DEFAULT_RIG_PRIOR_CENTER,
    rig_prior_sigma: tuple[float, float] = DEFAULT_RIG_PRIOR_SIGMA,
    rig_prior_floor: float = DEFAULT_RIG_PRIOR_FLOOR,
    bayer_excess_image: np.ndarray | None = None,
    bayer_excess_scale: float = 4096.0,
    subpixel_refine: bool = False,
    line_mask_corridor_px: float | None = None,
    intermediates_out: dict | None = None,
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
    # If line masking is requested, we MUST pass line_abc/line_confidence to
    # the coarse pass so the mask can be built. To keep soft-snap deferred to
    # after refinement, pass alpha_max=0.0 which makes soft_snap_to_line a
    # no-op (alpha is clipped to 0).
    coarse_line_abc = line_abc if line_mask_corridor_px is not None else None
    coarse_line_conf = line_confidence if line_mask_corridor_px is not None else 0.0
    # Forward coarse-stage intermediates into the same dict if requested.
    coarse_intermediates = {} if intermediates_out is not None else None
    coarse = predict_frame(
        image_bgr, model,
        wavelength=wavelength, device=device,
        tile=tile, overlap=overlap, batch_size=batch_size,
        presence_threshold=presence_threshold,
        autocast_dtype=autocast_dtype,
        line_abc=coarse_line_abc, line_confidence=coarse_line_conf,
        alpha_max=0.0,  # suppress coarse-pass snap; cascade snaps after pass-2
        rig_prior=rig_prior,
        rig_prior_bbox=rig_prior_bbox,
        rig_prior_center=rig_prior_center,
        rig_prior_sigma=rig_prior_sigma,
        rig_prior_floor=rig_prior_floor,
        bayer_excess_image=bayer_excess_image,
        bayer_excess_scale=bayer_excess_scale,
        subpixel_refine=subpixel_refine,  # so the fallback-to-coarse path
        # (pass-2 rejected by presence or confidence-drop check below) still
        # benefits from sub-pixel; cropping uses int(round(...)) regardless.
        line_mask_corridor_px=line_mask_corridor_px,
        intermediates_out=coarse_intermediates,
    )
    if intermediates_out is not None and coarse_intermediates is not None:
        intermediates_out.update(coarse_intermediates)
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

    bayer_crop: np.ndarray | None = None
    if bayer_excess_image is not None:
        bayer_crop = bayer_excess_image[y0:y1, x0:x1]
        bayer_crop = _reflect_pad(bayer_crop, refine_window, refine_window)

    wavelength_value = (
        WAVELENGTH_CHANNEL.get(wavelength, UNKNOWN_WAVELENGTH_CHANNEL)
        if wavelength is not None
        else UNKNOWN_WAVELENGTH_CHANNEL
    )
    arr = _preprocess_tile(
        crop, wavelength_value,
        bayer_excess_tile=bayer_crop,
        bayer_excess_scale=bayer_excess_scale,
    )
    batch = torch.from_numpy(arr[None]).to(device, non_blocking=True)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None and device.type == "cuda"
        else _NullCtx()
    )
    with autocast_ctx:
        out = model(batch)
    heatmap_logits = out["heatmap_logits"][0].float()  # full precision for sub-pixel
    heatmap_probs = torch.sigmoid(heatmap_logits)
    presence_prob = float(torch.sigmoid(out["presence_logits"][0]).max().item())

    flat = heatmap_probs.view(-1)
    refined_idx = int(flat.argmax().item())
    refined_value = float(flat.max().item())
    local_y, local_x = divmod(refined_idx, refine_window)
    if subpixel_refine:
        # Refine on logits (bf16-stable) — see predict_frame for the rationale.
        rx, ry = _subpixel_refine_peak(heatmap_logits[0], local_x, local_y)
        refined_x = float(x0) + rx
        refined_y = float(y0) + ry
    else:
        refined_x = float(x0 + local_x)
        refined_y = float(y0 + local_y)

    # Capture cascade intermediates BEFORE the fallback / snap so the
    # oracle records the pre-fallback + pre-snap (x, y) alongside the flag.
    cascade_pre_fallback_xy: tuple[float, float] = (refined_x, refined_y)

    # Only accept the refinement if the local heatmap actually has a peak; if
    # presence drops below threshold or the local peak is much weaker than
    # the coarse one, fall back to coarse.
    fell_back = False
    if presence_threshold is not None and presence_prob < presence_threshold:
        fell_back = True
    elif refined_value < 0.5 * coarse.pred_confidence:
        fell_back = True

    if intermediates_out is not None:
        intermediates_out['cascade_heatmap_logits'] = heatmap_logits[0].detach().cpu().numpy().copy()
        intermediates_out['cascade_crop_origin'] = np.asarray([x0, y0], dtype=np.int64)
        intermediates_out['cascade_local_argmax'] = np.asarray([local_x, local_y], dtype=np.int64)
        intermediates_out['cascade_refined_value'] = float(refined_value)
        intermediates_out['cascade_presence'] = float(presence_prob)
        intermediates_out['cascade_pre_fallback_xy'] = cascade_pre_fallback_xy
        intermediates_out['cascade_fell_back'] = fell_back

    if fell_back:
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

    if intermediates_out is not None:
        intermediates_out['final_pre_bias_xy'] = (float(refined_x), float(refined_y))
        intermediates_out['final_conf'] = float(final_conf)

    return FramePrediction(
        pred_x=refined_x, pred_y=refined_y, pred_confidence=final_conf,
    )
