"""Phase 2 training dataset: 1024 px native-resolution tiles with Gaussian heatmap targets.

Per DESIGN.md §4.1 and §5.2:
- Tile = 1024 × 1024 cropped from native 4K (no downsample — would lose the laser).
- For positive frames: crop is biased to include the label ~70% of the time;
  the other 30% are random crops that may or may not contain the laser, which
  give the model balanced exposure to both regimes.
- Negative frames: always random crops.
- Photometric augs only — geometric augs (flip/rotation) break the per-dive
  colinearity prior, which a downstream phase relies on.
- Input: 4 channels = chromaticity-normalized RGB (3) + wavelength channel (1,
  0.0 green / 1.0 red / 0.5 unknown).
- Target: 1024 × 1024 float32 Gaussian (σ ≈ 3 px) at the label when the label
  falls inside the crop; all-zero otherwise. Per-tile presence label is 1.0
  iff the laser pixel is in the crop, else 0.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import albumentations as A
import cv2
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from laser_detector.preprocessing.image_loader import ImageLoader

logger = logging.getLogger(__name__)


DEFAULT_TILE_SIZE = 1024
DEFAULT_HEATMAP_SIGMA_PX = 3.0
DEFAULT_POSITIVE_CENTER_P = 0.70
DEFAULT_LABEL_EDGE_PAD_PX = 64  # keep label this far from a biased-crop edge

WAVELENGTH_CHANNEL = {"red": 1.0, "green": 0.0}
UNKNOWN_WAVELENGTH_CHANNEL = 0.5


def _photometric_augs() -> A.Compose:
    """Per DESIGN.md §5.2: photometric only. No flip/rotate/affine."""
    return A.Compose(
        [
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.2),
            A.ImageCompression(quality_range=(70, 95), p=0.2),
        ]
    )


def _chromaticity_norm(image_rgb: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """RGB → chromaticity in [0, 1] per channel: c_i = R_i / sum(R+G+B), clipped.

    Returns float32 [H, W, 3]. Accepts uint8 (JPEG cache) or uint16 (linear
    cache from `CachingLinearImageLoader`). Reduces sensitivity to underwater
    attenuation by discarding overall brightness — model sees color *ratios*.
    Per DESIGN.md §4.1.
    """
    scale = 65535.0 if image_rgb.dtype == np.uint16 else 255.0
    rgb = image_rgb.astype(np.float32) / scale
    intensity = rgb.sum(axis=2, keepdims=True)
    intensity = np.maximum(intensity, eps)
    return rgb / intensity


def _make_gaussian_heatmap(
    label_x: float,
    label_y: float,
    height: int,
    width: int,
    sigma_px: float,
) -> np.ndarray:
    """Stamp a Gaussian peak (max=1.0) at (label_x, label_y). Out-of-bounds → zeros."""
    out = np.zeros((height, width), dtype=np.float32)
    if not (0.0 <= label_x < width and 0.0 <= label_y < height):
        return out
    # Limit work to a ±3σ patch (Gaussian is ~0 outside that).
    radius = int(np.ceil(3.0 * sigma_px))
    x0 = max(int(np.floor(label_x)) - radius, 0)
    x1 = min(int(np.ceil(label_x)) + radius + 1, width)
    y0 = max(int(np.floor(label_y)) - radius, 0)
    y1 = min(int(np.ceil(label_y)) + radius + 1, height)
    if x0 >= x1 or y0 >= y1:
        return out
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    sq = (xs - label_x) ** 2 + (ys - label_y) ** 2
    out[y0:y1, x0:x1] = np.exp(-sq / (2.0 * sigma_px * sigma_px))
    return out


def _pick_crop_origin(
    img_h: int,
    img_w: int,
    tile: int,
    rng: np.random.Generator,
    *,
    label_xy: tuple[float, float] | None,
    positive_center_p: float,
    edge_pad: int,
) -> tuple[int, int]:
    """Return (crop_x, crop_y) for a `tile`×`tile` window inside an `img_h`×`img_w` image.

    If `label_xy` is provided and `rng.random() < positive_center_p`, bias the
    crop so the label sits at least `edge_pad` from each tile edge (clamped to
    the image bounds — small images may force the label closer to the edge).
    """
    max_x = max(img_w - tile, 0)
    max_y = max(img_h - tile, 0)
    if label_xy is None or rng.random() >= positive_center_p:
        return rng.integers(0, max_x + 1), rng.integers(0, max_y + 1)

    lx, ly = label_xy
    # Range of crop origins that put the label in [edge_pad, tile - edge_pad).
    lo_x = int(np.ceil(lx - (tile - edge_pad)))
    hi_x = int(np.floor(lx - edge_pad))
    lo_y = int(np.ceil(ly - (tile - edge_pad)))
    hi_y = int(np.floor(ly - edge_pad))
    lo_x = max(lo_x, 0)
    hi_x = min(hi_x, max_x)
    lo_y = max(lo_y, 0)
    hi_y = min(hi_y, max_y)
    if lo_x > hi_x:
        lo_x = hi_x = max(0, min(int(lx) - tile // 2, max_x))
    if lo_y > hi_y:
        lo_y = hi_y = max(0, min(int(ly) - tile // 2, max_y))
    return rng.integers(lo_x, hi_x + 1), rng.integers(lo_y, hi_y + 1)


def _reflect_pad_to_tile(image_bgr: np.ndarray, tile: int) -> np.ndarray:
    """Pad with reflection if the image is smaller than `tile` in either dim."""
    h, w = image_bgr.shape[:2]
    pad_h = max(tile - h, 0)
    pad_w = max(tile - w, 0)
    if pad_h == 0 and pad_w == 0:
        return image_bgr
    return cv2.copyMakeBorder(
        image_bgr,
        top=0, bottom=pad_h, left=0, right=pad_w,
        borderType=cv2.BORDER_REFLECT_101,
    )


@dataclass(frozen=True)
class FrameRecord:
    """One row of training data, decoupled from polars for picklability + clarity."""

    image_id: int
    dive_id: int
    image_path: str
    image_checksum: str
    label_xy: tuple[float, float] | None  # None = negative frame
    wavelength: str | None  # "red" / "green" / None
    # Per-dive line fit (DESIGN.md §3.1). The trio (a, b, c) parameterizes
    # `a*x + b*y + c = 0` in frame coordinates, with (a, b) unit-normalized
    # so |a*x + b*y + c| is the perpendicular distance directly. None for
    # dives where Phase 0 couldn't fit a line (no positive labels). Used by
    # the L_line aux loss (DESIGN §5.1) and inference soft-snap (§6.2).
    line_abc: tuple[float, float, float] | None = None
    line_confidence: float = 0.0
    is_line_confident: bool = False


class LaserTileDataset(Dataset):
    """1024 × 1024 native-resolution tile crops with Gaussian heatmap targets.

    Args:
        records: per-frame metadata. Build from frames.parquet + dive_wavelengths.parquet.
        loader: any `ImageLoader`. Must be picklable (DataLoader workers re-import).
        tile_size: crop side (px). Default 1024.
        heatmap_sigma_px: σ of the Gaussian peak (px). Default 3.
        positive_center_p: P(crop biased to include label) for positive frames.
        edge_pad_px: in a biased crop, label sits at least this far from any tile edge.
        augment: enable photometric augmentations.
        seed: per-process RNG seed. Workers should add their worker_id (handled
            in `worker_init_fn`).
    """

    def __init__(
        self,
        records: list[FrameRecord],
        loader: ImageLoader,
        *,
        tile_size: int = DEFAULT_TILE_SIZE,
        heatmap_sigma_px: float = DEFAULT_HEATMAP_SIGMA_PX,
        positive_center_p: float = DEFAULT_POSITIVE_CENTER_P,
        edge_pad_px: int = DEFAULT_LABEL_EDGE_PAD_PX,
        augment: bool = True,
        seed: int = 0,
    ):
        self.records = records
        self.loader = loader
        self.tile_size = int(tile_size)
        self.heatmap_sigma_px = float(heatmap_sigma_px)
        self.positive_center_p = float(positive_center_p)
        self.edge_pad_px = int(edge_pad_px)
        self.augment = bool(augment)
        self._aug_pipeline = _photometric_augs() if augment else None
        self._seed = int(seed)
        self._rng: np.random.Generator | None = None  # lazy-init per worker

    def __len__(self) -> int:
        return len(self.records)

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            self._rng = np.random.default_rng(self._seed + worker_id)
        return self._rng

    def _load_image(self, rec: FrameRecord) -> np.ndarray | None:
        return self.loader.load(rec.image_path, rec.image_checksum)

    def __getitem__(self, idx: int) -> dict:
        # Skip frames that fail to decode rather than crashing the loader.
        # Bounded retries so a fully broken split fails loudly instead of looping.
        for offset in range(min(8, len(self.records))):
            real_idx = (idx + offset) % len(self.records)
            rec = self.records[real_idx]
            image_bgr = self._load_image(rec)
            if image_bgr is not None:
                break
            logger.warning(
                "Image failed to load (image_id=%d), skipping to next index", rec.image_id
            )
        else:
            raise RuntimeError(f"8 consecutive image loads failed starting at idx={idx}")

        rng = self._get_rng()

        image_bgr = _reflect_pad_to_tile(image_bgr, self.tile_size)
        h, w = image_bgr.shape[:2]
        crop_x, crop_y = _pick_crop_origin(
            h, w, self.tile_size, rng,
            label_xy=rec.label_xy,
            positive_center_p=self.positive_center_p,
            edge_pad=self.edge_pad_px,
        )
        crop = image_bgr[crop_y : crop_y + self.tile_size, crop_x : crop_x + self.tile_size]

        if self._aug_pipeline is not None:
            crop = self._aug_pipeline(image=crop)["image"]

        # OpenCV is BGR; standard imagenet preprocessing expects RGB.
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        chrom = _chromaticity_norm(rgb)  # [H, W, 3] float32

        wavelength_value = (
            WAVELENGTH_CHANNEL.get(rec.wavelength, UNKNOWN_WAVELENGTH_CHANNEL)
            if rec.wavelength is not None
            else UNKNOWN_WAVELENGTH_CHANNEL
        )
        wavelength_channel = np.full(
            (self.tile_size, self.tile_size, 1), wavelength_value, dtype=np.float32
        )
        # 4-channel input: [chrom_r, chrom_g, chrom_b, wavelength]
        x = np.concatenate([chrom, wavelength_channel], axis=2)
        # HWC → CHW for torch
        x_chw = np.transpose(x, (2, 0, 1)).copy()

        if rec.label_xy is None:
            heatmap = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
            presence = 0.0
        else:
            local_x = rec.label_xy[0] - crop_x
            local_y = rec.label_xy[1] - crop_y
            heatmap = _make_gaussian_heatmap(
                local_x, local_y, self.tile_size, self.tile_size, self.heatmap_sigma_px
            )
            presence = 1.0 if heatmap.max() > 0.0 else 0.0

        # Line params for the L_line aux loss. Always emit a (3,) tensor; the
        # trainer masks frames where line_abc is None or is_line_confident is
        # False. Default abc=(0,0,1) is a degenerate "line" — won't be used
        # because the valid_mask gates it out, but keeps tensor shapes stable.
        if rec.line_abc is None:
            line_abc = (0.0, 0.0, 1.0)
        else:
            line_abc = rec.line_abc

        return {
            "image": torch.from_numpy(x_chw),
            "heatmap": torch.from_numpy(heatmap).unsqueeze(0),
            "presence": torch.tensor(presence, dtype=torch.float32),
            "image_id": int(rec.image_id),
            "dive_id": int(rec.dive_id),
            "crop_offset": torch.tensor([crop_x, crop_y], dtype=torch.float32),
            "line_abc": torch.tensor(line_abc, dtype=torch.float32),
            "line_confidence": torch.tensor(rec.line_confidence, dtype=torch.float32),
            "is_line_confident": torch.tensor(rec.is_line_confident, dtype=torch.bool),
        }


class HardNegativeBalancedSampler:
    """50/50 positive / hard-negative frame sampler for the Phase 2 trainer.

    Per DESIGN.md §5.1: the presence head is trained on every frame with
    hard-negative mining. "Random sampling is dominated by trivial negatives
    and produces a useless presence head." This sampler:

    1. Each epoch yields `2 * n_positive` indices: half positive (sampled
       uniformly with replacement), half negative (sampled with replacement
       weighted by per-record hardness scores).
    2. Hardness scores start uniform — the first epoch is effectively random.
       The trainer calls `set_neg_score(record_idx, score)` after each epoch
       to update with the model's max heatmap response on each negative.

    Sampling positives with replacement to match negative count is the right
    move when negatives dominate (~10× as many in v1) — we'd otherwise drown
    the heatmap loss in tiles where the target is all zeros.
    """

    def __init__(
        self,
        records: list[FrameRecord],
        seed: int = 0,
        *,
        rank: int = 0,
        world_size: int = 1,
    ):
        is_pos = np.array([r.label_xy is not None for r in records], dtype=bool)
        self.pos_indices = np.where(is_pos)[0]
        self.neg_indices = np.where(~is_pos)[0]
        if len(self.pos_indices) == 0:
            raise ValueError(
                "HardNegativeBalancedSampler requires at least one positive frame"
            )
        # record_idx → position in self.neg_indices, used by set_neg_score.
        self._neg_array_of_record = {
            int(rec_idx): k for k, rec_idx in enumerate(self.neg_indices)
        }
        self.neg_scores = np.ones(len(self.neg_indices), dtype=np.float64)
        self._base_seed = int(seed)
        # set_epoch updates this; without that call, every epoch would emit
        # the same shuffle. Trainer is expected to call set_epoch.
        self._epoch = 0
        # Rank 0 owns the score-update RNG so independent calls (`set_neg_score`
        # via the trainer's hard-neg loop) don't desync the per-epoch shuffle
        # across ranks — the shuffle uses a separate epoch-derived seed.
        self._score_rng = np.random.default_rng(seed + 999_999)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"rank={self.rank} not in [0, {self.world_size})")

    def set_epoch(self, epoch: int) -> None:
        """Trainer must call this once per epoch on every rank with the same
        epoch number — that's what keeps the shuffle identical across ranks
        so sharding lines up. Mirrors `torch.utils.data.DistributedSampler`."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        # 2 × n_pos when negatives exist (50/50 balance); just n_pos otherwise
        # so DataLoader's progress + scheduler step counts match what runs.
        if len(self.neg_indices) == 0:
            total = len(self.pos_indices)
        else:
            total = 2 * len(self.pos_indices)
        # Per-rank length. Trim to a multiple of world_size (drop_last-style)
        # so each rank gets the same step count and DDP's all-reduce schedule
        # stays aligned.
        per_rank = total // self.world_size
        return per_rank

    def __iter__(self):
        # Deterministic per-epoch shuffle — same seed on every rank → same
        # `idx` array → consistent shard mapping when we slice by rank.
        rng = np.random.default_rng(self._base_seed + self._epoch)
        n = len(self.pos_indices)
        if len(self.neg_indices) > 0:
            pos_sample = rng.choice(self.pos_indices, size=n, replace=True)
            weights = self.neg_scores / self.neg_scores.sum()
            neg_sample = rng.choice(
                self.neg_indices, size=n, replace=True, p=weights
            )
            idx = np.concatenate([pos_sample, neg_sample])
        else:
            idx = rng.permutation(self.pos_indices)
        rng.shuffle(idx)
        # Drop the last (len(idx) % world_size) entries so per-rank counts match.
        per_rank = len(idx) // self.world_size
        usable = idx[: per_rank * self.world_size]
        my_slice = usable[self.rank :: self.world_size]
        return iter(int(i) for i in my_slice)

    def has_negatives(self) -> bool:
        return len(self.neg_indices) > 0

    def sample_neg_record_indices(self, k: int) -> np.ndarray:
        """Pick `min(k, n_neg)` negative-record indices uniformly without
        replacement. Used by the trainer to choose which negatives to score
        on rank 0; uses a separate RNG so it doesn't perturb the shared
        per-epoch shuffle seed."""
        if len(self.neg_indices) == 0:
            return np.empty(0, dtype=np.int64)
        k = min(k, len(self.neg_indices))
        chosen = self._score_rng.choice(len(self.neg_indices), size=k, replace=False)
        return self.neg_indices[chosen].astype(np.int64)

    def set_neg_score(self, record_idx: int, score: float) -> None:
        """Update one negative's sampling weight from a fresh model score.

        Floored at a small epsilon so unsampled negatives stay reachable —
        without that, a single low-score epoch could permanently silence a
        record that was just unlucky in its random crop."""
        arr = self._neg_array_of_record.get(int(record_idx))
        if arr is None:
            return
        self.neg_scores[arr] = max(float(score), 1e-3)


def build_records(
    frames: pl.DataFrame,
    wavelengths: pl.DataFrame,
    lines: pl.DataFrame | None = None,
    *,
    drop_superseded: bool = True,
) -> list[FrameRecord]:
    """Join Phase 0 frames + wavelengths (+ optional lines) into per-frame records.

    `drop_superseded=True` (default) excludes frames whose label was superseded
    upstream (typically because the labeler-error audit flagged the label as
    an outlier). Training on superseded labels regresses the heatmap to known-
    bad targets and contributes to the bimodal failure mode in the Phase 2
    BCE+pos_weight run.

    `lines` (optional) attaches per-dive line params + confidence so the
    trainer's L_line aux loss (DESIGN §5.1) and inference soft-snap (§6.2)
    can read them off the FrameRecord without separate lookups.

    Set drop_superseded=False for ablation runs that want to measure the
    impact of the filter.
    """
    if drop_superseded and "superseded" in frames.columns:
        frames = frames.filter(~pl.col("superseded"))
    joined = frames.join(
        wavelengths.select("dive_id", "wavelength"), on="dive_id", how="left"
    )
    if lines is not None:
        joined = joined.join(
            lines.select("dive_id", "line_a", "line_b", "line_c",
                         "line_confidence", "is_line_confident"),
            on="dive_id", how="left",
        )
    records: list[FrameRecord] = []
    for row in joined.iter_rows(named=True):
        is_pos = bool(row["is_positive"])
        label_xy = (
            (float(row["label_x"]), float(row["label_y"])) if is_pos else None
        )
        line_abc: tuple[float, float, float] | None = None
        line_conf = 0.0
        is_line_conf = False
        if lines is not None and row.get("line_a") is not None:
            line_abc = (
                float(row["line_a"]), float(row["line_b"]), float(row["line_c"]),
            )
            line_conf = float(row.get("line_confidence") or 0.0)
            is_line_conf = bool(row.get("is_line_confident") or False)
        records.append(
            FrameRecord(
                image_id=int(row["image_id"]),
                dive_id=int(row["dive_id"]),
                image_path=str(row["image_path"]),
                image_checksum=str(row["image_checksum"]),
                label_xy=label_xy,
                wavelength=row["wavelength"],
                line_abc=line_abc,
                line_confidence=line_conf,
                is_line_confident=is_line_conf,
            )
        )
    return records
