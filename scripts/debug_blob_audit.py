"""Render audit samples to disk for visual review of `_segment_blob`.

Writes JPEG cutouts (200×200 around the label) for:
  - The 15 largest-diameter audit measurements (suspected reflection artifacts).
  - A balanced sample of red and green dives at varied diameters.

Each cutout overlays:
  - Red dot at the labeled (x, y).
  - Green contour around the segmented connected component.
  - Filename embeds dive_id, diameter, wavelength.

Output: data/phase0/_debug_blobs/  (gitignored under data/)
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from laser_detector.preprocessing.audit import (
    MAX_BLOB_AREA_FRAC,
    PATCH_HALF_SIZE,
    _segment_blob,
)
from laser_detector.preprocessing.config import load_config
from laser_detector.preprocessing.image_loader import (
    CachingImageLoader,
    LocalFilesystemImageLoader,
)


CUT_HALF = 100  # 200×200 cutout around the label


def _segment_mask(image: np.ndarray, x: float, y: float) -> np.ndarray | None:
    """Mirror `_segment_blob`'s logic exactly: pick only the component
    containing the labeled pixel, reject when that component exceeds the area
    cap. Returns a full-frame mask set only inside that single component (or
    None if the audit would also reject)."""
    h, w = image.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return None
    x0, x1 = max(0, cx - PATCH_HALF_SIZE), min(w, cx + PATCH_HALF_SIZE + 1)
    y0, y1 = max(0, cy - PATCH_HALF_SIZE), min(h, cy + PATCH_HALF_SIZE + 1)
    patch = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    raw_mask = ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 200)).astype(np.uint8)
    if raw_mask.sum() == 0:
        return None

    _, components = cv2.connectedComponents(raw_mask)
    label_at_center = int(components[cy - y0, cx - x0])
    if label_at_center == 0:
        return None

    selected_pixels = int((components == label_at_center).sum())
    if selected_pixels > MAX_BLOB_AREA_FRAC * raw_mask.size:
        return None

    selected = ((components == label_at_center).astype(np.uint8)) * 255
    full = np.zeros(image.shape[:2], dtype=np.uint8)
    full[y0:y1, x0:x1] = selected
    return full


def _render(image: np.ndarray, x: float, y: float, out_path: Path) -> None:
    h, w = image.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx - CUT_HALF), min(w, cx + CUT_HALF)
    y0, y1 = max(0, cy - CUT_HALF), min(h, cy + CUT_HALF)
    cut = image[y0:y1, x0:x1].copy()

    mask = _segment_mask(image, x, y)
    if mask is not None:
        cut_mask = mask[y0:y1, x0:x1]
        contours, _ = cv2.findContours(cut_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cut, contours, -1, (0, 255, 0), 1)
    cv2.circle(cut, (cx - x0, cy - y0), 2, (0, 0, 255), -1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cut, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_config()
    if config.image_root is None:
        raise SystemExit("images.root must be set")

    loader = CachingImageLoader(
        inner=LocalFilesystemImageLoader(config.image_root),
        cache_dir=config.cache_dir,
        jpeg_quality=config.cache_jpeg_quality,
    )

    audit = pl.read_parquet(config.data_dir / "laser_size_audit.parquet")
    frames = pl.read_parquet(config.data_dir / "frames.parquet").select(
        "image_id", "image_path", "image_checksum", "label_x", "label_y"
    )
    audit_with_paths = audit.join(frames, on="image_id", how="left")

    out_dir = config.data_dir / "_debug_blobs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Top 15 largest blobs
    top = audit_with_paths.sort("blob_diameter_px", descending=True).head(15)
    for row in top.iter_rows(named=True):
        img = loader.load(row["image_path"], row["image_checksum"])
        if img is None:
            logging.warning("could not load image %s", row["image_path"])
            continue
        name = f"large_d{row['blob_diameter_px']:05.1f}_dive{row['dive_id']}_img{row['image_id']}_{row['wavelength']}.jpg"
        _render(img, row["label_x"], row["label_y"], out_dir / name)
        logging.info("wrote %s", name)

    # Balanced sample: 5 small/median/large per wavelength
    for wl in ["red", "green"]:
        sub = audit_with_paths.filter(pl.col("wavelength") == wl).sort("blob_diameter_px")
        n = sub.height
        if n == 0:
            continue
        idxs = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
        for i in idxs:
            row = sub.row(i, named=True)
            img = loader.load(row["image_path"], row["image_checksum"])
            if img is None:
                continue
            name = f"sample_{wl}_d{row['blob_diameter_px']:05.1f}_dive{row['dive_id']}_img{row['image_id']}.jpg"
            _render(img, row["label_x"], row["label_y"], out_dir / name)
            logging.info("wrote %s", name)

    print(f"\n{len(list(out_dir.glob('*.jpg')))} debug images in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
