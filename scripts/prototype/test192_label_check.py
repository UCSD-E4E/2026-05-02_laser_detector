"""Visual check: are the labels on dive 192 correct?

Sample 12 frames spanning the error spectrum (hits, borderlines, catastrophic),
crop ~150x150 px around each label, mark the label (green +) and prediction
(red x). Saves a grid figure for human inspection.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO = Path('/scratch/krg/c.crutchfield.642/Repos/school/e4e/fishsense/2026-05-02_laser_detector')
sys.path.insert(0, str(REPO / 'src'))

from laser_detector.preprocessing.image_loader import make_cached_image_loader
from laser_detector.preprocessing.config import load_config

OUT = REPO / 'notes/figures/test192_label_check.png'
OUT.parent.mkdir(parents=True, exist_ok=True)

config = load_config()
loader = make_cached_image_loader(
    config.image_root, Path(f'{config.cache_dir}_linear_npy'),
    pipeline='linear_npy', jpeg_quality=config.cache_jpeg_quality,
)

# Pull dive 192 frames + v3-calibrated predictions
pred = pl.read_parquet(REPO / 'data/audit/epoch_021_recipe_calibrated_subpx_v3_test/predictions_with_meta.parquet')
t192 = pred.filter(pl.col('dive_id') == 192).filter(pl.col('is_positive') & pl.col('pred_x').is_not_null())
# Apply v3 calibration (parquet was audited w/o calibration)
t192 = t192.with_columns([
    (pl.col('pred_x') - (-0.200)).alias('px_cal'),
    (pl.col('pred_y') - (-0.006)).alias('py_cal'),
])
t192 = t192.with_columns(((pl.col('px_cal')-pl.col('label_x'))**2 + (pl.col('py_cal')-pl.col('label_y'))**2).sqrt().alias('err'))

# Stratified sample by error magnitude
rng = np.random.default_rng(42)
buckets = {
    'hit (err<2)': t192.filter(pl.col('err') < 2),
    'borderline-low (3-5)': t192.filter((pl.col('err') >= 3) & (pl.col('err') < 5)),
    'borderline-high (5-10)': t192.filter((pl.col('err') >= 5) & (pl.col('err') < 10)),
    'catastrophic (>50)': t192.filter(pl.col('err') > 50),
}
n_per = 3
selected = []
for bucket, df in buckets.items():
    if df.height == 0:
        continue
    sample = df.sample(n=min(n_per, df.height), seed=42)
    for r in sample.iter_rows(named=True):
        selected.append((bucket, r))

CROP = 200  # half-size of crop around label
ncols = 4
nrows = (len(selected) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4.2*nrows))
axes = np.array(axes).reshape(nrows, ncols)
for i in range(nrows * ncols):
    ax = axes[i // ncols, i % ncols]
    ax.axis('off')

for i, (bucket, r) in enumerate(selected):
    ax = axes[i // ncols, i % ncols]
    img = loader.load(r['image_path'], r['image_checksum'])
    if img is None:
        ax.text(0.5, 0.5, 'image load failed', ha='center', va='center')
        continue
    # linear_npy pipeline returns uint16; convert to 8-bit via per-image
    # contrast stretch for human inspection (preserves laser visibility).
    img_rgb = img[..., ::-1].astype(np.float32)
    if img.dtype == np.uint16:
        # Use the 99.5th percentile of the crop to anchor display so blooms don't clip everything.
        lo, hi = np.percentile(img_rgb, [0.5, 99.5])
        img_rgb = np.clip((img_rgb - lo) / max(1e-6, hi - lo) * 255, 0, 255).astype(np.uint8)
    lx, ly = int(r['label_x']), int(r['label_y'])
    px, py = float(r['px_cal']), float(r['py_cal'])
    H, W = img.shape[:2]
    x0 = max(0, lx - CROP); x1 = min(W, lx + CROP)
    y0 = max(0, ly - CROP); y1 = min(H, ly + CROP)
    crop = img_rgb[y0:y1, x0:x1]
    ax.imshow(crop)
    # Label (green +)
    ax.plot(lx - x0, ly - y0, '+', color='lime', markersize=18, markeredgewidth=2)
    # Prediction (red x) — only if it falls inside the crop
    if x0 <= px <= x1 and y0 <= py <= y1:
        ax.plot(px - x0, py - y0, 'x', color='red', markersize=14, markeredgewidth=2)
        pred_str = f'pred in crop'
    else:
        pred_str = f'pred at ({px:.0f},{py:.0f}) outside crop'
    ax.set_title(f"img {r['image_id']} — {bucket}\n"
                 f"err={r['err']:.1f}px ({pred_str})", fontsize=9)
    ax.axis('off')

# Legend
legend = [
    mpatches.Patch(color='lime', label='label (+)'),
    mpatches.Patch(color='red',  label='prediction (x)'),
]
fig.legend(handles=legend, loc='upper right', fontsize=10)
fig.suptitle('test:192 (red, 60.9% failure rate) — label vs prediction inspection', fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig(OUT, dpi=110, bbox_inches='tight')
print(f'wrote {OUT}')
print(f'sampled {len(selected)} frames across {len(set(b for b,_ in selected))} error buckets')
