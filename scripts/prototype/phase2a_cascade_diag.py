"""Dump the CASCADE pass-2 heatmap on a few real frames to see if the
neighborhood around the peak gives meaningful parabolic shifts.

Hypothesis: because pass-2 is run on a 256x256 crop CENTERED on the coarse
argmax, the model finds the peak at ~(128, 128) and the heatmap is
quasi-symmetric — symmetric heatmap → parabolic shift = 0 (numerator zero).
That would explain why 95% of cascade frames have no sub-pixel shift even
though the global-tile diagnostic showed meaningful shifts.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path('/scratch/krg/c.crutchfield.642/Repos/school/e4e/fishsense/2026-05-02_laser_detector')
sys.path.insert(0, str(REPO / 'src'))

import polars as pl  # noqa: E402
from laser_detector.model import LaserDetector  # noqa: E402
from laser_detector.preprocessing.image_loader import make_cached_image_loader  # noqa: E402
from laser_detector.preprocessing.config import load_config  # noqa: E402
from laser_detector.train import TrainConfig  # noqa: E402
from laser_detector.inference import (  # noqa: E402
    predict_frame, _preprocess_tile, _reflect_pad,
    WAVELENGTH_CHANNEL, UNKNOWN_WAVELENGTH_CHANNEL,
    DEFAULT_RIG_PRIOR_BBOX, DEFAULT_RIG_PRIOR_CENTER, DEFAULT_RIG_PRIOR_SIGMA,
)

CKPT_PATH = REPO / 'data/phase2/checkpoints_sensor_bayer_50e_run3/epoch_021.pt'
VAL_PQ = REPO / 'data/audit/epoch_021_recipe_calibrated/predictions_with_meta.parquet'

DEVICE = torch.device('cuda:0')
REFINE_WIN = 256


def main():
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in ckpt['cfg'].items() if k in TrainConfig.__dataclass_fields__})
    model = LaserDetector(in_channels=cfg.in_channels, decoder_interpolation=cfg.decoder_interpolation).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    config = load_config()
    image_loader = make_cached_image_loader(
        config.image_root, Path(f'{config.cache_dir}_linear_npy'),
        pipeline='linear_npy', jpeg_quality=config.cache_jpeg_quality,
    )
    bayer_loader = make_cached_image_loader(
        config.image_root, Path(f'{config.cache_dir}_bayer_excess'),
        pipeline='bayer_excess',
    )

    df = pl.read_parquet(VAL_PQ).filter(pl.col('is_positive') & pl.col('pred_x').is_not_null())
    df = df.with_columns(
        ((pl.col('pred_x')-pl.col('label_x'))**2 + (pl.col('pred_y')-pl.col('label_y'))**2).sqrt().alias('err')
    )
    succ = df.filter(pl.col('err') <= 2).sample(n=5, seed=42)
    border = df.filter((pl.col('err') > 3) & (pl.col('err') <= 8)).sample(n=5, seed=42)
    samples = pl.concat([succ, border])

    shifts = []
    for row in samples.iter_rows(named=True):
        image_bgr = image_loader.load(row['image_path'], row['image_checksum'])
        bayer_img = bayer_loader.load(row['image_path'], row['image_checksum'])
        if image_bgr is None or bayer_img is None:
            continue
        # Get the coarse pred — re-use what's already in the parquet (it's the calibrated final);
        # for the diagnostic, simulate cascade by re-running the model on a 256x256 crop centered
        # on the (uncalibrated) coarse argmax. We don't have the uncalibrated pred handy, so use
        # the calibrated pred minus the calibration shift to approximate.
        cx = int(round(row['pred_x'] + 1.13))  # undo the calibration
        cy = int(round(row['pred_y'] + 2.07))
        h, w = image_bgr.shape[:2]
        half = REFINE_WIN // 2
        x0 = max(0, min(w - REFINE_WIN, cx - half))
        y0 = max(0, min(h - REFINE_WIN, cy - half))
        crop = image_bgr[y0:y0+REFINE_WIN, x0:x0+REFINE_WIN]
        crop_bayer = bayer_img[y0:y0+REFINE_WIN, x0:x0+REFINE_WIN]
        crop = _reflect_pad(crop, REFINE_WIN, REFINE_WIN)
        crop_bayer = _reflect_pad(crop_bayer, REFINE_WIN, REFINE_WIN)
        wavelength_value = WAVELENGTH_CHANNEL.get(row['wavelength'], UNKNOWN_WAVELENGTH_CHANNEL)
        arr = _preprocess_tile(crop, wavelength_value, bayer_excess_tile=crop_bayer)
        batch = torch.from_numpy(arr[None]).to(DEVICE)
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(batch)
        logits = out['heatmap_logits'][0, 0].float()  # [256, 256]
        probs = torch.sigmoid(logits)
        flat = probs.view(-1)
        idx = int(flat.argmax().item())
        ly, lx = divmod(idx, REFINE_WIN)
        if lx <= 0 or ly <= 0 or lx >= REFINE_WIN - 1 or ly >= REFINE_WIN - 1:
            print(f'{row["image_id"]}: cascade peak on edge ({lx},{ly}) - skip')
            continue
        P = probs.cpu().numpy()
        L = logits.cpu().numpy()
        p_c, p_xm, p_xp, p_ym, p_yp = P[ly,lx], P[ly,lx-1], P[ly,lx+1], P[ly-1,lx], P[ly+1,lx]
        l_c, l_xm, l_xp, l_ym, l_yp = L[ly,lx], L[ly,lx-1], L[ly,lx+1], L[ly-1,lx], L[ly+1,lx]
        # parabolic shift on logits
        den_x = l_xm - 2 * l_c + l_xp
        den_y = l_ym - 2 * l_c + l_yp
        dxL = 0.5 * (l_xm - l_xp) / den_x if abs(den_x) > 1e-12 else 0.0
        dyL = 0.5 * (l_ym - l_yp) / den_y if abs(den_y) > 1e-12 else 0.0
        den_xP = p_xm - 2 * p_c + p_xp
        den_yP = p_ym - 2 * p_c + p_yp
        dxP = 0.5 * (p_xm - p_xp) / den_xP if abs(den_xP) > 1e-12 else 0.0
        dyP = 0.5 * (p_ym - p_yp) / den_yP if abs(den_yP) > 1e-12 else 0.0
        clamp_dxL = 0.0 if not (-0.5 < dxL < 0.5) else dxL
        clamp_dyL = 0.0 if not (-0.5 < dyL < 0.5) else dyL
        print(f'img {row["image_id"]} err={row["err"]:.2f} peak=({lx},{ly}) (in 256x256)')
        print(f'  LOGITS: c={l_c:+.3f}  xm={l_xm:+.3f}  xp={l_xp:+.3f}  ym={l_ym:+.3f}  yp={l_yp:+.3f}')
        print(f'  PROBS:  c={p_c:.6f}  xm={p_xm:.6f}  xp={p_xp:.6f}  ym={p_ym:.6f}  yp={p_yp:.6f}')
        print(f'  parabolic dx={dxL:+.4f} (clamp -> {clamp_dxL:+.4f})  dy={dyL:+.4f} (clamp -> {clamp_dyL:+.4f})')
        shifts.append((clamp_dxL, clamp_dyL))
        print()
    if shifts:
        arr = np.array(shifts)
        print(f'=== cascade pass-2 sub-pixel shifts ===')
        print(f'  n: {len(shifts)}')
        print(f'  mean |dx|: {np.abs(arr[:,0]).mean():.4f}  mean |dy|: {np.abs(arr[:,1]).mean():.4f}')
        print(f'  count |dx|<0.01: {(np.abs(arr[:,0])<0.01).sum()} / {len(shifts)}')
        print(f'  count |dy|<0.01: {(np.abs(arr[:,1])<0.01).sum()} / {len(shifts)}')


if __name__ == '__main__':
    main()
