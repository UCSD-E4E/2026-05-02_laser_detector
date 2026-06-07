"""Diagnose why Phase 2A sub-pixel refinement does nothing.

Loads run3, runs on a handful of val frames, captures the heatmap LOGITS
(pre-sigmoid) AND the post-sigmoid probabilities at the argmax peak and its
4-neighborhood. For each frame, computes the parabolic shift from both
representations and reports whether refinement on logits would shift the
prediction farther than refinement on sigmoid probs.

Hypothesis: sigmoid saturates at strong peaks (logit > ~5 → prob > 0.99),
so the 3-point cross around the peak has all values near 1.0 and the
parabolic denominator collapses. Logits avoid the saturation.

Runs single-GPU, single-process. ~10 frames * ~3s each = ~30s.
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
from laser_detector.train import TrainConfig  # noqa: E402
from laser_detector.inference import (  # noqa: E402
    DEFAULT_RIG_PRIOR_BBOX, DEFAULT_RIG_PRIOR_CENTER, DEFAULT_RIG_PRIOR_SIGMA,
    _preprocess_tile, _reflect_pad, _rig_prior_for_tile, _subpixel_refine_peak,
    compute_tile_grid, WAVELENGTH_CHANNEL, UNKNOWN_WAVELENGTH_CHANNEL,
)
from laser_detector.preprocessing.config import load_config  # noqa: E402

CKPT_PATH = REPO / 'data/phase2/checkpoints_sensor_bayer_50e_run3/epoch_021.pt'
VAL_PQ = REPO / 'data/audit/epoch_021_recipe_calibrated/predictions_with_meta.parquet'

DEVICE = torch.device('cuda:0')
TILE = 1024
OVERLAP = 256
REFINE_WIN = 256


def parabolic_shift(v_c: float, v_xm: float, v_xp: float, v_ym: float, v_yp: float) -> tuple[float, float]:
    """Same as inference._subpixel_refine_peak's core math, returned for inspection."""
    den_x = v_xm - 2 * v_c + v_xp
    den_y = v_ym - 2 * v_c + v_yp
    dx = 0.5 * (v_xm - v_xp) / den_x if abs(den_x) > 1e-12 else 0.0
    dy = 0.5 * (v_ym - v_yp) / den_y if abs(den_y) > 1e-12 else 0.0
    return dx, dy


def main():
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in ckpt['cfg'].items() if k in TrainConfig.__dataclass_fields__})
    model = LaserDetector(in_channels=cfg.in_channels, decoder_interpolation=cfg.decoder_interpolation).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'loaded checkpoint epoch={ckpt.get("epoch", -1)}, in_channels={cfg.in_channels}')

    config = load_config()
    image_loader = make_cached_image_loader(
        config.image_root, Path(f'{config.cache_dir}_linear_npy'),
        pipeline='linear_npy', jpeg_quality=config.cache_jpeg_quality,
    )
    bayer_loader = make_cached_image_loader(
        config.image_root, Path(f'{config.cache_dir}_bayer_excess'),
        pipeline='bayer_excess',
    )

    # Pick a stratified sample: 5 success frames + 5 borderline-failure frames + 2 catastrophic.
    df = pl.read_parquet(VAL_PQ)
    df = df.filter(pl.col('is_positive') & pl.col('pred_x').is_not_null())
    df = df.with_columns(((pl.col('pred_x') - pl.col('label_x'))**2 + (pl.col('pred_y') - pl.col('label_y'))**2).sqrt().alias('err'))
    succ = df.filter(pl.col('err') <= 2).sample(n=5, seed=42)
    border = df.filter((pl.col('err') > 3) & (pl.col('err') <= 8)).sample(n=5, seed=42)
    catas = df.filter(pl.col('err') > 100).sample(n=2, seed=42)
    samples = pl.concat([succ, border, catas])
    print(f'sampled {samples.height} frames (5 success / 5 borderline / 2 catastrophic)\n')

    rows = []
    for row in samples.iter_rows(named=True):
        image_bgr = image_loader.load(row['image_path'], row['image_checksum'])
        bayer_img = bayer_loader.load(row['image_path'], row['image_checksum'])
        if image_bgr is None or bayer_img is None:
            print(f'skip {row["image_id"]}: image load failed')
            continue

        h, w = image_bgr.shape[:2]
        grid = compute_tile_grid(h, w, tile=TILE, overlap=OVERLAP)
        padded = _reflect_pad(image_bgr, grid.padded_h, grid.padded_w)
        bayer_padded = _reflect_pad(bayer_img, grid.padded_h, grid.padded_w)
        wavelength_value = WAVELENGTH_CHANNEL.get(row['wavelength'], UNKNOWN_WAVELENGTH_CHANNEL)

        # Run all tiles, find the winning one, capture both logits and sigmoid probs.
        rig_bbox = DEFAULT_RIG_PRIOR_BBOX
        rig_center = DEFAULT_RIG_PRIOR_CENTER
        rig_sigma = DEFAULT_RIG_PRIOR_SIGMA
        rig_floor = 1.0  # production recipe uses --rig-prior-floor 1.0 (pure bbox mask)

        best_val = -1.0
        best_tile_logits = None
        best_tile_probs = None
        best_local = (-1, -1)
        best_origin = (-1, -1)
        with torch.inference_mode():
            for ox, oy in grid.origins:
                tile_bgr = padded[oy:oy+TILE, ox:ox+TILE]
                tile_bayer = bayer_padded[oy:oy+TILE, ox:ox+TILE]
                arr = _preprocess_tile(tile_bgr, wavelength_value, bayer_excess_tile=tile_bayer)
                batch = torch.from_numpy(arr[None]).to(DEVICE)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = model(batch)
                logits = out['heatmap_logits'][0, 0].float()  # [H, W]
                probs = torch.sigmoid(logits)
                rig_mask = torch.from_numpy(
                    _rig_prior_for_tile(ox, oy, TILE, rig_bbox, rig_center, rig_sigma, rig_floor)
                ).to(probs.device)
                probs_masked = probs * rig_mask
                idx = int(probs_masked.view(-1).argmax().item())
                val = float(probs_masked.view(-1).max().item())
                if val > best_val:
                    best_val = val
                    ly, lx = divmod(idx, TILE)
                    best_local = (lx, ly)
                    best_origin = (ox, oy)
                    best_tile_logits = logits.detach().cpu().numpy()
                    best_tile_probs = probs_masked.detach().cpu().numpy()
        lx, ly = best_local
        # Inspect 3x3 around peak (pre-cascade)
        if lx <= 0 or ly <= 0 or lx >= TILE - 1 or ly >= TILE - 1:
            print(f'{row["image_id"]}: peak on edge ({lx}, {ly}), skip')
            continue
        L = best_tile_logits
        P = best_tile_probs
        l_c, l_xm, l_xp, l_ym, l_yp = L[ly, lx], L[ly, lx-1], L[ly, lx+1], L[ly-1, lx], L[ly+1, lx]
        p_c, p_xm, p_xp, p_ym, p_yp = P[ly, lx], P[ly, lx-1], P[ly, lx+1], P[ly-1, lx], P[ly+1, lx]
        dxL, dyL = parabolic_shift(l_c, l_xm, l_xp, l_ym, l_yp)
        dxP, dyP = parabolic_shift(p_c, p_xm, p_xp, p_ym, p_yp)
        rows.append({
            'image_id': row['image_id'], 'err': row['err'], 'lx': lx, 'ly': ly,
            'p_c': p_c, 'p_xm': p_xm, 'p_xp': p_xp, 'p_ym': p_ym, 'p_yp': p_yp,
            'l_c': l_c, 'l_xm': l_xm, 'l_xp': l_xp, 'l_ym': l_ym, 'l_yp': l_yp,
            'dx_probs': dxP, 'dy_probs': dyP, 'dx_logits': dxL, 'dy_logits': dyL,
        })

        print(f'img {row["image_id"]} err={row["err"]:.2f} px  peak=({lx},{ly})')
        print(f'  PROBS (sigmoid):  center={p_c:.6f}  xm={p_xm:.6f}  xp={p_xp:.6f}  ym={p_ym:.6f}  yp={p_yp:.6f}')
        print(f'  LOGITS (raw):     center={l_c:+.3f}    xm={l_xm:+.3f}    xp={l_xp:+.3f}    ym={l_ym:+.3f}    yp={l_yp:+.3f}')
        print(f'  parabolic shift on PROBS:  dx={dxP:+.4f}  dy={dyP:+.4f}')
        print(f'  parabolic shift on LOGITS: dx={dxL:+.4f}  dy={dyL:+.4f}')
        print()

    if rows:
        # Summary
        print('=== summary ===')
        print(f'  mean center prob:       {np.mean([r["p_c"] for r in rows]):.6f}')
        print(f'  mean center logit:      {np.mean([r["l_c"] for r in rows]):+.3f}')
        print(f'  mean |dx_probs|:        {np.mean([abs(r["dx_probs"]) for r in rows]):.4f}')
        print(f'  mean |dy_probs|:        {np.mean([abs(r["dy_probs"]) for r in rows]):.4f}')
        print(f'  mean |dx_logits|:       {np.mean([abs(r["dx_logits"]) for r in rows]):.4f}')
        print(f'  mean |dy_logits|:       {np.mean([abs(r["dy_logits"]) for r in rows]):.4f}')
        print(f'  count probs-shift>0.05: {sum(1 for r in rows if abs(r["dx_probs"]) > 0.05 or abs(r["dy_probs"]) > 0.05)} / {len(rows)}')
        print(f'  count logits-shift>0.05:{sum(1 for r in rows if abs(r["dx_logits"]) > 0.05 or abs(r["dy_logits"]) > 0.05)} / {len(rows)}')


if __name__ == '__main__':
    main()
