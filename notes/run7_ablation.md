# Run7 (HRNet-w18) inference-constraint ablation matrix

**Checkpoint**: `epoch_021.pt` (best epoch from run7)

**Split**: val (n=3414 positives, 211 unloadable from dive 249)

**Bias offset**: post-hoc, derived from each config's own val inliers (err≤5 median).

**Generated**: 2026-06-08 17:27

## Flag legend

- **R** = rig-prior (bbox mask, floor=1.0)
- **L** = line-mask (corridor ±25 px around dive line)
- **C** = cascade (pass-2 refinement crop)
- **S** = sub-pixel parabolic peak refinement
- **N** = soft-snap to line at the final pred
- **B** = bias-offset calibration (always applied post-hoc here)

## Matrix

| config | flags | raw hit_n3 | cal hit_n3 | raw bias (dx, dy) | fail | border | catas |
|---|---|---|---|---|---|---|---|
| 00_baseline | ----- | 0.8849 | 0.8873 | (-0.341, -0.053) | 363 | 208 | 136 |
| 01_rig_only | R---- | 0.8954 | 0.8970 | (-0.340, -0.053) | 332 | 210 | 101 |
| 02_line_only | -L--- | 0.8849 | 0.8873 | (-0.341, -0.053) | 363 | 208 | 136 |
| 03_cascade_only | --C-- | 0.8681 | 0.8814 | (-0.426, -0.355) | 382 | 226 | 136 |
| 04_subpix_only | ---S- | 0.8858 | 0.8901 | (-0.325, -0.050) | 354 | 198 | 136 |
| 05_production | RLCSN | 0.8942 | 0.9069 | (-0.452, -0.316) | 300 | 222 | 52 |
| 06_prod_minus_R | -LCSN | 0.8892 | 0.9022 | (-0.452, -0.315) | 315 | 221 | 70 |
| 07_prod_minus_L | R-CSN | 0.8805 | 0.8929 | (-0.454, -0.319) | 345 | 222 | 100 |
| 08_prod_minus_C | RL-SN | 0.9091 | 0.9137 | (-0.315, -0.051) | 278 | 197 | 56 |
| 09_prod_minus_S | RLC-N | 0.8920 | 0.9053 | (-0.419, -0.341) | 305 | 228 | 53 |

