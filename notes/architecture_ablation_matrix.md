# Architecture × inference-constraint ablation matrix

Side-by-side: run3 (ResNet-34 UNet, current production) vs run7 (HRNet-w18 + smp.Unet).
Identical 10-config grid on val. Bias offset is post-hoc, per-config from val inliers.

## Flag legend

- **R** = rig-prior (bbox mask, floor=1.0)
- **L** = line-mask (corridor ±25 px around fitted dive line)
- **C** = cascade (pass-2 refinement crop)
- **S** = sub-pixel parabolic peak refinement
- **N** = soft-snap to line (post-pred)

## Calibrated hit_n3 — side by side

| config | flags | run3 raw | run3 cal | run7 raw | run7 cal | Δ (cal) |
|---|---|---|---|---|---|---|
| 00_baseline | ----- | 0.8786 | 0.8808 | 0.8849 | 0.8873 | +0.0065 |
| 01_rig_only | R---- | 0.8908 | 0.8929 | 0.8954 | 0.8970 | +0.0040 |
| 02_line_only | -L--- | 0.8786 | 0.8808 | 0.8849 | 0.8873 | +0.0065 |
| 03_cascade_only | --C-- | 0.8790 | 0.8818 | 0.8681 | 0.8814 | -0.0003 |
| 04_subpix_only | ---S- | 0.8811 | 0.8814 | 0.8858 | 0.8901 | +0.0087 |
| 05_production | RLCSN | 0.9081 | 0.9078 | 0.8942 | 0.9069 | -0.0009 |
| 06_prod_minus_R | -LCSN | 0.8976 | 0.8994 | 0.8892 | 0.9022 | +0.0028 |
| 07_prod_minus_L | R-CSN | 0.7194 | 0.8954 | 0.8805 | 0.8929 | -0.0025 |
| 08_prod_minus_C | RL-SN | 0.9063 | 0.9066 | 0.9091 | 0.9137 | +0.0071 |
| 09_prod_minus_S | RLC-N | 0.9047 | 0.9075 | 0.8920 | 0.9053 | -0.0022 |

## Failure counts (n_fail / borderline+mid not shown / catastrophic)

| config | flags | run3 fail / catas | run7 fail / catas |
|---|---|---|---|
| 00_baseline | ----- | 384 / 167 | 363 / 136 |
| 01_rig_only | R---- | 345 / 123 | 332 / 101 |
| 02_line_only | -L--- | 384 / 167 | 363 / 136 |
| 03_cascade_only | --C-- | 381 / 167 | 382 / 136 |
| 04_subpix_only | ---S- | 382 / 167 | 354 / 136 |
| 05_production | RLCSN | 297 / 76 | 300 / 52 |
| 06_prod_minus_R | -LCSN | 324 / 106 | 315 / 70 |
| 07_prod_minus_L | R-CSN | 337 / 123 | 345 / 100 |
| 08_prod_minus_C | RL-SN | 301 / 76 | 278 / 56 |
| 09_prod_minus_S | RLC-N | 298 / 76 | 305 / 53 |

## Raw bias per config (calibrated post-hoc)

| config | flags | run3 (dx, dy) | run7 (dx, dy) |
|---|---|---|---|
| 00_baseline | ----- | (-0.277, +0.040) | (-0.341, -0.053) |
| 01_rig_only | R---- | (-0.277, +0.041) | (-0.340, -0.053) |
| 02_line_only | -L--- | (-0.277, +0.040) | (-0.341, -0.053) |
| 03_cascade_only | --C-- | (-0.259, -0.018) | (-0.426, -0.355) |
| 04_subpix_only | ---S- | (-0.262, +0.051) | (-0.325, -0.050) |
| 05_production | RLCSN | (-0.000, +0.002) | (-0.452, -0.316) |
| 06_prod_minus_R | -LCSN | (-0.201, -0.005) | (-0.452, -0.315) |
| 07_prod_minus_L | R-CSN | (+0.920, +2.037) | (-0.454, -0.319) |
| 08_prod_minus_C | RL-SN | (-0.253, +0.055) | (-0.315, -0.051) |
| 09_prod_minus_S | RLC-N | (-0.250, -0.012) | (-0.419, -0.341) |
