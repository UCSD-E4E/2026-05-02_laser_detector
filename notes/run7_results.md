# Run7 results — HRNet-w32 encoder retrain

**Checkpoint**: `/scratch/krg/c.crutchfield.642/Repos/school/e4e/fishsense/2026-05-02_laser_detector/data/phase2/checkpoints_sensor_bayer_50e_run7_hrnet_w18/epoch_021.pt` (best epoch from checkpoints_sensor_bayer_50e_run7_hrnet_w18)

**Recipe**: same as production — `--soft-snap-inference --rig-prior --cascade --rig-prior-floor 1.0 --subpixel-refine --line-mask-corridor-px 25 --pixel-bias-offset DX DY`

**Generated**: 2026-06-08 12:24

---

## A/B vs run3 (current production)

| | val hit_n3 | test hit_n3 | val raw bias |
|---|---|---|---|
| run3 + v3 + line mask | 0.9081 | 0.8615 | (-0.20, -0.006) |
| **run7 (HRNet-w32) + v3 + line mask** | **0.9069** | **0.8620** | (-0.452, -0.316) |

val delta vs run3: -0.0012

test delta vs run3: +0.0005

## Failure breakdown

| split | fail | borderline | catastrophic |
|---|---|---|---|
| val (run3 baseline) | 296 | 191 | 76 |
| val (run7) | 300 | 222 | 52 |
| test (run3 baseline) | 565 | 394 | 105 |
| test (run7) | 563 | 403 | 107 |

