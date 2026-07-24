# Run7 results — HRNet-w32 encoder retrain

**Checkpoint**: `/scratch/krg/c.crutchfield.642/Repos/school/e4e/fishsense/2026-05-02_laser_detector/data/phase2/checkpoints_sensor_bayer_50e_run7_hrnet_w18/epoch_021.pt` (best epoch from checkpoints_sensor_bayer_50e_run7_hrnet_w18)

**Recipe**: same as production — `--soft-snap-inference --rig-prior --cascade --rig-prior-floor 1.0 --subpixel-refine --line-mask-corridor-px 25 --pixel-bias-offset DX DY`

**Generated**: 2026-06-08 12:24 (bf16 stack). **fp32 refit added 2026-07-24 — see below.**

## ⚠️ fp32 refit (2026-07-24, issue #13)

Full val + test fp32 audits on the same recipe. Val-inlier bias:
`(−0.436, −0.441)` — materially different from run3's fp32 val bias
of `(−0.179, −0.023)`. Original bf16-derived run7 bias was
`(−0.452, −0.316)`; new fp32 y-shift is larger.

| split | before (bf16 + own bias) | after (fp32 + own bias) | Δ |
|---|---:|---:|---:|
| val hit_n3 | 0.9069 | **0.9088** | +0.19 pp |
| test hit_n3 | 0.8620 | **0.8632** | +0.12 pp |

**Caveat — offset transfer is imperfect on run7.** Val-derived
`(−0.436, −0.441)` gives essentially flat hit_n3 on test (within noise
across all bias choices) and a +0.56 pp hit_n4 gain, but degrades
median test error from 0.915 → 1.099 px. Per-wavelength on test, green
residual sign flips vs val (green val (−0.178, −0.660); green test
(+0.040, −0.062)) while red aligns. The offset optimizes zero mean
signed residual (right for downstream 3D pixel-space triangulation);
it does not optimize hit_n3 or median error.

For downstream users of run7:
- If pixel-space predictions must be unbiased (3D reconstruction) → use
  the val-derived offset.
- If median-frame tightness matters more than mean bias → no offset;
  hit_n3 is within noise either way.

run3 shows neither of these tensions and its val-derived offset
transfers cleanly on both metrics.

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

