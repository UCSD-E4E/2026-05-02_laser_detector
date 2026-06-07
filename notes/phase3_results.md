# Phase 3 results

Phase 3 split into three sub-phases:
- 3.0 — closure / repo state sync after the Phase 2 v3 discovery
- 3.1 — targeted cheap inference-time fixes for the residual failure modes
- 3.2 — (deferred) focused retrain if 3.1 reveals leverage
- 3.3 — (deferred) exploratory architectural alternatives

## 3.1a — rig-prior bbox y_max tightening (superseded by 3.1d)

Tightened the rig-prior bbox y_max from 2180 to 1700 to clip wandering
catastrophic predictions that land well below any legitimate label
(green label y_max = 1552 val / 1624 test; red 1520 / 1514).

Measured on val: hit_n3 0.8951 → 0.9019 (+0.68 pp). Catastrophic
failures cut 123 → 100. Borderline rate unchanged. Test audit was
killed mid-run; not re-collected because 3.1d supersedes this.

CLI: `--rig-prior-bbox-ymax INT`. Implementation in
`src/laser_detector/train.py` (`inference_rig_prior_bbox_ymax`) and
`src/laser_detector/inference.py:_run_val_inference`.

## 3.1b — test:192 root-cause investigation

test:192 is the worst dive in the test split: 192 frames, 60.9%
failure rate, all red wavelength. The orchestrator analysis showed
79% borderline + 6% catastrophic + 15% middle failures, with
predictions tracking labels well on average (median dx ≈ +0.4,
dy ≈ +0.9 after v3) but heavy-tailed (mean dx=9, dy=32).

Visual inspection of 12 stratified frames + dedicated review of the
6 catastrophic frames found:
- **Labels are correct** on every reviewed frame (10/12 hits and
  borderlines have visible laser on label; 6/6 catastrophics have
  the laser on the labeled location).
- **The label spread is genuinely wide**: x_std=47, y_std=157
  (3-5× typical for a test red dive). Reflects real camera/rig
  motion in this dive, not noisy labels.
- The 6 catastrophic predictions land at almost identical
  coordinates: ~(2217, 2080). A fixed-location distractor that the
  model consistently picks. Same failure mode as val:427.
- 1 of 6 (img 51957) is a "reflected laser" (specular reflection
  elsewhere in the frame), 5 of 6 are the (2217, 2080) lock-on.

Verdict: not a label-quality problem; the model has a learned
preference for the (2217, 2080) location on this dive. Not addressable
by the line mask (the distractor sits 5.3 px from the dive line —
inside any reasonable corridor). Phase 3.2 territory.

JPEGs of the 6 catastrophic frames + 2 reference hits are at
[notes/figures/test192_review/](figures/test192_review/) (marked +
unmarked variants).

## 3.1d — per-dive line corridor mask (PRODUCTION)

Mask: zero heatmap pixels farther than `corridor_px` from the fitted
dive line `a*x + b*y + c = 0`. Per-dive geometric constraint, much
tighter than the static rig bbox.

Sizing: population label-to-line perpendicular distance p99 is
2.93 px val / 8.98 px test. A ±25 px corridor is safe (includes p99
of all labels with margin) while masking the val:427 distractor at
203 px off-line and similar cases.

CLI: `--line-mask-corridor-px FLOAT`. Implemented as
`_line_mask_for_tile` in `inference.py`, multiplied into
`heatmap_probs` alongside the rig prior mask in both
`predict_frame` and the cascade's coarse pass (with `alpha_max=0`
to suppress the coarse soft-snap, since snap is deferred to after
pass-2).

### A/B (production stack: run3 + v3 + calib (−0.20, −0.006))

| metric | v3 baseline | + line mask ±25 | delta |
|---|---|---|---|
| val hit_n3 | 0.8951 | **0.9081** | **+1.30 pp** |
| val hit_n4 | 0.9184 | 0.9320 | +1.36 pp |
| val fail | 338 | 296 | −42 |
| val borderline | 188 | 191 | +3 |
| val catastrophic | 123 | 76 | **−47 (−38%)** |
| test hit_n3 | 0.8544 | **0.8615** | **+0.71 pp** |
| test hit_n4 | 0.8863 | 0.8914 | +0.51 pp |
| test fail | 594 | 565 | −29 |
| test borderline | 396 | 394 | −2 |
| test catastrophic | 135 | 105 | **−30 (−22%)** |

The mask attacks the catastrophic distractor mode specifically;
borderline rate is unchanged.

### Per wavelength on test

| wavelength | hit_n3 | n | fail | catas |
|---|---|---|---|---|
| red | 0.8530 | 3028 | 445 | 54 |
| green | **0.8859** | 1052 | 120 | 51 |

Green improves more than red. Green dives have more off-line
distractors, which the mask kills cleanly.

### Worst-dive impact

- val:427 (green): 0.6132 baseline → 0.8019 line mask = **+18.87 pp**.
  This was the catastrophic distractor dive from the Phase 2 case
  study. Line mask is the targeted fix.
- test:192 (red): 0.3906 → 0.4271 (v3) → 0.4323 (+ line mask) = +0.5 pp
  on top of v3. As predicted from the (2217, 2080) being on-line.

## Cumulative across the session

| stack | val hit_n3 | test hit_n3 |
|---|---|---|
| baseline (bf16 prod, manual calibration) | 0.8498 | 0.8120 |
| + v3 (fp32 sigmoid + sub-pixel parabolic) | 0.8951 | 0.8544 |
| + line mask (±25 px) | **0.9081** | **0.8615** |
| **cumulative gain** | **+5.83 pp** | **+4.95 pp** |

Catastrophic failures:
- val: 158 (baseline) → 123 (v3) → 76 (line mask) = −52%
- test: 205 (baseline) → 135 (v3) → 105 (line mask) = −49%

All gains are from inference-time changes; no retraining required.

## Remaining failure population (test, n=565)

- 394 borderline (3 < err ≤ 10) — **70% of remaining failures**
- 66 middle (10 < err ≤ 50)
- 105 catastrophic (err > 50) — concentrated in test:192 (107 hits
  out of 192 frames is the largest single contributor)

The dominant residual is the borderline mode. The line mask did
not move it (and was not expected to: it's a peak-precision
problem, not a peak-location problem). Closing that mode requires
either:
- Architectural changes that produce sharper peaks (HRNet, offset
  regression head)
- Multi-scale inference ensemble (cheap exploration)

Phase 3.2 and 3.3 considerations are tracked in DESIGN.md
ongoing-work notes; not yet committed direction.
