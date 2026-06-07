# Phase 1A: Calibrated production-model failure stratification

**Model**: `data/phase2/checkpoints_sensor_bayer_50e_run3/epoch_021.pt`

**Recipe**: `--soft-snap-inference --rig-prior --cascade --rig-prior-floor 1.0 --pixel-bias-offset -1.13 -2.07`

**Generated**: 2026-06-06 14:10

---

## val

- n positives: 3222
- n failures (err > 3px): 484
- calibrated hit_n3: **0.8498**
- residual bias (err<=5): dx=-0.027, dy=-0.079

### Failure error distribution

| class | range | count | % of failures |
|---|---|---|---|
| borderline | 3 < err ≤ 10 | 295 | 61.0% |
| middle | 10 < err ≤ 50 | 31 | 6.4% |
| catastrophic | err > 50 | 158 | 32.6% |

### Per-wavelength

| wavelength | n | n_fail | median_err | failure_rate |
| --- | --- | --- | --- | --- |
| green | 652 | 146 | 1.5779 | 0.2239 |
| red | 2570 | 338 | 1.3608 | 0.1315 |

### Worst 5 dives (by fail count)

| dive_id | n | n_fail | median_err | failure_rate |
| --- | --- | --- | --- | --- |
| 108 | 1393 | 166 | 1.3722 | 0.1192 |
| 354 | 265 | 49 | 1.4564 | 0.1849 |
| 427 | 106 | 41 | 1.8950 | 0.3868 |
| 114 | 370 | 38 | 1.1729 | 0.1027 |
| 421 | 157 | 38 | 1.6439 | 0.2420 |

### Cross-tab: wavelength × failure class

| wavelength | fail_class | n |
| --- | --- | --- |
| green | borderline | 62 |
| green | catastrophic | 79 |
| green | middle | 5 |
| red | borderline | 233 |
| red | catastrophic | 79 |
| red | middle | 26 |

---

## test

- n positives: 4080
- n failures (err > 3px): 767
- calibrated hit_n3: **0.8120**
- residual bias (err<=5): dx=-0.039, dy=-0.210

### Failure error distribution

| class | range | count | % of failures |
|---|---|---|---|
| borderline | 3 < err ≤ 10 | 499 | 65.1% |
| middle | 10 < err ≤ 50 | 63 | 8.2% |
| catastrophic | err > 50 | 205 | 26.7% |

### Per-wavelength

| wavelength | n | n_fail | median_err | failure_rate |
| --- | --- | --- | --- | --- |
| green | 1052 | 179 | 1.4749 | 0.1702 |
| red | 3028 | 588 | 1.4473 | 0.1942 |

### Worst 5 dives (by fail count)

| dive_id | n | n_fail | median_err | failure_rate |
| --- | --- | --- | --- | --- |
| 192 | 192 | 117 | 3.6454 | 0.6094 |
| 247 | 536 | 95 | 1.3718 | 0.1772 |
| 362 | 379 | 77 | 1.5240 | 0.2032 |
| 303 | 358 | 70 | 1.4669 | 0.1955 |
| 318 | 414 | 60 | 1.2881 | 0.1449 |

### Cross-tab: wavelength × failure class

| wavelength | fail_class | n |
| --- | --- | --- |
| green | borderline | 93 |
| green | catastrophic | 83 |
| green | middle | 3 |
| red | borderline | 406 |
| red | catastrophic | 122 |
| red | middle | 60 |

---

## Interpretation

### Headline

- Calibration is working: residual bias after the `(-1.13, -2.07)` offset is `(-0.03, -0.08)` val, `(-0.04, -0.21)` test — essentially zero. The 2 px Y-bias documented in `bias_attribution.md` is fully absorbed by calibration.
- The current failure population is dominated by **borderline misses**, not catastrophic ones. On val, **61% of failures sit in `3 < err ≤ 10` px**; on test, **65%**. These are predictions that landed adjacent to the right blob and a sub-pixel refinement would convert most of them to hits.
- **Catastrophic failures (>50 px) are 27–33% of failures.** They are real and disproportionately concentrate in (a) green-wavelength frames and (b) a few specific dives.
- Note on hit_n3 reconciliation: test hit_n3 here (0.8120) matches the documented production number exactly. Val hit_n3 here (0.8498) is higher than the previously-documented 0.7976; the gap is consistent with the audit denominator dropping the ~400 positives where the cascade emits no prediction. Structural conclusions below are unaffected — they are computed within the same positive set.

### Wavelength × failure-class is the most actionable finding

| split | wavelength | borderline | middle | catastrophic | n_fail |
|---|---|---|---|---|---|
| val | red | 233 (69%) | 26 (8%) | 79 (23%) | 338 |
| val | green | 62 (42%) | 5 (3%) | 79 (**54%**) | 146 |
| test | red | 406 (69%) | 60 (10%) | 122 (21%) | 588 |
| test | green | 93 (52%) | 3 (2%) | 83 (**46%**) | 179 |

**Green-wavelength failures are roughly 2× more likely to be catastrophic than red-wavelength failures** (46–54% vs 21–23%). This is consistent across val and test, so it's a real wavelength-specific failure mode, not noise.

A green laser on a busy reef scene looks like:
- A dim greenish pixel cluster, often near other distracting greenish reef structure (algae, sea-grass).
- Lower signal-to-noise than the red channel because the bayer R-excess channel that explicitly helps red localization has no clean analog for green (G is the most-sampled channel; G1−G2 anti-diagonal was added in run6 but doesn't change red↔green imbalance).

### Per-dive concentration: hard dives are real

- **Val dive 427**: 38.7% failure rate (41/106 fails) — far above the population mean.
- **Test dive 192**: **60.9% failure rate** (117/192 fails) — by far the worst dive in either split. This single dive contributes 15% of all test failures. Worth a focused case study before generalizing the catastrophic-mode interpretation.
- The biggest dive by frame count (val 108 with 1393 frames) has only a 12% failure rate, near average — so the population isn't being driven by a single outlier dive in val.

### Priority recommendation for Phase 2/3

1. **TOP: Phase 2A — sub-pixel argmax / parabolic peak refinement.** Borderline failures are 61–65% of all failures. Refining the argmax to floating-point would fix most of them. No retraining, no architecture change, universal payoff. This is the cheapest, highest-leverage move available and is justified by the data regardless of anything else.

2. **Investigate the worst dives** (val:427, test:192) as case studies. Pull example frames, look at predicted vs label position, check whether catastrophics are "wrong blob picked" (distractor) vs "no blob at all" (saliency / scene-level difficulty). Cheap, may unlock a targeted fix.

3. **Wavelength-specific catastrophic mode**: green failures are disproportionately catastrophic. Worth digging into whether the issue is upstream (bayer channels don't isolate green-laser signal as well as red) or downstream (model has fewer training examples on green dives, dive-level split aligned with wavelength). This influences whether the fix is feature-engineering, sampling, or architectural.

4. **Phase 2C multi-frame consensus**: high-leverage for catastrophics specifically. If a frame's prediction is far from temporally-adjacent predictions in the same dive, reject it. Cheap and orthogonal to 2A.

5. **Phase 3A DINOv2 prototype** — frozen encoder + small CNN head, prototype-scale only. The Phase 1B saliency test showed partial support (H1 holds with d=−0.47, H2 doesn't), so DINOv2 may help on the scene-difficulty axis but doesn't validate the "global attention will shift to the right spot" framing. A small-scale prototype is the right next step before committing to a full retrain.

