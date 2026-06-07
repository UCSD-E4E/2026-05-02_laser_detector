# Phase 2: worst-dive case study

Phase 1A identified two outlier dives that dominate their split's failure
counts. Drilling into them reveals two fundamentally different failure modes,
which informs the Phase 2 prioritization.

## val:427 — catastrophic distractor lock-on

- 106 positive frames, **38.7% failure rate** (41 / 106)
- All green wavelength
- Label position is extremely tight: `label_x_std = 8 px`, `label_y_std = 22 px` —
  the laser hits essentially the same sensor location frame after frame (rig is stable).
- Failure breakdown:
  - borderline 3–10 px: 6 (15%)
  - middle 10–50 px: 0
  - catastrophic >50 px: **35 (85%)**

### Catastrophic predictions cluster on consistent distractor blobs

The 35 catastrophic predictions land in just **two tight clusters**, not random
locations:

| cluster | example prediction | offset from label |
|---|---|---|
| A | `(~2065, ~1935)` | dx ≈ −23, **dy ≈ +560** |
| B | `(~1982, ~1960)` | dx ≈ −106, **dy ≈ +584** |

Same `pred_x` to within ~3 px across many frames; same `pred_y` to within
~1 px across many frames. This is not noise — the model is **consistently
picking the same distractor blob** below the true laser on each failure.

A green-laser scene around the rig's laser-position prior plausibly has a
secondary green object near the bottom of the rig prior bbox that the model
locks onto when the true laser signal is weak.

### What helps val:427

- **Not sub-pixel refinement (2A)**: the prediction is in the wrong tile entirely.
- **Not rolling-median consensus (2C)**: consecutive failures share the same
  wrong location, so the median is also wrong.
- **Tightened rig prior**: the distractor cluster is at `y ≈ 1935-1965` which
  is just inside the current bbox `y_max = 2180`. Shrinking the bbox's lower
  bound to e.g. `y_max = 1700` would hard-zero the distractor cluster on this
  dive without affecting label positions in val (label `y_max = 1428`).
  This is the cheapest available fix; needs a population check to confirm no
  legitimate labels live near `y_max`.
- **Wavelength-specific feature engineering** (deferred, Phase 2.5): green-laser
  scenes lack the bayer-excess R channel asymmetry that helps red localization.
  A green-laser-specific feature would address the root cause.

## test:192 — borderline noise dominated

- 192 positive frames, **60.9% failure rate** (117 / 192)
- All red wavelength
- Failure breakdown:
  - borderline 3–10 px: 92 (79%)
  - middle 10–50 px: 18 (15%)
  - catastrophic >50 px: 7 (6%)
- Error distribution: `p50=3.65`, `p75=7.51`, `p90=10.93`, `p99=1205.75`
  — median is right at the failure threshold.

### Borderline failures are scattered, not systematic

Across the 92 borderline failures: `median dx = +1.12`, `median dy = +0.69`,
`mean dx = +0.62`, `mean dy = +0.09`. The 1 px median bias is small enough that
these are scattered random misses around the calibrated zero, not a
per-dive systematic offset. The dive is just hard — every frame's prediction
is at the edge of acceptable error and many tip over the 3 px threshold.

### What helps test:192

- **Sub-pixel refinement (2A)**: load-bearing. Integer argmax is ±0.5 px per
  axis. Most failures are 3–5 px off; pulling a 3.5 px miss down to 3.0 px
  flips it to a hit. A 0.5 px reduction in median noise translates to a
  meaningful fraction of borderline → hit conversions.
- **Multi-frame consensus (2C)**: marginal — only 7 catastrophic frames to
  reject.
- **Investigation of per-frame difficulty**: is this dive dimmer, lower-contrast,
  motion-blurred? If a known capture quality issue, it could be quarantined
  or flagged separately rather than dragging overall metrics down.

## Implication

The two dives need different fixes. Phase 2A (sub-pixel) directly addresses
test:192 and the population's 61–65% borderline mode. The catastrophic mode
in val:427 (green-distractor lock-on) needs either tightened rig prior or a
green-specific feature — both deferred until the wider Phase 2A measurement
is in.
