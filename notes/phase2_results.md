# Phase 2 results: sub-pixel refinement + consensus + worst-dive case study

Production stack: run3 epoch_021 + recipe + `--pixel-bias-offset -1.13 -2.07`.
Comparison: same stack with `--subpixel-refine` (Phase 2A) added.

## Phase 2A — three iterations

The initial implementation (v1/v2) gave near-zero gain because of a bf16 sigmoid
argmax tie-break artifact. See `bias_attribution.md` ⚠️ top section for the
full root-cause and revision. v3 is the corrected implementation.

### v1: parabolic peak on bf16-saturated sigmoid probs (no gain)

| split | baseline | + sub-pixel v1 | delta |
|---|---|---|---|
| val  | 0.8498 | 0.8501 | +0.0003 |
| test | 0.8120 | 0.8125 | +0.0005 |

Only 5% of frames showed any shift. v_c, v_xp, v_yp at peak all saturated to
exactly 1.0 in bf16; parabolic fit produced exactly dx=+0.5 which the strict
clamp `−0.5 < dx < 0.5` rejected → 0 shift.

### v3: fp32 sigmoid + parabolic peak on logits (production)

Two-line fix in `src/laser_detector/inference.py`: split out
`heatmap_logits = out["heatmap_logits"].float()` before sigmoid, refine the
peak on logits (bf16-stable, monotonic with sigmoid so same peak location).

| split | baseline (bf16) | v3 (fp32 + sub-pixel, calibrated) | delta |
|---|---|---|---|
| val  | 0.8498 | **0.8951** | **+4.53 pp** |
| test | 0.8120 | **0.8544** | **+4.24 pp** |

Val-derived calibration offset for v3 = `(−0.20, −0.006)`. Held out on test:
0.8544 with val-derived offset vs. 0.8549 with test-derived = 0.05 pp gap,
no meaningful overfit. Failure-class breakdown:

| split | scenario | n_fail | borderline (3–10) | middle (10–50) | catastrophic (>50) |
|---|---|---|---|---|---|
| val  | baseline | 484 | 295 (61%) | 31 | 158 (33%) |
| val  | v3       | 338 | 188 (56%) | 27 | 123 (36%) |
| test | baseline | 767 | 499 (65%) | 63 | 205 (27%) |
| test | v3       | 594 | 396 (67%) | 63 | 135 (23%) |

Borderline drop 36% / 21% (val / test); catastrophic drop 22% / 34%. Worst-dive
test:192 lifts from 0.391 → 0.427 (+3.7 pp); val:427 unchanged (catastrophic
distractor mode, not fixable by sub-pixel).

## Re-evaluation of run5 and run6 under v3 inference

The original `bias_attribution.md` framing claimed run5 (bilinear decoder)
and run6 (G_diff channel) each reduced architectural bias relative to run3.
Those measurements used bf16 inference, which inflates the measured bias.
Both runs were re-audited under v3 (fp32 sigmoid + sub-pixel) with
val-derived calibration applied to test.

| run | inference | val cal | test cal | val raw bias |
|---|---|---|---|---|
| run3 (production) | bf16 (pre-fix) | 0.8498 | 0.8120 | (−1.13, −2.07) |
| run5 (bilinear) | bf16 (pre-fix) | 0.7751 | 0.7961 | (−1.08, −1.90) |
| run6 (G_diff)   | bf16 (pre-fix) | 0.7923 | 0.7912 | (−0.08, −1.93) |
| **run3** | **v3** | **0.8951** | **0.8544** | **(−0.20, −0.006)** |
| run5 | v3 | 0.8693 | 0.8458 | (+0.11, +0.45) |
| run6 | v3 | 0.8768 | 0.8287 | (+0.62, −0.06) |

Under v3 inference, **run3 is the best architecture on both val and test**.
The architectural choices in run5 (bilinear decoder upsample) and run6
(G_diff anti-diagonal Bayer channel) did not win in either precision regime
— gaps to run3 are slightly smaller under v3 but in the same direction.
The "G_diff cracks the X bias" framing in `bias_attribution.md` is refuted
by the v3 numbers: run6's raw X bias under v3 is actually larger (+0.62 px)
than run3's (−0.20 px).

## Phase 2C: rolling-median consensus — NEGATIVE

Tested as a post-processing step on the baseline parquet using a window=5 rolling
median over dive-ordered frames, with predictions farther than threshold T from
the local median snapped to the median. Results across T:

| split | T=10 | T=30 | T=50 | T=100 |
|---|---|---|---|---|
| val  | -0.278 | -0.113 | -0.058 | -0.018 |
| test | -0.334 | -0.165 | -0.099 | -0.039 |

All deltas are NEGATIVE. The failure mode includes consecutive consistent
failures (val:427 green-laser distractor lock-on, see notes/phase2_worst_dives.md)
whose local median is at the WRONG location, so good neighboring predictions get
snapped onto the wrong cluster. Naive rolling consensus is not viable; a smarter
variant (confidence-weighted, per-frame veto) is a Phase 2 follow-up.

## Worst-dive case study

See [phase2_worst_dives.md](phase2_worst_dives.md). Two qualitatively different
failure modes:
- **val:427** (green): 85% catastrophic, distractor lock-on at consistent wrong location.
  Cheapest fix is to tighten the rig-prior bbox lower-y; needs a pop check on label range.
- **test:192** (red): 79% borderline, noise-dominated. Sub-pixel refinement is the
  expected primary intervention; this A/B reports the measured lift on this dive.

**Dive val:427** baseline hit_n3 = 0.6132 → sub-pixel = 0.6132 (delta +0.0000)

**Dive test:192** baseline hit_n3 = 0.3906 → sub-pixel = 0.3906 (delta +0.0000)

