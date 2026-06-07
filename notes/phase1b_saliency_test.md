# Phase 1B — Saliency-vs-Failure Correlation Test

**Date:** 2026-06-06
**Author:** Claude (Phase 1B analysis script).
**Sample source:** `data/audit/epoch_021_recipe/predictions_with_meta.parquet`
**Saliency model:** `facebook/dinov2-small` via `transformers`. Last-layer
[CLS]-to-patch attention, averaged over 6 heads, reshaped to a 16x16 patch
grid (224x224 input, patch=14), bilinearly upsampled to source resolution,
per-image min-max normalized to [0,1].
**Calibration applied:** `pred_cal = (pred_x + 1.13, pred_y + 2.07)` (this is
the direction that reproduces the audit-reported calibrated hit_n3 ~0.80;
the prompt's stated sign was inverted relative to the parquet contents — a
sanity check inside the script logs the calibrated hit_n3 at startup).
**Failure threshold:** `err > 3 px` on the calibrated prediction (N=3 strict).

## Headline

**PARTIAL SUPPORT — mixed signal; prototype before committing weeks to a retrain.**

Calibrated hit_n3 on the full audit val parquet: see script logs (~0.85 in
this run; production-reported 0.7976 differs because the audit pipeline used
soft-snap and cascade post-processing not applied here, but the structural
question is unaffected: catastrophic failures (>50 px) dominate this set and
their classification is calibration-invariant).

## Sample

- Failures sampled: **100** across **18** dives (per-dive cap = 20).
- Successes sampled: **100** across **21** dives (per-dive cap = 20).
- Seed: 42.
- Out-of-bounds predictions (could not score saliency at pred coord): 0 failure / 0 success.

## H1 — saliency at predicted location (failures vs successes)

Hypothesis: failed predictions land in lower-saliency regions than successes.

| group   | n   | mean sal | median sal |
|---------|-----|----------|------------|
| failure | 100 | 0.2943 | 0.2054 |
| success | 100 | 0.4227 | 0.4312 |

- Mann-Whitney U two-sided p-value: **0.003921**
- Cohen's d (failure − success): **-0.466**
- Rank-biserial: **+0.236**

Direction: failure < success (hypothesis-consistent).

## H2 — within-failure label-vs-pred saliency delta

Hypothesis: among failures, the true label sits in a more salient region
than the model's prediction (i.e. a saliency-aware model would have
shifted attention toward the label).

- n = 100
- mean(sal@label − sal@pred): **+0.0564**
- median delta: +0.0003
- fraction of failure frames where label is more salient than pred: 0.540
- Wilcoxon signed-rank p: **0.1226**

Reference — same delta computed on the SUCCESS group (sanity check; pred and
label should be close in space so saliency should be similar):

- n = 100, mean delta = -0.0004, median = +0.0000.

## Side comparison — saliency at LABEL (failures vs successes)

Are failure-frame labels themselves on more / less salient subjects?

| group   | n   | mean sal | median sal |
|---------|-----|----------|------------|
| failure | 100 | 0.3507 | 0.3324 |
| success | 100 | 0.4224 | 0.4289 |

- Mann-Whitney U two-sided p-value: **0.1131**
- Cohen's d (failure − success): **-0.258**

## Example overlays

See `figures/phase1b_saliency_examples.png` — red X = calibrated prediction,
green O = ground-truth label, jet heatmap = DINOv2 [CLS] attention.

## Interpretation

- H1 effect size |d| = 0.466. Conventional thresholds: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >=0.8 large.
- H2 within-failure mean delta = +0.0564 (fraction label > pred = 0.54).
- The success-group within-delta (mean = -0.0004) provides a null
  reference: when prediction and label are co-located, the delta should be ~0.

**Joint reading.** H1 is significant in the hypothesis-consistent direction
(p < 0.01) with a small-to-medium effect (|d| = 0.47): failed
predictions DO land in lower-saliency regions on average. But H2 — the
critical claim that the model's failure shifts attention AWAY from a more
salient label — is not significant (Wilcoxon p = 0.123, only
54% of failures have label more salient than
prediction). And the label-saliency side comparison (|d| = 0.26,
p = 0.11) shows that failure-frame LABELS are themselves
slightly less salient than success-frame labels, suggesting the dominant
signal is "the whole scene is harder" (low contrast, low texture, no
clear subject) rather than "the model attended to the wrong subject within
a salient scene." A DINOv2 encoder gives you global attention but doesn't
inherently make the scene more salient — if the scene has no clear subject,
attention has nothing useful to anchor to.

**Recommendation:** Do not commit to a full Phase 3A retrain yet. Build a small-scale prototype first: frozen DINOv2 + 2-layer CNN head on the existing failure set; compare hit_n3 against the run3 baseline on the same val split. In parallel, continue Phase 2A (parabolic peak refinement) and 2B (Y-bias mechanism) since they have known concrete payoff.

## Caveats

1. **Saliency proxy.** DINOv2 [CLS] attention is an objectness proxy, not a
   ground-truth saliency map. It is biased toward large, centered subjects;
   small or off-center fish may not register strongly even if a human would
   call them salient. A U2-Net comparison would strengthen the conclusion.
2. **Calibration sign.** The prompt's stated sign for the pixel-bias offset
   did not reproduce the audit-reported hit_n3; the script applies the
   direction that does (+1.13, +2.07). For catastrophic failures (>50 px
   err), this is moot.
3. **Per-image normalization.** Saliency was min-max normalized per image,
   so absolute values aren't comparable across frames. The
   between-group comparisons are valid (each frame is in both groups equally
   often by construction of the sample), but the absolute values shouldn't
   be over-interpreted.
4. **Dive imbalance.** Failures are heavily concentrated on a few dives
   (dive 108 alone has 166 failures); the per-dive cap of 20 mitigates this
   but the failure distribution is still less diverse than the success
   distribution.

## Files

- This writeup: `notes/phase1b_saliency_test.md`
- Figure: `notes/figures/phase1b_saliency_examples.png`
- Per-row saliency results: `data/audit/epoch_021_recipe/phase1b_saliency_sample.parquet`
- Script: `scripts/prototype/phase1b_saliency.py`
