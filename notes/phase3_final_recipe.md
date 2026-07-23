# Phase 3 final state — recipe + architecture conclusions

**Closed**: 2026-06-08. After Phase 3.1 (line mask, ~+1 pp val) the project hit
a label-noise floor around val 0.91 / test 0.86. Phase 3.2 architecture
exploration (run7 HRNet-w18) and the 10-config inference-constraint matrix
on both architectures confirm we're at the ceiling. **No production changes
recommended.**

## Production recipe

```
--checkpoint data/phase2/checkpoints_sensor_bayer_50e_run3/epoch_021.pt
--soft-snap-inference
--rig-prior --rig-prior-floor 1.0
--cascade
--subpixel-refine
--line-mask-corridor-px 25
--pixel-bias-offset -0.179 -0.023
```

**Post-close revision (2026-07-23)**: `--pixel-bias-offset` recalibrated
under fp32 inference (issue #13); previous value was `-0.200 -0.006`
against Ada bf16. See `bias_attribution.md` for the fp32 refit
rationale and `HOW_TO_USE.md`'s "bf16 at inference — DISABLED" section.

This is run3 (ResNet-34 UNet, 6-ch bayer_excess) + fp32 inference +
logit-based parabolic peak refinement + line mask + bias offset. See
`phase3_results.md` for the cumulative trajectory:

- val hit_n3: 0.8498 (bf16 baseline) → 0.9081 (bf16 + old bias) → **0.9100** (fp32 + new bias, +0.19 pp reproducibility fix)
- test hit_n3: 0.8120 → 0.8615 (bf16 + old bias); fp32 test refit pending

## Architecture × inference-constraint ablation matrix (final)

Two architectures × 10 configs on val. See
`architecture_ablation_matrix.md` for the full table.

Key results:

| config | flags | run3 cal | run7 cal | Δ (run7−run3) |
|---|---|---|---|---|
| 00 baseline | ----- | 0.8808 | 0.8873 | +0.65 |
| 05 production | RLCSN | **0.9078** | 0.9069 | −0.09 |
| 08 prod − C | RL-SN | 0.9066 | **0.9137** | +0.71 |

### Conclusions

1. **Architecture is near-irrelevant.** All 10 configs across both architectures
   land within ±0.87 pp. The HRNet-w18 retrain (run7) does not meaningfully beat
   ResNet-34 + UNet (run3) on any config. Cross-architecture parity is strong
   evidence we're at a label-noise floor, not a model-capacity ceiling.

2. **Cascade is architecture-specific, not universal.**
   - On run3: prod (0.9078) vs prod − C (0.9066) → cascade adds ~+0.12 pp.
   - On run7: prod (0.9069) vs prod − C (0.9137) → cascade HURTS by −0.68 pp.
   The cascade pass-2 helps when the coarse pass needs sharpening; HRNet's
   coarse pass is already sharper than the cascade refinement crop, so
   cascade actively replaces good coarse predictions with worse refined ones.

3. **Line mask is conditional, not universal.**
   - Line-only (config 02) is identically zero gain over baseline on both
     architectures — no predictions are ever far enough off-line to mask.
   - But prod − L loses 1.2-1.4 pp on both architectures: the line mask
     contributes meaningfully only when combined with the other constraints
     that surface borderline-line cases.

4. **Sub-pixel parabolic refinement is genuinely small.** +0.3 pp on run7
   (config 04 vs 00), basically zero on run3. The bf16 sigmoid fix was the
   load-bearing change; the parabolic refinement itself is marginal.

5. **The 0.91 val ceiling is the labeler-click-noise floor.** Perpendicular
   label-to-line σ ≈ 0.77 px (clean dives); along-line variance is harder to
   measure but bounded by the same source. The 70% borderline failure rate
   matches the click-noise σ almost exactly.

## What was tried this session that did not work

| intervention | result | reason |
|---|---|---|
| Multi-scale inference ensemble (0.95×/1.0×/1.05×) | val mean err 2.3 → 240 px | model brittle to scale; no scale aug in training |
| Naive rolling-median consensus | −0.06 to −0.33 hit_n3 | consecutive failures anchor median to wrong location |
| run5 bilinear decoder retrain | tied run3 under v3 | architectural choice didn't matter once bf16 was fixed |
| run6 G_diff bayer channel retrain | tied run3 under v3 | same — confounded by bf16 in original measurement |
| run7 HRNet-w18 retrain | tied run3 under v3 | architecture-irrelevant at this ceiling |

## What was NOT tried (and why it's also unlikely to help)

- **σ=1.5 retrain (was queued as run8)**: cancelled before launch.
  Hypothesis was "sharper supervision → sharper peak → better sub-pixel."
  Reframed by label-noise diagnosis: smaller σ would ask the model to learn
  the labeler-click noise distribution. Won't help if click noise is the floor.

- **Sub-pixel offset regression head**: deferred indefinitely. Same label-
  noise floor argument applies. Could be worth revisiting if relabeling
  campaign (below) drops the floor materially.

- **Multi-frame consensus (smart variant — confidence-weighted)**: deferred.
  Naive version was negative; smart version is unlikely to find enough
  signal beyond the line mask we already have.

## What could still help (label-side, not model-side)

| direction | cost | expected impact |
|---|---|---|
| `hit_blob` metric on existing audits | hours | more honest deployment number; may show we're already at ~98% under the right metric |
| Multi-click variance test (~30 frames, 3 labelers, 5 clicks each) | ~1.25 hr labeler time | direct measurement of click σ; validates the floor argument |
| Relabel ~300 borderline-failure val frames with multi-click consensus | days of labeler time | tightens labels; may flip ~half of borderlines to hits |
| Bundle-adjustment-based label refinement in IMWUT 3D pipeline | days of dev | downstream win on 3D reconstruction quality (publication-relevant) |

The IMWUT bundle adjustment work is the highest-EV next play; the detector
side is done.

## Files of record

- `notes/bias_attribution.md` — bf16 sigmoid + revised attribution
- `notes/phase1a_calibrated_stratification.md` — failure population analysis
- `notes/phase1b_saliency_test.md` — DINOv2 saliency hypothesis (partial)
- `notes/phase2_worst_dives.md` — val:427, test:192 case studies
- `notes/phase2_results.md` — sub-pixel + line mask + run5/6/7 comparison
- `notes/phase3_results.md` — Phase 3.1 line mask + bbox tightening
- `notes/run7_results.md` — HRNet-w18 trained run
- `notes/run7_ablation.md` — run7's 10-config matrix
- `notes/architecture_ablation_matrix.md` — run3 vs run7 side-by-side
- `notes/phase3_final_recipe.md` — this file
- `notes/imwut_ba_findings.md` — downstream IMWUT bundle-adjustment study
  (no, BA doesn't help; depth floor is unmeasured along-line click σ)
