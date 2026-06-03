# Current state — 2026-06-03 (post run4 retrain — calibration retained, root cause still open)

A quick reference for picking up tomorrow or after a server outage.

## Latest — run4_centered retrain did NOT eliminate the bias; production stays on run3 + calibration

Hypothesis from 2026-05-30 was that the Bayer-excess upsample (`np.repeat` placing
half-res values at full-res top-left instead of supercell centroid) was the
sole/dominant source of the ~(−1.13, −2.07) px label-prediction bias. Tested
end-to-end:

1. Replaced `np.repeat` with `cv2.resize(..., INTER_LINEAR)` (commit `0dba88e`)
   in `_decode_raw_bayer_excess` — centroid-aligned upsample.
2. Rebuilt the bayer_excess cache at `data/image_cache_bayer_excess_centered/`
   (33090 decoded, 230 expected stale-path failures matching the old cache exactly).
3. Trained `run4_centered` from scratch (50 epochs target). Killed at epoch 37
   (best epoch 19, subsample val_hit_n3 = 0.566). Patience > 10 with train_loss
   still dropping (overfitting territory).
4. Paired full eval on val + test, with and without `--pixel-bias-offset`:

| split | metric          | run3 e021 no-bias | run3 e021 +bias | run4 e019 no-bias | run4 e019 +bias |
|-------|-----------------|------------------:|----------------:|------------------:|----------------:|
| val   | hit_rate_n3     |            0.5255 |          0.7976 |            0.5042 |          0.7777 |
| val   | hit_rate_n4     |            0.7498 |          0.8436 |            0.7241 |          0.8276 |
| val   | median_pix_err  |              2.05 |            1.41 |              2.88 |            1.45 |
| test  | hit_rate_n3     |            0.5034 |          0.8120 |            0.4466 |          0.7973 |
| test  | hit_rate_n4     |            0.7627 |          0.8583 |            0.7324 |          0.8321 |
| test  | median_pix_err  |              2.99 |            1.45 |              3.15 |            1.43 |

Key observations:
- **The bias is still present in run4.** Same `(−1.13, −2.07)` calibration lifts
  it by +27/+35 pp, identical magnitude to run3's lift — so the Bayer-excess
  upsample shift was NOT the dominant cause (or possibly not a cause at all).
- **Run4 is slightly worse than run3** across all configs (~1.5–2 pp behind on
  hit_n3 with bias, up to −5.7 pp on test no-bias). Same architecture, similar
  best-epoch, just slightly unluckier init or training noise.
- Production stays on `run3/epoch_021.pt` + recipe + `--pixel-bias-offset -1.13 -2.07`.

### Investigation: where IS the bias from?

Eliminated:
- Bayer-excess upsample shift (run4 falsified it).
- Heatmap encode/decode origin mismatch (prior agent verified, encode + decode
  both treat pixels as integer point-samples).
- Tile-stitch off-by-half (`compute_tile_grid` uses integer origins; `local_x + ox`
  is exact translation).
- Audit-script coordinate transform (raw pass-through from inference).
- Eval-rotation bug (`_inverse_rotate_label` mismatch) — affects only 15 val
  frames with `flip != 0`, immaterial to the systemic bias.

Partial explanation found:
- **Annotator-vs-photometric-centroid offset**: on 200 val frames, the laser's
  photometric centroid sits ~(+0.36, +0.41) px below-left of the click label.
  Per-wavelength: red (+0.34, +0.36), green (+0.53, +0.58). This is ~25–30%
  of the bias magnitude. If labels click the saturated core and the model
  learns photometric centroid features, this is consistent with the sign.
  But it doesn't explain the remaining ~75%.

Still unexplained:
- The dx:dy ratio is ~1:2 (Y bias is 2× X bias). Photometric offset is ~1:1.
  Bayer-excess shift was ~1:1. Something Y-asymmetric is contributing the
  bulk of the bias and we haven't identified it.
- Candidates for next investigation if the calibration ever stops working:
  rolling-shutter / chromatic aberration / chromaticity-norm interpolation
  shift / per-rig optical center asymmetry / a hidden integer cast somewhere
  in the heatmap path.

**Practical takeaway**: the calibration constant is empirically correct and
data-validated on the held-out test set; we don't need to understand the
mechanism to ship it. But the assumption that the Bayer-excess upsample was
the dominant root cause is wrong.

**Update 2026-06-03 — synthetic ablation cracks the question** (full writeup
in [bias_attribution.md](bias_attribution.md)): feeding the production
checkpoint synthetic Gaussian inputs at 81 known sub-pixel positions
produces a deterministic median bias of **dx=−1.00 ± 0.78, dy=−3.70 ± 0.57**.
That matches the real-data X bias (−1.13) exactly and exceeds the real Y
bias (−2.07) — meaning the architectural shift is ~(−1.0, −3.7) and real
blob characteristics partially attenuate the Y component to −2.07.
Calibration constant is therefore a deterministic correction for a model-
architecture artifact (likely smp UNet's nearest-mode decoder upsample),
**not** a data-fitting hack — defensible for publication.

### Side work in this iteration (kept regardless of run4 outcome)

- Added `--bayer-excess-cache-dir` flag to `scripts/eval_checkpoint.py` and
  `scripts/audit_failures.py` so the centered-cache experiments could be A/B'd
  cleanly against the buggy-cache checkpoints.
- Hardened `LocalFilesystem*Loader.load()` in `preprocessing/image_loader.py`
  to catch `OSError` on `path.exists()` (e.g., expired Kerberos, FUSE crash,
  stale path), log a warning, and return None — caller writes a null prediction
  for the frame instead of crashing the whole eval. Fixes the eval-crash
  encountered when the CIFS Kerberos ticket expired mid-run.

### Open items

- **DEPRECATED** — DESIGN.md §10 Risks entry "Bayer-excess upsample shift" now
  partially wrong: the shift is real but is not the dominant bias source.
  Worth updating when DESIGN.md gets its sensor-coord sync.
- **Kerberos**: user action — run `kinit` whenever to refresh the NAS auth
  ticket (yesterday's expired 06/02 00:57 PDT). Not blocking inference
  (loader is now resilient) but blocks any new ORF decode (training/prewarm).
- **Dive 249**: 211 frames have stale paths in `frames.parquet` upstream of
  Phase 0; not fixable without re-ingesting or re-mapping. Caps val at 94%
  fraction_localized. Test set unaffected.

## Previous — found a 2-px systematic decode bias; calibrating it out lifts hit_n3 0.526 → 0.798

After a step-back audit of the 47% miss population, the failure-mode
stratification on `data/audit/epoch_021_recipe/predictions_with_meta.parquet`
surfaced a **constant ~(−1.13, −2.07) px residual on correct (hit_n3=True)
predictions**, essentially uniform across rigs (every rig 1/2/4/6/10 lands
in dx ∈ [−1.07, −1.26], dy ∈ [−2.07, −2.25]) and wavelengths (red
(−1.10, −2.12), green (−1.26, −1.89)). The bias was pushing ~1037 frames
sitting in `border_3to5` past the 3-px threshold.

Mechanism (smoking gun, [image_loader.py:148-150](src/laser_detector/preprocessing/image_loader.py#L148-L150)):
the Bayer-excess channels are produced at half-res
and upsampled with `np.repeat(np.repeat(half, 2, axis=0), 2, axis=1)`,
which places each supercell value at the full-res top-left `(2i, 2j)`
rather than its centroid `(2i+0.5, 2j+0.5)`. The 6-ch model fuses
chromaticity (correctly aligned) with Bayer-excess (shifted +0.5 down-right)
and learns a centroid pulled toward the shifted features. Sign matches;
magnitude is amplified through the U-Net decoder. This is **specific to
6-ch sensor+Bayer checkpoints**; 4-ch JPEG checkpoints should be exempt.

LOO-cross-validated (derive bias from N−1 dives, apply to held-out): every
dive improves, aggregate lift on val = **+27.15 pp**. Wavelength-symmetric;
a single global scalar pair suffices.

End-to-end validation with `--pixel-bias-offset -1.13 -2.07` on run3
epoch_021 + full deployment recipe. **Both val (calibration source) and
test (held-out generalization check) confirm the lift:**

| split | metric              | baseline | + bias-offset |    Δ          |
|-------|---------------------|---------:|--------------:|--------------:|
| val   | hit_rate_n3         |   0.5255 |    **0.7976** | **+27.21 pp** |
| val   | hit_rate_n4         |   0.7498 |    **0.8436** | +9.38 pp      |
| val   | median_pixel_error  |   2.05   |      **1.41** | −0.64         |
| val   | fraction_localized  |   0.9385 |      0.9385   | 0             |
| test  | hit_rate_n3         |   0.5034 |    **0.8120** | **+30.86 pp** |
| test  | hit_rate_n4         |   0.7627 |    **0.8583** | +9.56 pp      |
| test  | median_pixel_error  |   2.99   |      **1.45** | −1.54         |
| test  | fraction_localized  |   1.000  |      1.000    | 0             |

Test beats val (0.812 vs 0.798) because test had no cache-miss frames (val
pays a ~3 pp penalty for dive 249's 211 missing frames). Calibrated `(dx,
dy)` was derived solely on val inliers and applied unchanged to test —
**confirms the bias is an architectural property of the data pipeline,
not a val-specific artifact**.

The full scoreboard since project inception (val canonical):

```
JPEG baseline ............. 0.477
sensor 6-ch baseline ...... 0.485   (+0.008)
+ full deployment recipe .. 0.526   (+0.041)
+ pixel-bias-offset ....... 0.798   (+0.272)  ← single change, no retrain

held-out test .............. 0.812  ← confirms generalization
```

This single change is larger than every previous improvement combined and
makes the prior chase for a wavelength-fix moot (post-correction, red 0.802
vs green 0.776 — gap closed and slightly inverted).

**Production now:** `run3/epoch_021.pt` with the deployment recipe
**AND** `--pixel-bias-offset -1.13 -2.07`.

**Open follow-ups (blocked on NAS being back):**
1. **Proper fix:** patch `_decode_raw_bayer_excess` to use centered bilinear
   upsampling (or post-shift by −0.5 px), rebuild bayer_excess cache, and
   retrain. The retrained model may or may not surface additional headroom
   beyond the calibration constant — worth measuring.
2. **Dive 249 prewarm:** 211/283 of dive 249's val frames have no cache
   (linear_npy AND bayer_excess), so they were silently sentineled as null
   preds in eval. Independent of bias: prewarming them yields +3 pp on the
   canonical metric. Needs ORF decode → needs NAS.

**Other notes from the strategy audit (not pursued for now):**
- Per-dive RANSAC line-snap re-scoring was prototyped offline
  (`scripts/prototype/colinearity_rescore.py`); does not help because 87%
  of misses are *along-track* wrong-target picks, not perpendicular drift.
  Would need top-K heatmap peaks emitted by the detector to matter.
- DSNT/subpixel regression, focal loss, temporal: all explicitly rejected
  in DESIGN.md (covered in the strategy review).
- HRNet, σ-per-wavelength sweep, parabolic peak refinement: Phase 4/5
  followups in DESIGN.md, not yet exercised. Parabolic peak refinement in
  particular was flagged as "cheap, may close 1–2 px of the gap" — now
  largely closed by this calibration, but still worth measuring after the
  proper Bayer-excess fix retrain to see if there's residual headroom.

Audit artifacts:
- `data/audit/epoch_021_recipe/stratification/` — per-axis failure
  breakdowns, label-noise floor analysis, dive 108 deep-dive, dive 249
  cache-coverage analysis.
- `data/phase2/checkpoints_sensor_bayer_50e_run3/bias_calibration/eval_bias-1.13_-2.07.log` —
  end-to-end validation run.

## Previous — `--wavelength-balance` retrain DID NOT improve green; rolling back

Tried `--wavelength-balance` (inverse-frequency reweighting of positives by
wavelength group; commit `60a5788`). Run is at
`data/phase2/checkpoints_sensor_bayer_wlbal_50e/epoch_011.pt`. It hit the
same subsample peak as run3 (0.5503) in *half* the epochs (11 vs 21), so the
sampler accelerated convergence — but the per-wavelength audit (both with
the deployment recipe, apples-to-apples) shows green REGRESSED:

| frame-weighted hit_n3 | green | red  | Δ (red − green) |
|-----------------------|-------|------|-----------------|
| **run3 + recipe** *(deployment)*     | **0.529** | 0.563 | +0.034 |
| wlbal + recipe                        | 0.387 | 0.606 | +0.219 |

Overall canonical `hit_n3` for wlbal+recipe = 0.5366 vs run3+recipe 0.5255
(+1.1 pp), but that gain came from red improving (+0.04) while green
collapsed (−0.14) — masked by the ~4:1 red/green frame imbalance in val.
Green q4 dropped 0.51 → 0.30 (q4 < q1 inversion — high line-confidence
frames got worst), suggesting wlbal made the model *overconfidently wrong*
on green at test time and soft-snap can't rescue (blend weight 1−pred_conf
collapses when pred_conf is high). Likely overfit to the repeatedly-sampled
green training examples.

**Decision: production stays on `run3/epoch_021.pt` with the deployment
recipe below.** wlbal artifacts kept for the record; not adopted.

Audit artifacts for both:
- `data/audit/epoch_021/` (run3 no-flags) and `data/audit/epoch_021_recipe/`
  (run3 with recipe — added 2026-05-30 for the wlbal A/B).
- `data/audit/epoch_011/` (wlbal with recipe).

## Run3 baseline — deployment model

The camera-coords refactor is **DONE** (commit `cda6dc3` and the Bayer-excess
and sensor-coords cache work that followed). The sensor-coords + Bayer-excess
(linear cache, 6-channel) pipeline trained end-to-end through `run3` and
early-stopped at epoch 31, best at epoch 21.

**Best checkpoint:**
`data/phase2/checkpoints_sensor_bayer_50e_run3/epoch_021.pt`

Canonical full-val numbers (3625 frames, no inference flags):

| metric           | value |
|------------------|-------|
| hit_rate_n3      | 0.485 |
| hit_rate_n4      | 0.717 |
| presence_auroc   | 0.906 |

With the deployment inference recipe below: **hit_rate_n3 = 0.526** — +4.8 pp
over the prior JPEG production peak (0.477), with bounded predictions.

The structural green-vs-red wavelength gap that motivated the whole refactor
is effectively **closed frame-weighted** (green ≈ 0.53, red ≈ 0.51) and
reduced from Δ ≈ 0.27 to Δ ≈ 0.06 dive-averaged. Worst-dive (427, green)
mean_err fell from ~829 px on the JPEG run to 234 px (3.5× better).

The camera-coords refactor is **DONE** (commit `cda6dc3` and the Bayer-excess
and sensor-coords cache work that followed). The sensor-coords + Bayer-excess
(linear cache, 6-channel) pipeline trained end-to-end through `run3` and
early-stopped at epoch 31, best at epoch 21.

**New best checkpoint:**
`data/phase2/checkpoints_sensor_bayer_50e_run3/epoch_021.pt`

Canonical full-val numbers (3625 frames, no inference flags):

| metric           | value |
|------------------|-------|
| hit_rate_n3      | 0.485 |
| hit_rate_n4      | 0.717 |
| presence_auroc   | 0.906 |

With the deployment inference recipe below: **hit_rate_n3 = 0.526** — +4.8 pp
over the prior JPEG production peak (0.477), with bounded predictions.

The structural green-vs-red wavelength gap that motivated the whole refactor
is effectively **closed frame-weighted** (green ≈ 0.53, red ≈ 0.51) and
reduced from Δ ≈ 0.27 to Δ ≈ 0.06 dive-averaged. Worst-dive (427, green)
mean_err fell from ~829 px on the JPEG run to 234 px (3.5× better).

Audit artifacts: `data/audit/epoch_021/{summary.png, plots/*.png,
per_dive_metrics.parquet, wavelength_x_lineq.parquet, predictions_with_meta.parquet}`.

## Deployment inference recipe — sensor 6-ch checkpoints

For `epoch_021.pt` (and future sensor + Bayer-excess checkpoints):

```bash
uv run python scripts/eval_checkpoint.py \
  --checkpoint <ckpt> --image-pipeline linear_npy \
  --soft-snap-inference --rig-prior --cascade --rig-prior-floor 1.0
```

→ `hit_n3 = 0.5255, hit_n4 = 0.7498, mean_err = 28.9 px` (mean is bounded
within the empirical laser bbox — no 1000-px catastrophic confusers).

**Alternative** if precision (mean_err) matters more than hit rate:
`--rig-prior-floor 0.0` (max Gaussian center bias) → `hit_n3 = 0.514`,
`mean_err = 9.59 px`.

**Individual flag contributions** (vs baseline 0.4850, full results in
`data/phase2/checkpoints_sensor_bayer_50e_run3/ab_inference/` and `…/rig_floor_sweep/`):

- `--cascade` (Phase 5 refinement crop): **+2.0 pp** — the heavy lifter,
  directly attacks the bimodal catastrophic-confuser failure mode.
- `--rig-prior`: −5.3 pp alone but **synergistic with cascade** (+2.7 pp
  combined). The bbox clamps confusers globally; cascade then refines
  locally to find the laser within bounds.
- `--soft-snap-inference`: +0.2 pp solo. Essentially a no-op on this
  checkpoint — the trained sensor-coords model relies on the per-dive line
  prior less than the prior JPEG model did. Cheap to leave on.
- `--rig-prior-floor 1.0` vs default `0.5`: **+1.3 pp** on top. The
  Gaussian center bias was net-negative — it was quietly pulling correct
  off-center predictions toward the empirical center. Pure bbox wins.

## Where we are

- **Phase 0**: refreshed 2026-05-04. 264 dives, 33,320 frames after
  upstream supersession dropped ~28% of positive labels (43,834 →
  31,469). All `superseded=False` in the parquet because upstream
  filters server-side.
- **Phase 1**: classical-CV baseline — done (commit `fc27aaa`).
- **Phase 2** (production training):
  - JPEG 50e (2026-05-06): best `checkpoints_bce_clean_50e/epoch_007.pt` →
    `hit_rate_n3=0.477` (per-epoch subsample).
  - **Sensor 6-ch 50e (run3, 2026-05-28)**: best
    `checkpoints_sensor_bayer_50e_run3/epoch_021.pt` → canonical full-val
    `hit_n3=0.485` (no flags) / **0.526** (deployment recipe). Early-stopped
    epoch 31. Wavelength gap closed frame-weighted. See "Latest" section.
- **Phase 3**: code wired and committed. **L_line is harmful at every λ
  tested** (0.1 → 0.001 collapse, 0.01 → 0.000 collapse). Soft-snap
  inference *is* fine and gives +0.3pp on Phase 2's epoch_002 (0.383 →
  0.386). See memory `l_line_aux_loss_harmful.md`. L_line training term
  is shelved until we have a warm-start or windowed variant.
- **Phase 4** (sweep): **inference-time** A/B done on run3 epoch_021 — see
  "Latest" section and `data/phase2/.../ab_inference/` + `.../rig_floor_sweep/`.
  Training-time sweep (`pos_weight`, `σ`, sampler weights) still pending;
  audit pointed at wavelength-balanced sampling as the next target.
- **Failure audit**: done. `scripts/audit_failures.py` produces
  per-dive metrics, wavelength × line-quartile crosstab, per-dive
  overlay plots, and a summary chart. Outputs in
  `data/audit/<checkpoint-stem>/`.

## Top checkpoints

Cleaned data, 4-GPU DDP, BCE+pos_weight=1000:

| run | ckpt | hit_n3 | hit_n4 | AUROC | FPR | mean_err |
| --- | --- | --- | --- | --- | --- | --- |
| Sensor 6-ch 50e (2026-05-28) | **epoch_021** (recipe) | **0.526** | **0.749** | 0.854 | — | 28.9 |
| Sensor 6-ch 50e (2026-05-28) | epoch_021 (no flags) | 0.485 | 0.717 | 0.906 | 0.122 | ~70 |
| JPEG 50e (2026-05-06)        | epoch_007             | 0.477   | 0.614 | 0.857 | 0.099 | 265 |
| JPEG 10e (2026-05-04)        | epoch_002             | 0.383   | 0.540 | 0.854 | 0.146 | 281 |

Sensor 6-ch numbers are canonical full-val (3625 frames); "recipe" row uses
`--soft-snap-inference --rig-prior --cascade --rig-prior-floor 1.0` — see
"Deployment inference recipe" above. Earlier JPEG numbers are per-epoch
subsample, so the headline `0.526 > 0.477` is mildly favorable for run3
(canonical-vs-subsample apples-to-oranges); a clean canonical comparison
would need re-evaluating `epoch_007.pt` through the same path.

Locations on the server (relative to repo root):

- `data/phase2/checkpoints_bce_clean_50e/epoch_*.pt` — production 50e
- `data/phase2/checkpoints_bce_clean/epoch_*.pt` — prior 10e
- `data/phase2/checkpoints_phase3_l001/` — failed L_line λ=0.01
- `data/phase2/checkpoints_phase3/` — failed L_line λ=0.1
- `data/phase2/checkpoints_bce/`, `checkpoints/`, `checkpoints_lr1e3/` —
  earlier dirty-data + focal-collapse runs

## MLflow

Server: `https://mlflow.krg.ucsd.edu`, experiment
`2026-05-02_laser_detector` (id 2).

| run | name / tag | run_id |
| --- | --- | --- |
| BCE 50e clean (best) | `phase2_train` | `8083d9380c124f26b7922365b98503c0` |
| Phase 3 λ=0.01 (failed) | `phase2_train` | (search by date 2026-05-06 morning) |
| Phase 3 λ=0.1 (failed) | `phase2_train` | (search by tag) |
| BCE 10e clean | `phase2_train` | (latest with `world_size=4`) |
| eval-only | tag `phase2_eval_only` | search by tag |

Note: the 50e run's MLflow `final/*` metrics are on the *epoch-17*
weights, not best. Use `eval_checkpoint.py` results for the canonical
number. See memory `end_of_run_final_val_uses_last_weights.md`.

## Audit findings (epoch_007, 2026-05-06)

`data/audit/epoch_007/` has:

- `summary.png` — two-panel chart: per-dive median-vs-mean scatter
  (catastrophic dives sit far above y=x) + per-frame error histogram
  showing **bimodal** "right or completely lost" distribution.
- `per_dive_metrics.parquet` — sorted worst-first.
- `wavelength_x_lineq.parquet` — slice table.
- `plots/<dive_id>.png` — labels (red) and predictions (cyan) overlaid
  for all 26 val dives.

Key takeaways:

1. **Wavelength gap is structural**: red hit_n3 ≈ 0.57, green ≈ 0.30.
   Persists across line-confidence quartiles. Mixed-color frames are
   rare (75/31k) so it's not a label-quality issue. See memory
   `wavelength_performance_gap.md`.
2. **Bimodal errors**: model is either within 1-3 px (hits) or 1000+ px
   off (catastrophic confusers). Very few in between. This points
   toward Phase 5 cascade (refinement crop) being a high-leverage move
   alongside any retraining.
3. **Line prior buys little**: red q1→q4 is 0.56→0.62, green q3→q4 is
   non-monotonic.

Worst dives (mean_err): 427 (green, 829), 354 (green, 600), 422
(green, 586), 421 (green, 514), 460 (green, 448), 114 (red, 382), 400
(green, 341), 455 (green, 339).

## Next steps

The dominant remaining gap (red 0.57 vs green 0.30) is structural:
JPEG/CLAHE saturates bright laser blobs in all RGB channels and
destroys the wavelength selectivity that chromaticity normalization
relies on. Per-dive contrast measurements (`/tmp/check_chrom.py`)
showed green=0.07-0.08 vs red=0.16-0.40. **The fix is to re-extract
the cache from ORF without CLAHE, in 16-bit linear data.**

1. **Re-extract cache from ORF (step 2)** — pipeline ready,
   blocked on NAS mount:
   - `LocalFilesystemLinearRawImageLoader` uses rawpy directly (no
     CLAHE, gamma=1, 16-bit). Deliberately deviates from CLAUDE.md's
     "use fishsense-core" guidance — laser-specific.
   - `CachingLinearImageLoader` writes lossless 16-bit PNGs.
   - `_chromaticity_norm` is dtype-aware (uint8 + uint16 both work).
   - `scripts/prewarm_linear_cache.py` wraps both, parallel decode.
   - **Blocker**: `/home/c.crutchfield/mnt/fishsense_data/REEF/data`
     not mounted. Need NAS up before kickoff.
   - Smoke validation: run `/tmp/smoke_linear.py` on 5 dives to
     confirm green chromaticity contrast jumps from ~0.08 to >0.4.
   - Full re-extract: ~6-8h wall-clock at the standard worker count.
2. **Phase 5 cascade** stub committed:
   `inference.predict_frame_with_cascade` does pass-1 global tiled +
   pass-2 refinement crop around argmax. Not yet wired into eval or
   training; A/B it after step 2.
3. **Wavelength sampler weights** — defer until step 2 results land
   (likely fixes most of the gap on its own).
4. **Phase 4 sweep** — only after we've closed the wavelength gap.
5. **Fix `train.py` final-val** — load best_checkpoint before the
   final-val pass so MLflow numbers aren't misleading.

## Linear-pipeline kickoff (when NAS is back)

```bash
# 1. Smoke: 5-dive chromaticity check (laptop-friendly once NAS is mounted)
uv run python /tmp/smoke_linear.py     # expect green contrast jump

# 2. Full re-extract (parallel decode; ~6-8h)
uv run python scripts/prewarm_linear_cache.py --splits train val test

# 3. New training run on linear cache (10h)
#    Need to point training scripts at the new cache via a CLI flag
#    or settings override — not yet wired; will be a small follow-up.
```

## Open items

- **NAS-path issues** on dives 219, 249 — frames missing on disk;
  filter-loadable drops them silently. Investigation upstream pending.
- **Resume with extended schedule** untested — resuming a 5-epoch run
  with `--epochs 10` may have scheduler-state discontinuity.
  Workaround: re-run from scratch when extending.

## Line-prior leakage caveat (unchanged)

The per-dive RANSAC line is fit (Phase 0 §3.1) from **every positive
label in the dive**, including val and test labels. Soft-snap (§6.2)
on val/test therefore uses the labels we're scoring against — a
dive-level information leak.

How to interpret numbers reported in this repo:

- **Without soft-snap**: leakage-free. Lower bound on production.
- **With soft-snap**: leaks dive-level info. Upper bound assuming the
  dive's line is already known (existing dives, not first-contact).

The 50e run's 0.477 was reported with soft-snap on. Leakage uplift is
small (+0.3pp on epoch_002), but worth flagging when comparing.

For production deployment on a brand-new dive, soft-snap stays off
until the §6.3 cold-start bootstrap finishes.

The `L_line` aux loss does NOT leak: only sees train-dive batches.
Val/test weights are never directly touched by their own line params.
The leakage is purely an *inference-time, soft-snap-only* phenomenon.

## Memories worth re-reading

In `~/.claude/projects/.../memory/`:

- `l_line_aux_loss_harmful.md` — don't re-enable without warm-start
- `wavelength_performance_gap.md` — green is the bottleneck
- `end_of_run_final_val_uses_last_weights.md` — re-eval after early stop
- `superseded_label_filtering.md`, `negative_frames_are_sparse.md`,
  `bf16_focal_loss_nan.md`, `training_run_preferences.md`,
  `dive_data_quality_issues.md`, `wavelength_data_mismatch.md`
