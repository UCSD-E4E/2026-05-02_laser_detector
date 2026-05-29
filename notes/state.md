# Current state — 2026-05-29 (post-run3)

A quick reference for picking up tomorrow or after a server outage.

## Latest — run3 done; deployment recipe established

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
