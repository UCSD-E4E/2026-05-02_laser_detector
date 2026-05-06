# Current state — 2026-05-06 afternoon

A quick reference for picking up tomorrow or after a server outage.

## Where we are

- **Phase 0**: refreshed 2026-05-04. 264 dives, 33,320 frames after
  upstream supersession dropped ~28% of positive labels (43,834 →
  31,469). All `superseded=False` in the parquet because upstream
  filters server-side.
- **Phase 1**: classical-CV baseline — done (commit `fc27aaa`).
- **Phase 2**: BCE+pos_weight=1000 production run, 50 epochs, soft-snap
  inference, no L_line. Early-stopped at epoch 17 (best=epoch 7).
  **Best**: `checkpoints_bce_clean_50e/epoch_007.pt` →
  `hit_rate_n3=0.477, hit_rate_n4=0.614, auroc=0.857, fpr=0.099`.
- **Phase 3**: code wired and committed. **L_line is harmful at every λ
  tested** (0.1 → 0.001 collapse, 0.01 → 0.000 collapse). Soft-snap
  inference *is* fine and gives +0.3pp on Phase 2's epoch_002 (0.383 →
  0.386). See memory `l_line_aux_loss_harmful.md`. L_line training term
  is shelved until we have a warm-start or windowed variant.
- **Phase 4** (sweep): not started. Audit findings should drive sweep
  axes — see "Next steps" below.
- **Failure audit**: done. `scripts/audit_failures.py` produces
  per-dive metrics, wavelength × line-quartile crosstab, per-dive
  overlay plots, and a summary chart. Outputs in
  `data/audit/<checkpoint-stem>/`.

## Top checkpoints

Cleaned data, 4-GPU DDP, BCE+pos_weight=1000:

| run | ckpt | hit_n3 | hit_n4 | AUROC | FPR | mean_err |
| --- | --- | --- | --- | --- | --- | --- |
| 50e (2026-05-06) | **epoch_007** | **0.477** | **0.614** | 0.857 | 0.099 | 265 |
| 10e (2026-05-04) | epoch_002 | 0.383 | 0.540 | 0.854 | 0.146 | 281 |

50-epoch run extended Phase 2 by 7 epochs of useful learning before
val plateaued; +9.4 pp hit_n3 from the same recipe just running longer.
Soft-snap inference enabled (caveat: see "Line-prior leakage" below).

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

1. **Close the wavelength gap** — highest-leverage knob:
   - Per-wavelength sampler weights (4× upweight green in
     `HardNegativeBalancedSampler`).
   - Heavier hue/saturation augmentation (currently photometric augs
     are mild).
   - Two-headed model (separate red vs green prediction heads).
2. **Phase 5 cascade** — refinement crop around argmax. The bimodal
   error shape says ~half of misses are "wrong tile selected, right
   region nearby" candidates that a refinement pass could rescue.
3. **Phase 4 sweep** — only after #1 has narrowed the gap, then sweep
   pos_weight × σ × presence_threshold.
4. **Fix `train.py` final-val** — load best_checkpoint before the
   final-val pass so MLflow numbers aren't misleading.

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
