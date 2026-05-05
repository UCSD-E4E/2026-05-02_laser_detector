# Current state — 2026-05-04 evening

A quick reference for picking up tomorrow or after a server outage.

## Where we are

- **Phase 0**: refreshed today (2026-05-04 morning). 264 dives, 33,320
  frames after upstream supersession dropped ~28% of positive labels
  (43,834 → 31,469). All `superseded=False` in the parquet because
  upstream filters server-side.
- **Phase 1**: classical-CV baseline — done (commit `fc27aaa`).
- **Phase 2**: BCE+pos_weight production run done. 4 checkpoints
  full-val'd. **Best**: epoch_002 on cleaned data → `hit_rate_n3=0.383`,
  `hit_rate_n4=0.540`, `auroc=0.854`, `fpr=0.146`.
- **Phase 3 code**: wired and committed today, **not yet trained on**:
  - `L_line` aux loss (`--lambda-line 0.1` → DESIGN default)
  - Soft-snap-to-line inference (`--soft-snap-inference`)
  - Resume from checkpoint (`--resume auto`)
  - Early stopping (`--early-stop-patience N`)
- **Phase 4**: hyperparameter sweep harness — not started.

## Production checkpoints

Cleaned data, 4-GPU DDP, 10 epochs, BCE+pos_weight=1000:

| ckpt | hit_n3 | hit_n4 | AUROC | FPR | mean_err |
| --- | --- | --- | --- | --- | --- |
| **epoch_002** | **0.383** | **0.540** | 0.854 | 0.146 | 281 |
| epoch_005 | 0.086 | 0.142 | 0.865 | **0.068** | 271 |
| epoch_008 | 0.203 | 0.427 | 0.872 | 0.167 | **216** |
| epoch_009 | 0.095 | 0.227 | **0.872** | 0.141 | 247 |

Locations on the server (relative to repo root):

- `data/phase2/checkpoints_bce_clean/epoch_*.pt` — production cleaned data
- `data/phase2/checkpoints_bce/epoch_*.pt` — yesterday's dirty data run
- `data/phase2/checkpoints/` — yesterday's lr=3e-4 focal collapse run
- `data/phase2/checkpoints_lr1e3/` — yesterday's lr=1e-3 focal collapse run

## MLflow runs

Server: `https://mlflow.krg.ucsd.edu`, experiment `2026-05-02_laser_detector` (id 2).

| run name | notes | run_id |
| --- | --- | --- |
| `phase2_train` (focal lr=3e-4) | failed: focal collapse | `06644ed4e7ae43b99d3b587aea290c0e` |
| `phase2_train` (focal lr=1e-3) | failed: same collapse | `825ad00473f64d7da0a2716e36045c05` |
| `phase2_train` (BCE dirty data) | first BCE escape | `45ea5165dc8d48939c621fb201fce25f` |
| `phase2_train` (BCE clean data) | best Phase 2 result so far | (see MLflow UI; latest with `world_size=4` tag) |
| `phase2_eval_*` | per-checkpoint full-val standalone runs | search by tag `phase2_eval_only` |

## What to launch first when the server comes back

The 50-epoch DESIGN-spec run with the Phase 3 stack:

```bash
uv run torchrun --standalone --nproc_per_node=4 scripts/run_train.py \
  --epochs 50 \
  --batch-size 16 \
  --num-workers 4 --prefetch-factor 2 \
  --warmup-steps 1000 \
  --heatmap-loss bce --heatmap-pos-weight 1000 \
  --lambda-line 0.1 \
  --soft-snap-inference --soft-snap-alpha-max 0.3 \
  --early-stop-patience 10 \
  --checkpoint-dir data/phase2/checkpoints_phase3
```

ETA at 12 min/epoch on cleaned data: ~10 h, less if early stopping
fires. Resume with `--resume auto --checkpoint-dir <same path>` if the
run dies partway.

## Open items

- **NAS-path issues** on dives 219, 249 — frames missing on disk;
  filter-loadable drops them silently. Investigation upstream pending.
- **Failure audit script** not written yet; see
  [laptop_friendly_tasks.md](laptop_friendly_tasks.md#1-failure-audit-script-scriptsaudit_failurespy-most-valuable).
- **Resume with extended schedule** not tested — resuming a 5-epoch run
  with `--epochs 10` may have scheduler-state discontinuity, see the
  same notes file.
- **Soft-snap A/B eval** in progress at the time of writing
  (`bp3jk3f3t`); ETA ~17:58. Will tell us if the snap helps the
  existing checkpoint before we commit to a 50-epoch run with it on.

## Memories worth re-reading

In `~/.claude/projects/.../memory/`:

- `superseded_label_filtering.md` — why filter is a no-op until Phase 0 re-runs
- `negative_frames_are_sparse.md` — only 4% of corpus is negative → hard-neg mining essential
- `bf16_focal_loss_nan.md` — fp32 loss under bf16 autocast required
- `training_run_preferences.md` — full resume required for big runs (now wired)
- `dive_data_quality_issues.md` — dives 237/219/249 NAS issues
- `wavelength_data_mismatch.md` — ~42% of dives are mixed-color
