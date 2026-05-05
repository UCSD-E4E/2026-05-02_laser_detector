# Laptop-friendly tasks (no GPU server)

Things that can be done on a 3060 mobile / any local dev machine without
the 4× RTX 4500 Ada server. Listed in order of "useful to land before the
next 50-epoch run."

## 1. Failure audit script — `scripts/audit_failures.py` (most valuable)

DESIGN.md §7.3 calls for failure auditing on the worst dives. We have
the data to do this from today's evaluations but haven't written the
script. Inputs:

- A checkpoint (we have `epoch_002.pt` from the cleaned-data run +
  `epoch_004.pt` from the dirty-data run as comparison points).
- Phase 0 parquets (frames, dive_lines, dive_wavelengths, dive_splits).
- Per-frame predictions from MLflow (or re-run via
  `scripts/eval_checkpoint.py`).

Outputs the script should produce:

- Per-dive metrics table sorted worst-first by `mean_pixel_error`. Joins
  predictions ↔ frames ↔ dive_wavelengths ↔ dive_lines so we can group
  by wavelength + line_confidence quartile.
- Aggregate metrics sliced by `wavelength × line_confidence_q1..q4`.
  Tells us whether failures concentrate in low-line-confidence dives
  (which would mean Phase 3 soft-snap is the right targeted fix) vs.
  spread across the corpus (which would mean a harder problem).
- A `predictions_with_meta.parquet` cache so subsequent ad-hoc analysis
  doesn't have to re-run inference.
- Plots: 8 worst dives' labels (red dots) and predictions (cyan dots)
  overlaid on a representative frame. PNG output to `data/audit/<dive_id>.png`.

What "predict_frame on a laptop" needs: the eval frames + the
checkpoint. ~30 sample frames per dive × 8 dives = 240 ORF/JPEGs. That
fits on a laptop. Or use the cached JPEGs (much smaller). The
`scripts/eval_checkpoint.py` script already does the inference part if
you want full numbers; the new script is the **slicing + plotting**
layer over those predictions.

CPU inference at 1024² is slow (~10–15 s/frame on CPU vs ~1 s on GPU)
but 240 frames × 12 s = ~50 min. Acceptable for a one-time audit. With
the 3060 it'd be ~3-5 min total.

## 2. Visualize soft-snap on a few frames

We added soft-snap to inference (§6.2) but haven't seen it on a real
image yet. Load `epoch_002.pt`, take ~10 cached val JPEGs from
confident-line dives, run `predict_frame` with and without
`line_abc`/`line_confidence`, plot:

- the heatmap (sigmoid output)
- argmax position
- snapped position
- the dive's RANSAC line
- the ground-truth label

Confirms the snap is doing the right thing visually before we burn a
50-epoch run on `--soft-snap-inference`. If the snap is consistently
moving things toward the label, we're set; if it's moving the wrong
direction or causing weirdness, we have time to fix it.

Easy laptop task: ~50 lines of matplotlib + the existing
`predict_frame` API.

## 3. Hyperparameter-sweep harness for Phase 4

Phase 4 is hyperparameter sweeps over `pos_weight`, `σ`, `λ_line`,
`presence_threshold`. A tiny harness that takes a YAML / JSON of
configs and emits one `uv run torchrun ...` invocation per config,
writing results into a single MLflow `Parent` run with child runs per
config. Doesn't need the GPU server to *write*, only to *run* — design
+ scaffolding can land tonight, sweep itself runs tomorrow.

Reference: MLflow's nested-run API is
`mlflow.start_run(nested=True)`. The parent run holds the manifest of
configs; each child holds one training run.

## 4. Resume + early stop end-to-end test

We unit-tested the resume mechanism on a 4-dive smoke (it works — see
the smoke transcript in commit `8011776`). A meaningful end-to-end test
would be:

- Train 5 epochs on a subset
- Kill at epoch 3
- Resume with `--epochs 5 --resume auto` → verify the last 2 epochs
  pick up cleanly with no metric discontinuity
- Resume with `--epochs 8` (extending the schedule) → verify the
  scheduler re-extends properly (this is currently NOT tested and may
  be where resume falls down on a real long run)

CPU is fine for this with a tiny dataset.

## 5. Polish & explore

- Run the existing test suite (`uv run pytest tests/`). 82 tests today;
  ensure they still pass after any local edits.
- `git log --oneline` shows 8 commits ahead of origin. Review the diff
  before pushing.
- Read MLflow runs in the `2026-05-02_laser_detector` experiment to
  build intuition about the loss curves vs. the metric curves —
  especially the `step_loss` series that landed in commit `1837ad2`.

## What to skip on laptop

- Anything that needs the full 331 GB image cache.
- Multi-epoch training on the full corpus (you'd be running 24+ h on a
  laptop GPU for what the cluster does in 3 h).
- DDP (single-GPU only on the laptop; testing DDP-specific paths needs
  the multi-GPU server).
