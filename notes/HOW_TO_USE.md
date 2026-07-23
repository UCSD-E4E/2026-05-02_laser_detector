# How to use this detector

The fishsense laser-dot detector. Given a 4K underwater dive frame, returns
either `(pred_x, pred_y, confidence)` or `no laser detected`.

This document covers how to USE the production system: setup, running
inference, expected outputs, performance, known limits. Project history and
experimental analysis is in the other `notes/phase*.md` files; you don't
need to read those to use the detector.

---

## Quick start

```bash
# One-time setup
nix develop                           # enters the devShell (uv, python 3.13, CUDA libs)
uv sync                                # installs deps from uv.lock

# Sanity check: load the production model
nix develop --command uv run python -c "
import torch
from laser_detector.model import LaserDetector
ckpt = torch.load('data/phase2/checkpoints_sensor_bayer_50e_run3/epoch_021.pt',
                  map_location='cpu', weights_only=False)
print(f'production model: epoch {ckpt[\"epoch\"]}, in_channels={ckpt[\"cfg\"][\"in_channels\"]}')
"
```

If that prints `production model: epoch 21, in_channels=6`, you're set.

---

## Production model

**Checkpoint**: `data/phase2/checkpoints_sensor_bayer_50e_run3/epoch_021.pt`

- ResNet-34 + UNet (segmentation-models-pytorch)
- 6-channel input: chromaticity (3) + wavelength (1) + Bayer-excess G/R (2)
- 50-epoch training, early-stopped at epoch 21 with val_hit_n3 ≈ 0.6 on the
  per-epoch subsample
- Bias is calibrated post-hoc, not baked into the checkpoint — see
  `--pixel-bias-offset` below

## Production inference recipe

The full production inference command line, applied to every dataset audit:

```bash
nix develop --command uv run torchrun --standalone --nproc_per_node=4 \
  scripts/audit_failures.py \
  --checkpoint data/phase2/checkpoints_sensor_bayer_50e_run3/epoch_021.pt \
  --split val \
  --image-pipeline linear_npy \
  --bayer-excess-cache-dir data/image_cache_bayer_excess \
  --soft-snap-inference \
  --rig-prior --rig-prior-floor 1.0 \
  --cascade \
  --subpixel-refine \
  --line-mask-corridor-px 25 \
  --pixel-bias-offset -0.179 -0.023 \
  --out-dir data/audit/your_run_name
```

The flag set was tuned in the Phase 2-3 ablation matrix and is the optimal
production configuration on this checkpoint. **Don't change individual flags
without re-reading `notes/architecture_ablation_matrix.md`** — several flags
are conditional (e.g. `--line-mask-corridor-px` adds zero alone but +1.4 pp
when combined with the others).

### What each flag does

| flag | purpose |
|---|---|
| `--soft-snap-inference` | After argmax, blend the prediction α=0.3 toward the projection onto the dive's fitted line. Skipped on cold-start dives where line confidence is low. |
| `--rig-prior --rig-prior-floor 1.0` | Multiply the heatmap by a static bbox mask centered on the empirical laser-position distribution in sensor coords. Floor=1.0 makes it a pure hard bbox; outside the bbox is zeroed. |
| `--cascade` | Two-pass inference: pass-1 finds coarse argmax on the full 4K tiled grid; pass-2 re-runs the model on a 256×256 crop around the coarse argmax and refines. Falls back to coarse if pass-2 presence drops below threshold. |
| `--subpixel-refine` | Parabolic-fit sub-pixel peak refinement on the heatmap LOGITS (not probs — see "bf16 caveat" below). Adds ~0.3 px of localization precision. |
| `--line-mask-corridor-px 25` | Per-dive geometric corridor: zero heatmap pixels >25 px perpendicular to the fitted dive line. Kills the val:427-style distractor mode. Only fires on frames with high line confidence. |
| `--pixel-bias-offset -0.179 -0.023` | Constant per-checkpoint calibration. Subtracts (dx, dy) from the final prediction. Empirically derived from val inliers; works on test too. |

### Output format

Predictions land in `<out-dir>/predictions_with_meta.parquet`:

| column | type | description |
|---|---|---|
| `image_id` | int | per-frame ID from `frames.parquet` |
| `dive_id` | int | per-dive ID |
| `image_path` | str | local cache path (uses `image_checksum` for keying) |
| `image_checksum` | str | content-hash, stable across recomputes |
| `label_x`, `label_y` | float | ground-truth label if available; else null |
| `pred_x`, `pred_y` | float | calibrated prediction in sensor-pixel coords |
| `pred_confidence` | float | per-frame presence sigmoid (max over tiles) |
| `is_positive` | bool | whether ground truth says a laser is visible |
| `wavelength` | str | "red" or "green" |
| `line_confidence` | float | per-dive RANSAC line fit quality |

A null `pred_x` means "no laser detected" (cascade rejected, OR presence
below `--presence-threshold`). Non-null with low confidence means
"detected, but uncertain."

**Coordinate frame** (important for downstream 3D reconstruction): by default,
`pred_x`, `pred_y` are in **raw pixel space** — same coordinate frame as the
raw image the detector consumed. Labels in `frames.parquet` are in
**rectified (undistorted) pixel space** because the labeling UI renders via
`RectifiedImage(RawImage(...))`. See issue #9. Empirical impact on hit_n3 is
negligible (median 0.02 px, p99 1.01 px displacement) but coord-frame
mismatch matters for downstream 3D pipelines that expect rectified inputs.

To emit rectified predictions, run once to build the per-rig intrinsics
parquet:

```bash
nix develop --command uv run python scripts/ingest_camera_intrinsics.py \
  --frames data/frames.parquet \
  --out data/rig_intrinsics.parquet
# (prompts for API credentials, or reads FISHSENSE_USERNAME / FISHSENSE_PASSWORD env vars)
```

Then pass `--rectify-output --rig-intrinsics-path data/rig_intrinsics.parquet`
to `eval_checkpoint.py` or `audit_failures.py`. Predictions in the parquet
will be in the rectified frame.

---

## Common entry points

### 1. Evaluate a checkpoint on val or test

```bash
nix develop --command uv run torchrun --standalone --nproc_per_node=4 \
  scripts/eval_checkpoint.py \
  --checkpoint data/phase2/checkpoints_sensor_bayer_50e_run3/epoch_021.pt \
  --split test \
  --image-pipeline linear_npy \
  --bayer-excess-cache-dir data/image_cache_bayer_excess \
  --soft-snap-inference --rig-prior --rig-prior-floor 1.0 --cascade \
  --subpixel-refine --line-mask-corridor-px 25 \
  --pixel-bias-offset -0.179 -0.023 \
  --no-mlflow
```

Same flag set, slightly less metadata in the output. Use this when you want
summary metrics, not per-frame predictions.

### 2. Audit failures (per-frame predictions + stratification)

Use the full production-recipe command from the top of this doc. The
resulting parquet feeds `scripts/stratify_failures_calibrated.py` for the
breakdown by dive, wavelength, error class.

### 3. Train a new model (only if you really need to)

The Phase 2-3 work explored the architecture space extensively — see
`notes/phase3_final_recipe.md`. Two architectures (ResNet-34 UNet, HRNet-w18)
land in the same ~0.91 val ceiling. Don't retrain expecting a metric lift;
retrain only if you're changing the data, the wavelength set, the rig, or
the input pipeline.

To retrain run3 from scratch:

```bash
bash scripts/launch_run_train.sh        # production recipe (run3 equivalent)
# or
bash scripts/launch_run7_hrnet.sh       # HRNet-w18 variant (matches run7)
```

Both launchers wrap `torchrun --standalone --nproc_per_node=4 scripts/run_train.py`
with the right flags. Read each launcher before running — they have
checkpoint-dir paths baked in that you'll want to change to avoid clobbering
existing runs.

---

## Performance

| split | hit_n3 (calibrated) | hit_n4 |
|---|---|---|
| val | 0.9081 | 0.9320 |
| test | 0.8615 | 0.8914 |

Where:
- `hit_n3` = prediction within 3 px of the human label
- `hit_n4` = prediction within 4 px
- Calibrated = with the bias offset above applied

These numbers reflect the **label-noise floor**, not the model's intrinsic
precision. The detector emits predictions whose perpendicular σ to the
dive's fitted line (~1 px) is *tighter* than the human label perpendicular σ
(~0.8 px after RANSAC outlier rejection, ~2-3 px overall). The 3-10 px
borderline failures that dominate the residual are click variance from the
labels, not model error. See `notes/phase3_final_recipe.md` and
`notes/imwut_ba_findings.md` for the full analysis.

---

## Environment / setup gotchas

### nix + uv

This project uses a **nix devShell** to provide system-level dependencies
(CUDA libs, FFmpeg, etc.) and **uv** to manage Python packages. Every
command needs the nix wrapper:

```bash
# WRONG — won't find uv or torch
python -c "import torch"

# RIGHT
nix develop --command uv run python -c "import torch"
```

### Kerberos / NAS

Raw ORF frames live on a CIFS-mounted NAS (`~/mnt/fishsense_data/`) and
require a valid Kerberos ticket. If you see `[Errno 126] Required key not
available`, run `kinit`.

**Most training and eval runs DON'T need NAS** because the local caches
(`data/image_cache_*`) cover train 99.9% / val 94.2% / test 100% of frames.
The 211 missing val frames are dive 249's stale paths (see "known issues"
below). Only operations that need to decode fresh ORFs (rebuilding caches,
inspecting new data) require live NAS.

### Bayer-excess cache

For 6-channel models (everything from run3 onward), inference reads from
`data/image_cache_bayer_excess/`. Pre-warmed for all train/val/test frames.
If you train a new variant that needs a different cache (e.g. the
`bayer_excess_diff` 7-channel for run6), pre-warm it with:

```bash
nix develop --command uv run python scripts/prewarm_bayer_excess_cache.py \
  --splits train val test \
  --cache-dir data/image_cache_bayer_excess \
  --workers 8
```

This takes ~2 hours on 8 workers with NAS access.

---

## Known issues / things to be careful about

### bf16 at inference — DISABLED (issue #13)

**Inference now runs in fp32 end-to-end.** `predict_frame` and
`predict_frame_with_cascade` default `autocast_dtype=None`, and
`_run_val_inference` no longer forwards `cfg.use_bf16`.

The earlier "float first, then sigmoid" fix solved the *sigmoid*
tie-break (multiple pixels rounding to `1.0` after saturation). It did
**not** solve the *logit* tie-break: two competing pixels can round to
identical bf16 buckets in the model's forward pass, and the `.float()`
cast afterwards is lossless zero-padding — the precision is already
gone. `flat.max(dim=1)` then breaks the tie by row-major index. Because
cuDNN picks different tensor-core kernels on different SMs, the same
weights + same input produce different bf16 logits on Ada (SM89) vs
Ampere (SM80) vs Hopper — argmax outcomes differ by up to ~200 px on
frames where the peak margin is <1 bf16 ulp. Empirically 3/4 sampled
positives have sub-ulp margins on this checkpoint, so this is normal,
not tail behavior.

Running the head outside autocast — or simplest, disabling autocast on
the entire inference path — makes the argmax IEEE-754 stable and
therefore reproducible across GPUs. On our RTX 4500 Ada, dropping bf16
from inference does not slow it down (kernel-launch dominated, not
tensor-core dominated). The 3222-frame val hit_n3 goes from 0.9081
(bf16 + old bias) to 0.9100 (fp32 + new bias) — small win, big
reproducibility gain.

Training still uses bf16 autocast (fp32 for training would materially
slow it down). If you write new code that runs a sigmoid-then-argmax on
autocasted heatmaps, either wrap it in `autocast(dtype=torch.bfloat16,
enabled=False)` or promote to fp32 *before* the forward pass.

### Dive 249 path discrepancy

The `frames.parquet` references 211 paths for dive 249 that don't exist
in the source data. The detector loader catches this and returns None
(those frames are dropped from train/val). This is an upstream data
issue, not a detector bug. If you fix the upstream paths, those frames
become usable for training again — re-run the bayer-excess cache prewarm
to pick them up.

### Caches and disk

The local caches are large:
- `data/image_cache_linear_npy/` — ~330 GB
- `data/image_cache_bayer_excess/` — ~7 GB
- `data/image_cache_bayer_excess_diff/` — ~10 GB (only if running run6)

`/scratch` is wiped on reinstalls. If you need persistent caches, copy them
to NFS first. See `~/.claude/projects/<this>/memory/machine-storage-layout.md`
for the storage layout.

### Geometric augmentation is OFF

Per `CLAUDE.md`: "Rotation and flip break the per-dive colinearity prior.
Photometric augs (hue, brightness, blur, noise) are fine and important."

If you add geometric augmentation, the rig prior + line mask + line fit all
silently degrade because they're per-dive-coords-aware. Don't.

### Splits are dive-level, never frame-level

If you re-split, do it at the dive level. Frame-level leakage hides the
per-dive line + wavelength priors and inflates val/test metrics. The
`dive_splits.parquet` is the source of truth.

---

## When to retrain vs. when not to

Don't retrain if:
- You want more accuracy. The architecture is at the label-noise floor;
  more training won't help. See `notes/phase3_final_recipe.md`.
- You're considering smaller σ supervision. Cancelled before launch; see
  the same doc for why.

Do retrain if:
- The data distribution changes (new rig, new wavelength, new dive style,
  new camera).
- You're cleaning up labels (e.g. multi-click consensus) and want to test
  whether tighter labels move the metric.
- You're swapping a sub-system (e.g. switching from `bayer_excess` to a
  new sensor extraction).

---

## Quick reference: documents in this repo

- `CLAUDE.md` — load-bearing constraints (read first when picking this up)
- `DESIGN.md` — architectural design, success criteria, phasing
- `GOAL.md` — original problem framing
- `README.md` — human-facing summary
- `notes/HOW_TO_USE.md` — this file
- `notes/phase3_final_recipe.md` — what we landed on and why, full picture
- `notes/imwut_ba_findings.md` — downstream IMWUT bundle-adjustment study
- `notes/bias_attribution.md` — bf16 finding + revision history
- `notes/phase1*.md`, `notes/phase2*.md`, `notes/phase3*.md`, `notes/run7_*.md`,
  `notes/architecture_ablation_matrix.md` — experimental record
