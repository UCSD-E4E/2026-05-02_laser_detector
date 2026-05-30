# Laser Detector — Design Doc

## 1. Overview

A post-processing detector that locates the laser dot in fishsense dive imagery. Given a frame, it returns either `(x, y)` of the laser center or `no_laser`. Real-time is a future goal; this design targets offline batch inference.

### Success criteria

- **Hit rate**: predicted pixel lies within the on-screen laser blob on ≥ 95% of positive frames in held-out dives.
- **False-positive rate**: ≤ 2% of negative (no-laser) frames produce a prediction with confidence above the operating threshold.
- **Generalization**: metrics computed on **dive-level holdout** — never frame-level.

### Eventual deployment target

- **Hardware**: UUV (edge). For now: server GPU.
- **Throughput**: 20 fps on video. For now: per-image batch inference.
- Architectural choices today shouldn't preclude that: prefer backbones that quantize / TensorRT-port cleanly (ResNet, HRNet, MobileNet families) and avoid exotic ops.

### Non-goals (for v1)

- Sub-pixel localization (laser blob is 3–8 px; argmax inside it is fine).
- Real-time inference *now* (instrument latency, but optimize accuracy first).
- Multi-laser detection (assume one laser per frame).
- Cross-rig generalization (different rigs in the future will get separate models — see §2 and §10).
- Temporal modeling (single-frame inference for v1; video-temporal smoothing is a future feature).

### Tooling

- **Environment**: `uv` (Astral) — `pyproject.toml` is the source of truth, `uv.lock` is checked in. Add deps with `uv add <pkg>`; run commands with `uv run <cmd>`. No `requirements.txt` / `setup.py` / `environment.yml`.
- **ML framework**: **PyTorch**. Default ecosystem choices:
  - `segmentation-models-pytorch` for U-Net + ResNet-34 encoder (saves writing the decoder by hand).
  - `timm` if we need a backbone outside `smp`.
  - `albumentations` for augmentation (handles keypoint coords through transforms correctly).
  - `torchmetrics` for metrics.
  - `mlflow` Python client for tracking + registry.

---

## 2. Data

### Source

- **API**: `fishsense-sdk` (`UCSD-E4E/fishsense-lite`)
- **Volume**: ~60,000 labels.
- **Per-dive count**: dozens of labels per dive (~30–50). Implies ~1,000–2,000 dives.
- **Label format**: single `(x, y)` point per labeled frame.
- **Negatives**: included; frames with no visible laser are explicitly labeled.
- **Per-dive invariants**: each dive is nominally single-color (**red or green** in the v1 corpus; the rig has moved to green-only going forward but a large red backlog remains) and uses a fixed-rig laser, so all positive labels in a dive are colinear in image space (modulo noise and label error). **Mixed-color dives exist (~42%)**: a dive labeled mostly "Red Laser" with a handful of "Green Laser" labels (or vice versa) is treated as the dominant color — the minority is almost always an annotator slip and is dropped/down-weighted by the line-fit outlier check (§3.1).
- **Laser blob size**: phase-0 audit on 91 sampled positives (after `_segment_blob` was tightened to require the labeled pixel sit on a thresholded component and to reject blobs >25% of the patch area): **median 8 px, p10 3.6, p25 5, p75 13, p90 19, max 35**. Sliced by wavelength: **green** median 5.6 / p90 11; **red** median 11 / p90 19 — red is consistently ~2× larger across the distribution (real, due to either physical divergence in water or the close-range frames in red dives, not segmentation bias). The 3 px small-blob tail still drives the input-resolution choice in §4.1; the typical-blob radius drives σ in §4.3.
- **File format**: Olympus ORF (Olympus Tough TG-6, 4K). Decoded via `fishsense-core`'s `RawImage` (rawpy → auto-gamma → CLAHE → 8-bit BGR) so this detector's input distribution matches the rest of the fishsense ecosystem. Decoded JPEGs are cached on disk (keyed by checksum) since ORF decode is slow.

### Rig assumptions

All current dives use similar rigs, so the colinearity and fixed-geometry priors apply uniformly across the dataset. Future rig changes will get a separate model, not a unified one — this is a deliberate scoping decision. Tag every label with `rig_id` (or default to `rig=v1`) from Phase 0 so the dataset is partitioned correctly when new rigs arrive.

### Label quality

Not all labels are equal. Cleaning pass (described in §3) uses the colinearity invariant to identify outliers.

### Splits

**Dive-level**, not frame-level:
- Train: 80% of dives
- Val: 10% of dives
- Test: 10% of dives

Frame-level splits would leak the per-dive line and wavelength priors (computed from labels) into validation. Dive-level holdout is non-negotiable.

Stratify the split on the wavelength tag (red/green, computed in §3.2) so both colors are present in each set.

---

## 3. Offline preprocessing

Run once after each data pull. Outputs are persisted alongside the dataset and logged as MLflow artifacts.

### 3.1 Per-dive line fit

For each dive, RANSAC-fit a 2D line through positive labels.

**Output per dive**:
- `line_params`: `(a, b, c)` for `ax + by + c = 0`, normalized
- `inlier_count`, `inlier_fraction`
- `line_confidence`: `λ_max / λ_min` of the centered covariance, or equivalently the spread along vs. perpendicular to the fit line. High confidence = points spread along the line.

Dives with `line_confidence < τ_line` are flagged "line-ambiguous" and excluded from prior-dependent steps. Threshold tuned on validation.

**Label cleaning**: for each dive, compute `label_noise_mad = 1.4826 * MAD(perp_distance over all positive labels)` — a robust estimator of the population σ. Flag labels whose perpendicular distance exceeds `k * label_noise_mad`, with `k ≈ 3`. **Use `label_noise_mad`, not `residual_std`, for the threshold**: `residual_std` is computed on RANSAC inliers only, so it underestimates the true label-noise scale and would flag the RANSAC outliers a second time regardless of population spread. (`residual_std` is still kept as an inlier-tightness diagnostic for `is_line_confident`.) Phase-0 result: ~14.6% of positives flagged at k=3, mostly real label noise plus genuine minority-color labels in mixed dives.

### 3.2 Per-dive wavelength tag

Each dive is nominally single-color (red or green, see §2). The fishsense `LaserLabel.label` string usually carries the color word ("Red Laser" / "Green Laser"), so:

1. **Label-string majority** (primary): take the most common color word across the dive's positive labels. Mixed dives (~42% of the v1 corpus) are resolved by majority — the minority is almost always an annotator slip.
2. **Color-cluster fallback**: if no label_string yields a color or the top two are tied, sample the brightest pixel in a 11×11 patch around each label, average across the dive to get a `dive_color` BGR vector, then KMeans-cluster k=2 across all such dives. The redder centroid (largest R−G) → "red"; the other → "green". Two-way clustering only — the v1 corpus has no blue dives.

**Output per dive**: `wavelength ∈ {red, green}` (or None for the rare degenerate case), plus `wavelength_source ∈ {label_string, color_cluster}` and the `dive_color` BGR vector for diagnostics.

For new dives at inference (cold start), the same procedure runs after enough high-confidence predictions accumulate. See §6.3.

### 3.3 Per-frame inputs

Persisted to disk in a frame-level table indexed by `(dive_id, frame_id)`:
- `image_path`
- `label_xy` (or `null` for negatives)
- `dive_id`, `wavelength`, `line_params`, `line_confidence`
- `is_positive` (boolean)

---

## 4. Model

### 4.1 Inputs

**Resolution is constrained by laser size.** Native frames are 4K (~3840 × 2160). Worst-case laser blob is 3 px native. Any meaningful downscale loses the target, and full-frame 4K through a U-Net doesn't fit reasonable GPU memory at usable batch size. **Tiled inference is the strategy.**

#### Tiling

- **Tile size**: 1024 × 1024 at native resolution.
- **Overlap**: 256 px (25%) → stride 768 px.
- **Tiles per 4K frame**: 5 horizontal × 3 vertical ≈ 15 tiles.
- **Heatmap merge**: take max across overlapping pixels. Average doesn't make sense for a peaked signal.
- **Padding**: reflect-pad if a tile would extend past the image edge.

A tile is what the model sees; the laser is 3–8 px inside a 1024-px input, which is small but well above the resolution floor.

#### Training-time crops

Don't train on full frames — train on random 1024 × 1024 crops at native resolution.

- **Positive frames**: bias the crop to include the labeled pixel ~70% of the time (so most crops contain the laser); the remaining ~30% are random crops not containing the laser, treated as negatives at the tile level. This gives the model balanced exposure to both regimes.
- **Negative frames**: random crops.
- This is also free augmentation — different crops every epoch.

#### Line-aware tile selection at inference (known dives only)

For dives with `line_confidence > τ_line`:
- Compute which of the 15 tiles intersect a band around the dive's laser line.
- Only run those tiles (typically 3–6 of 15).
- Massive efficiency win for known dives, and a head start toward the eventual 20-fps target.

For unknown dives, run all tiles.

#### Other input pipeline

- **Color preprocessing**: chromaticity normalization (divide RGB by intensity, clipped) applied as a preprocessing layer. Reduces sensitivity to underwater color attenuation.
- **Wavelength conditioning**: one extra input channel set to `0.0` for green and `1.0` for red (the two colors in the v1 corpus). Concatenated to the chromaticity-normalized RGB → 4-channel input. Cold-start dives (§6.3) get `0.5` until clustered.

### 4.2 Backbone

U-Net with a ResNet-34 encoder (ImageNet-pretrained), or HRNet-W18. Either is appropriate; ResNet-34 U-Net is simpler and the default choice. We have post-processing latency budget for v1, so no need for a phone-grade backbone yet.

Both options TensorRT-port cleanly, so neither blocks the eventual UUV deployment. Avoid backbones with attention-only paths or unusual ops (Swin, deformable convs, etc.) until accuracy data shows they're needed — they're harder to ship to edge.

### 4.3 Heads

Two heads share the backbone, both operating on a single 1024 × 1024 tile:

- **Heatmap head**: 1 channel, full tile resolution (1024 × 1024). Sigmoid activation. Target is a Gaussian centered at the labeled pixel (when in-crop). **Default σ ≈ 3 px** at native resolution — chosen from the (cleaned) Phase-0 blob audit: median diameter 8 px ⇒ radius ~4 px ⇒ σ at half-radius is ~2–3 px. Treat σ as a hyperparameter to sweep in [2, 5] px during Phase 4; the red-vs-green asymmetry (red blobs ~2× larger) suggests a wavelength-conditional σ may be warranted. A learned per-frame σ (predicted from local image statistics) is a Phase-3+ option if heatmap loss saturates. Tiles without a label inside (negative-tile case) have an all-zero target.
- **Presence head** (per-tile): scalar logit. Mean-pooled feature from the bottleneck → small MLP → sigmoid. Trained on every tile.

**Frame-level presence** is computed at inference: positive iff *any* tile's presence head fires above τ_presence (or equivalently, the merged heatmap's max exceeds threshold). The two are equivalent in practice; tracking both is cheap and useful for diagnostics.

Sub-pixel offset head: **not used.** "Anywhere within the laser is good enough" makes it unnecessary.

### 4.4 Output at inference

`(heatmap, presence_logit)` → see §6 for post-processing into a final prediction.

---

## 5. Training

### 5.1 Losses

```
L = λ_hm * L_heatmap + λ_pres * L_presence + λ_line * L_line
```

- **`L_heatmap`** (BCE with `pos_weight`, **not** focal): applied to every tile against the Gaussian target. The Phase-2 implementation tried CenterNet-style penalty-reduced focal loss first and observed a hard collapse to the trivial "predict 0 everywhere" minimum at any LR — `(p^α) · log(1-p)` summed over ~1M near-zero target pixels per tile dwarfs the 1-positive-pixel reward, and `(1-target)^β` doesn't soften it enough at σ=3 over a 1024² tile. BCE with `pos_weight ≈ 1000` (set as 30× the actual ~30:1M active-pixel ratio) inverts the imbalance arithmetically and converges cleanly on the same data. Focal is kept in code as `model.focal_heatmap_loss` for ablation but is not the default. Loss is computed in fp32 even under bf16 autocast — bf16's 7-bit mantissa is too coarse for the log-of-clamped-sigmoid; this is a silent regression risk if anyone moves the loss back inside the autocast block.
- **`L_presence`** (BCE): applied on all frames, with hard-negative mining — sample 50/50 from positives and from "hard" negatives (negatives with high heatmap response in the previous epoch). Random sampling is dominated by trivial negatives and produces a useless presence head. Especially load-bearing for this corpus: only ~5% of frames are negatives, so uniform shuffling barely shows the presence head any negative class. Implemented as `data.HardNegativeBalancedSampler` with rank-aware sharding under DDP; rank 0 owns the score-update RNG and broadcasts updated `neg_scores` after each epoch so all ranks shuffle consistently.
- **`L_line`** (line-consistency aux loss): for positive frames in dives with `line_confidence > τ_line`, penalize the perpendicular distance from the predicted heatmap centroid (soft-argmax) to the dive's line. Weighted by `line_confidence`. Skipped for line-ambiguous dives. **Phase 3** — not yet wired.

Default weights: `λ_hm = 1.0`, `λ_pres = 0.5`, `λ_line = 0.1` (Phase 3+). Tuned via MLflow sweep in Phase 4.

### 5.2 Augmentation

Photometric (always on):
- Hue jitter (broad — color shift is the primary domain variation)
- Brightness, contrast, saturation jitter
- Gaussian blur, noise
- JPEG compression artifacts

Geometric (off by default):
- Horizontal flip and rotation **break** the per-dive line. Either disable, or recompute the line per augmented sample (expensive). Default: disabled. Crops are also disabled — we want full-frame inputs.

### 5.3 Optimizer

- AdamW, LR `3e-4`, cosine decay
- Batch size: `--batch-size` is **per-rank** under torchrun. Phase 2 uses bs=16/rank × 4 ranks = global bs=64 on 4× RTX 4500 Ada (24 GB each, ~16 GB used at this config — bs=20–24/rank is feasible if needed).
- Mixed precision: bf16 forward (`torch.autocast`), losses in fp32 (cast `.float()` before computing — see §5.1).
- Epochs: ~50, with early stopping on val hit-rate. **Early stopping not yet wired**; current 10-epoch runs overshoot the localization peak and rely on best-checkpoint selection.
- Warmup: **1000 absolute steps** (was "1 epoch linear" before Phase 2 — at full-corpus 4272 batches/epoch a 1-epoch warmup leaves LR sub-1e-5 for ~10% of training and the model never escapes init). The CLI flag `--warmup-steps` overrides `warmup_epochs`.
- Distributed training: `torchrun --standalone --nproc_per_node=4` for 4-GPU DDP. Rank 0 owns checkpoint saves, MLflow logging, final-val, and latency benchmark. Val inference shards across all ranks and gathers to rank 0.

### 5.4 Per-epoch evaluation strategy

Per-epoch full-val on 4309 frames at ~1 s/frame = ~72 min/epoch is unworkable across 10+ epochs. Each epoch's `[best]` selection runs on a **stratified 200-frame subsample** (positive/negative ratio preserved); a single full-val pass at end-of-training produces the canonical metrics under `final_val_*`. The subsample is fixed across epochs (deterministic seed) so per-epoch trajectories are comparable.

Subsample variance is high — at 3-pixel tolerance with ~190 positives, a single run's hit_rate can swing ±0.10 between epochs purely from sampling noise. The "best by subsample" checkpoint has empirically agreed with "best by full-val" in Phase 2 runs, but this should be verified by re-evaluating the selected checkpoint on the full val split (`scripts/eval_checkpoint.py`).

### 5.5 Inference-prior independence and the line-leakage caveat

The model takes **only image + wavelength**. The line prior never enters the model directly — it only appears as auxiliary supervision (`L_line`, §5.1) and as optional post-processing (§6.2). This keeps the model usable on new dives where no line is available yet.

**Leakage caveat**: the line for any given dive is fit (Phase 0 §3.1) from **every positive label in that dive**, including val and test labels. So when soft-snap inference (§6.2) runs on val/test:

- the soft-snap projection is informed by the exact labels we're then scoring against — a dive-level information leak.
- this is structural to the line-prior approach: the prior is *per-dive*, and even the dive-level train/val split can't separate "labels used to fit the line" from "labels used to score the model" within the same val dive.

**How to interpret reported numbers**:
- **Without soft-snap**: leakage-free. Lower bound on production performance.
- **With soft-snap**: includes leakage. Upper bound assuming the dive's line is already known (which is true for dives that have run through Phase 0 — i.e. existing dives, not first-contact ones).
- For production (new dive, no line yet): see §6.3 cold-start. Soft-snap is off until enough high-confidence predictions accumulate to fit a line.

The `L_line` aux loss does **not** leak into val/test: it only sees train-dive batches and only contributes gradient during training. The val pass evaluates a model whose weights were never directly touched by val-dive line params.

---

## 6. Inference pipeline

```
image (4K), dive_id
    ↓
[lookup wavelength for dive_id; if unknown, use bootstrap (§6.3)]
    ↓
[if known dive with confident line: select tiles intersecting the line band]
[else: select all 15 tiles]
    ↓
For each selected tile:
    chromaticity-norm + wavelength channel
        ↓
    model → (tile_heatmap, tile_presence_logit)
    ↓
Merge tile heatmaps into frame heatmap (max in overlaps)
    ↓
frame_presence = max(sigmoid(tile_presence_logits))  ≈ max(frame_heatmap)
    ↓
if frame_presence < τ_presence: return no_laser
    ↓
[if dive_id has high-confidence line: soft-snap toward line (§6.2)]
    ↓
(x, y) = argmax(frame_heatmap)  # native resolution
```

### 6.1 Presence gate

Single threshold `τ_presence`, calibrated on the val set against the false-positive target (≤ 2%).

### 6.2 Optional line refinement

If the dive's `line_confidence` exceeds `τ_line`, project the argmax toward the line:

```
final_xy = (1 - α) * argmax_xy + α * project(argmax_xy, line)
α = clip(sigmoid(line_confidence - τ_line) * (1 - pred_confidence), 0, alpha_max)
```

`α` is small (≤ `alpha_max`, default 0.3) when prediction confidence is high and the model is already near the line; larger when the model is uncertain. Implemented as `inference.soft_snap_to_line` and gated by `cfg.inference_soft_snap`.

**Eval-time vs production behavior** — see §5.5. Reporting val/test metrics with soft-snap on is appropriate only for "existing dive" performance; new-dive (cold-start) performance is the no-snap number until §6.3 bootstrap completes.

### 6.3 Cold-start (new dives)

For dives without a known wavelength or line:
1. Run the model with a default wavelength channel (e.g., 0.5 — neutral) on a batch of frames.
2. Cluster the colors at high-confidence predictions to assign green/blue.
3. Re-run with the correct wavelength channel.
4. Once enough high-confidence predictions accumulate, fit a line and (optionally) re-run with line refinement.

Cold-start performance will be slightly worse than known-dive performance; this is acceptable.

---

## 7. Evaluation

### 7.1 Metrics

Computed on the **test dive split**:

- **Hit rate (primary)**: fraction of positive frames where prediction lies within the on-screen laser blob.
  - Per-frame tolerance: derived from a local blob segmentation around the ground-truth label (color/brightness threshold in a 31×31 patch). Hit = prediction inside that blob.
  - Fixed-tolerance fallback (sanity check): prediction within `N` px of label, with `N = 3` (worst-case blob radius at z = 6 m, no divergence) for the strict variant and `N = 4` (typical with divergence) for the lenient variant. Report both.
- **Mean pixel error** (positive frames only).
- **Presence AUROC**: discrimination between positive and negative frames.
- **False-positive rate at operating point**: fraction of negative frames flagged as positive at `τ_presence`.

### 7.2 Slicing

Always report metrics sliced by:
- `wavelength` (green vs. blue)
- `line_confidence` quartile (does the model lean on the line prior or stand alone?)
- Object-distance proxy if available (close vs. far frames)

A single aggregate number hides bimodal failures.

### 7.3 Failure auditing

For each test dive:
- Plot the dive's labels, the fitted line, and the model predictions.
- Flag dives with > 2σ worse hit rate than the median.
- These are candidates for label-quality issues, unusual conditions, or dive-specific failure modes.

---

## 8. MLflow integration

The MLflow server is the system of record for experiments and artifacts.

### 8.1 Tracking

Per training run, log:

**Params**:
- Model config (backbone, head dims, sigma, etc.)
- Loss weights, optimizer config, augmentation config
- Data version / commit / row count
- Split seed and dive-level split manifest hash

**Metrics** (per epoch, stepped by epoch index):
- Train: `train_loss`, `train_loss_heatmap`, `train_loss_presence`, (Phase 3+) `train_loss_line`
- Val: `val_hit_rate_n3`, `val_hit_rate_n4`, `val_mean_pixel_error`, `val_presence_auroc`, `val_fpr_at_threshold`, `val_recall_at_threshold`
- Sliced val metrics by wavelength (`wavelength_<wl>/...`) and by line-confidence quartile (`line_q<1-4>/...`)
- Hard-negative scoring overhead: `hard_negative_score_seconds`, `hard_negative_score_n`

**Metrics** (per training step, stepped by global batch index, every `log_every_n_steps` batches — default 50):
- `step_loss`, `step_loss_heatmap`, `step_loss_presence`, `lr`. Gives a smooth loss curve in the MLflow UI; per-epoch series alone is too coarse (~10 points) to be useful for debugging.

**Metrics** (per run, logged once at end):
- `final_val_*`: full-val pass on all val frames after training, the canonical eval (per-epoch uses a 200-frame subsample for speed).
- `latency_bs1_ms`, `latency_bs8_ms`: ms/frame at 4K full-tile-grid on rank 0's GPU. Tracked from Phase 2 onward so we have a record well before the UUV port; a regression here is a real signal.

**Artifacts**:
- Best checkpoint
- Per-dive line fits (parquet)
- Per-dive wavelength tags (parquet)
- Failure audit plots for the worst dives
- Confusion-style plots (presence histogram split by ground-truth label)

### 8.2 Model registry

Register the deployed model under a name like `fishsense-laser-detector`. Stages: `Staging` → `Production`. Promotion gated on all of:
- Hit rate ≥ 95% on test
- FPR ≤ 2% at operating point
- No dive in the test set with hit rate < 80% (bimodal-failure guard)

### 8.3 Reproducibility

- Pin Python and PyTorch versions in the run's `MLproject` env.
- Log the data manifest hash (so we know which 60k labels were used).
- Log the random seed.

---

## 9. Implementation phases

Each phase ends with metrics logged to MLflow and a go/no-go before moving on.

### Phase 0 — Data & preprocessing (no model) ✓ done

- ~~Pull labels via fishsense-sdk; build the frame-level table with `rig_id` tagged.~~ Done.
- ~~Per-dive RANSAC line fit and confidence~~ — done. Output: `dive_lines.parquet` with `line_confidence`, `is_line_confident` per dive.
- ~~Per-dive wavelength clustering~~ — done. 99% of dives resolve via `label_string` (the SDK label name carries the color word); the `dive_color` clustering fallback exists but rarely fires.
- ~~Laser-size audit~~ — done. Confirmed: median 8.6 px diameter; red blobs ~2× green (median 11 vs 5.6). σ=3 is a compromise; wavelength-conditional σ is a Phase 4 sweep candidate.
- ~~Dive-level splits~~ — done, stratified by wavelength.
- **Re-run cadence**: each Phase 0 re-run picks up upstream label changes (e.g., outlier supersession). Image cache (~331 GB) hits by checksum, so re-runs are ~20 s once warm. `scripts/run_phase0.py --force` recomputes every step.

### Phase 1 — Classical CV baseline ✓ done

- ~~Wavelength-specific color mask, brightest blob, scored by line proximity~~ — done.
- Run via `scripts/run_baseline.py --split val/test`. Logs to MLflow with tag `phase1_baseline`.
- Establishes the floor a learned model has to beat.

### Phase 2 — Supervised heatmap detector ⏳ in progress

- ResNet-34 U-Net + per-tile presence head, no line aux loss yet.
- DDP training across 4 GPUs (`torchrun --nproc_per_node=4 scripts/run_train.py`).
- 4-channel input: chromaticity-normalized RGB + wavelength channel.
- **Loss formulation iteration:** focal collapsed → BCE+pos_weight=1000 escapes (see §5.1).
- **First production result (10 epochs, 2026-05-04, cleaned data):** epoch-2 best on subsample, full-val on best-checkpoint TBD; at the time of writing trending toward hit_rate_n3 ≈ 0.35, hit_rate_n4 ≈ 0.55, AUROC ≈ 0.91, FPR@0.5 ≈ 0.15. Latency ~950 ms/frame at 4K full-tile-grid (15 tiles, batch=1 or batch=8 — compute-bound on this GPU).
- **Observed failure mode** to address in Phase 3: bimodal — when the heatmap argmax hits, it hits within 4 px on >50% of frames; when it misses, mean error is 200+ px (argmax wandered into the wrong tile). Soft-snap-to-line is the targeted fix.

### Phase 3 — Add line aux loss + line post-refinement (next)

- Add `L_line`, evaluate impact on hit_rate (especially on the tail of catastrophic misses).
- Add inference-time soft-snap to line. The 200+ px misses in Phase 2 are exactly the failure mode this targets.
- Wire **early stopping on val hit-rate** before the run (DESIGN said this in §5.3 but Phase 2 didn't have it).
- Wire **resume from checkpoint** before any 50-epoch+ run.
- **Decision point**: does the prior help, especially on hard slices? If not, drop it — complexity isn't free.

### Phase 4 — Hardening & registry promotion

- Hyperparameter sweep (`pos_weight`, `σ` per wavelength, `λ_hm`/`λ_pres`/`λ_line`, presence threshold).
- Failure audit on worst dives (per DESIGN §7.3) — iterate on data quality if a few dives dominate the loss.
- Sub-pixel argmax (parabolic peak refinement) at inference — cheap, may close 1–2 px of the gap.
- Promote best run to `Production` in the model registry.

### Phase 5 (revisit if accuracy gap remains) — Two-stage cascade

- If Phase 4 stalls below 95%, switch to: presence head finds candidate tiles → focused 256² high-res heatmap on candidates only. Reduces pos:neg pixel ratio from 1:1M to 1:65k.
- Or swap ResNet-34 backbone for HRNet-W18 (higher-resolution branches throughout).
- Both are documented in §4.2 as deferred until accuracy data justifies them. Phase 2's bimodal failure is an *argmax-localization* failure, not a *pos:neg-imbalance* failure, so Phase 3 (line prior) should be tried first.

---

## 10. Risks and open questions

### Risks

- **Tiny target**: the laser is 3–8 px across. This is the dominant risk and the reason input resolution gets its own subsection (§4.1). If full-frame-high-res training doesn't fit the GPU, fall back to tiled inference. Don't paper over this with interpolation tricks — the model needs to see real laser pixels.
- **Label noise**: 60k labels are not all equal. **Mitigated upstream**: the labeling team marks outlier labels with `superseded=True` and they're filtered server-side before reaching us; the corpus dropped 28% (43834 → 31469 positives) between 2026-05-03 and 2026-05-04 as a result. `build_records()` and the entry-point scripts also filter `superseded=True` defensively in case the SDK behavior changes. Within-line label noise still exists but is much smaller.
- **Small object + class imbalance**: the standard pitfall for keypoint detection. The CenterNet penalty-reduced focal loss collapses on this corpus regardless of LR (the 1-positive-pixel-out-of-1M imbalance has the trivial solution at `pred=0` everywhere as a stable global minimum). **Mitigation**: BCE with `pos_weight ≈ 1000` (see §5.1) escapes cleanly. Hard-negative mining (5% negative rate makes random sampling useless) is in `data.HardNegativeBalancedSampler`. If those still aren't enough, two-stage cascade (Phase 5).
- **Subsample variance for checkpoint selection**: 200-frame per-epoch val subsample has ±0.10 hit_rate variance at strict 3-px tolerance. Subsample picks the right "best" epoch in practice but the absolute number is noisy. Final full-val + per-checkpoint full-val on `best_epoch.pt` close the loop.
- **Cold-start performance**: new dives don't have a line or wavelength tag. Bootstrap procedure should work but is untested. May need to instrument production for graceful degradation.
- **Distribution shift between dives**: water clarity, depth, fish density vary widely. Augmentation is the first defense; if it isn't enough, per-dive batch-norm calibration or test-time adaptation is a fallback.
- **Future rig changes**: when new rigs arrive, the per-rig prior assumption breaks. Mitigation: `rig_id` tagged from Phase 0 so we can train per-rig models without re-touching the dataset. Single unified cross-rig model is explicitly out of scope.
- **UUV deployment gap**: the model trained on a server GPU has to eventually run at 20 fps on edge. We instrument latency from Phase 2 to keep the gap honest, but real port effort is deferred. Avoiding exotic backbone ops is the cheap insurance we're paying now. Current per-frame latency on RTX 4500 Ada at 4K with 15 tiles: ~950 ms (compute-bound; bs=1 ≈ bs=8). UUV target is 50 ms/frame; 19× gap to close — Phase 5 cascade alone can plausibly cut tile count from 15 to ~3, and TensorRT INT8 typically yields 2–4× more.
- **Disk/I/O pressure during training**: at 4 ranks × 6 dataloader workers × prefetch=2 the random-access pattern on a 331 GB image cache can saturate NVMe even when most pages are RAM-cached, contending with other server processes. Phase 2 production runs use 4 ranks × 4 workers × prefetch=2 as a stable middle ground. Bumping further is fine in isolation but should be coordinated with anything else doing heavy disk on the host.
- **Line-prior leakage in val/test reporting**: the dive-level RANSAC line is fit from every positive label in the dive, including val/test labels. Soft-snap (§6.2) on val/test therefore uses the very labels we're scoring against. Mitigation: report numbers both with and without soft-snap so the leakage-free lower bound is always visible. Production cold-start is no-snap-until-bootstrap, so the with-snap number is "existing-dive" performance, not "first-contact" performance.
- **Bayer-excess upsample shift (2026-05-30, 6-ch checkpoints only)**: the per-frame Bayer-excess channels (G_excess, R_excess) are produced at half resolution and upsampled to full res with `np.repeat(...,2,axis=0)` along both axes. `np.repeat` writes each half-res supercell value `[i, j]` into the full-res block `[2i:2i+2, 2j:2j+2]`, placing the value at the supercell top-left rather than its true centroid `(2i+0.5, 2j+0.5)`. The 6-ch model fuses correctly-aligned chromaticity channels with shifted Bayer-excess channels and learns a centroid pulled ~(−1.13, −2.07) px toward the shifted features. **Empirically uniform across rigs and wavelengths.** Workaround: subtract the calibrated offset at inference via `--pixel-bias-offset DX DY` (plumbed through `cfg.inference_pixel_bias_offset_{x,y}`). On run3 epoch_021 this lifts hit_rate_n3 0.526 → 0.798. Proper fix: change `_decode_raw_bayer_excess` (`src/laser_detector/preprocessing/image_loader.py:148-150`) to use centered bilinear upsampling, rebuild the bayer_excess cache, and retrain — pending NAS access. Calibration is checkpoint-specific; 4-ch JPEG checkpoints are exempt (offset = 0). _DESIGN doc gap: the sensor-coord + 6-channel + Bayer-excess pipeline predates this doc revision; §4.1's "4-channel input" is no longer the production configuration. A sync of §3–4 to reflect the sensor-coord refactor is pending separate work._

### Open questions

Operational items still in motion:

- **NAS-path issues** for some dives (notably 219, 249) — frames are missing on disk; pre-warm currently flags them and the trainer drops them. Investigation upstream is in progress.
- **Resume-from-checkpoint** is required for any run > 10 epochs and isn't wired yet. Blocking before the next 50-epoch run.
- **Early stopping on val hit-rate** — stated in §5.3, not yet wired. Will land alongside resume.

Resolved on 2026-05-02:
- Native resolution: **4K**. Drives the tiling strategy in §4.1.
- Per-frame depth / object-distance metadata: **not available**. Removed from auxiliary-task consideration.
- Eventual deployment: **20 fps on UUV**, video. Captured in §1.
- Mixed-color frames: **none** — single color per dive (overstated; ~42% of dives have a minority-color label, resolved by majority-vote per §3.2).
- Rigs: similar in current data; future rigs get separate models.

Resolved during Phase 2 (2026-05-03 → 2026-05-04):
- **MLflow server**: live at `https://mlflow.krg.ucsd.edu`, basic-auth via the `mlflow-oidc-auth` plugin. Per-step + per-epoch metrics both flowing.
- **Heatmap loss**: BCE with `pos_weight=1000`, not focal — see §5.1.
- **Warmup**: 1000 absolute steps, not 1 epoch — see §5.3.
- **Checkpoint selection**: best by subsample hit_rate during the run, then full-val on the chosen checkpoint via `scripts/eval_checkpoint.py` for the canonical number.
- **Outlier labels**: filtered upstream via `superseded=True`. Defensive filter in `build_records()` and entry-point scripts.
