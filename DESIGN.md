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
- **Per-dive invariants**: each dive is single-color (green or blue) and uses a fixed-rig laser, so all positive labels in a dive are colinear in image space (modulo noise and label error).
- **Laser blob size**: at z = 6 m, ~2.66 × 2.66 px without divergence (≈ 3 px diameter); with divergence ~6–8 px diameter. Closer objects produce larger blobs. **The 3 px floor drives the input-resolution choice in §4.1.**
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

Stratify the split on the green/blue tag (computed in §3.2) so both colors are present in each set.

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

**Label cleaning**: drop labels with perpendicular distance to the dive's line greater than `k * inlier_residual_std`, with `k ≈ 3`. Logged so we can audit dropped labels.

### 3.2 Per-dive wavelength tag

Each dive is single-color but the wavelength field isn't recorded. Recover it:

1. For each labeled positive frame, sample a 5×5 patch around the labeled pixel; take the mean RGB.
2. Average those means across all positive labels in the dive → one `dive_color` vector per dive.
3. Cluster all `dive_color` vectors across the dataset with k=2 (or 3 if a third "ambiguous" cluster appears).
4. Assign `green` / `blue` tag per dive based on which cluster centroid is greener vs. bluer.

**Output per dive**: `wavelength ∈ {green, blue}`, plus the `dive_color` vector (kept for diagnostics).

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
- **Wavelength conditioning**: one extra input channel set to `0.0` for green and `1.0` for blue. Concatenated to the chromaticity-normalized RGB → 4-channel input.

### 4.2 Backbone

U-Net with a ResNet-34 encoder (ImageNet-pretrained), or HRNet-W18. Either is appropriate; ResNet-34 U-Net is simpler and the default choice. We have post-processing latency budget for v1, so no need for a phone-grade backbone yet.

Both options TensorRT-port cleanly, so neither blocks the eventual UUV deployment. Avoid backbones with attention-only paths or unusual ops (Swin, deformable convs, etc.) until accuracy data shows they're needed — they're harder to ship to edge.

### 4.3 Heads

Two heads share the backbone, both operating on a single 1024 × 1024 tile:

- **Heatmap head**: 1 channel, full tile resolution (1024 × 1024). Sigmoid activation. Target is a Gaussian centered at the labeled pixel (when in-crop) with a small fixed sigma (σ = 2 px at native resolution — matches the worst-case blob). Tiles without a label inside (negative-tile case) have an all-zero target.
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

- **`L_heatmap`** (focal heatmap loss, à la CenterNet): applied only on positive frames. Penalizes the predicted heatmap against the Gaussian target. Focal weighting handles the severe positive-pixel imbalance (one Gaussian blob in a 512×512 frame).
- **`L_presence`** (BCE): applied on all frames, with hard-negative mining — sample 50/50 from positives and from "hard" negatives (negatives with high heatmap response in the previous epoch). Random sampling is dominated by trivial negatives and produces a useless presence head.
- **`L_line`** (line-consistency aux loss): for positive frames in dives with `line_confidence > τ_line`, penalize the perpendicular distance from the predicted heatmap centroid (soft-argmax) to the dive's line. Weighted by `line_confidence`. Skipped for line-ambiguous dives.

Default weights: `λ_hm = 1.0`, `λ_pres = 0.5`, `λ_line = 0.1`. Tuned via MLflow sweep.

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
- Batch size: as large as fits on the GPU (target 16+)
- Mixed precision (bf16 if supported, else fp16)
- Epochs: ~50, with early stopping on val hit-rate
- Warmup: 1 epoch linear

### 5.4 Inference-prior independence

The model takes **only image + wavelength**. The line prior never enters the model directly — it only appears as auxiliary supervision and as optional post-processing (§6.2). This keeps the model usable on new dives where no line is available yet.

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
α = sigmoid_blend(line_confidence, prediction_confidence)
```

`α` is small (≤ 0.3) when prediction confidence is high and the model is already near the line; larger when the model is uncertain.

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

**Metrics** (per epoch):
- Train and val: total loss, heatmap loss, presence loss, line loss
- Val: hit rate (blob-tolerance and fixed-tolerance N=3, N=4), mean pixel error, presence AUROC, FPR@τ
- Sliced val metrics by wavelength
- **Inference latency**: ms/frame on the training GPU at batch=1 and batch=8. Tracked from Phase 2 onward so we have a record well before the UUV port; a regression here is a real signal.

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

### Phase 0 — Data & preprocessing (no model)

- Pull labels via fishsense-sdk; build the frame-level table with `rig_id` tagged.
- Implement per-dive RANSAC line fit and confidence.
- Implement per-dive wavelength clustering.
- **Laser-size audit**: for a sample of positive frames at 4K native, segment the laser blob locally and compute its pixel-diameter distribution. Confirms the 3–8 px assumption and informs σ choice for the heatmap target.
- Build dive-level splits, stratified by wavelength.
- **Deliverable**: cleaned dataset + per-dive metadata + blob-size distribution, all in MLflow.
- **Decision point**: confirm line-fit confidence distribution looks sane; confirm wavelength clusters are clean; confirm blob-size assumption holds.

### Phase 1 — Classical CV baseline

- Per dive: compute a wavelength-specific color mask, find the brightest blob, score by closeness to the dive's line.
- No learning. This is the floor.
- **Deliverable**: hit rate / FPR baseline numbers in MLflow.
- **Decision point**: how much headroom does the learned model need to provide?

### Phase 2 — Supervised heatmap detector

- ResNet-34 U-Net, heatmap + presence head, no line aux loss yet.
- Train, evaluate, log.
- **Deliverable**: first learned-model metrics.
- **Decision point**: where does it fail? Slicing tells us whether to add the line aux loss next or focus on color robustness.

### Phase 3 — Add line aux loss + line post-refinement

- Add `L_line`, evaluate impact.
- Add inference-time soft-snap to line, evaluate impact.
- **Decision point**: does the prior help, especially on hard slices? If not, drop it — complexity isn't free.

### Phase 4 — Hardening & registry promotion

- Hyperparameter sweep (loss weights, sigma, presence threshold).
- Failure audit on worst dives; iterate on data quality if a few dives dominate the loss.
- Promote best run to `Production` in the model registry.

---

## 10. Risks and open questions

### Risks

- **Tiny target**: the laser is 3–8 px across. This is the dominant risk and the reason input resolution gets its own subsection (§4.1). If full-frame-high-res training doesn't fit the GPU, fall back to tiled inference. Don't paper over this with interpolation tricks — the model needs to see real laser pixels.
- **Label noise**: 60k labels are not all equal. The colinearity check catches gross outliers but not within-line errors. May need spot human re-labeling on low-performing dives.
- **Small object + class imbalance**: the standard pitfall for keypoint detection. Focal heatmap loss + hard-negative mining are the standard mitigations; if they're insufficient, consider higher input resolution or two-stage cascade (revisit).
- **Cold-start performance**: new dives don't have a line or wavelength tag. Bootstrap procedure should work but is untested. May need to instrument production for graceful degradation.
- **Distribution shift between dives**: water clarity, depth, fish density vary widely. Augmentation is the first defense; if it isn't enough, per-dive batch-norm calibration or test-time adaptation is a fallback.
- **Future rig changes**: when new rigs arrive, the per-rig prior assumption breaks. Mitigation: `rig_id` tagged from Phase 0 so we can train per-rig models without re-touching the dataset. Single unified cross-rig model is explicitly out of scope.
- **UUV deployment gap**: the model trained on a server GPU has to eventually run at 20 fps on edge. We instrument latency from Phase 2 to keep the gap honest, but real port effort is deferred. Avoiding exotic backbone ops is the cheap insurance we're paying now.

### Open questions

All design-shaping open questions resolved as of 2026-05-02. Remaining operational unknown:

- **MLflow server URL / auth**. User is putting this together. Needed before Phase 2 logging starts; doesn't block Phase 0.

Resolved on 2026-05-02:
- Native resolution: **4K**. Drives the tiling strategy in §4.1.
- Per-frame depth / object-distance metadata: **not available**. Removed from auxiliary-task consideration.
- Eventual deployment: **20 fps on UUV**, video. Captured in §1.
- Mixed-color frames: **none** — single color per dive.
- Rigs: similar in current data; future rigs get separate models.
