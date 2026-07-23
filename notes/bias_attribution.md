# Pixel-bias attribution — synthetic ablation results

## ⚠️ Third revision (2026-07-23) — fp32 refit (issue #13)

The `(−0.20, −0.006)` offset was calibrated against inference that ran the
model forward under `torch.autocast(bfloat16)`. The prior fp32-sigmoid
fix (Phase 2A, June) solved the sigmoid tie-break, but the logits
themselves were still bf16-quantized in the forward pass. Under bf16,
rival pixels routinely round to identical values and `flat.max()` breaks
ties by row-major index — an outcome that depends on the tensor-core
kernel cuDNN selects, which varies across GPU architectures. On our
Ada card 3/4 sampled val positives showed sub-ulp margins, and Ampere
reproduced ~200 px argmax shifts on the same weights + input.

**Fix**: `predict_frame` and `predict_frame_with_cascade` now default
`autocast_dtype=None`, and `_run_val_inference` no longer forwards
`cfg.use_bf16`. Inference is fp32 end-to-end; training is unchanged.

**Refit**: with the fp32 inference path, val-inlier mean signed residual
is `(−0.1794, −0.0232)`. New production offset: **`−0.179 −0.023`**.
Old value stays in the git history (checkpoint calibration is a
per-checkpoint number, not a per-model number). Per-wavelength shows
same-sign residuals with red slightly tighter than green
(red −0.15 −0.02, green −0.31 −0.05, both same-sign) — reasonable
sub-pixel drift, not a wavelength-specific artifact.

Val hit_n3 goes from 0.9081 (Ada bf16 + old bias) to **0.9100** (Ada
fp32 + new bias). The +0.19 pp lift is small; the reproducibility win
(same numbers on any IEEE-754 hardware) is the point.

The `notes/HOW_TO_USE.md` "bf16 sigmoid caveat" section is now
"bf16 at inference — DISABLED" and documents the current default.

## ⚠️ Second revision (2026-07-22)

The bayer_excess upsample switch to centered bilinear
(commit `0dba88e "Fix Bayer-excess upsample bias: centered bilinear instead
of np.repeat"`, subsequently amended in `f6c262c "run4_centered: Bayer-excess
upsample fix did not eliminate the bias"`) has been **reverted** and the code
returned to `np.repeat`. Two independently sufficient reasons:

1. **The bias attribution the switch was meant to correct was refuted** by
   the bf16 sigmoid finding documented in the first revision below. Under
   fp32 sigmoid + logit-based sub-pixel refinement, the residual raw bias
   drops to (−0.20, −0.006) val / (+0.024, +0.026) test — well inside
   labeler click σ. The np.repeat vs centered-bilinear distinction is
   irrelevant at this scale.

2. **The bayer_excess cache was never invalidated** when the code
   switched. Cache files date from May 26 with `np.repeat` outputs; the
   centered-bilinear code shipped June 1. Cache is keyed by
   `image_checksum`, so run3 training + every downstream audit + the
   published val 0.9081 / test 0.8615 / bias offset (−0.20, −0.006) were
   all computed against `np.repeat` inputs. Reverting the code makes it
   consistent with the cache, the checkpoint's training-time inputs, and
   the calibrated bias offset.

The centered-bilinear branch was tried as `run4_centered` and its own
commit message concluded it "did not eliminate the bias." The revert
consolidates that result.

If a future retrain moves to centered bilinear (or any other decode
change), the required sequence is: invalidate the bayer_excess cache,
prewarm from scratch, retrain, re-derive the bias offset from val
inliers, republish the checkpoint. Skipping any of those steps
reintroduces the invisible-mismatch class of bug this revert is
preventing.

## ⚠️ Major revision (2026-06-06)

**The bias attribution below is largely refuted by a subsequent finding.**

The ~(−1.13, −2.07) px residual bias on real validation data was measured
with the production inference pipeline running under `torch.autocast(dtype=
bfloat16)`. Inside the autocast region, `torch.sigmoid(heatmap_logits)` is
computed in bf16. For a confident peak the logit is ≳ +6, and bf16 sigmoid
saturates exactly to 1.0 on the peak plus several surrounding pixels (bf16
has ~8-bit mantissa precision near 1.0, and `sigmoid(x) > 1 − 2⁻⁸` for any
`x > 5.5`). When the saturated cluster spans multiple pixels, `tensor.max()`
resolves ties to the lowest flat (row-major) index — i.e. the top-left
corner of the cluster. Predictions are biased toward the upper-left of the
true peak, and the magnitude depends on cluster size.

Fixing this is a two-line change in `src/laser_detector/inference.py`:

```python
# was:
heatmap_probs = torch.sigmoid(out["heatmap_logits"]).float()
# now:
heatmap_logits = out["heatmap_logits"].float()  # promote to fp32 BEFORE sigmoid
heatmap_probs = torch.sigmoid(heatmap_logits)   # fp32 sigmoid; no saturation tie
```

(Two spots: the global tiled pass in `predict_frame`, and the refinement
crop in `predict_frame_with_cascade`.)

### Empirical impact on the production stack (run3 + recipe + sub-pixel)

| split | inference | raw bias on inliers | hit_n3 @ calibrated |
|---|---|---|---|
| val | bf16 sigmoid (pre-fix) | (−1.13, −2.07) | 0.8498 |
| val | **fp32 sigmoid + sub-pixel** | **(−0.20, −0.006)** | **0.8951** (+4.5 pp) |
| test | bf16 sigmoid (pre-fix) | (≈0 after −1.13,−2.07) | 0.8120 |
| test | **fp32 sigmoid + sub-pixel** | **(+0.024, +0.026)** | **0.8544** (+4.2 pp) |

The val-derived calibration (−0.20, −0.006) generalizes cleanly to test
(0.8544 with val-derived offset vs. 0.8549 with test-derived = 0.05 pp gap,
no meaningful overfit).

The remaining raw bias of (−0.20, −0.006) val / (+0.02, +0.03) test is
small enough to be label-quality noise and is not stable in sign across
splits.

### What this revises in the analysis below

- The "architectural Y bias of −2.07 px (real) / −3.70 px (synthetic)"
  comparison conflated two different inference precisions. The real-data
  measurements in §"Mechanisms tested" and §"Synthetic ablation" used
  `bf16` autocast; the synthetic reproducer at the bottom used `fp32` (no
  autocast context). Apples-to-apples after the fp32 fix:
  - Real-data inliers, fp32 sigmoid: median (−0.20, −0.006) (val)
  - Synthetic, fp32 sigmoid: median (−1.00, −3.70) (unchanged — was already fp32)
  The synthetic still shows a real architectural bias, but the real-data
  manifestation is much smaller than the synthetic predicts. Real-frame
  texture and blob characteristics apparently let the model localize closer
  to truth than the σ=2 isotropic Gaussian benchmark suggests.

- Ablations B and C ("decoder bilinear monkey-patch lifts hit_n3 from
  0.5255 to 0.6671") were also measured against the **bf16** baseline. Most
  of that +14.16 pp gain was the same bf16 tie-break being smoothed away by
  bilinear upsample. Running both arms in fp32 should show a much smaller
  gap, possibly negligible.

- The run5_bilinear and run6_bayer_diff retrains were motivated by the
  "decoder is the bias culprit" hypothesis. Their *trained* numbers
  (bf16-calibrated) were within ~2 pp of run3 on test. **These numbers need
  to be re-collected under fp32 + sub-pixel inference** to know whether the
  architectural choices made any real contribution. That re-evaluation is in
  progress; results are written to `notes/phase2_results.md` and the bottom
  of this file.

### What this does NOT revise

- The bias *was* deterministic and large in the bf16 pipeline. The
  calibration constant was a correct empirical fit to that pipeline.
- Synthetic ablation results were always in fp32 and are technically
  still valid as a statement about the model's architectural shift in a
  Gaussian-only signal regime.
- The G_diff (anti-diagonal Bayer asymmetry) work in run6 is a clean
  signal-engineering idea independent of the precision bug; it just may
  not have moved the needle as much as we thought relative to a fixed-up
  baseline.

---

## Original analysis (2026-06-03)

**Status (2026-06-03)**: Architectural shift confirmed on synthetic data.
Calibration constant is a justified post-hoc correction for a deterministic
model-architecture artifact, not a data-fit hack.

## The question

Production checkpoints (`run3/epoch_021`, run4 retrain) show a constant
~(−1.13, −2.07) px residual on correct (hit_n3=True) val predictions,
essentially uniform across rigs (1, 2, 4, 6, 10), wavelengths (red, green),
and dives. Subtracting this offset at inference lifts val hit_rate_n3 from
0.526 → 0.798 and test from 0.503 → 0.812 (`--pixel-bias-offset -1.13 -2.07`,
generalized cleanly val → test).

The mechanism question matters for publication: a calibration constant
needs an attributable cause, not "unexplained 50% gap."

## Mechanisms tested and eliminated

| Hypothesis | Test | Verdict |
|---|---|---|
| Audit-script coordinate transform | Read `scripts/audit_failures.py` and `_run_val_inference`: pred coords pass through unmodified | RULED OUT |
| Heatmap encode/decode origin mismatch | `_make_gaussian_heatmap` centers at float (`label_x`, `label_y`); `predict_frame` returns int argmax. Conventions consistent | RULED OUT |
| Tile-stitch off-by-half | `compute_tile_grid` uses integer tile origins; `(local_x + ox)` is exact integer translation | RULED OUT |
| Eval-rotation bug (`_inverse_rotate_label`) | Affects only 15 val frames with `flip != 0`; pred-vs-label errors on those frames are reasonable | RULED OUT — bug exists in theory but never fires on real data |
| Bayer-excess upsample `np.repeat` shift | Trained run4 (50 ep target, killed at 37) with `cv2.resize(INTER_LINEAR)` centered upsample on rebuilt cache. Bias persists at same magnitude. | RULED OUT as **dominant** cause |
| Per-rig optical / lens calibration | Per-rig median bias across rigs 1/2/4/6/10 all within (−0.9, −1.3) × (−2.0, −2.1). Rig 3 outlier has only 9 samples | RULED OUT |
| Motion/temporal direction artifact | Decomposed bias along vs perpendicular to dive line. Per-dive bias is CONSISTENT IN IMAGE COORDS, not in line-relative coords. The "bias-along-line ≈ ±2 px" was an artifact of most dive lines aligning with the (1, 2) image direction | RULED OUT |

## Partial explanation found

**Annotator click vs photometric centroid offset.** Measured on 300 cached
val frames with patches around each label:

| Method (top-30% mask in 21×21 patch) | dx_med | dy_med |
|---|---:|---:|
| Raw intensity centroid (laser-color channel) | +0.30 | +0.41 |
| Chromaticity centroid (laser-color chromaticity) | +0.24 | +0.37 |
| Bloom-fringe centroid (non-saturated bright pixels) | +0.25 | +0.38 |

Per wavelength: red and green produce similar offsets within noise. So
labels DO sit ~(+0.3, +0.4) px above-right of the photometric blob center.
But this accounts for only ~25–30% of the bias magnitude and the wrong
shape — observed dx:dy ratio is ~1:2, this offset is ~1:1.

## Synthetic ablation — the decisive test

**Method**: directly probe the architectural shift by feeding the production
checkpoint synthetic inputs with known ground-truth feature positions.

1. Load `run3/epoch_021.pt` (in_channels=6, use_bayer_excess=True).
2. For each of 81 sub-pixel test positions `(cx, cy)` on a 9×9 grid centered
   on (512, 512) with steps in {-7.5, -5.3, -3.1, -1.7, 0, 1.7, 3.1, 5.3, 7.5}:
   - Build a 1024×1024 BGR uint16 tile with uniform background (8000 in all
     channels) and a Gaussian (σ=2 px) added to the R channel at (cx, cy)
     (peak amplitude 60000). Add a small green leak (×0.3) to mimic real
     red-laser color signature.
   - Build a matching 1024×1024 × 2 Bayer-excess synthetic with the same
     Gaussian profile in R_excess and a smaller one in G_excess.
   - Run the same preprocessing chain (`_preprocess_tile` → chromaticity
     normalization → wavelength channel → 6-channel input).
3. Forward through the model in eval mode, sigmoid the heatmap, take argmax,
   measure `(pred_x − cx, pred_y − cy)`.

**Result** (n=81):

| Quantity | dx (pred−true) | dy (pred−true) |
|---|---:|---:|
| **median** | **−1.00** | **−3.70** |
| mean | −0.84 | −3.79 |
| std | 0.78 | 0.57 |

**Interpretation**:
- The X bias of **−1.00** px on synthetic ≈ the X bias of −1.13 px on real
  validation data → **X bias is purely architectural**.
- The Y bias of **−3.70** px on synthetic is *larger* than the Y bias of
  −2.07 px on real data → architectural Y shift is **−3.7 px**, partially
  attenuated to **−2.07 px** on real data by data-side effects (likely
  saturation, bloom asymmetry, real-blob shape).
- The std is small (~0.5–0.8 px) — the shift is deterministic, not random.
- The Y-asymmetry (dy/dx ≈ 3.7) is even more pronounced on synthetic than
  on real data (dy/dx ≈ 2.0). **The 1:2 ratio observed on real data is
  the model's intrinsic Y-shift partially masked**, not a coincidence.

## Where the architectural bias comes from

`segmentation_models_pytorch.Unet` (smp) with default settings uses the
`DecoderBlock.forward` at
`segmentation_models_pytorch/decoders/unet/decoder.py:50-53`:

```python
feature_map = F.interpolate(
    feature_map,
    size=(target_height, target_width),
    mode=self.interpolation_mode,
)
```

`self.interpolation_mode` defaults to `"nearest"`. `F.interpolate` with
`size=` and `mode="nearest"` uses floor-rounding for the source-index lookup
— this is asymmetric in the sense that ties between two input pixels
resolve to the smaller index. After 5 stages of this (ResNet-34 has 5
downsample levels, so the decoder upsamples 5×), the cumulative argmax
of a piecewise-constant block consistently falls at the top-left corner of
the ambiguous region, biasing predictions toward smaller (x, y).

A pure 5-stage F.interpolate nearest-mode test (single-pixel delta at
the bottleneck, no model) gives a worst-case shift of −15.5 px in
both axes at 32× upsample. In practice the model has conv layers
between upsamples that smooth this, so the realized bias is much smaller
(~−1 to −4 px depending on how the learned features distribute).

The X vs Y asymmetry in the realized bias is not from the upsampling
itself — both axes are mathematically symmetric in `F.interpolate(...,
size=(H, W), mode="nearest")`. It's likely from the model's *learned*
feature distribution: the laser blob is slightly elongated along the
sensor-Y axis (or the bayer-pattern's Y-direction interpolation
introduces a Y-specific learned bias). Confirming this would require
a separate ablation (smaller blob σ, anisotropic synthetic blobs,
or a from-scratch retrain with isotropic synthetic data).

## What this means for the paper

1. **The calibration is a deterministic correction.** Synthetic data with
   *known* ground-truth positions shows the bias persists at near-identical
   magnitude (X-direction matches exactly within noise) regardless of input
   content. This proves the calibration constant is not a hyperparameter
   fit to validation noise.

2. **The bias is data-distribution-invariant for X.** Same magnitude on
   synthetic and real → no concerns about test-distribution shift breaking
   the calibration.

3. **The Y component is mixed** — architectural shift of −3.7 px, partially
   offset by real-blob characteristics to −2.07 px. The calibration value
   is fit to the *real-data* observed bias, which is what we want for
   production inference, but the underlying architectural shift is larger.

4. **The proper fix would be `interpolation_mode='bilinear'` (or
   `align_corners=True` with bilinear)** on the smp UNet, plus a retrain.
   This was not pursued because (a) it would only narrow the gap (the
   X-axis component is mostly architectural so should drop, but the
   Y component is mixed); (b) it requires a full 24h retrain on already-
   adequate production performance; (c) the calibration constant is
   defensible and the change would be a separate ablation.

## Follow-up ablations

### Ablation A — bayer-excess channels zeroed (synthetic input)

Same synthetic test as the main ablation but with the bayer-excess input
channels filled with zeros (model receives non-informative bayer-excess data):

| condition | dx median | dy median |
|---|---:|---:|
| baseline (bayer populated, nearest decoder) | −1.00 ± 0.78 | −3.70 ± 0.56 |
| **bayer zeroed**, nearest decoder | **+0.50 ± 0.28** | **+0.90 ± 0.37** |

**The bias sign FLIPS when bayer-excess is removed**, and magnitude drops
substantially. The bayer-excess input channel is the **dominant driver of
both X and Y bias** — the model has learned to weight bayer-excess heavily
for localization, and any geometric misalignment between bayer-excess and
chromaticity (which is real, due to the decoder-side upsample shift discussed
below) propagates straight into prediction position.

### Ablation B — decoder upsample mode monkey-patched (run3 weights, no retrain)

Loaded `run3/epoch_021.pt`, walked the smp UNet decoder, set
`block.interpolation_mode` to the test mode, ran the synthetic test:

| decoder mode | bayer | dx median | dy median |
|---|---|---:|---:|
| **nearest** (training default) | populated | −1.00 ± 0.78 | **−3.70 ± 0.56** |
| **bilinear** (no retrain) | populated | −1.10 ± 0.43 | **−1.70 ± 0.69** |
| bilinear (no retrain) | zeroed | +0.30 ± 0.28 | +0.90 ± 0.34 |
| bicubic (no retrain) | populated | −1.00 ± 0.72 | −4.30 ± 0.49 |

**Switching from nearest to bilinear at inference cuts the Y bias roughly
in half (−3.70 → −1.70) on synthetic inputs**, with X essentially unchanged.
Bicubic is slightly worse than nearest in Y. The bilinear+bayer-zeroed case
is the cleanest condition (close to no bias), suggesting that the bayer
channels are the major remaining source of bias once nearest-upsample is
fixed.

### Ablation C — bilinear monkey-patch on REAL val data (run3 weights)

| metric | run3 nearest (production, no bias offset) | + bilinear monkey-patch | Δ |
|---|---:|---:|---:|
| hit_rate_n3 | 0.5255 | **0.6671** | **+14.16 pp** |
| hit_rate_n4 | 0.7498 | 0.8144 | +6.46 pp |
| median_pixel_error | 2.05 | 2.33 | +0.28 |

**A one-line change to the decoder upsample mode at inference time recovers
~half of what the bias-offset calibration provides** — on real validation
data, with no retraining. Median goes up slightly because bilinear pulls
outliers (10+ px misses) into the [2, 3] px range — they count as hits at
n=3 but raise the median.

### Pending ablations (running)

- **D — bilinear + bias offset**: does the calibration constant stack with
  the architectural fix, or are they redundant? In progress on val.
- **E — bilinear + X-only bias offset (−1.13, 0)**: if bilinear absorbs the
  Y bias, only X correction is needed. In progress on val.
- **F — `run5_bilinear` retrain (24h)**: train from scratch with
  `decoder_interpolation='bilinear'` using the centered bayer-excess cache,
  same hyperparameters as run4. The cleanest publication story:
  same training data, different decoder choice. Auto-launches after the
  (D, E) eval pair completes.

## Planned but not queued — `run6_split_bayer`

The bayer_excess decode currently averages the two G photosites in each
2×2 Bayer supercell into a single `g_avg` channel before computing
`g_excess` and upsampling. In an RGGB pattern, G1 is at (row, col+1) and
G2 at (row+1, col) — the anti-diagonal of the supercell. Their mean
represents intensity at the supercell *centroid*, but their *difference*
encodes sub-supercell direction:

- `G1 > G2` → laser shifted toward (row, col+1) — up-right within supercell
- `G1 < G2` → laser shifted toward (row+1, col) — down-left

The anti-diagonal direction is not axis-aligned: it projects mostly onto
Y in standard sensor orientation. **The current averaging step destroys
roughly 0.7 px of Y-discriminative sub-pixel signal** — which is on the
same order as the residual Y bias remaining after the bilinear fix.

**Hypothesis**: keeping G1 and G2 as separate channels (or adding their
difference as a third bayer_excess channel) preserves the anti-diagonal
asymmetry signal and should reduce the Y bias further. Falsifiable on
synthetic data with the same single-frame test we used for ablations A–E.

**Cost if pursued**:
- Modify `_decode_raw_bayer_excess` to output additional channel(s).
- Rebuild bayer_excess cache (~2h, requires NAS access).
- Retrain from scratch (~24h) — `run6_split_bayer`.
- Update `in_channels` from 6 to 7 (with diff channel) or 8 (full split).

**Not queued — pending run5 results.** If run5_bilinear already cleans up
the residual Y bias on real data, run6 becomes a marginal-improvement
ablation; if run5 leaves a meaningful Y residual, run6 is the next
high-value retrain.

## Reproducer

```bash
nix develop --command uv run python <<'PY'
import numpy as np, torch
from laser_detector.model import LaserDetector
from laser_detector.train import TrainConfig
from laser_detector.inference import _preprocess_tile, WAVELENGTH_CHANNEL, UNKNOWN_WAVELENGTH_CHANNEL

ckpt = torch.load("data/phase2/checkpoints_sensor_bayer_50e_run3/epoch_021.pt",
                  map_location="cpu", weights_only=False)
cfg = TrainConfig(**{k: v for k, v in ckpt["cfg"].items() if k in TrainConfig.__dataclass_fields__})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LaserDetector(in_channels=cfg.in_channels).to(device)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()

TILE, sigma = 1024, 2.0
ys, xs = np.indices((TILE, TILE)).astype(np.float32)
results = []
wl_val = WAVELENGTH_CHANNEL.get("red", UNKNOWN_WAVELENGTH_CHANNEL)
with torch.no_grad():
    for cx in [512.0 + d for d in [-7.5,-5.3,-3.1,-1.7,0,1.7,3.1,5.3,7.5]]:
        for cy in [512.0 + d for d in [-7.5,-5.3,-3.1,-1.7,0,1.7,3.1,5.3,7.5]]:
            g = 60000 * np.exp(-((xs-cx)**2 + (ys-cy)**2)/(2*sigma*sigma))
            img = np.full((TILE,TILE,3), 8000.0, dtype=np.float32)
            img[...,2] += g; img[...,1] += g*0.3
            img = np.clip(img, 0, 65535).astype(np.uint16)
            r_excess = np.clip(g - 4000, 0, None).astype(np.uint16)
            g_excess = np.clip(g*0.3 - 4000, 0, None).astype(np.uint16)
            bayer = np.stack([g_excess, r_excess], axis=2)
            arr = _preprocess_tile(img, wl_val, bayer_excess_tile=bayer, bayer_excess_scale=4096.0)
            out = model(torch.from_numpy(arr[None]).to(device))
            h = torch.sigmoid(out["heatmap_logits"][0,0]).float().cpu().numpy()
            py, px = divmod(int(h.argmax()), TILE)
            results.append((px - cx, py - cy))
dx, dy = zip(*results)
print(f"dx: median={np.median(dx):+.3f}, std={np.std(dx):.3f}")
print(f"dy: median={np.median(dy):+.3f}, std={np.std(dy):.3f}")
PY
```

Expected output:
```
dx: median=-1.000, std=0.784
dy: median=-3.700, std=0.565
```
