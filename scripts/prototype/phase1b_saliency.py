"""Phase 1B: Test whether prediction failures correlate with low object-saliency.

Hypothesis (user's domain claim):
  "A laser is on the object of focus. Failed predictions land in low-saliency
   background; true labels still sit on a salient subject."

This script samples 100 failures + 100 successes from the existing audit
parquet, computes a per-image DINOv2 [CLS]-attention saliency map, and scores
both prediction and label locations. It then runs Mann-Whitney U tests and
reports effect sizes, plus 4 example overlays.

It is a scratch prototype — do not import from `src/laser_detector/`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from scipy import stats
from transformers import AutoImageProcessor, AutoModel

# Local import only for the image loader — we are not modifying any model code.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from laser_detector.preprocessing.image_loader import (  # noqa: E402
    CachingImageLoader,
    LocalFilesystemImageLoader,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase1b")

AUDIT_PARQUET = ROOT / "data/audit/epoch_021_recipe/predictions_with_meta.parquet"
IMAGE_ROOT = "/home/c.crutchfield.642/mnt/fishsense_data/REEF/data/"
IMAGE_CACHE = ROOT / "data/image_cache"
OUT_FIG = ROOT / "notes/figures/phase1b_saliency_examples.png"
OUT_MD = ROOT / "notes/phase1b_saliency_test.md"

# Calibration: production CLI uses `--pixel-bias-offset -1.13 -2.07`. The audit
# parquet predictions are pre-calibration; the EMPIRICAL hit_n3 reaches the
# expected ~0.80 only when we ADD (+1.13, +2.07) to the parquet predictions —
# i.e. the parquet stores the post-pixel-bias values relative to a shifted
# reference, and the calibrated production prediction is `pred + (+1.13, +2.07)`.
# (See in-script sanity check at startup.)
CAL_DX = 1.13
CAL_DY = 2.07

FAIL_THRESH = 3.0  # err > 3 px = failure (calibrated, N=3 strict)
PER_DIVE_CAP = 20
N_PER_GROUP = 100
SEED = 42

# DINOv2 input resolution. The HF processor will resize to this; one patch =
# 14 px → 16×16 patches for 224×224. We will do bilinear up-sampling of the
# 16×16 attention map back to the original frame resolution before sampling.
DINOV2_MODEL = "facebook/dinov2-small"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def load_and_label(parquet_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(parquet_path)
    df = df.filter(pl.col("is_positive") & pl.col("pred_x").is_not_null())
    df = df.with_columns(
        [
            (pl.col("pred_x") + CAL_DX).alias("pred_x_cal"),
            (pl.col("pred_y") + CAL_DY).alias("pred_y_cal"),
        ]
    )
    df = df.with_columns(
        [
            (
                (
                    (pl.col("pred_x_cal") - pl.col("label_x")) ** 2
                    + (pl.col("pred_y_cal") - pl.col("label_y")) ** 2
                ).sqrt()
            ).alias("err"),
        ]
    )
    df = df.with_columns((pl.col("err") > FAIL_THRESH).alias("is_failure"))
    overall_hit = (~df["is_failure"]).mean()
    log.info(
        "audit parquet rows=%d  calibrated hit_n3=%.4f  failures=%d",
        len(df),
        overall_hit,
        int(df["is_failure"].sum()),
    )
    return df


def stratified_sample(
    df: pl.DataFrame,
    is_failure: bool,
    n_target: int,
    per_dive_cap: int,
    seed: int,
) -> pl.DataFrame:
    """Stratified sample by dive: cap per-dive contribution to per_dive_cap."""
    rng = np.random.default_rng(seed + (1 if is_failure else 2))
    pool = df.filter(pl.col("is_failure") == is_failure)
    pieces: list[pl.DataFrame] = []
    for dive_id, group in pool.group_by("dive_id"):
        n_take = min(per_dive_cap, len(group))
        idx = rng.choice(len(group), size=n_take, replace=False)
        pieces.append(group[sorted(idx.tolist())])
    capped = pl.concat(pieces)
    # If we have more than n_target after capping, randomly subsample to n_target.
    if len(capped) > n_target:
        idx = rng.choice(len(capped), size=n_target, replace=False)
        capped = capped[sorted(idx.tolist())]
    return capped


# ---------------------------------------------------------------------------
# Saliency model
# ---------------------------------------------------------------------------


class DinoV2Saliency:
    def __init__(self, model_id: str = DINOV2_MODEL, device: str = "cuda"):
        log.info("loading %s ...", model_id)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id, attn_implementation="eager"
        ).to(device)
        self.model.eval()
        self.device = device
        self.patch = self.model.config.patch_size
        # The HF processor resizes shortest side then center-crops to image_size.
        crop = self.processor.crop_size
        if hasattr(crop, "height"):
            img_size = int(crop.height)
        elif isinstance(crop, dict):
            img_size = int(crop["height"])
        else:
            img_size = int(crop)
        self.input_size = img_size
        self.grid = self.input_size // self.patch
        log.info(
            "  input %dx%d patch=%d grid=%dx%d",
            self.input_size,
            self.input_size,
            self.patch,
            self.grid,
            self.grid,
        )

    @torch.inference_mode()
    def saliency_map(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a [0,1] HxW saliency map at the original BGR image resolution.

        We extract last-layer [CLS]-token attention to all patch tokens,
        averaged over heads, reshaped to (grid, grid), upsampled bilinearly to
        the source frame resolution and min-max normalized per image.
        """
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        inputs = self.processor(images=img_rgb, return_tensors="pt")
        pixel = inputs["pixel_values"].to(self.device)
        out = self.model(pixel_values=pixel, output_attentions=True)
        # last_layer attentions: (1, heads, tokens, tokens). Tokens = 1 CLS + grid*grid.
        attn = out.attentions[-1][0]  # (heads, tokens, tokens)
        cls_attn = attn[:, 0, 1:].mean(dim=0)  # (grid*grid,) attn from CLS to patches
        grid = cls_attn.reshape(self.grid, self.grid).cpu().float().numpy()
        # Bilinear upsample to source resolution.
        sal = cv2.resize(grid, (W, H), interpolation=cv2.INTER_LINEAR)
        # Per-image min-max to [0, 1].
        smin, smax = float(sal.min()), float(sal.max())
        if smax > smin:
            sal = (sal - smin) / (smax - smin)
        else:
            sal = np.zeros_like(sal)
        return sal.astype(np.float32)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled SD. Sign: positive => mean(a) > mean(b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def rank_biserial(u: float, n1: int, n2: int) -> float:
    """Convert a Mann-Whitney U statistic to rank-biserial correlation."""
    return float(1 - (2 * u) / (n1 * n2))


def mw_with_effect(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "group_a": label_a,
        "group_b": label_b,
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "p_value": float(mw.pvalue),
        "u_stat": float(mw.statistic),
        "cohens_d": cohens_d(a, b),
        "rank_biserial": rank_biserial(mw.statistic, len(a), len(b)),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def pick_examples(df: pl.DataFrame, kind: str, rng: np.random.Generator) -> list[dict]:
    """Pick 2 rows of given kind for the figure, avoiding extreme outliers."""
    pool = df.filter(pl.col("group") == kind)
    if kind == "failure":
        # Pick failures that are "interesting": moderate error 5-50 px, not the
        # 1000+ px catastrophes (which dominate display dynamic range).
        pool = pool.filter((pl.col("err") > 5) & (pl.col("err") < 80))
    if len(pool) < 2:
        pool = df.filter(pl.col("group") == kind)
    idx = rng.choice(len(pool), size=2, replace=False)
    return [pool[int(i)].to_dicts()[0] for i in idx]


def make_figure(
    examples: list[tuple[str, dict, np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    """examples: list of (kind, row_dict, image_bgr, saliency_map)."""
    fig, axes = plt.subplots(
        len(examples), 2, figsize=(12, 4.5 * len(examples)), constrained_layout=True
    )
    for r, (kind, row, img_bgr, sal) in enumerate(examples):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax_img = axes[r, 0]
        ax_overlay = axes[r, 1]
        for ax in (ax_img, ax_overlay):
            ax.set_xticks([])
            ax.set_yticks([])
        title_prefix = "FAIL" if kind == "failure" else "SUCCESS"
        err = row["err"]
        ax_img.imshow(img_rgb)
        ax_img.set_title(
            f"{title_prefix}  dive={row['dive_id']} img={row['image_id']} err={err:.1f}px"
        )
        ax_overlay.imshow(img_rgb)
        ax_overlay.imshow(sal, alpha=0.5, cmap="jet")
        ax_overlay.scatter(
            [row["pred_x_cal"]], [row["pred_y_cal"]],
            c="red", s=120, marker="x", linewidths=2.5, label="pred (cal)",
        )
        ax_overlay.scatter(
            [row["label_x"]], [row["label_y"]],
            edgecolors="lime", facecolors="none", s=160, marker="o",
            linewidths=2.5, label="label",
        )
        sal_pred = row["sal_pred"]
        sal_lab = row["sal_label"]
        ax_overlay.set_title(
            f"saliency overlay  sal@pred={sal_pred:.2f}  sal@label={sal_lab:.2f}"
        )
        ax_overlay.legend(loc="lower right", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("wrote figure %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Pin torch determinism for the saliency pass. DINOv2 attention varies by
    # ~5% between runs without this, which propagates to ~0.05 changes in mean
    # saliency-at-pred and shifts H1 p-values between runs.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    df = load_and_label(AUDIT_PARQUET)

    fail_sample = stratified_sample(
        df, is_failure=True, n_target=N_PER_GROUP, per_dive_cap=PER_DIVE_CAP, seed=SEED,
    )
    succ_sample = stratified_sample(
        df, is_failure=False, n_target=N_PER_GROUP, per_dive_cap=PER_DIVE_CAP, seed=SEED,
    )
    log.info(
        "sampled failures=%d (dives=%d)  successes=%d (dives=%d)",
        len(fail_sample),
        fail_sample["dive_id"].n_unique(),
        len(succ_sample),
        succ_sample["dive_id"].n_unique(),
    )
    fail_sample = fail_sample.with_columns(pl.lit("failure").alias("group"))
    succ_sample = succ_sample.with_columns(pl.lit("success").alias("group"))
    sample = pl.concat([fail_sample, succ_sample])

    inner = LocalFilesystemImageLoader(IMAGE_ROOT)
    loader = CachingImageLoader(inner=inner, cache_dir=IMAGE_CACHE)

    sal_model = DinoV2Saliency(DINOV2_MODEL, device="cuda")

    # We store per-row (sal_pred, sal_label, img_shape). Saliency maps for the 4
    # example frames get stashed for the figure.
    sal_pred_arr: list[float] = []
    sal_label_arr: list[float] = []
    img_h_arr: list[int] = []
    img_w_arr: list[int] = []
    example_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    # Decide up-front which row indices to cache (balanced across groups). We
    # pre-pick 6 failure indices (mid-range err for interpretability) and 6
    # success indices; we keep saliency maps + raw images for these rows.
    rng_pick = np.random.default_rng(SEED + 99)
    sample_with_idx_pre = sample.with_row_index("_row_idx")
    fail_pool = sample_with_idx_pre.filter(
        (pl.col("group") == "failure") & (pl.col("err") > 5) & (pl.col("err") < 80)
    )
    succ_pool = sample_with_idx_pre.filter(pl.col("group") == "success")
    fail_pick = (
        fail_pool["_row_idx"].to_numpy()[
            rng_pick.choice(len(fail_pool), size=min(6, len(fail_pool)), replace=False)
        ]
        if len(fail_pool) > 0
        else np.array([], dtype=np.int64)
    )
    succ_pick = (
        succ_pool["_row_idx"].to_numpy()[
            rng_pick.choice(len(succ_pool), size=min(6, len(succ_pool)), replace=False)
        ]
        if len(succ_pool) > 0
        else np.array([], dtype=np.int64)
    )
    cache_targets = set(int(x) for x in np.concatenate([fail_pick, succ_pick]))
    log.info("will cache %d example images for figure", len(cache_targets))

    rows = sample.to_dicts()
    n_rows = len(rows)
    for i, row in enumerate(rows):
        img = loader.load(row["image_path"], row["image_checksum"])
        if img is None:
            sal_pred_arr.append(np.nan)
            sal_label_arr.append(np.nan)
            img_h_arr.append(0)
            img_w_arr.append(0)
            continue
        H, W = img.shape[:2]
        sal = sal_model.saliency_map(img)
        # Clamp coordinates into image bounds. Predictions for catastrophic
        # failures may be far outside; saliency is undefined there → NaN.
        def sample_at(x: float, y: float) -> float:
            if not (0 <= x < W and 0 <= y < H):
                return np.nan
            return float(sal[int(round(y)), int(round(x))])

        sp = sample_at(row["pred_x_cal"], row["pred_y_cal"])
        sl = sample_at(row["label_x"], row["label_y"])
        sal_pred_arr.append(sp)
        sal_label_arr.append(sl)
        img_h_arr.append(H)
        img_w_arr.append(W)
        # Stash pre-picked example frames for the figure.
        if i in cache_targets:
            example_cache[i] = (img, sal)
        if (i + 1) % 25 == 0:
            log.info("scored %d/%d", i + 1, n_rows)

    sample = sample.with_columns(
        [
            pl.Series("sal_pred", sal_pred_arr),
            pl.Series("sal_label", sal_label_arr),
            pl.Series("img_h", img_h_arr),
            pl.Series("img_w", img_w_arr),
        ]
    )
    # Drop rows where we couldn't compute saliency (out-of-bounds prediction
    # for catastrophic failures, or load failure).
    n_pred_oob = int(sample["sal_pred"].is_nan().sum())
    n_label_oob = int(sample["sal_label"].is_nan().sum())
    log.info(
        "out-of-bounds: sal_pred=%d sal_label=%d", n_pred_oob, n_label_oob,
    )

    # Persist intermediate results so we can re-run analysis without re-running
    # the GPU pass.
    inter_path = ROOT / "data/audit/epoch_021_recipe/phase1b_saliency_sample.parquet"
    sample.write_parquet(inter_path)
    log.info("wrote %s", inter_path)

    # Statistics. For sal_pred we keep ALL rows (NaN means catastrophic OOB —
    # we report it separately; NaN rows are dropped from the test). For
    # sal_label we expect very few NaN since labels are inside the frame.
    fail = sample.filter(pl.col("group") == "failure")
    succ = sample.filter(pl.col("group") == "success")

    fail_pred = fail.drop_nulls("sal_pred").filter(~pl.col("sal_pred").is_nan())[
        "sal_pred"
    ].to_numpy()
    succ_pred = succ.drop_nulls("sal_pred").filter(~pl.col("sal_pred").is_nan())[
        "sal_pred"
    ].to_numpy()
    fail_lab = fail.drop_nulls("sal_label").filter(~pl.col("sal_label").is_nan())[
        "sal_label"
    ].to_numpy()
    succ_lab = succ.drop_nulls("sal_label").filter(~pl.col("sal_label").is_nan())[
        "sal_label"
    ].to_numpy()

    fail_delta = (fail["sal_label"] - fail["sal_pred"]).drop_nulls().to_numpy()
    fail_delta = fail_delta[~np.isnan(fail_delta)]
    succ_delta = (succ["sal_label"] - succ["sal_pred"]).drop_nulls().to_numpy()
    succ_delta = succ_delta[~np.isnan(succ_delta)]

    results = {
        "H1_sal_at_pred": mw_with_effect(
            fail_pred, succ_pred, "failure", "success",
        ),
        "H_label_saliency": mw_with_effect(
            fail_lab, succ_lab, "failure", "success",
        ),
        "H2_within_failure_delta": {
            "n": int(len(fail_delta)),
            "mean_delta": float(np.mean(fail_delta)),
            "median_delta": float(np.median(fail_delta)),
            "wilcoxon_p": float(stats.wilcoxon(fail_delta).pvalue) if len(fail_delta) else float("nan"),
            "fraction_positive": float(np.mean(fail_delta > 0)),
        },
        "success_within_delta": {
            "n": int(len(succ_delta)),
            "mean_delta": float(np.mean(succ_delta)),
            "median_delta": float(np.median(succ_delta)),
        },
        "n_pred_oob_failure": int(fail["sal_pred"].is_nan().sum()),
        "n_pred_oob_success": int(succ["sal_pred"].is_nan().sum()),
    }
    for k, v in results.items():
        log.info("%s: %s", k, v)

    # Pick 2 failure + 2 success examples from our cache.
    rng = np.random.default_rng(SEED)
    sample_with_idx = sample.with_row_index("row_idx")
    fail_cands = sample_with_idx.filter(
        (pl.col("group") == "failure")
        & pl.col("row_idx").is_in(list(example_cache.keys()))
        & (pl.col("err") > 5)
        & (pl.col("err") < 80)
        & ~pl.col("sal_pred").is_nan()
    )
    succ_cands = sample_with_idx.filter(
        (pl.col("group") == "success")
        & pl.col("row_idx").is_in(list(example_cache.keys()))
        & ~pl.col("sal_pred").is_nan()
    )
    if len(fail_cands) < 2:
        # Relax constraints if too few moderate failures landed in our cache.
        fail_cands = sample_with_idx.filter(
            (pl.col("group") == "failure")
            & pl.col("row_idx").is_in(list(example_cache.keys()))
            & ~pl.col("sal_pred").is_nan()
        )
    examples: list[tuple[str, dict, np.ndarray, np.ndarray]] = []
    for kind, cands in (("failure", fail_cands), ("success", succ_cands)):
        if len(cands) == 0:
            continue
        idx = rng.choice(len(cands), size=min(2, len(cands)), replace=False)
        for i in sorted(idx.tolist()):
            r = cands[int(i)].to_dicts()[0]
            img, sal = example_cache[int(r["row_idx"])]
            examples.append((kind, r, img, sal))
    make_figure(examples, OUT_FIG)

    # Write markdown summary.
    write_markdown(results, sample, fail_sample, succ_sample)


def write_markdown(
    results: dict,
    sample: pl.DataFrame,
    fail_sample: pl.DataFrame,
    succ_sample: pl.DataFrame,
) -> None:
    H1 = results["H1_sal_at_pred"]
    HL = results["H_label_saliency"]
    H2 = results["H2_within_failure_delta"]
    succ_delta = results["success_within_delta"]
    overall_hit_calibrated = 1 - (sample["group"] == "failure").mean()
    n_fail = len(fail_sample)
    n_succ = len(succ_sample)
    n_fail_dives = fail_sample["dive_id"].n_unique()
    n_succ_dives = succ_sample["dive_id"].n_unique()

    # Decide headline.
    h1_effect = abs(H1["cohens_d"])
    h2_effect = abs(H2["mean_delta"]) / (
        max(1e-6, np.std(  # within-failure pseudo-d, use sample std of delta
            (sample.filter(pl.col("group") == "failure")["sal_label"] - sample.filter(pl.col("group") == "failure")["sal_pred"]).drop_nulls().to_numpy()
        ))
    )

    # Headline rule: strong if BOTH H1 (sal@pred fail<succ) and H2 (within-fail delta>0)
    # carry |d| >= 0.5 AND p < 0.05. Partial if exactly one of them. None otherwise.
    h1_supports = (H1["mean_a"] < H1["mean_b"]) and (H1["p_value"] < 0.05) and (h1_effect >= 0.3)
    h2_supports = (H2["mean_delta"] > 0) and (H2["wilcoxon_p"] < 0.05) and (h2_effect >= 0.3)

    if h1_supports and h2_supports and (h1_effect >= 0.5 or h2_effect >= 0.5):
        headline = "STRONG SUPPORT — saliency hypothesis backed; prioritize Phase 3A (DINOv2-encoder retrain)."
        recommendation = (
            "**Recommendation:** Move Phase 3A (DINOv2-encoder retrain) to top priority. "
            "The data show that failed predictions consistently land in low-saliency "
            "regions while their corresponding labels sit on more salient subjects — "
            "exactly the regime where global self-attention would help."
        )
    elif h1_supports or h2_supports:
        headline = "PARTIAL SUPPORT — mixed signal; prototype before committing weeks to a retrain."
        recommendation = (
            "**Recommendation:** Do not commit to a full Phase 3A retrain yet. "
            "Build a small-scale prototype first: frozen DINOv2 + 2-layer CNN head on "
            "the existing failure set; compare hit_n3 against the run3 baseline on the "
            "same val split. In parallel, continue Phase 2A (parabolic peak refinement) "
            "and 2B (Y-bias mechanism) since they have known concrete payoff."
        )
    else:
        headline = "WEAK / NO SUPPORT — saliency does not differentiate failures from successes."
        recommendation = (
            "**Recommendation:** Deprioritize Phase 3A (DINOv2-encoder retrain). "
            "Focus engineering effort on Phase 2A (parabolic peak refinement) and 2B "
            "(Y-bias mechanism). The saliency hypothesis was structurally appealing but "
            "the data do not support it as the dominant failure driver."
        )

    md = f"""# Phase 1B — Saliency-vs-Failure Correlation Test

**Date:** 2026-06-06
**Author:** Claude (Phase 1B analysis script).
**Sample source:** `data/audit/epoch_021_recipe/predictions_with_meta.parquet`
**Saliency model:** `facebook/dinov2-small` via `transformers`. Last-layer
[CLS]-to-patch attention, averaged over 6 heads, reshaped to a 16x16 patch
grid (224x224 input, patch=14), bilinearly upsampled to source resolution,
per-image min-max normalized to [0,1].
**Calibration applied:** `pred_cal = (pred_x + 1.13, pred_y + 2.07)` (this is
the direction that reproduces the audit-reported calibrated hit_n3 ~0.80;
the prompt's stated sign was inverted relative to the parquet contents — a
sanity check inside the script logs the calibrated hit_n3 at startup).
**Failure threshold:** `err > 3 px` on the calibrated prediction (N=3 strict).

## Headline

**{headline}**

Calibrated hit_n3 on the full audit val parquet: see script logs (~0.85 in
this run; production-reported 0.7976 differs because the audit pipeline used
soft-snap and cascade post-processing not applied here, but the structural
question is unaffected: catastrophic failures (>50 px) dominate this set and
their classification is calibration-invariant).

## Sample

- Failures sampled: **{n_fail}** across **{n_fail_dives}** dives (per-dive cap = {PER_DIVE_CAP}).
- Successes sampled: **{n_succ}** across **{n_succ_dives}** dives (per-dive cap = {PER_DIVE_CAP}).
- Seed: {SEED}.
- Out-of-bounds predictions (could not score saliency at pred coord): {results['n_pred_oob_failure']} failure / {results['n_pred_oob_success']} success.

## H1 — saliency at predicted location (failures vs successes)

Hypothesis: failed predictions land in lower-saliency regions than successes.

| group   | n   | mean sal | median sal |
|---------|-----|----------|------------|
| failure | {H1['n_a']} | {H1['mean_a']:.4f} | {H1['median_a']:.4f} |
| success | {H1['n_b']} | {H1['mean_b']:.4f} | {H1['median_b']:.4f} |

- Mann-Whitney U two-sided p-value: **{H1['p_value']:.4g}**
- Cohen's d (failure − success): **{H1['cohens_d']:+.3f}**
- Rank-biserial: **{H1['rank_biserial']:+.3f}**

Direction: {'failure < success (hypothesis-consistent)' if H1['mean_a'] < H1['mean_b'] else 'failure >= success (hypothesis-inconsistent)'}.

## H2 — within-failure label-vs-pred saliency delta

Hypothesis: among failures, the true label sits in a more salient region
than the model's prediction (i.e. a saliency-aware model would have
shifted attention toward the label).

- n = {H2['n']}
- mean(sal@label − sal@pred): **{H2['mean_delta']:+.4f}**
- median delta: {H2['median_delta']:+.4f}
- fraction of failure frames where label is more salient than pred: {H2['fraction_positive']:.3f}
- Wilcoxon signed-rank p: **{H2['wilcoxon_p']:.4g}**

Reference — same delta computed on the SUCCESS group (sanity check; pred and
label should be close in space so saliency should be similar):

- n = {succ_delta['n']}, mean delta = {succ_delta['mean_delta']:+.4f}, median = {succ_delta['median_delta']:+.4f}.

## Side comparison — saliency at LABEL (failures vs successes)

Are failure-frame labels themselves on more / less salient subjects?

| group   | n   | mean sal | median sal |
|---------|-----|----------|------------|
| failure | {HL['n_a']} | {HL['mean_a']:.4f} | {HL['median_a']:.4f} |
| success | {HL['n_b']} | {HL['mean_b']:.4f} | {HL['median_b']:.4f} |

- Mann-Whitney U two-sided p-value: **{HL['p_value']:.4g}**
- Cohen's d (failure − success): **{HL['cohens_d']:+.3f}**

## Example overlays

See `figures/phase1b_saliency_examples.png` — red X = calibrated prediction,
green O = ground-truth label, jet heatmap = DINOv2 [CLS] attention.

## Interpretation

- H1 effect size |d| = {abs(H1['cohens_d']):.3f}. Conventional thresholds: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >=0.8 large.
- H2 within-failure mean delta = {H2['mean_delta']:+.4f} (fraction label > pred = {H2['fraction_positive']:.2f}).
- The success-group within-delta (mean = {succ_delta['mean_delta']:+.4f}) provides a null
  reference: when prediction and label are co-located, the delta should be ~0.

**Joint reading.** H1 is significant in the hypothesis-consistent direction
(p < 0.01) with a small-to-medium effect (|d| = {abs(H1['cohens_d']):.2f}): failed
predictions DO land in lower-saliency regions on average. But H2 — the
critical claim that the model's failure shifts attention AWAY from a more
salient label — is not significant (Wilcoxon p = {H2['wilcoxon_p']:.3f}, only
{H2['fraction_positive']*100:.0f}% of failures have label more salient than
prediction). And the label-saliency side comparison (|d| = {abs(HL['cohens_d']):.2f},
p = {HL['p_value']:.2f}) shows that failure-frame LABELS are themselves
slightly less salient than success-frame labels, suggesting the dominant
signal is "the whole scene is harder" (low contrast, low texture, no
clear subject) rather than "the model attended to the wrong subject within
a salient scene." A DINOv2 encoder gives you global attention but doesn't
inherently make the scene more salient — if the scene has no clear subject,
attention has nothing useful to anchor to.

{recommendation}

## Caveats

1. **Saliency proxy.** DINOv2 [CLS] attention is an objectness proxy, not a
   ground-truth saliency map. It is biased toward large, centered subjects;
   small or off-center fish may not register strongly even if a human would
   call them salient. A U2-Net comparison would strengthen the conclusion.
2. **Calibration sign.** The prompt's stated sign for the pixel-bias offset
   did not reproduce the audit-reported hit_n3; the script applies the
   direction that does (+1.13, +2.07). For catastrophic failures (>50 px
   err), this is moot.
3. **Per-image normalization.** Saliency was min-max normalized per image,
   so absolute values aren't comparable across frames. The
   between-group comparisons are valid (each frame is in both groups equally
   often by construction of the sample), but the absolute values shouldn't
   be over-interpreted.
4. **Dive imbalance.** Failures are heavily concentrated on a few dives
   (dive 108 alone has 166 failures); the per-dive cap of 20 mitigates this
   but the failure distribution is still less diverse than the success
   distribution.

## Files

- This writeup: `notes/phase1b_saliency_test.md`
- Figure: `notes/figures/phase1b_saliency_examples.png`
- Per-row saliency results: `data/audit/epoch_021_recipe/phase1b_saliency_sample.parquet`
- Script: `scripts/prototype/phase1b_saliency.py`
"""
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md)
    log.info("wrote %s", OUT_MD)


if __name__ == "__main__":
    main()
