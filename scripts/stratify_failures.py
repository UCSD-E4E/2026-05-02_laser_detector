"""Failure-mode stratification for the epoch_021_recipe audit.

Reads predictions_with_meta.parquet, computes pixel error, defines failure as
err > 3 px, and breaks the failure population down along the axes called out in
the diagnostic plan: wavelength, rig_id, line_confidence quartile, distance
from per-wavelength rig-prior centroid, presence-vs-localization, and per-dive
concentration. Also fits a residual distribution on the correct predictions to
estimate the label-noise floor.

Outputs land under data/audit/epoch_021_recipe/stratification/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path("data/audit/epoch_021_recipe")
SRC = ROOT / "predictions_with_meta.parquet"
OUT = ROOT / "stratification"
PLOTS = OUT / "plots"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

HIT_TOL = 3.0


def add_err(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        ((pl.col("pred_x") - pl.col("label_x")) ** 2
         + (pl.col("pred_y") - pl.col("label_y")) ** 2).sqrt().alias("err")
    )


def summarize_axis(pos: pl.DataFrame, by: list[str], name: str) -> pl.DataFrame:
    g = (
        pos.group_by(by)
        .agg(
            n=pl.len(),
            n_fail=(pl.col("fail").cast(pl.Int64)).sum(),
            hit_rate_n3=(~pl.col("fail")).cast(pl.Float64).mean(),
            median_err=pl.col("err").median(),
            mean_err=pl.col("err").mean(),
        )
        .with_columns(failure_rate=pl.col("n_fail") / pl.col("n"))
        .sort(by)
    )
    total_fail = int(pos["fail"].sum())
    g = g.with_columns(share_of_failures=pl.col("n_fail") / max(total_fail, 1))
    print(f"\n=== {name} ===")
    print(g)
    g.write_parquet(OUT / f"by_{name}.parquet")
    return g


def main() -> None:
    df = pl.read_parquet(SRC)
    print(f"loaded {df.shape[0]} rows, columns: {list(df.columns)}")

    # Positives only: failure is undefined for frames with no label.
    pos = df.filter(pl.col("is_positive"))
    n_pos = pos.height
    print(f"positives: {n_pos} (of {df.height})")

    # A frame has no detection if pred_x/pred_y are null. Treat those as a
    # localization failure with infinite error so they always count as misses.
    pos = pos.with_columns(
        no_pred=pl.col("pred_x").is_null() | pl.col("pred_y").is_null(),
    )
    pos = add_err(pos)
    # Replace null errors (no-pred frames) with +inf so the comparison still works.
    pos = pos.with_columns(
        err=pl.when(pl.col("no_pred")).then(float("inf")).otherwise(pl.col("err")),
    )
    pos = pos.with_columns(fail=pl.col("err") > HIT_TOL)

    n_fail = int(pos["fail"].sum())
    print(f"failures (err>{HIT_TOL}): {n_fail} / {n_pos} = {n_fail / n_pos:.3f}")
    print(f"no-prediction frames among positives: {int(pos['no_pred'].sum())}")

    # --- Axis 1: wavelength ---
    summarize_axis(pos, ["wavelength"], "wavelength")

    # --- Axis 2: rig_id ---
    summarize_axis(pos, ["rig_id"], "rig_id")
    summarize_axis(pos, ["rig_id", "wavelength"], "rig_id_x_wavelength")

    # --- Axis 3: line_confidence quartiles ---
    lc = pos["line_confidence"].drop_nulls().to_numpy()
    edges = np.quantile(lc, [0.25, 0.5, 0.75])
    print(f"line_confidence quartile edges: {edges}")
    pos = pos.with_columns(
        line_q=pl.when(pl.col("line_confidence").is_null()).then(pl.lit("none"))
        .when(pl.col("line_confidence") <= edges[0]).then(pl.lit("q1"))
        .when(pl.col("line_confidence") <= edges[1]).then(pl.lit("q2"))
        .when(pl.col("line_confidence") <= edges[2]).then(pl.lit("q3"))
        .otherwise(pl.lit("q4"))
        .alias("line_q"),
    )
    summarize_axis(pos, ["line_q"], "line_q")
    summarize_axis(pos, ["wavelength", "line_q"], "wavelength_x_line_q")

    # --- Axis 4: distance from rig-prior center ---
    # The "rig prior" is the per-(rig_id, wavelength) center where the laser
    # tends to land. Use the median pred location among hits (err<=3) so the
    # prior isn't contaminated by failures. Fall back to the median over all
    # frames for cells with too few hits.
    hits = pos.filter(~pl.col("fail") & ~pl.col("no_pred"))
    rig_prior = (
        hits.group_by(["rig_id", "wavelength"])
        .agg(
            prior_x=pl.col("pred_x").median(),
            prior_y=pl.col("pred_y").median(),
            n_prior=pl.len(),
        )
    )
    print("\nrig priors (from hits):")
    print(rig_prior)
    rig_prior.write_parquet(OUT / "rig_priors.parquet")

    pos = pos.join(rig_prior.select(["rig_id", "wavelength", "prior_x", "prior_y"]),
                   on=["rig_id", "wavelength"], how="left")
    # Distance from the rig prior — use the *label* position so we're asking
    # "is the laser in an unusual spot for this rig+wavelength?".
    pos = pos.with_columns(
        dist_from_prior=((pl.col("label_x") - pl.col("prior_x")) ** 2
                        + (pl.col("label_y") - pl.col("prior_y")) ** 2).sqrt(),
    )
    dp = pos["dist_from_prior"].drop_nulls().to_numpy()
    edges_dp = np.quantile(dp, [0.25, 0.5, 0.75])
    print(f"dist_from_prior quartile edges (label-vs-prior): {edges_dp}")
    pos = pos.with_columns(
        prior_q=pl.when(pl.col("dist_from_prior").is_null()).then(pl.lit("none"))
        .when(pl.col("dist_from_prior") <= edges_dp[0]).then(pl.lit("near_q1"))
        .when(pl.col("dist_from_prior") <= edges_dp[1]).then(pl.lit("q2"))
        .when(pl.col("dist_from_prior") <= edges_dp[2]).then(pl.lit("q3"))
        .otherwise(pl.lit("far_q4"))
        .alias("prior_q"),
    )
    summarize_axis(pos, ["prior_q"], "prior_distance")

    # --- Axis 5: presence vs. localization ---
    # `pred_confidence` is ~binary (mostly 1.0). Treat <1.0 as "model unsure"
    # and use no_pred as the strict presence-failure indicator.
    pos = pos.with_columns(
        low_conf=pl.col("pred_confidence") < 1.0,
    )
    pos = pos.with_columns(
        fail_class=pl.when(~pl.col("fail")).then(pl.lit("hit"))
        .when(pl.col("no_pred")).then(pl.lit("presence_nopred"))
        .when(pl.col("low_conf")).then(pl.lit("presence_lowconf"))
        .otherwise(pl.lit("localization"))
        .alias("fail_class"),
    )
    summarize_axis(pos, ["fail_class"], "fail_class")
    summarize_axis(pos, ["wavelength", "fail_class"], "wavelength_x_fail_class")

    # --- Axis 6: per-dive concentration ---
    per_dive = (
        pos.group_by("dive_id")
        .agg(
            wavelength=pl.col("wavelength").first(),
            n=pl.len(),
            n_fail=pl.col("fail").cast(pl.Int64).sum(),
            mean_err=pl.col("err").filter(pl.col("err").is_finite()).mean(),
        )
        .with_columns(failure_rate=pl.col("n_fail") / pl.col("n"))
        .sort("n_fail", descending=True)
    )
    per_dive.write_parquet(OUT / "per_dive_failures.parquet")
    print("\nper-dive failure totals (sorted, top 15):")
    print(per_dive.head(15))

    cum = per_dive["n_fail"].to_numpy().cumsum()
    total = n_fail
    print("\nConcentration (cum failures captured by top-K dives):")
    for k in (1, 5, 10, 20, 50, per_dive.height):
        if k <= per_dive.height:
            print(f"  top {k:>3} of {per_dive.height} dives -> {cum[k - 1]} / {total} = {cum[k - 1] / total:.3f}")

    # --- Residual distribution on correct predictions ---
    correct = pos.filter(~pl.col("fail") & ~pl.col("no_pred"))
    rx = (correct["pred_x"] - correct["label_x"]).to_numpy()
    ry = (correct["pred_y"] - correct["label_y"]).to_numpy()
    rerr = np.sqrt(rx ** 2 + ry ** 2)
    print("\nResiduals on HITS only (err<=3):")
    print(f"  n={len(rx)}")
    print(f"  mean dx={rx.mean():+.3f} px, dy={ry.mean():+.3f} px  (bias)")
    print(f"  median dx={np.median(rx):+.3f} px, dy={np.median(ry):+.3f} px")
    print(f"  std   dx={rx.std():.3f} px, dy={ry.std():.3f} px")
    print(f"  err quantiles: 25%={np.quantile(rerr, 0.25):.3f}  50%={np.quantile(rerr, 0.5):.3f}  "
          f"75%={np.quantile(rerr, 0.75):.3f}  95%={np.quantile(rerr, 0.95):.3f}")

    # Per-wavelength residual bias.
    print("\nResidual bias by wavelength (hits only):")
    bias_rows = []
    for wl in correct["wavelength"].unique().to_list():
        sub = correct.filter(pl.col("wavelength") == wl)
        sx = (sub["pred_x"] - sub["label_x"]).to_numpy()
        sy = (sub["pred_y"] - sub["label_y"]).to_numpy()
        print(f"  {wl}: n={len(sx)}  mean dx={sx.mean():+.3f}  mean dy={sy.mean():+.3f}  "
              f"std dx={sx.std():.3f}  std dy={sy.std():.3f}")
        bias_rows.append({"wavelength": wl, "n": len(sx),
                          "mean_dx": float(sx.mean()), "mean_dy": float(sy.mean()),
                          "std_dx": float(sx.std()), "std_dy": float(sy.std())})
    pl.DataFrame(bias_rows).write_parquet(OUT / "residual_bias_by_wavelength.parquet")

    # ---- Plots ----
    # Residual scatter on hits.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(rx, ry, s=4, alpha=0.35)
    axes[0].axhline(0, color="k", linewidth=0.5)
    axes[0].axvline(0, color="k", linewidth=0.5)
    axes[0].set_xlim(-3.5, 3.5)
    axes[0].set_ylim(-3.5, 3.5)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("pred_x - label_x  [px]")
    axes[0].set_ylabel("pred_y - label_y  [px]")
    axes[0].set_title(f"Residuals on HITS (n={len(rx)})")
    axes[0].grid(alpha=0.3)
    axes[1].hist(rerr, bins=np.linspace(0, 3, 31), edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel("|residual| [px]")
    axes[1].set_ylabel("count")
    axes[1].set_title("Hit-error magnitude (label-noise floor)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "residuals_on_hits.png", dpi=120)
    plt.close(fig)

    # Per-dive Pareto.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(1, per_dive.height + 1)
    share = cum / total
    ax.plot(x, share, marker="o", linewidth=1.2)
    for k in (5, 10, 20):
        if k <= per_dive.height:
            ax.axvline(k, color="gray", linestyle="--", alpha=0.5)
            ax.text(k + 0.2, 0.05, f"top {k} -> {share[k - 1]:.2f}", fontsize=8)
    ax.set_xlabel("dive rank (sorted by # failures)")
    ax.set_ylabel("cumulative share of all failures")
    ax.set_title("Per-dive concentration of failures")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "per_dive_pareto.png", dpi=120)
    plt.close(fig)

    # Failure rate by wavelength x line_q heatmap.
    wl_lq = (
        pos.group_by(["wavelength", "line_q"])
        .agg(n=pl.len(), fr=pl.col("fail").cast(pl.Float64).mean())
        .sort(["wavelength", "line_q"])
    )
    print("\nwavelength x line_q failure rate:")
    print(wl_lq)
    wl_lq.write_parquet(OUT / "wavelength_x_line_q_failure_rate.parquet")

    # Save the enriched per-frame frame for follow-ups.
    pos.write_parquet(OUT / "per_frame_enriched.parquet")
    print(f"\nwrote outputs to {OUT}")


if __name__ == "__main__":
    main()
