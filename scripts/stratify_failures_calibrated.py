"""Failure-mode stratification for a calibrated audit dir.

Copy of stratify_failures.py with an argparse front so we can point at any
audit directory (e.g. data/audit/epoch_021_recipe_calibrated{,_test}).

Outputs land under <audit-dir>/stratification/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

HIT_TOL = 3.0


def add_err(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        ((pl.col("pred_x") - pl.col("label_x")) ** 2
         + (pl.col("pred_y") - pl.col("label_y")) ** 2).sqrt().alias("err")
    )


def summarize_axis(pos: pl.DataFrame, by: list[str], name: str, out_dir: Path) -> pl.DataFrame:
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
    g.write_parquet(out_dir / f"by_{name}.parquet")
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", type=Path, required=True,
                    help="Directory containing predictions_with_meta.parquet")
    args = ap.parse_args()

    root = args.audit_dir
    src = root / "predictions_with_meta.parquet"
    out = root / "stratification"
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(src)
    print(f"loaded {df.shape[0]} rows from {src}")

    pos = df.filter(pl.col("is_positive"))
    n_pos = pos.height
    print(f"positives: {n_pos} (of {df.height})")

    pos = pos.with_columns(
        no_pred=pl.col("pred_x").is_null() | pl.col("pred_y").is_null(),
    )
    pos = add_err(pos)
    pos = pos.with_columns(
        err=pl.when(pl.col("no_pred")).then(float("inf")).otherwise(pl.col("err")),
    )
    pos = pos.with_columns(fail=pl.col("err") > HIT_TOL)

    n_fail = int(pos["fail"].sum())
    print(f"failures (err>{HIT_TOL}): {n_fail} / {n_pos} = {n_fail / n_pos:.3f}")
    print(f"no-prediction frames among positives: {int(pos['no_pred'].sum())}")

    summarize_axis(pos, ["wavelength"], "wavelength", out)
    summarize_axis(pos, ["rig_id"], "rig_id", out)
    summarize_axis(pos, ["rig_id", "wavelength"], "rig_id_x_wavelength", out)

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
    summarize_axis(pos, ["line_q"], "line_q", out)
    summarize_axis(pos, ["wavelength", "line_q"], "wavelength_x_line_q", out)

    hits = pos.filter(~pl.col("fail") & ~pl.col("no_pred"))
    rig_prior = (
        hits.group_by(["rig_id", "wavelength"])
        .agg(
            prior_x=pl.col("pred_x").median(),
            prior_y=pl.col("pred_y").median(),
            n_prior=pl.len(),
        )
    )
    rig_prior.write_parquet(out / "rig_priors.parquet")

    pos = pos.join(rig_prior.select(["rig_id", "wavelength", "prior_x", "prior_y"]),
                   on=["rig_id", "wavelength"], how="left")
    pos = pos.with_columns(
        dist_from_prior=((pl.col("label_x") - pl.col("prior_x")) ** 2
                        + (pl.col("label_y") - pl.col("prior_y")) ** 2).sqrt(),
    )
    dp = pos["dist_from_prior"].drop_nulls().to_numpy()
    edges_dp = np.quantile(dp, [0.25, 0.5, 0.75])
    pos = pos.with_columns(
        prior_q=pl.when(pl.col("dist_from_prior").is_null()).then(pl.lit("none"))
        .when(pl.col("dist_from_prior") <= edges_dp[0]).then(pl.lit("near_q1"))
        .when(pl.col("dist_from_prior") <= edges_dp[1]).then(pl.lit("q2"))
        .when(pl.col("dist_from_prior") <= edges_dp[2]).then(pl.lit("q3"))
        .otherwise(pl.lit("far_q4"))
        .alias("prior_q"),
    )
    summarize_axis(pos, ["prior_q"], "prior_distance", out)

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
    summarize_axis(pos, ["fail_class"], "fail_class", out)
    summarize_axis(pos, ["wavelength", "fail_class"], "wavelength_x_fail_class", out)

    # NEW: borderline vs catastrophic split (Phase 1A question B/E).
    pos = pos.with_columns(
        err_class=pl.when(~pl.col("fail")).then(pl.lit("hit"))
        .when(pl.col("err") <= 10.0).then(pl.lit("borderline_3to10"))
        .when(pl.col("err") <= 50.0).then(pl.lit("mid_10to50"))
        .when(pl.col("err").is_finite()).then(pl.lit("catastrophic_gt50"))
        .otherwise(pl.lit("no_pred"))
        .alias("err_class"),
    )
    summarize_axis(pos, ["err_class"], "err_class", out)
    summarize_axis(pos, ["wavelength", "err_class"], "wavelength_x_err_class", out)

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
    per_dive.write_parquet(out / "per_dive_failures.parquet")
    print("\nper-dive failure totals (sorted, top 15):")
    print(per_dive.head(15))

    cum = per_dive["n_fail"].to_numpy().cumsum()
    total = n_fail
    print("\nConcentration (cum failures captured by top-K dives):")
    for k in (1, 5, 10, 20, 50, per_dive.height):
        if k <= per_dive.height:
            print(f"  top {k:>3} of {per_dive.height} dives -> {cum[k - 1]} / {total} = {cum[k - 1] / total:.3f}")

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

    bias_rows = []
    for wl in correct["wavelength"].unique().to_list():
        sub = correct.filter(pl.col("wavelength") == wl)
        sx = (sub["pred_x"] - sub["label_x"]).to_numpy()
        sy = (sub["pred_y"] - sub["label_y"]).to_numpy()
        bias_rows.append({"wavelength": wl, "n": len(sx),
                          "mean_dx": float(sx.mean()), "mean_dy": float(sy.mean()),
                          "std_dx": float(sx.std()), "std_dy": float(sy.std())})
    pl.DataFrame(bias_rows).write_parquet(out / "residual_bias_by_wavelength.parquet")

    pos.write_parquet(out / "per_frame_enriched.parquet")
    print(f"\nwrote outputs to {out}")


if __name__ == "__main__":
    main()
