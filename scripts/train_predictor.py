#!/usr/bin/env python3
"""Train the production LightGBM point predictor for the live season.

Two phases (features built once):
1. EVAL   — train on 2016-17..2024-25, holdout 2025-26 (the first DEFCON/
            BPS-overhaul season) to measure honest out-of-sample quality.
2. PROD   — retrain on ALL seasons through 2025-26 (val = last 8 GWs of
            2025-26 for early stopping) and save to models/prod_2026-27.

This is the committed, reproducible recipe for the model of record
(replaces the ad-hoc runs that produced models/no_xp and models/full_pregame).
Includes the full pre-game feature set with fpl_xp — see EP_FORMULA.md for
why xP is point-in-time safe (and it is snapshotted live pre-deadline by
fpl_optimizer.data.collectors.fpl_live).

Usage:
    python scripts/train_predictor.py [--out models/prod_2026-27] [--no-eval]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent

EVAL_TRAIN_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]
EVAL_HOLDOUT = "2025-26"
PROD_SEASONS = EVAL_TRAIN_SEASONS + [EVAL_HOLDOUT]

PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 127,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
    "n_estimators": 2000,
    "verbose": -1,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
}


def _split_val(df: pd.DataFrame, seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/val split: val = last 8 GWs of the last season in *seasons*."""
    subset = df[df["season"].isin(seasons)].copy()
    last = seasons[-1]
    last_rows = subset[subset["season"] == last]
    max_gw = int(last_rows["GW"].max())
    val_mask = (subset["season"] == last) & (subset["GW"] > max_gw - 8)
    return subset[~val_mask].copy(), subset[val_mask].copy()


def _report(preds: np.ndarray, holdout: pd.DataFrame, label: str) -> dict:
    actual = holdout["target"].values
    valid = ~np.isnan(actual)
    mae = float(np.mean(np.abs(preds[valid] - actual[valid])))
    rmse = float(np.sqrt(np.mean((preds[valid] - actual[valid]) ** 2)))
    corr = float(np.corrcoef(preds[valid], actual[valid])[0, 1])
    print(f"\n=== {label} ===")
    print(f"  MAE: {mae:.4f}  RMSE: {rmse:.4f}  Corr: {corr:.4f}")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        m = (holdout["position"] == pos).values & valid
        if m.sum():
            print(f"    {pos}: MAE={np.mean(np.abs(preds[m] - actual[m])):.4f} "
                  f"(n={m.sum()})")
    hd = holdout.copy()
    hd["pred"] = preds
    gw_corrs = [
        g["pred"].corr(g["target"])
        for _, g in hd.groupby("GW")
        if len(g) > 20
    ]
    mean_gw_corr = float(np.nanmean(gw_corrs))
    print(f"  Mean per-GW corr: {mean_gw_corr:.4f}")

    out = {"mae": mae, "rmse": rmse, "corr": corr, "gw_corr": mean_gw_corr}
    # Outcome-stratified breakdown (Zeros/Blanks/Tickers/Haulers)
    try:
        from fpl_optimizer.prediction.stratified_metrics import (
            format_report, stratified_metrics,
        )

        strat = stratified_metrics(actual[valid], preds[valid])
        print(format_report(strat, label=f"stratified: {label}"))
        out["stratified"] = strat
    except Exception as exc:  # metrics module optional — never break training
        print(f"  (stratified metrics unavailable: {exc})")
    return out


def main() -> None:
    from fpl_optimizer.prediction.feature_pipeline import FeaturePipeline
    from fpl_optimizer.prediction.id_resolver import IDResolver
    from fpl_optimizer.prediction.model import PointPredictor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "prod_2026-27")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--no-eval", action="store_true",
                        help="skip the 2025-26 holdout evaluation phase")
    args = parser.parse_args()

    print(f"Building features for {len(PROD_SEASONS)} seasons...")
    t0 = time.time()
    resolver = IDResolver(args.data_dir)
    df = FeaturePipeline(args.data_dir, resolver, PROD_SEASONS).build()
    print(f"Features: {len(df)} rows x {len(df.columns)} cols "
          f"in {time.time() - t0:.0f}s")

    # Drop synthetic/unplayed gameweeks: fpl_live's live rebuild writes
    # upcoming-GW rows with all stats zeroed. A real GW always has at least
    # one player with points (appearance minimum), so an all-zero-target
    # (season, GW) group can only be synthetic.
    gw_max_target = df.groupby(["season", "GW"])["target"].transform("max")
    synthetic = ~(gw_max_target > 0)  # catches all-zero AND all-NaN groups
    if synthetic.any():
        dropped = df.loc[synthetic, ["season", "GW"]].drop_duplicates()
        print(f"Dropping {int(synthetic.sum())} synthetic/unplayed rows: "
              + ", ".join(f"{s} GW{int(g)}" for s, g in dropped.itertuples(index=False)))
        df = df[~synthetic]

    metrics: dict = {}

    if not args.no_eval:
        train_df, val_df = _split_val(df, EVAL_TRAIN_SEASONS)
        holdout = df[df["season"] == EVAL_HOLDOUT].copy()
        print(f"\nEVAL phase: train={len(train_df)} val={len(val_df)} "
              f"holdout({EVAL_HOLDOUT})={len(holdout)}")
        eval_model = PointPredictor(params=PARAMS, early_stopping_rounds=50)
        eval_model.train(train_df, val_df)
        preds = eval_model.predict(holdout)
        metrics["eval_2025-26"] = _report(preds, holdout, f"HOLDOUT {EVAL_HOLDOUT}")

    print(f"\nPROD phase: training on all {len(PROD_SEASONS)} seasons...")
    train_df, val_df = _split_val(df, PROD_SEASONS)
    print(f"  train={len(train_df)} val={len(val_df)}")
    prod = PointPredictor(params=PARAMS, early_stopping_rounds=50)
    prod.train(train_df, val_df)

    # Atomic promote: train into a staging dir, keep the previous model as
    # a .prev rollback, then rename — readers (gameweek.py) never see a torn
    # model dir, and the old model+report survive for comparison.
    staging = args.out.with_name(args.out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    prod.save(staging)
    metrics["train_seasons"] = PROD_SEASONS
    metrics["params"] = PARAMS
    (staging / "training_report.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8"
    )
    meta_path = staging / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["train_seasons"] = PROD_SEASONS
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    prev = args.out.with_name(args.out.name + ".prev")
    if args.out.exists():
        if prev.exists():
            shutil.rmtree(prev)
        args.out.rename(prev)
    staging.rename(args.out)
    print(f"\nProduction model saved to {args.out} "
          f"(previous model kept at {prev.name})")

    fi = prod.feature_importance()
    if not fi.empty:
        print("\nTop 15 features:")
        for i, row in fi.head(15).iterrows():
            print(f"  #{i + 1:>2} {row['feature']:<35} {row['importance']:.0f}")


if __name__ == "__main__":
    main()
