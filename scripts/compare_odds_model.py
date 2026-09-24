#!/usr/bin/env python3
"""Compare prediction models with and without odds features.

Trains two LightGBM models using temporal cross-validation:
  1. Baseline: all features EXCEPT odds
  2. Odds-enhanced: all features INCLUDING odds

Reports MAE/RMSE improvement overall and per position, plus
odds feature importance ranking.

Usage:
    python scripts/compare_odds_model.py
    python scripts/compare_odds_model.py --data-dir data --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fpl_optimizer.prediction.evaluation import TemporalCV, ALL_SEASONS, HOLDOUT_SEASON
from fpl_optimizer.prediction.feature_pipeline import FeaturePipeline
from fpl_optimizer.prediction.features.odds import FEATURE_COLS as ODDS_FEATURE_COLS
from fpl_optimizer.prediction.id_resolver import IDResolver
from fpl_optimizer.prediction.model import PointPredictor, POSITIONS

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare prediction models with/without odds features."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory (default: data/)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def build_features(data_dir: Path) -> pd.DataFrame:
    """Build the full feature DataFrame."""
    seasons = [s for s in ALL_SEASONS if s != HOLDOUT_SEASON]
    resolver = IDResolver(data_dir)
    pipeline = FeaturePipeline(data_dir=data_dir, id_resolver=resolver, seasons=seasons)
    return pipeline.build()


def strip_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with odds feature columns dropped."""
    cols_to_drop = [c for c in ODDS_FEATURE_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def run_cv(df: pd.DataFrame, label: str) -> dict:
    """Run temporal CV and return results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    cv = TemporalCV()
    results = cv.evaluate(df)

    print(f"\n  Overall MAE:  {results['mae']:.4f}")
    print(f"  Overall RMSE: {results['rmse']:.4f}")
    print(f"\n  Per-position MAE:")
    for pos in POSITIONS:
        mae = results["per_position_mae"].get(pos, float("nan"))
        print(f"    {pos:4s}: {mae:.4f}")

    print(f"\n  Per-fold results:")
    for fold in results["fold_results"]:
        print(
            f"    Fold {fold['fold']} ({fold['test_season']}): "
            f"MAE={fold['mae']:.4f}, RMSE={fold['rmse']:.4f}, "
            f"n={fold['n_test']}"
        )

    return results


def print_comparison(baseline: dict, enhanced: dict) -> None:
    """Print side-by-side comparison."""
    print(f"\n{'='*60}")
    print("  COMPARISON: Baseline vs Odds-Enhanced")
    print(f"{'='*60}")

    mae_diff = enhanced["mae"] - baseline["mae"]
    rmse_diff = enhanced["rmse"] - baseline["rmse"]
    mae_pct = 100.0 * mae_diff / baseline["mae"] if baseline["mae"] > 0 else 0

    print(f"\n  Overall MAE:  {baseline['mae']:.4f} -> {enhanced['mae']:.4f}  "
          f"({mae_diff:+.4f}, {mae_pct:+.2f}%)")
    print(f"  Overall RMSE: {baseline['rmse']:.4f} -> {enhanced['rmse']:.4f}  "
          f"({rmse_diff:+.4f})")

    print(f"\n  Per-position MAE change:")
    for pos in POSITIONS:
        base_mae = baseline["per_position_mae"].get(pos, float("nan"))
        enh_mae = enhanced["per_position_mae"].get(pos, float("nan"))
        diff = enh_mae - base_mae
        print(f"    {pos:4s}: {base_mae:.4f} -> {enh_mae:.4f}  ({diff:+.4f})")

    # Per-fold comparison (only folds that have odds data: 2020-21+)
    print(f"\n  Per-fold MAE change:")
    for b_fold, e_fold in zip(baseline["fold_results"], enhanced["fold_results"]):
        diff = e_fold["mae"] - b_fold["mae"]
        has_odds = b_fold["test_season"] >= "2020-21"
        marker = " [has odds]" if has_odds else " [no odds]"
        print(
            f"    {b_fold['test_season']}: {b_fold['mae']:.4f} -> "
            f"{e_fold['mae']:.4f}  ({diff:+.4f}){marker}"
        )


def print_odds_feature_importance(df: pd.DataFrame) -> None:
    """Train a single model on all data and show odds feature importance."""
    print(f"\n{'='*60}")
    print("  ODDS FEATURE IMPORTANCE (trained on all CV data)")
    print(f"{'='*60}")

    # Use a simple train/val split from the last two seasons
    train_df = df[df["season"] < "2023-24"]
    val_df = df[df["season"] == "2023-24"]

    if train_df.empty or val_df.empty:
        print("  Not enough data to compute importance")
        return

    predictor = PointPredictor()
    predictor.train(train_df, val_df)

    importance = predictor.feature_importance()
    if importance.empty:
        print("  No importance data available")
        return

    # Show top 20 features
    print(f"\n  Top 20 features by importance (gain):")
    for _, row in importance.head(20).iterrows():
        is_odds = row["feature"] in ODDS_FEATURE_COLS
        marker = " *** ODDS ***" if is_odds else ""
        print(f"    {row['feature']:35s} {row['importance']:10.1f}{marker}")

    # Show just odds features
    odds_rows = importance[importance["feature"].isin(ODDS_FEATURE_COLS)]
    if not odds_rows.empty:
        total_imp = importance["importance"].sum()
        odds_imp = odds_rows["importance"].sum()
        print(f"\n  Odds features total importance: {odds_imp:.1f} / {total_imp:.1f} "
              f"({100*odds_imp/total_imp:.1f}%)")
        for _, row in odds_rows.iterrows():
            rank = (importance["feature"] == row["feature"]).values.argmax() + 1
            print(f"    {row['feature']:35s} rank #{rank} ({row['importance']:.1f})")


def main() -> None:
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = Path(args.data_dir)

    print("Building feature pipeline (this may take a few minutes)...")
    full_df = build_features(data_dir)
    print(f"Feature DataFrame: {full_df.shape[0]} rows x {full_df.shape[1]} columns")

    # Check if odds features are present
    odds_cols_present = [c for c in ODDS_FEATURE_COLS if c in full_df.columns]
    if not odds_cols_present:
        print("\nERROR: No odds features found in DataFrame.")
        print("Run `python scripts/collect_odds.py` first to collect odds data.")
        sys.exit(1)

    n_with_odds = full_df[odds_cols_present[0]].notna().sum()
    print(f"Rows with odds data: {n_with_odds}/{len(full_df)} "
          f"({100*n_with_odds/len(full_df):.1f}%)")

    # Run CV for both models
    baseline_df = strip_odds_features(full_df)
    baseline_results = run_cv(baseline_df, "BASELINE (no odds features)")

    enhanced_results = run_cv(full_df, "ODDS-ENHANCED (with odds features)")

    # Print comparison
    print_comparison(baseline_results, enhanced_results)

    # Show feature importance
    print_odds_feature_importance(full_df)

    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
