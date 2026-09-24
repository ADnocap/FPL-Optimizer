"""Verify that the feature pipeline data fixes work on real data.

Run from project root:
    python scripts/verify_data_fixes.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25",
]


def main() -> None:
    from fpl_optimizer.prediction.id_resolver import IDResolver
    from fpl_optimizer.prediction.feature_pipeline import FeaturePipeline

    print("=" * 70)
    print("VERIFYING DATA FIXES")
    print("=" * 70)

    id_resolver = IDResolver(DATA_DIR)
    pipeline = FeaturePipeline(DATA_DIR, id_resolver, SEASONS)

    print("\nBuilding features (this takes a few minutes)...")
    df = pipeline.build()

    print(f"\nResult: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Seasons: {sorted(df['season'].unique())}")

    # ---- Check 1: Position coverage ----
    print("\n" + "=" * 70)
    print("FIX 1: Position coverage")
    print("=" * 70)
    pos_na = df["position"].isna().sum()
    pos_total = len(df)
    print(f"  NaN positions: {pos_na:,} / {pos_total:,} ({pos_na/pos_total*100:.1f}%)")

    for season in SEASONS:
        sdf = df[df["season"] == season]
        na = sdf["position"].isna().sum()
        pct = na / len(sdf) * 100 if len(sdf) > 0 else 0
        print(f"  {season}: {len(sdf):,} rows, {na:,} NaN position ({pct:.1f}%)")

    pos_counts = df["position"].value_counts(dropna=False)
    print(f"\n  Position distribution:\n{pos_counts.to_string()}")

    # ---- Check 2: FBref features ----
    print("\n" + "=" * 70)
    print("FIX 2: FBref prior-season features")
    print("=" * 70)
    fbref_cols = [
        "prev_sot_per90", "prev_pass_cmp_pct", "prev_prog_dist_per90",
        "prev_tkl_int_per90", "prev_blocks_per90", "prev_gls_per90",
    ]
    for col in fbref_cols:
        if col in df.columns:
            na_pct = df[col].isna().mean() * 100
            non_na = df[col].notna().sum()
            print(f"  {col}: {non_na:,} non-NaN ({100-na_pct:.1f}% coverage)")
        else:
            print(f"  {col}: MISSING COLUMN")

    # ---- Check 3: Opponent rolling features ----
    print("\n" + "=" * 70)
    print("FIX 3: Opponent rolling features")
    print("=" * 70)
    opp_cols = ["opp_goals_conceded_r5", "opp_pts_conceded_r5"]
    for col in opp_cols:
        if col in df.columns:
            na_pct = df[col].isna().mean() * 100
            non_na = df[col].notna().sum()
            print(f"  {col}: {non_na:,} non-NaN ({100-na_pct:.1f}% coverage)")

            # Per season
            for season in SEASONS:
                sdf = df[df["season"] == season]
                s_na_pct = sdf[col].isna().mean() * 100 if len(sdf) > 0 else 100
                print(f"    {season}: {100-s_na_pct:.1f}% coverage")
        else:
            print(f"  {col}: MISSING COLUMN")

    # ---- Overall NaN summary ----
    print("\n" + "=" * 70)
    print("OVERALL: Feature NaN rates (top 15)")
    print("=" * 70)
    feature_cols = [
        c for c in df.columns
        if c not in {"code", "element", "season", "GW", "position", "target", "total_points"}
    ]
    nan_pct = (df[feature_cols].isna().mean() * 100).sort_values(ascending=False)
    print(nan_pct.head(15).round(1).to_string())

    # ---- After dropping NaN position: how many rows remain? ----
    print("\n" + "=" * 70)
    print("SUMMARY: Usable data after position/target filter")
    print("=" * 70)
    usable = df.dropna(subset=["position", "target"])
    print(f"  Before: {len(df):,} rows")
    print(f"  After:  {len(usable):,} rows")
    print(f"  Retained: {len(usable)/len(df)*100:.1f}%")
    print(f"  Positions: {dict(usable['position'].value_counts())}")

    # Compare with old numbers
    print("\n  BEFORE fixes: 126,737 usable rows (from 215,087)")
    print(f"  AFTER  fixes: {len(usable):,} usable rows (from {len(df):,})")
    improvement = len(usable) - 126_737
    print(f"  Improvement:  +{improvement:,} rows ({improvement/126737*100:.1f}%)")


if __name__ == "__main__":
    main()
