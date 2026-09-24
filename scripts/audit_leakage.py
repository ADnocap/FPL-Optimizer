#!/usr/bin/env python3
"""Audit the prediction model for lookahead bias.

Checks:
1. Are predicted points suspiciously correlated with actual points?
2. Does the model predict haulers accurately?
3. Compare captain picks vs what random/naive would achieve.
4. Check if transfer-ins are always the top scorer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fpl_optimizer.data.downloader import DEFAULT_DATA_DIR
from fpl_optimizer.data.loader import SeasonDataLoader
from fpl_optimizer.prediction.integration import PredictionIntegrator

SEASON = "2024-25"


def main():
    data_dir = DEFAULT_DATA_DIR
    pred_data_dir = data_dir.parent if data_dir.name == "raw" else data_dir

    print(f"Loading {SEASON} data...")
    loader = SeasonDataLoader(SEASON, data_dir)

    # Load BOTH models
    print("Loading prediction models...")
    v1_integrator = PredictionIntegrator.from_model(
        Path("models/point_predictor"), pred_data_dir, SEASON
    )
    v2_integrator = PredictionIntegrator.from_model(
        Path("models/point_predictor_v2"), pred_data_dir, SEASON
    )

    # Build comparison: predicted vs actual for every (player, gw)
    print("\n=== PREDICTION ACCURACY ANALYSIS ===\n")

    rows = []
    for gw in range(1, 39):
        eids = loader.get_all_element_ids(gw)
        for eid in eids:
            data = loader.get_player_gw(eid, gw)
            if data is None:
                continue
            actual = int(data["total_points"])
            v1_pred = v1_integrator.get_predicted_points(eid, gw)
            v2_pred = v2_integrator.get_predicted_points(eid, gw)
            pos = loader.get_player_position(eid)
            rows.append({
                "gw": gw, "eid": eid, "actual": actual,
                "v1_pred": v1_pred, "v2_pred": v2_pred,
                "pos": pos.name if pos else "?",
            })

    df = pd.DataFrame(rows)

    # Overall correlation
    v1_corr = df["actual"].corr(df["v1_pred"])
    v2_corr = df["actual"].corr(df["v2_pred"])
    v1_mae = (df["actual"] - df["v1_pred"]).abs().mean()
    v2_mae = (df["actual"] - df["v2_pred"]).abs().mean()

    print(f"V1 (original) — Corr: {v1_corr:.4f}, MAE: {v1_mae:.4f}")
    print(f"V2 (retrained) — Corr: {v2_corr:.4f}, MAE: {v2_mae:.4f}")
    print(f"V1 vs V2 pred corr: {df['v1_pred'].corr(df['v2_pred']):.4f}")

    # Check per-GW correlation
    print(f"\n=== PER-GW CORRELATION (V1 vs actual) ===\n")
    print(f"{'GW':>4} {'Corr':>7} {'MAE':>7} {'N':>5}")
    gw_corrs = []
    for gw in range(1, 39):
        gw_df = df[df["gw"] == gw]
        corr = gw_df["actual"].corr(gw_df["v1_pred"])
        mae = (gw_df["actual"] - gw_df["v1_pred"]).abs().mean()
        gw_corrs.append(corr)
        print(f"{gw:4d} {corr:7.3f} {mae:7.3f} {len(gw_df):5d}")

    print(f"\nAvg per-GW corr: {np.mean(gw_corrs):.4f}")
    print(f"Median per-GW corr: {np.median(gw_corrs):.4f}")

    # Check haulers: when actual > 10, how well did the model predict?
    print(f"\n=== HAULER DETECTION (actual >= 10 pts) ===\n")
    haulers = df[df["actual"] >= 10]
    non_haulers = df[df["actual"] < 10]
    print(f"Haulers: {len(haulers)} rows")
    print(f"  V1 mean pred for haulers: {haulers['v1_pred'].mean():.2f}")
    print(f"  V1 mean pred for non-haulers: {non_haulers['v1_pred'].mean():.2f}")
    print(f"  Actual mean for haulers: {haulers['actual'].mean():.2f}")
    print(f"  V1 correctly ranks haulers > non-haulers on average: "
          f"{'YES' if haulers['v1_pred'].mean() > non_haulers['v1_pred'].mean() else 'NO'}")

    # Big haulers (>= 15)
    big_haulers = df[df["actual"] >= 15]
    print(f"\nBig haulers (>= 15 pts): {len(big_haulers)} rows")
    print(f"  V1 mean pred: {big_haulers['v1_pred'].mean():.2f} (actual: {big_haulers['actual'].mean():.2f})")

    # Top predicted vs top actual per GW
    print(f"\n=== TOP PREDICTED vs TOP ACTUAL PER GW ===\n")
    print(f"{'GW':>4} {'Top Pred Name':>25} {'Pred':>6} {'Actual':>6} | {'Top Actual Name':>25} {'Actual':>6} {'Pred':>6}")

    # Build name map
    name_map = {}
    for _, row in loader._merged_gw[["element", "name"]].drop_duplicates("element").iterrows():
        name_map[int(row["element"])] = str(row["name"])

    top_pred_is_top_actual = 0
    top_pred_in_top5_actual = 0

    for gw in range(1, 39):
        gw_df = df[df["gw"] == gw].copy()

        # Top predicted
        top_pred_row = gw_df.loc[gw_df["v1_pred"].idxmax()]
        top_pred_eid = int(top_pred_row["eid"])
        top_pred_name = name_map.get(top_pred_eid, str(top_pred_eid))[:25]

        # Top actual
        top_actual_row = gw_df.loc[gw_df["actual"].idxmax()]
        top_actual_eid = int(top_actual_row["eid"])
        top_actual_name = name_map.get(top_actual_eid, str(top_actual_eid))[:25]

        # Top 5 actual
        top5_actual = set(gw_df.nlargest(5, "actual")["eid"].astype(int))

        if top_pred_eid == top_actual_eid:
            top_pred_is_top_actual += 1
        if top_pred_eid in top5_actual:
            top_pred_in_top5_actual += 1

        print(f"{gw:4d} {top_pred_name:>25} {top_pred_row['v1_pred']:6.1f} {top_pred_row['actual']:6.0f} | "
              f"{top_actual_name:>25} {top_actual_row['actual']:6.0f} {top_actual_row['v1_pred']:6.1f}")

    print(f"\nTop predicted = top actual: {top_pred_is_top_actual}/38 ({top_pred_is_top_actual/38*100:.0f}%)")
    print(f"Top predicted in top 5 actual: {top_pred_in_top5_actual}/38 ({top_pred_in_top5_actual/38*100:.0f}%)")

    # Distribution of predictions
    print(f"\n=== PREDICTION DISTRIBUTION ===\n")
    print(f"V1 predictions: mean={df['v1_pred'].mean():.2f}, std={df['v1_pred'].std():.2f}, "
          f"min={df['v1_pred'].min():.2f}, max={df['v1_pred'].max():.2f}")
    print(f"Actual points:  mean={df['actual'].mean():.2f}, std={df['actual'].std():.2f}, "
          f"min={df['actual'].min():.2f}, max={df['actual'].max():.2f}")

    # Check for suspiciously tight predictions
    print(f"\n=== RESIDUAL ANALYSIS ===\n")
    residuals = df["actual"] - df["v1_pred"]
    print(f"Residuals: mean={residuals.mean():.3f}, std={residuals.std():.3f}")
    print(f"  |residual| < 1: {(residuals.abs() < 1).sum()} ({(residuals.abs() < 1).mean()*100:.1f}%)")
    print(f"  |residual| < 2: {(residuals.abs() < 2).sum()} ({(residuals.abs() < 2).mean()*100:.1f}%)")
    print(f"  |residual| < 3: {(residuals.abs() < 3).sum()} ({(residuals.abs() < 3).mean()*100:.1f}%)")

    # Smoking gun check: rank correlation within each GW
    print(f"\n=== RANK CORRELATION (Spearman) PER GW ===\n")
    rank_corrs = []
    for gw in range(1, 39):
        gw_df = df[df["gw"] == gw]
        spearman = gw_df["actual"].corr(gw_df["v1_pred"], method="spearman")
        rank_corrs.append(spearman)
    print(f"Avg Spearman rank correlation: {np.mean(rank_corrs):.4f}")
    print(f"A truly clean model should get ~0.2-0.4 Spearman per GW")
    print(f"An oracle (leaking) would get ~0.8-1.0 Spearman per GW")

    # Compare v1 vs v2
    print(f"\n=== V1 vs V2 COMPARISON ===\n")
    print("If V1 was trained on 2024-25 and V2 was not, V1 should be much better:")
    v1_rank_corrs = []
    v2_rank_corrs = []
    for gw in range(1, 39):
        gw_df = df[df["gw"] == gw]
        v1_rank_corrs.append(gw_df["actual"].corr(gw_df["v1_pred"], method="spearman"))
        v2_rank_corrs.append(gw_df["actual"].corr(gw_df["v2_pred"], method="spearman"))

    print(f"V1 avg Spearman: {np.mean(v1_rank_corrs):.4f}")
    print(f"V2 avg Spearman: {np.mean(v2_rank_corrs):.4f}")
    print(f"Difference: {np.mean(v1_rank_corrs) - np.mean(v2_rank_corrs):.4f}")
    if abs(np.mean(v1_rank_corrs) - np.mean(v2_rank_corrs)) < 0.02:
        print("=> Models are very similar — V1 likely NOT trained on 2024-25")
    else:
        print("=> SIGNIFICANT DIFFERENCE — V1 may have been trained on 2024-25!")


if __name__ == "__main__":
    main()
