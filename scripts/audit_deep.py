#!/usr/bin/env python3
"""Deep audit: what's driving the high prediction correlation?

Checks:
1. Spearman among starters only (mins > 0) vs all players
2. fpl_xp alone as a baseline
3. Feature importance of the model
4. teams.csv snapshot leakage check
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
from fpl_optimizer.prediction.model import PointPredictor

SEASON = "2024-25"


def main():
    data_dir = DEFAULT_DATA_DIR
    pred_data_dir = data_dir.parent if data_dir.name == "raw" else data_dir

    loader = SeasonDataLoader(SEASON, data_dir)

    print("Loading prediction model...")
    integrator = PredictionIntegrator.from_model(
        Path("models/point_predictor"), pred_data_dir, SEASON
    )

    # Build full dataset
    rows = []
    for gw in range(1, 39):
        eids = loader.get_all_element_ids(gw)
        for eid in eids:
            data = loader.get_player_gw(eid, gw)
            if data is None:
                continue
            actual = int(data["total_points"])
            minutes = int(data.get("minutes", 0))
            pred = integrator.get_predicted_points(eid, gw)

            # Get fpl_xp if available
            xp = data.get("xP", None)
            if xp is not None:
                try:
                    xp = float(xp)
                except (ValueError, TypeError):
                    xp = None

            rows.append({
                "gw": gw, "eid": eid, "actual": actual,
                "pred": pred, "minutes": minutes, "fpl_xp": xp,
            })

    df = pd.DataFrame(rows)

    # ================================================================
    # 1. WHAT'S DRIVING THE HIGH SPEARMAN?
    # ================================================================
    print("\n" + "=" * 60)
    print("1. SPEARMAN BREAKDOWN: ALL vs STARTERS ONLY")
    print("=" * 60)

    all_spearman = []
    starters_spearman = []
    xp_spearman_all = []
    xp_spearman_starters = []

    for gw in range(1, 39):
        gw_df = df[df["gw"] == gw]
        gw_starters = gw_df[gw_df["minutes"] > 0]

        all_sp = gw_df["actual"].corr(gw_df["pred"], method="spearman")
        all_spearman.append(all_sp)

        if len(gw_starters) > 10:
            st_sp = gw_starters["actual"].corr(gw_starters["pred"], method="spearman")
            starters_spearman.append(st_sp)

        # fpl_xp baseline
        gw_xp = gw_df.dropna(subset=["fpl_xp"])
        if len(gw_xp) > 10:
            xp_sp = gw_xp["actual"].corr(gw_xp["fpl_xp"], method="spearman")
            xp_spearman_all.append(xp_sp)

            gw_xp_st = gw_xp[gw_xp["minutes"] > 0]
            if len(gw_xp_st) > 10:
                xp_sp_st = gw_xp_st["actual"].corr(gw_xp_st["fpl_xp"], method="spearman")
                xp_spearman_starters.append(xp_sp_st)

    print(f"\nModel Spearman (all players):      {np.mean(all_spearman):.4f}")
    print(f"Model Spearman (starters only):    {np.mean(starters_spearman):.4f}")
    print(f"  => Drop when removing non-starters: {np.mean(all_spearman) - np.mean(starters_spearman):.4f}")
    print(f"\nfpl_xp Spearman (all players):     {np.mean(xp_spearman_all):.4f}")
    print(f"fpl_xp Spearman (starters only):   {np.mean(xp_spearman_starters):.4f}")

    pct_non_starters = (df["minutes"] == 0).mean()
    print(f"\n% non-starters in dataset: {pct_non_starters*100:.1f}%")
    print(f"(These are easy to predict as 0 pts, inflating Spearman)")

    # ================================================================
    # 2. FEATURE IMPORTANCE
    # ================================================================
    print("\n" + "=" * 60)
    print("2. FEATURE IMPORTANCE (top 20)")
    print("=" * 60)

    predictor = PointPredictor.load(Path("models/point_predictor"))
    fi = predictor.feature_importance()
    print(f"\n{'Rank':>4} {'Feature':<35} {'Importance':>12}")
    print("-" * 55)
    for i, (_, row) in enumerate(fi.head(20).iterrows(), 1):
        print(f"{i:4d} {row['feature']:<35} {row['importance']:12.0f}")

    # ================================================================
    # 3. CAPTAIN SIMULATION: what would random/naive achieve?
    # ================================================================
    print("\n" + "=" * 60)
    print("3. CAPTAIN HIT RATE SIMULATION")
    print("=" * 60)

    # For each GW, take the top 11 predicted players as our "squad"
    # Check how often the top-predicted is the actual top scorer
    captain_hit = 0
    captain_top3 = 0
    random_hit_rates = []

    np.random.seed(42)
    for gw in range(1, 39):
        gw_df = df[df["gw"] == gw].copy()
        # Get top 11 by prediction (simplified squad)
        top11 = gw_df.nlargest(11, "pred")
        if len(top11) < 11:
            continue

        captain_eid = top11.iloc[0]["eid"]  # highest predicted
        actual_top_eid = top11.loc[top11["actual"].idxmax(), "eid"]

        if captain_eid == actual_top_eid:
            captain_hit += 1

        # Top-3 actual
        top3_actual = set(top11.nlargest(3, "actual")["eid"])
        if captain_eid in top3_actual:
            captain_top3 += 1

        # Monte Carlo: random captain from the 11
        hits = 0
        for _ in range(10000):
            rand_idx = np.random.randint(0, len(top11))
            rand_eid = top11.iloc[rand_idx]["eid"]
            if rand_eid == actual_top_eid:
                hits += 1
        random_hit_rates.append(hits / 10000)

    print(f"\nTop-predicted captain = top scorer in squad: {captain_hit}/38 ({captain_hit/38*100:.0f}%)")
    print(f"Top-predicted captain in top-3 actual:       {captain_top3}/38 ({captain_top3/38*100:.0f}%)")
    print(f"Random captain = top scorer (Monte Carlo):   {np.mean(random_hit_rates)*100:.1f}%")
    print(f"\nExpected random rate for 11 players: ~9.1%")
    print(f"(The model captain rate should be somewhere between random and oracle)")

    # ================================================================
    # 4. PREDICTION vs ACTUAL FOR SPECIFIC SUSPICIOUS PICKS
    # ================================================================
    print("\n" + "=" * 60)
    print("4. SUSPICIOUS CAPTAIN PICKS - DETAILED")
    print("=" * 60)

    name_map = {}
    for _, row in loader._merged_gw[["element", "name"]].drop_duplicates("element").iterrows():
        name_map[int(row["element"])] = str(row["name"])

    # These were the suspicious picks from the report
    suspicious = [
        (5, "Jack Hinshelwood"),
        (10, "Ola Aina"),
        (12, "Aaron Wan-Bissaka"),
        (13, "Kevin Schade"),
        (23, "Dango Ouattara"),
        (26, "Marco Asensio"),
        (28, "Marc Cucurella"),
        (30, "Sandro Tonali"),
        (31, "Jacob Murphy"),
        (36, "Morgan Gibbs-White"),
    ]

    for gw, name_hint in suspicious:
        gw_df = df[df["gw"] == gw].copy()
        # Find the player
        match = gw_df[gw_df["eid"].isin(
            [eid for eid, n in name_map.items() if name_hint.lower() in n.lower()]
        )]
        if match.empty:
            print(f"\nGW{gw}: {name_hint} — not found")
            continue

        player = match.iloc[0]
        # Also show top 5 predicted and top 5 actual in that GW
        top5_pred = gw_df.nlargest(5, "pred")
        top5_actual = gw_df.nlargest(5, "actual")

        rank_pred = int((gw_df["pred"] >= player["pred"]).sum())
        rank_actual = int((gw_df["actual"] >= player["actual"]).sum())
        n_players = len(gw_df)

        print(f"\nGW{gw}: {name_hint}")
        print(f"  Predicted: {player['pred']:.1f} (rank {rank_pred}/{n_players})")
        print(f"  Actual:    {player['actual']:.0f} (rank {rank_actual}/{n_players})")
        print(f"  Minutes:   {player['minutes']:.0f}")
        print(f"  fpl_xp:    {player['fpl_xp']}" if pd.notna(player['fpl_xp']) else "  fpl_xp:    N/A")
        print(f"  Top 5 predicted: ", end="")
        for _, r in top5_pred.iterrows():
            n = name_map.get(int(r['eid']), '?')[:15].encode('ascii', 'replace').decode()
            print(f"{n}({r['pred']:.1f}->{r['actual']:.0f}) ", end="")
        print()


if __name__ == "__main__":
    main()
