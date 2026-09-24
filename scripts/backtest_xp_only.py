"""Backtest 2024-25 using only raw FPL xP as predictions."""

import logging
import pandas as pd
from pathlib import Path

from fpl_optimizer.data.loader import SeasonDataLoader
from fpl_optimizer.optimizer.backtest import SeasonBacktester

logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR = Path("data/raw")
SEASON = "2024-25"


def main():
    loader = SeasonDataLoader(SEASON, DATA_DIR)

    # Extract raw xP for every (element_id, gw) pair
    merged = pd.read_csv(DATA_DIR / SEASON / "gws" / "merged_gw.csv", encoding="utf-8")
    merged["xP"] = pd.to_numeric(merged["xP"], errors="coerce").fillna(0.0)

    # For DGWs: sum xP across fixtures (same as backtest does for total_points)
    xp_agg = merged.groupby(["element", "GW"])["xP"].sum().reset_index()

    all_preds: dict[tuple[int, int], float] = {}
    for _, row in xp_agg.iterrows():
        all_preds[(int(row["element"]), int(row["GW"]))] = float(row["xP"])

    print(f"Loaded {len(all_preds)} xP predictions for {SEASON}")
    print(f"Gameweeks: 1-{loader.get_num_gameweeks()}")

    # Monkey-patch the backtester to use our xP predictions directly
    bt = SeasonBacktester(loader)

    # Override _load_predictions to return our xP dict
    bt._load_predictions = lambda season: all_preds
    bt.model_dir = Path("dummy")  # trigger prediction mode

    result = bt.run(season=SEASON)

    print("\n" + "=" * 60)
    print(f"SEASON: {result.season}")
    print(f"TOTAL POINTS: {result.total_points}")
    print(f"TRANSFERS MADE: {result.transfers_made}")
    print(f"HIT COST: {result.hits_taken}")
    print(f"CHIPS USED: {result.chips_used}")
    print("=" * 60)

    print("\nPer-GW breakdown:")
    print(f"{'GW':>4} {'Gross':>6} {'Hit':>4} {'Net':>6} {'Captain':>8} {'In':>15} {'Out':>15}")
    for gw in result.gw_results:
        ins = ",".join(str(x) for x in gw.transfers_in) if gw.transfers_in else "-"
        outs = ",".join(str(x) for x in gw.transfers_out) if gw.transfers_out else "-"
        print(f"{gw.gw:>4} {gw.gross_points:>6} {gw.hit_cost:>4} {gw.net_points:>6} {gw.captain_id:>8} {ins:>15} {outs:>15}")


if __name__ == "__main__":
    main()
