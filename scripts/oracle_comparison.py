#!/usr/bin/env python3
"""Compare: model predictions vs oracle (actual points) vs no-transfer baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fpl_optimizer.data.downloader import DEFAULT_DATA_DIR
from fpl_optimizer.data.loader import SeasonDataLoader
from fpl_optimizer.engine.engine import FPLGameEngine
from fpl_optimizer.engine.state import ChipState, EngineAction, GameState, PlayerSlot, Squad
from fpl_optimizer.optimizer.squad_selection import select_squad
from fpl_optimizer.optimizer.transfer_optimizer import optimize_transfers
from fpl_optimizer.optimizer.types import build_candidate_pool, to_engine_action
from fpl_optimizer.prediction.integration import PredictionIntegrator
from fpl_optimizer.utils.constants import GW1_FREE_TRANSFERS, STARTING_BUDGET, Position


def run_optimizer_season(loader, engine, predicted_points_fn, label, max_xfers=5):
    """Run MILP optimizer through a full season with given prediction function."""
    num_gws = min(loader.get_num_gameweeks(), 38)

    # GW1: initial squad selection
    pp1 = predicted_points_fn(1)
    candidates = build_candidate_pool(loader, 1, pp1)
    result = select_squad(candidates, budget=STARTING_BUDGET)

    # Build initial state
    players = []
    for eid in result.squad_element_ids:
        pos = loader.get_player_position(eid)
        price = loader.get_player_price(eid, 1)
        if pos and price > 0:
            players.append(PlayerSlot(element_id=eid, position=pos, purchase_price=price, selling_price=price))

    eid_to_idx = {p.element_id: i for i, p in enumerate(players)}
    lineup = [eid_to_idx[eid] for eid in result.lineup_element_ids if eid in eid_to_idx]
    bench = [eid_to_idx[eid] for eid in result.bench_element_ids if eid in eid_to_idx]
    captain_idx = eid_to_idx.get(result.captain_id, 0)
    vice_idx = eid_to_idx.get(result.vice_captain_id, 1)

    squad = Squad(players=players, lineup=lineup, bench=bench, captain_idx=captain_idx, vice_captain_idx=vice_idx)
    state = GameState(squad=squad, bank=STARTING_BUDGET - result.total_cost,
                      free_transfers=GW1_FREE_TRANSFERS, chips=ChipState(), current_gw=1, total_points=0)

    gw_points_list = []

    for gw in range(1, num_gws + 1):
        pp = predicted_points_fn(gw)
        candidates = build_candidate_pool(loader, gw, pp)

        if not candidates:
            # No-op
            action = EngineAction()
        else:
            try:
                opt_result = optimize_transfers(state, candidates, max_transfers=max_xfers)
                action = to_engine_action(opt_result)
            except RuntimeError:
                action = EngineAction()

        try:
            new_state, result = engine.step(state, action)
        except ValueError:
            action = EngineAction()
            new_state, result = engine.step(state, action)

        gw_points_list.append(result.gw_points)
        state = new_state

    total = state.total_points
    gross = sum(gw_points_list)
    print(f"\n{label}:")
    print(f"  Total (net):  {total}")
    print(f"  Gross:        {gross}")
    print(f"  Hits:         {gross - total}")
    print(f"  Avg GW gross: {np.mean(gw_points_list):.1f}")
    print(f"  Top 5 GWs:    {sorted(gw_points_list, reverse=True)[:5]}")
    print(f"  Bottom 5 GWs: {sorted(gw_points_list)[:5]}")
    return total, gross, gw_points_list


def main():
    season = "2024-25"
    data_dir = DEFAULT_DATA_DIR
    pred_data_dir = data_dir.parent if data_dir.name == "raw" else data_dir

    loader = SeasonDataLoader(season, data_dir)
    engine = FPLGameEngine(loader)

    # 1. Oracle mode: optimizer sees ACTUAL points
    print("=== Running Oracle (perfect foresight) ===")
    oracle_total, oracle_gross, _ = run_optimizer_season(
        loader, engine,
        lambda gw: None,  # None = use actual total_points in build_candidate_pool
        "ORACLE (actual points)", max_xfers=5
    )

    # 2. Prediction mode: optimizer sees model predictions
    print("\n=== Running Prediction Model ===")
    integrator = PredictionIntegrator.from_model(
        Path("models/point_predictor"), pred_data_dir, season,
    )

    def pred_fn(gw):
        all_eids = loader.get_all_element_ids(gw)
        return {eid: integrator.get_predicted_points(eid, gw) for eid in all_eids}

    pred_total, pred_gross, _ = run_optimizer_season(
        loader, engine, pred_fn, "PREDICTIONS (LightGBM)", max_xfers=5
    )

    # 3. No-transfer baseline: just pick best GW1 squad, never transfer
    print("\n=== Running No-Transfer Baseline ===")
    no_xfer_total, no_xfer_gross, _ = run_optimizer_season(
        loader, engine, pred_fn, "NO TRANSFERS (GW1 squad only)", max_xfers=0
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Oracle (perfect knowledge):     {oracle_total:>6} net ({oracle_gross} gross)")
    print(f"LightGBM predictions, 5 xfers:  {pred_total:>6} net ({pred_gross} gross)")
    print(f"No transfers (pred GW1 squad):  {no_xfer_total:>6} net ({no_xfer_gross} gross)")
    print(f"\nPrediction efficiency: {pred_total/oracle_total*100:.1f}% of oracle")
    print(f"Transfer value: +{pred_total - no_xfer_total} points from transfers")
    print(f"\nFor context:")
    print(f"  Best human 2024-25: ~2810")
    print(f"  Best human ever:    ~2844")


if __name__ == "__main__":
    main()
