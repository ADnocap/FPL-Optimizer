#!/usr/bin/env python3
"""Honest full-season backtest of the live setup (model + MILP + engine).

Trains an evaluation model ONLY on seasons before the target season (no
leakage), predicts every (player, GW) of the target season, then replays the
season GW-by-GW: initial squad MILP at GW1, weekly transfer MILP, and the
full rules engine (auto-subs, captain failover, FT banking, hits).

No chips are played (upside not counted) and no availability filtering is
possible historically — both make this an underestimate of careful live play.

Usage:
    python scripts/backtest_season.py --season 2025-26 [--max-transfers 1]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent

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

ALL_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]


def main() -> None:
    from fpl_optimizer.data.loader import SeasonDataLoader
    from fpl_optimizer.engine.engine import FPLGameEngine
    from fpl_optimizer.engine.state import EngineAction, GameState, PlayerSlot, Squad
    from fpl_optimizer.optimizer.squad_selection import select_squad
    from fpl_optimizer.optimizer.transfer_optimizer import optimize_transfers
    from fpl_optimizer.optimizer.types import build_candidate_pool, to_engine_action
    from fpl_optimizer.prediction.feature_pipeline import FeaturePipeline
    from fpl_optimizer.prediction.id_resolver import IDResolver
    from fpl_optimizer.prediction.model import PointPredictor
    from fpl_optimizer.utils.constants import GW1_FREE_TRANSFERS, STARTING_BUDGET

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--max-transfers", type=int, default=None,
                        help="cap weekly transfers (None = optimizer decides, hits allowed)")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    target = args.season
    train_seasons = ALL_SEASONS[: ALL_SEASONS.index(target)]

    print(f"=== Backtest {target} | train on {train_seasons[0]}..{train_seasons[-1]} "
          f"| max_transfers={args.max_transfers} ===")

    t0 = time.time()
    resolver = IDResolver(args.data_dir)
    df = FeaturePipeline(args.data_dir, resolver, train_seasons + [target]).build()
    print(f"Features: {len(df)} rows in {time.time() - t0:.0f}s")

    # Drop synthetic/unplayed GWs (same guard as train_predictor)
    gw_max_target = df.groupby(["season", "GW"])["target"].transform("max")
    df = df[gw_max_target > 0]

    # Train the eval model strictly on pre-target seasons
    train_full = df[df["season"].isin(train_seasons)]
    last = train_seasons[-1]
    max_gw = int(train_full[train_full["season"] == last]["GW"].max())
    val_mask = (train_full["season"] == last) & (train_full["GW"] > max_gw - 8)
    predictor = PointPredictor(params=PARAMS, early_stopping_rounds=50)
    predictor.train(train_full[~val_mask].copy(), train_full[val_mask].copy())

    # Predict the target season
    target_df = df[df["season"] == target]
    preds = predictor.predict(target_df)
    predictions: dict[tuple[int, int], float] = {}
    for pred, (_, row) in zip(preds, target_df.iterrows()):
        eid = resolver.element_id_from_code(int(row["code"]), target)
        if eid is not None:
            key = (eid, int(row["GW"]))
            predictions[key] = predictions.get(key, 0.0) + float(pred)
    print(f"Predictions: {len(predictions)} (element, GW) pairs")

    # Replay the season
    loader = SeasonDataLoader(target, args.data_dir / "raw")
    engine = FPLGameEngine(loader)
    gws = sorted({int(g) for _, g in predictions})

    def pred_for_gw(gw: int) -> dict[int, float]:
        eids = loader.get_all_element_ids(gw)
        return {eid: predictions.get((eid, gw), 0.0) for eid in eids}

    first_gw = gws[0]
    init = select_squad(
        build_candidate_pool(loader, first_gw, pred_for_gw(first_gw)),
        budget=STARTING_BUDGET,
    )
    players = []
    for eid in init.squad_element_ids:
        pos = loader.get_player_position(eid)
        price = loader.get_player_price(eid, first_gw)
        players.append(PlayerSlot(eid, pos, price, price))
    idx = {p.element_id: i for i, p in enumerate(players)}
    squad = Squad(
        players=players,
        lineup=[idx[e] for e in init.lineup_element_ids if e in idx],
        bench=[idx[e] for e in init.bench_element_ids if e in idx],
        captain_idx=idx.get(init.captain_id, 0),
        vice_captain_idx=idx.get(init.vice_captain_id, 1),
    )
    state = GameState(
        squad=squad,
        bank=STARTING_BUDGET - init.total_cost,
        free_transfers=GW1_FREE_TRANSFERS,
        current_gw=first_gw,
    )

    gross = hits = xfers = 0
    gw_scores = []
    for gw in gws:
        if gw == first_gw:
            # The GW1 squad was just picked freely: play it as selected
            # (no transfers — FPL has none before the GW1 deadline).
            action = to_engine_action(init)
            action.transfers_in, action.transfers_out = [], []
        else:
            try:
                opt = optimize_transfers(
                    state,
                    build_candidate_pool(loader, gw, pred_for_gw(gw)),
                    max_transfers=args.max_transfers,
                    squad_teams={
                        p.element_id: loader.get_player_team(p.element_id)
                        for p in state.squad.players
                        if loader.get_player_team(p.element_id) is not None
                    },
                )
                action = to_engine_action(opt)
            except Exception:
                action = EngineAction()
        try:
            state, res = engine.step(state, action)
        except ValueError:
            state, res = engine.step(state, EngineAction())
        gross += res.gw_points
        hits += res.hit_cost
        xfers += len(action.transfers_out)
        gw_scores.append(res.net_points)
        if gw % 10 == 0 or gw == gws[-1]:
            print(f"  GW{gw:>2}: net {res.net_points:>3} | total {state.total_points}")

    print(f"\n=== RESULT {target} (max_transfers={args.max_transfers}) ===")
    print(f"  Net points:   {state.total_points}")
    print(f"  Gross points: {gross}  |  hits: -{hits}  |  transfers: {xfers}")
    print(f"  Avg net/GW:   {np.mean(gw_scores):.1f}  |  best GW: {max(gw_scores)}"
          f"  |  worst GW: {min(gw_scores)}")
    print("  (no chips played; availability filtering not possible historically)")


if __name__ == "__main__":
    main()
