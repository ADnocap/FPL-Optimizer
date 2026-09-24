#!/usr/bin/env python3
"""Retrain the LightGBM predictor from scratch and retest.

Uses the exact same hyperparameters as the original model.
Saves to models/point_predictor_v2/ to avoid overwriting.
Then runs the MILP optimizer comparison on the holdout season.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25",
]
HOLDOUT = "2024-25"
MODEL_DIR = Path("models/point_predictor_v2")

TUNED_PARAMS = {
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


def main():
    from fpl_optimizer.data.downloader import DEFAULT_DATA_DIR
    from fpl_optimizer.prediction.id_resolver import IDResolver
    from fpl_optimizer.prediction.feature_pipeline import FeaturePipeline
    from fpl_optimizer.prediction.model import PointPredictor

    data_dir = DEFAULT_DATA_DIR.parent if DEFAULT_DATA_DIR.name == "raw" else DEFAULT_DATA_DIR
    print(f"Data dir: {data_dir}")

    # ================================================================
    # STEP 1: Build features for ALL seasons
    # ================================================================
    print("\n=== STEP 1: Building feature pipeline for all 9 seasons ===")
    t0 = time.time()
    id_resolver = IDResolver(data_dir)
    pipeline = FeaturePipeline(data_dir, id_resolver, SEASONS)
    df = pipeline.build()
    print(f"Feature pipeline complete: {len(df)} rows x {len(df.columns)} cols in {time.time()-t0:.1f}s")
    print(f"Seasons present: {sorted(df['season'].unique())}")
    print(f"Feature columns: {len([c for c in df.columns if c not in {'code','element','season','GW','position','target','total_points'}])}")

    # ================================================================
    # STEP 2: Split train / validation / holdout
    # ================================================================
    print("\n=== STEP 2: Splitting data ===")
    train_seasons = [s for s in SEASONS if s != HOLDOUT]
    train_full = df[df["season"].isin(train_seasons)].copy()
    holdout_df = df[df["season"] == HOLDOUT].copy()

    # Validation: last 8 GWs of last training season (2023-24)
    last_season = train_seasons[-1]
    last_season_data = train_full[train_full["season"] == last_season]
    max_gw = int(last_season_data["GW"].max())
    val_cutoff = max_gw - 8

    val_mask = (train_full["season"] == last_season) & (train_full["GW"] > val_cutoff)
    val_df = train_full[val_mask].copy()
    train_df = train_full[~val_mask].copy()

    print(f"Train: {len(train_df)} rows ({len(train_df['season'].unique())} seasons)")
    print(f"Val:   {len(val_df)} rows (last 8 GWs of {last_season})")
    print(f"Holdout: {len(holdout_df)} rows ({HOLDOUT})")

    # Check for NaN in target
    train_target_nan = train_df["target"].isna().sum()
    val_target_nan = val_df["target"].isna().sum()
    print(f"Train target NaN: {train_target_nan}, Val target NaN: {val_target_nan}")

    # ================================================================
    # STEP 3: Train the model
    # ================================================================
    print("\n=== STEP 3: Training LightGBM (4 position-specific models) ===")
    t0 = time.time()
    predictor = PointPredictor(params=TUNED_PARAMS, early_stopping_rounds=50)
    train_metrics = predictor.train(train_df, val_df)
    print(f"Training complete in {time.time()-t0:.1f}s")
    print(f"Training MAE per position: {train_metrics}")

    # ================================================================
    # STEP 4: Save the model
    # ================================================================
    print(f"\n=== STEP 4: Saving model to {MODEL_DIR} ===")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    predictor.save(MODEL_DIR)
    print("Model saved.")

    # ================================================================
    # STEP 5: Evaluate on holdout
    # ================================================================
    print("\n=== STEP 5: Evaluating on holdout (2024-25) ===")
    holdout_preds = predictor.predict(holdout_df)
    holdout_actual = holdout_df["target"].values

    valid_mask = ~np.isnan(holdout_actual)
    preds_valid = holdout_preds[valid_mask]
    actual_valid = holdout_actual[valid_mask]

    mae = np.mean(np.abs(preds_valid - actual_valid))
    rmse = np.sqrt(np.mean((preds_valid - actual_valid) ** 2))
    corr = np.corrcoef(preds_valid, actual_valid)[0, 1]

    print(f"Holdout MAE:  {mae:.4f}")
    print(f"Holdout RMSE: {rmse:.4f}")
    print(f"Holdout Corr: {corr:.4f}")

    # Per-position
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_mask = (holdout_df["position"] == pos).values & valid_mask
        if pos_mask.sum() > 0:
            pos_mae = np.mean(np.abs(holdout_preds[pos_mask] - holdout_actual[pos_mask]))
            pos_n = pos_mask.sum()
            print(f"  {pos}: MAE={pos_mae:.4f} (n={pos_n})")

    # ================================================================
    # STEP 6: Compare old vs new predictions
    # ================================================================
    print("\n=== STEP 6: Comparing old vs new model predictions ===")
    old_predictor = PointPredictor.load(Path("models/point_predictor"))
    old_preds = old_predictor.predict(holdout_df)

    old_mae = np.mean(np.abs(old_preds[valid_mask] - actual_valid))
    old_corr = np.corrcoef(old_preds[valid_mask], actual_valid)[0, 1]

    print(f"Old model: MAE={old_mae:.4f}, Corr={old_corr:.4f}")
    print(f"New model: MAE={mae:.4f}, Corr={corr:.4f}")

    pred_diff = np.mean(np.abs(holdout_preds[valid_mask] - old_preds[valid_mask]))
    pred_corr = np.corrcoef(holdout_preds[valid_mask], old_preds[valid_mask])[0, 1]
    print(f"Old vs New prediction difference: MAE={pred_diff:.4f}, Corr={pred_corr:.4f}")

    # ================================================================
    # STEP 7: Run MILP optimizer with new predictions on holdout
    # ================================================================
    print("\n=== STEP 7: Running MILP optimizer with new predictor ===")
    from fpl_optimizer.data.loader import SeasonDataLoader
    from fpl_optimizer.engine.engine import FPLGameEngine
    from fpl_optimizer.engine.state import ChipState, EngineAction, GameState, PlayerSlot, Squad
    from fpl_optimizer.optimizer.squad_selection import select_squad
    from fpl_optimizer.optimizer.transfer_optimizer import optimize_transfers
    from fpl_optimizer.optimizer.types import build_candidate_pool, to_engine_action
    from fpl_optimizer.prediction.integration import PredictionIntegrator
    from fpl_optimizer.utils.constants import INITIAL_FREE_TRANSFERS, STARTING_BUDGET

    loader = SeasonDataLoader(HOLDOUT, DEFAULT_DATA_DIR)
    engine = FPLGameEngine(loader)

    # Build integrator from new model
    new_integrator = PredictionIntegrator.from_model(MODEL_DIR, data_dir, HOLDOUT)
    print(f"New integrator: {len(new_integrator)} predictions")

    # Also build from old model for comparison
    old_integrator = PredictionIntegrator.from_model(
        Path("models/point_predictor"), data_dir, HOLDOUT
    )
    print(f"Old integrator: {len(old_integrator)} predictions")

    def run_season(integrator, label, max_xfers=5):
        def pred_fn(gw):
            eids = loader.get_all_element_ids(gw)
            return {eid: integrator.get_predicted_points(eid, gw) for eid in eids}

        pp1 = pred_fn(1)
        candidates = build_candidate_pool(loader, 1, pp1)
        result = select_squad(candidates, budget=STARTING_BUDGET)

        players = []
        for eid in result.squad_element_ids:
            pos = loader.get_player_position(eid)
            price = loader.get_player_price(eid, 1)
            if pos and price > 0:
                players.append(PlayerSlot(
                    element_id=eid, position=pos,
                    purchase_price=price, selling_price=price,
                ))
        eid_to_idx = {p.element_id: i for i, p in enumerate(players)}
        lineup = [eid_to_idx[e] for e in result.lineup_element_ids if e in eid_to_idx]
        bench = [eid_to_idx[e] for e in result.bench_element_ids if e in eid_to_idx]
        squad = Squad(
            players=players, lineup=lineup, bench=bench,
            captain_idx=eid_to_idx.get(result.captain_id, 0),
            vice_captain_idx=eid_to_idx.get(result.vice_captain_id, 1),
        )
        state = GameState(
            squad=squad, bank=STARTING_BUDGET - result.total_cost,
            free_transfers=INITIAL_FREE_TRANSFERS, chips=ChipState(),
            current_gw=1, total_points=0,
        )

        gw_pts = []
        for gw in range(1, 39):
            pp = pred_fn(gw)
            cands = build_candidate_pool(loader, gw, pp)
            try:
                opt = optimize_transfers(state, cands, max_transfers=max_xfers)
                action = to_engine_action(opt)
            except RuntimeError:
                action = EngineAction()
            try:
                state, res = engine.step(state, action)
            except ValueError:
                state, res = engine.step(state, EngineAction())
            gw_pts.append(res.gw_points)

        gross = sum(gw_pts)
        net = state.total_points
        hits = gross - net
        print(f"  {label}: {net} net ({gross} gross, {hits} hits)")
        return net, gross

    # Run with both models
    print()
    new_net, new_gross = run_season(new_integrator, "NEW model (retrained)")
    old_net, old_gross = run_season(old_integrator, "OLD model (original)")

    # Also run oracle and 1-xfer
    def oracle_pred_fn(gw):
        return None  # Uses actual points

    print()
    # Run new model with 1 xfer for realistic comparison
    new_1x_net, _ = run_season(new_integrator, "NEW model (1 xfer/GW)", max_xfers=1)

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*60}")
    print(f"FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"OLD model, 5 xfer/GW:  {old_net} net ({old_gross} gross)")
    print(f"NEW model, 5 xfer/GW:  {new_net} net ({new_gross} gross)")
    print(f"NEW model, 1 xfer/GW:  {new_1x_net} net")
    print(f"\nPrediction quality:")
    print(f"  Old: MAE={old_mae:.4f}, Corr={old_corr:.4f}")
    print(f"  New: MAE={mae:.4f}, Corr={corr:.4f}")
    print(f"\nFor context:")
    print(f"  Best human 2024-25:   ~2810")
    print(f"  Best human ever:      ~2844")


if __name__ == "__main__":
    main()
