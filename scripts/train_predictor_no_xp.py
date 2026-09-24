#!/usr/bin/env python3
"""Train LightGBM predictor with full pre-game feature set.

Features:
- fpl_xp (FPL's ep_this, pre-match, available 2020-21+)
- synthetic_ep (reconstructed EP from form + fixture offset + playing prob)
- playing_prob, fixture_offset (EP components for the model to learn from)
- Set-piece flags (is_penalty_taker, is_corner_taker, is_freekick_taker)
- Separate transfers_in/out rolling features
- Value/selected momentum
- FPL xG/xA rolling 3 (shorter window)
- Derived: transfer_ratio, ICT/BPS momentum, bonus_rate, nailedness

Outputs:
- Feature analysis (correlations, importance)
- Model saved to models/full_pregame/
- Holdout (2024-25) evaluation metrics
- Comparison with MILP optimizer
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

HOLDOUT = "2024-25"
TRAIN_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24",
]

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

MODEL_DIR = Path("models/full_pregame")


def analyze_features(df: pd.DataFrame) -> None:
    """Print feature analysis: coverage, correlations with target."""
    from fpl_optimizer.prediction.model import _NON_FEATURE_COLS

    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]

    print(f"\n{'='*70}")
    print("FEATURE ANALYSIS")
    print(f"{'='*70}")
    print(f"Total features: {len(feature_cols)}")
    print(f"Total rows: {len(df)}")

    # Check key features are present
    key_feats = ["fpl_xp", "synthetic_ep", "playing_prob", "fixture_offset"]
    for f in key_feats:
        present = f in feature_cols
        print(f"  {f}: {'YES' if present else 'MISSING'}")

    # Coverage by feature
    print(f"\n{'Feature':<35} {'Coverage':>8} {'Corr w/ target':>15}")
    print("-" * 60)

    target = df["target"]
    rows = []
    for col in sorted(feature_cols):
        valid = df[col].notna()
        coverage = valid.mean() * 100
        if valid.sum() > 100 and target[valid].notna().sum() > 100:
            corr = df.loc[valid, col].corr(target[valid])
        else:
            corr = float("nan")
        rows.append((col, coverage, corr))

    # Sort by absolute correlation
    rows.sort(key=lambda x: abs(x[2]) if not np.isnan(x[2]) else 0, reverse=True)
    for col, cov, corr in rows:
        corr_str = f"{corr:>+.4f}" if not np.isnan(corr) else "N/A"
        print(f"  {col:<35} {cov:>6.1f}% {corr_str:>15}")

    # New features specifically
    new_features = [
        "fpl_xp", "synthetic_ep", "playing_prob", "fixture_offset",
        "is_penalty_taker", "is_corner_taker", "is_freekick_taker",
        "set_piece_order_sum",
        "transfers_in_rolling_3", "transfers_in_rolling_5",
        "transfers_out_rolling_3", "transfers_out_rolling_5",
        "value_momentum", "selected_momentum",
        "fpl_xg_rolling_3", "fpl_xa_rolling_3", "fpl_xgi_rolling_3",
        "transfer_ratio_3", "ict_form_delta", "bps_form_delta",
        "nailedness", "bonus_rate_5",
    ]
    print(f"\n{'='*70}")
    print("NEW FEATURES SPOTLIGHT")
    print(f"{'='*70}")
    print(f"{'Feature':<35} {'Coverage':>8} {'Corr w/ target':>15}")
    print("-" * 60)
    for col in new_features:
        if col in df.columns:
            valid = df[col].notna()
            coverage = valid.mean() * 100
            if valid.sum() > 100:
                corr = df.loc[valid, col].corr(target[valid])
            else:
                corr = float("nan")
            corr_str = f"{corr:>+.4f}" if not np.isnan(corr) else "N/A"
            print(f"  {col:<35} {cov:>6.1f}% {corr_str:>15}")
        else:
            print(f"  {col:<35}  MISSING")


def train_and_eval(df: pd.DataFrame) -> dict:
    """Train model and evaluate on holdout."""
    from fpl_optimizer.prediction.model import PointPredictor

    train_full = df[df["season"].isin(TRAIN_SEASONS)].copy()
    holdout_df = df[df["season"] == HOLDOUT].copy()

    # Validation split: last 8 GWs of last training season
    last_season = TRAIN_SEASONS[-1]
    last_data = train_full[train_full["season"] == last_season]
    max_gw = int(last_data["GW"].max())
    val_mask = (train_full["season"] == last_season) & (train_full["GW"] > max_gw - 8)
    val_df = train_full[val_mask].copy()
    train_df = train_full[~val_mask].copy()

    print(f"\nTrain: {len(train_df)} rows, Val: {len(val_df)} rows, Holdout: {len(holdout_df)} rows")

    predictor = PointPredictor(params=PARAMS, early_stopping_rounds=50)
    metrics = predictor.train(train_df, val_df)

    preds = predictor.predict(holdout_df)
    actual = holdout_df["target"].values
    valid = ~np.isnan(actual)

    mae = np.mean(np.abs(preds[valid] - actual[valid]))
    rmse = np.sqrt(np.mean((preds[valid] - actual[valid]) ** 2))
    corr = np.corrcoef(preds[valid], actual[valid])[0, 1]

    print(f"\n{'='*70}")
    print("HOLDOUT RESULTS ({})".format(HOLDOUT))
    print(f"{'='*70}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Corr: {corr:.4f}")

    # Per position
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_mask = (holdout_df["position"] == pos).values & valid
        if pos_mask.sum() > 0:
            pos_mae = np.mean(np.abs(preds[pos_mask] - actual[pos_mask]))
            pos_corr = np.corrcoef(preds[pos_mask], actual[pos_mask])[0, 1]
            print(f"    {pos}: MAE={pos_mae:.4f}, Corr={pos_corr:.4f}, N={pos_mask.sum()}")

    # Per-GW correlation
    holdout_df = holdout_df.copy()
    holdout_df["pred"] = preds
    gw_corrs = []
    for gw in sorted(holdout_df["GW"].unique()):
        gw_data = holdout_df[holdout_df["GW"] == gw]
        if len(gw_data) > 20:
            gc = gw_data["pred"].corr(gw_data["target"])
            gw_corrs.append(gc)
    mean_gw_corr = np.nanmean(gw_corrs)
    print(f"\n  Mean per-GW correlation: {mean_gw_corr:.4f} (across {len(gw_corrs)} GWs)")

    # Feature importance
    fi = predictor.feature_importance()
    if not fi.empty:
        print(f"\n  Top 20 features by importance:")
        for i, row in fi.head(20).iterrows():
            print(f"    #{i+1:>2} {row['feature']:<35} {row['importance']:.0f}")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    predictor.save(MODEL_DIR)
    print(f"\n  Model saved to {MODEL_DIR}")

    return {
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "mean_gw_corr": mean_gw_corr,
        "predictions": {
            (int(holdout_df.iloc[i]["element"]), int(holdout_df.iloc[i]["GW"])): float(preds[i])
            for i in range(len(preds))
        },
    }


def run_optimizer(predictions: dict) -> None:
    """Run MILP optimizer with the new predictions."""
    from fpl_optimizer.data.downloader import DEFAULT_DATA_DIR
    from fpl_optimizer.data.loader import SeasonDataLoader
    from fpl_optimizer.engine.engine import FPLGameEngine
    from fpl_optimizer.engine.state import EngineAction, GameState, PlayerSlot, Squad
    from fpl_optimizer.optimizer.squad_selection import select_squad
    from fpl_optimizer.optimizer.transfer_optimizer import optimize_transfers
    from fpl_optimizer.optimizer.types import build_candidate_pool, to_engine_action
    from fpl_optimizer.utils.constants import GW1_FREE_TRANSFERS, STARTING_BUDGET

    loader = SeasonDataLoader(HOLDOUT, DEFAULT_DATA_DIR)
    engine = FPLGameEngine(loader)

    def pred_fn(gw):
        eids = loader.get_all_element_ids(gw)
        return {eid: predictions.get((eid, gw), 0.0) for eid in eids}

    pp1 = pred_fn(1)
    cands = build_candidate_pool(loader, 1, pp1)
    result = select_squad(cands, budget=STARTING_BUDGET)

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
    squad = Squad(
        players=players,
        lineup=[eid_to_idx[e] for e in result.lineup_element_ids if e in eid_to_idx],
        bench=[eid_to_idx[e] for e in result.bench_element_ids if e in eid_to_idx],
        captain_idx=eid_to_idx.get(result.captain_id, 0),
        vice_captain_idx=eid_to_idx.get(result.vice_captain_id, 1),
    )
    state = GameState(
        squad=squad, bank=STARTING_BUDGET - result.total_cost,
        free_transfers=GW1_FREE_TRANSFERS, current_gw=1,
    )

    print(f"\n{'='*70}")
    print(f"MILP OPTIMIZER — {HOLDOUT}")
    print(f"{'='*70}")

    total_gross = 0
    total_hits = 0
    total_xfers = 0
    gw_points = []
    for gw in range(1, 39):
        pp = pred_fn(gw)
        c = build_candidate_pool(loader, gw, pp)
        try:
            opt = optimize_transfers(state, c)
            action = to_engine_action(opt)
        except RuntimeError:
            action = EngineAction()
        try:
            state, res = engine.step(state, action)
        except ValueError:
            state, res = engine.step(state, EngineAction())
        total_gross += res.gw_points
        total_hits += res.hit_cost
        total_xfers += len(action.transfers_out)
        gw_points.append(res.gw_points)
        if gw % 10 == 0 or gw == 38:
            print(f"  GW{gw:>2}: {res.gw_points:>3}pts (running total: {state.total_points})")

    print(f"\n  Final: net={state.total_points}, gross={total_gross}, hits={total_hits}, xfers={total_xfers}")
    print(f"  Avg pts/GW: {total_gross/38:.1f}")
    print(f"\n  Reference: Oracle=4756, Best human=~2810")


def main():
    from fpl_optimizer.data.downloader import DEFAULT_DATA_DIR
    from fpl_optimizer.prediction.id_resolver import IDResolver
    from fpl_optimizer.prediction.feature_pipeline import FeaturePipeline

    data_dir = DEFAULT_DATA_DIR.parent
    all_seasons = TRAIN_SEASONS + [HOLDOUT]

    print("=" * 70)
    print("PREDICTOR TRAINING — full pre-game features (xP + synthetic EP)")
    print("=" * 70)
    print(f"Train: {TRAIN_SEASONS}")
    print(f"Holdout: {HOLDOUT}")
    print()

    # Build features
    print("Building feature pipeline...")
    t0 = time.time()
    resolver = IDResolver(data_dir)
    pipeline = FeaturePipeline(data_dir, resolver, all_seasons)
    df = pipeline.build()
    elapsed = time.time() - t0
    print(f"Built {len(df)} rows x {len(df.columns)} cols in {elapsed:.0f}s")

    # Verify key features
    for f in ["fpl_xp", "synthetic_ep", "playing_prob"]:
        if f in df.columns:
            n = df[f].notna().sum()
            print(f"  {f}: {n}/{len(df)} non-null ({100*n/len(df):.1f}%)")

    # Feature analysis
    analyze_features(df)

    # Train and evaluate
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")
    t0 = time.time()
    result = train_and_eval(df)
    print(f"\nTraining + eval took {time.time()-t0:.0f}s")

    # Run optimizer
    run_optimizer(result["predictions"])


if __name__ == "__main__":
    main()
