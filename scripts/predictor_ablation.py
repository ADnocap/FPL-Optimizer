#!/usr/bin/env python3
"""Ablation study: effect of xP on predictor quality and optimizer score.

Variants:
1. UNSHIFTED — fpl_xp as-is (current GW's ep_this)
2. SHIFTED — fpl_xp shifted by 1 GW (previous GW's ep_this)
3. NO_XP — fpl_xp dropped entirely
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


def train_and_eval(df, label, model_dir):
    from fpl_optimizer.prediction.model import PointPredictor

    train_full = df[df["season"].isin(TRAIN_SEASONS)].copy()
    holdout_df = df[df["season"] == HOLDOUT].copy()

    last_season = TRAIN_SEASONS[-1]
    last_data = train_full[train_full["season"] == last_season]
    max_gw = int(last_data["GW"].max())
    val_mask = (train_full["season"] == last_season) & (train_full["GW"] > max_gw - 8)
    val_df = train_full[val_mask].copy()
    train_df = train_full[~val_mask].copy()

    predictor = PointPredictor(params=PARAMS, early_stopping_rounds=50)
    metrics = predictor.train(train_df, val_df)

    preds = predictor.predict(holdout_df)
    actual = holdout_df["target"].values
    valid = ~np.isnan(actual)

    mae = np.mean(np.abs(preds[valid] - actual[valid]))
    rmse = np.sqrt(np.mean((preds[valid] - actual[valid]) ** 2))
    corr = np.corrcoef(preds[valid], actual[valid])[0, 1]

    # Per position
    pos_maes = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_mask = (holdout_df["position"] == pos).values & valid
        if pos_mask.sum() > 0:
            pos_maes[pos] = np.mean(np.abs(preds[pos_mask] - actual[pos_mask]))

    # Feature importance for fpl_xp
    xp_rank = None
    xp_pct = 0
    feat_names = predictor._feature_names
    if "MID" in predictor._models:
        importances = predictor._models["MID"].feature_importance(importance_type="gain")
        total_imp = sum(importances)
        if "fpl_xp" in feat_names:
            xp_idx = feat_names.index("fpl_xp")
            xp_imp = importances[xp_idx]
            xp_pct = 100 * xp_imp / total_imp
            ranked = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
            xp_rank = ranked.index(xp_idx) + 1

    # Save model
    model_dir.mkdir(parents=True, exist_ok=True)
    predictor.save(model_dir)

    return {
        "label": label,
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "pos_maes": pos_maes,
        "xp_rank": xp_rank,
        "xp_pct": xp_pct,
        "n_features": len(feat_names),
        "model_dir": model_dir,
        "predictions": {
            (int(holdout_df.iloc[i]["element"]), int(holdout_df.iloc[i]["GW"])): float(preds[i])
            for i in range(len(preds))
        },
    }


def run_optimizer(predictions, label, max_xfers=None):
    from fpl_optimizer.data.downloader import DEFAULT_DATA_DIR
    from fpl_optimizer.data.loader import SeasonDataLoader
    from fpl_optimizer.engine.engine import FPLGameEngine
    from fpl_optimizer.engine.state import ChipState, EngineAction, GameState, PlayerSlot, Squad
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

    total_gross = 0
    total_hits = 0
    total_xfers = 0
    for gw in range(1, 39):
        pp = pred_fn(gw)
        c = build_candidate_pool(loader, gw, pp)
        try:
            opt = optimize_transfers(state, c, max_transfers=max_xfers)
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

    return state.total_points, total_gross, total_hits, total_xfers


def main():
    from fpl_optimizer.data.downloader import DEFAULT_DATA_DIR
    from fpl_optimizer.prediction.id_resolver import IDResolver
    from fpl_optimizer.prediction.feature_pipeline import FeaturePipeline

    data_dir = DEFAULT_DATA_DIR.parent
    all_seasons = TRAIN_SEASONS + [HOLDOUT]

    print("=" * 70)
    print("PREDICTOR ABLATION STUDY: xP variants")
    print("=" * 70)
    print(f"Train: {TRAIN_SEASONS}")
    print(f"Holdout: {HOLDOUT}")
    print()

    # Build features once
    print("Building feature pipeline...")
    t0 = time.time()
    resolver = IDResolver(data_dir)
    pipeline = FeaturePipeline(data_dir, resolver, all_seasons)
    df = pipeline.build()
    print(f"Built {len(df)} rows x {len(df.columns)} cols in {time.time()-t0:.0f}s")

    has_xp = "fpl_xp" in df.columns
    print(f"fpl_xp in features: {has_xp}")
    if has_xp:
        xp_nonnull = df["fpl_xp"].notna().sum()
        print(f"fpl_xp non-null: {xp_nonnull}/{len(df)} ({100*xp_nonnull/len(df):.1f}%)")
    print()

    # Create variants
    variants = []

    # 1. Unshifted
    df_unshifted = df.copy()
    variants.append(("UNSHIFTED", df_unshifted, Path("models/ablation_unshifted")))

    # 2. Shifted
    if has_xp:
        df_shifted = df.copy().sort_values(["season", "element", "GW"])
        df_shifted["fpl_xp"] = df_shifted.groupby(["season", "element"])["fpl_xp"].shift(1)
        variants.append(("SHIFTED", df_shifted, Path("models/ablation_shifted")))

    # 3. No xP
    if has_xp:
        df_noxp = df.copy()
        df_noxp["fpl_xp"] = float("nan")
        variants.append(("NO_XP", df_noxp, Path("models/ablation_noxp")))

    # Train and evaluate each variant
    results = []
    for label, df_variant, model_dir in variants:
        print(f"\n{'='*70}")
        print(f"  VARIANT: {label}")
        print(f"{'='*70}")
        t0 = time.time()
        r = train_and_eval(df_variant, label, model_dir)
        print(f"  Time: {time.time()-t0:.0f}s")
        print(f"  Holdout MAE:  {r['mae']:.4f}")
        print(f"  Holdout RMSE: {r['rmse']:.4f}")
        print(f"  Holdout Corr: {r['corr']:.4f}")
        for pos, mae in r["pos_maes"].items():
            print(f"    {pos}: MAE={mae:.4f}")
        if r["xp_rank"]:
            print(f"  fpl_xp rank: #{r['xp_rank']} ({r['xp_pct']:.1f}%)")
        print(f"  Features: {r['n_features']}")

        # Run optimizer
        print(f"\n  MILP Optimizer on {HOLDOUT}:")
        for mx in [0, 1, 5, None]:
            net, gross, hits, xfers = run_optimizer(
                r["predictions"], label, max_xfers=mx,
            )
            mx_label = "unlimited" if mx is None else str(mx)
            print(f"    {mx_label:>9} xfer/GW: net={net:>5} (gross={gross:>5} hits={hits:>4} xfers={xfers:>4})")

        results.append(r)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':>12} {'MAE':>7} {'RMSE':>7} {'Corr':>7} {'xP rank':>8} {'xP %':>6}")
    for r in results:
        xp_str = f"#{r['xp_rank']}" if r["xp_rank"] else "N/A"
        xp_pct_str = f"{r['xp_pct']:.1f}%" if r["xp_rank"] else "N/A"
        print(f"{r['label']:>12} {r['mae']:>7.4f} {r['rmse']:>7.4f} {r['corr']:>7.4f} {xp_str:>8} {xp_pct_str:>6}")

    print(f"\nReference: Oracle=4756, Best human=~2810")


if __name__ == "__main__":
    main()
