#!/usr/bin/env python3
"""Within-season A/B: do the player-prop odds features improve prediction?

Props snapshots only exist for 2025-26, so the standard eval harness
(train <=2024-25, holdout 2025-26) can never learn them — the training
data has them all-NaN. This experiment trains INSIDE the props era:

    train = 2016-17..2024-25 + 2025-26 GW1-22 (val = 2025-26 GW23-30)
    eval  = 2025-26 GW31-38

Config A: full current pipeline including the 6 props features.
Config B: same minus props_xg, props_xa, props_sot, props_card_prob,
          props_has_line, props_n_books.

Compared on stratified metrics (overall + MID/FWD, where the attacking
prop markets should pay) and per-GW Spearman.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
PROPS_FEATURES = [
    "props_xg", "props_xa", "props_sot", "props_card_prob",
    "props_has_line", "props_n_books",
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
SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]


def main() -> None:
    from fpl_optimizer.prediction.feature_pipeline import FeaturePipeline
    from fpl_optimizer.prediction.id_resolver import IDResolver
    from fpl_optimizer.prediction.model import PointPredictor
    from fpl_optimizer.prediction.stratified_metrics import (
        format_report, per_gw_spearman, stratified_metrics,
    )

    data_dir = REPO_ROOT / "data"
    t0 = time.time()
    df = FeaturePipeline(data_dir, IDResolver(data_dir), SEASONS).build()
    gw_max = df.groupby(["season", "GW"])["target"].transform("max")
    df = df[gw_max > 0]
    print(f"Features: {len(df)} rows x {len(df.columns)} cols "
          f"in {time.time() - t0:.0f}s")

    is_2526 = df["season"] == "2025-26"
    train = df[~is_2526 | (df["GW"] <= 22)].copy()
    val = df[is_2526 & df["GW"].between(23, 30)].copy()
    evald = df[is_2526 & (df["GW"] >= 31)].copy()
    print(f"train={len(train)} val={len(val)} eval(GW31-38)={len(evald)}")

    for label, drop in (("A: with props feats", []),
                        ("B: without props feats", PROPS_FEATURES)):
        tr = train.drop(columns=drop, errors="ignore")
        va = val.drop(columns=drop, errors="ignore")
        ev = evald.drop(columns=drop, errors="ignore")
        model = PointPredictor(params=PARAMS, early_stopping_rounds=50)
        model.train(tr, va)
        preds = model.predict(ev)
        actual = ev["target"].values
        ok = ~np.isnan(actual)
        print("\n" + format_report(
            stratified_metrics(actual[ok], preds[ok]), label=label))
        for pos in ("MID", "FWD"):
            m = (ev["position"] == pos).values & ok
            print(format_report(
                stratified_metrics(actual[m], preds[m]), label=f"{label} [{pos}]"))
        sp = per_gw_spearman(actual[ok], preds[ok], ev.loc[ok, "GW"].values)
        per_gw = "  ".join(f"GW{int(g)}:{r:.3f}" for g, r in sorted(sp["per_gw"].items()))
        print(f"{label} per-GW Spearman: mean={sp['mean']:.4f}  [{per_gw}]")

        if not drop:
            imp = model.feature_importance().reset_index(drop=True)
            print(f"\nProps feature importance (gain, {len(imp)} features):")
            for feat in PROPS_FEATURES:
                match = imp.index[imp["feature"] == feat]
                if len(match):
                    rank = int(match[0]) + 1
                    gain = imp.loc[match[0], "importance"]
                    print(f"  {feat:<18} rank #{rank:<4} gain={gain:.1f}")


if __name__ == "__main__":
    main()
