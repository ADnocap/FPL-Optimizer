#!/usr/bin/env python3
"""Within-season A/B: do the DEFCON features improve DEFCON-era prediction?

The standard eval harness (train <=2024-25, holdout 2025-26) cannot measure
features whose source columns only exist from 2025-26 — the eval model never
sees them non-NaN. This experiment trains INSIDE the DEFCON era:

    train = 2016-17..2024-25 + 2025-26 GW1-30 (val = 2025-26 GW23-30)
    eval  = 2025-26 GW31-38

Config A: full current pipeline (122 cols).
Config B: same minus the 5 DEFCON features (defcon_rolling_3/5, cbi_rolling_5,
          cbit_rolling_5, cbirt_rolling_5).

Compared on stratified metrics, overall and DEF/MID only (where DEFCON pays).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFCON_FEATURES = [
    "defcon_rolling_3", "defcon_rolling_5", "cbi_rolling_5",
    "cbit_rolling_5", "cbirt_rolling_5",
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
        format_report, stratified_metrics,
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

    for label, drop in (("A: with DEFCON feats", []),
                        ("B: without DEFCON feats", DEFCON_FEATURES)):
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
        for pos in ("DEF", "MID"):
            m = (ev["position"] == pos).values & ok
            print(format_report(
                stratified_metrics(actual[m], preds[m]), label=f"{label} [{pos}]"))


if __name__ == "__main__":
    main()
