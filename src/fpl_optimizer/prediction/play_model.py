"""Optional LightGBM P(play) classifier (P(minutes > 0) in a GW).

The horizon predictor's default P(play) is a calibrated lookup on
``playing_prob`` and the lead ``k``
(:func:`fpl_optimizer.prediction.horizon.p_play_from_playing_prob`).  This
classifier uses more of the as-of row (recent minutes/starts, nailedness,
price, ownership, transfers...) and is sharper: on the 2025-26 holdout
(trained <= 2024-25) log-loss 0.287 / Brier 0.088 / AUC 0.946, vs 0.358 /
0.109 / 0.917 for a logistic fit on recent minutes.  It predicts the as-of
GW; later GWs keep the lookup's decay profile with ``k``
(:func:`apply_play_model`).

Used by ``scripts/chip_eval.py --pplay-dir`` (auto-subs, captain failover
and Bench Boost values depend on who plays).  Train with
``scripts/chip_eval.py train-pplay --features <features.parquet>``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

POS_INT = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}

PPLAY_FEATURES = [
    "mins_rolling_3", "mins_rolling_5", "mins_std_5", "starts_rolling_5",
    "playing_prob", "nailedness", "minutes_share_5", "games_played",
    "season_total_mins", "value", "selected_norm", "gw_phase", "fpl_xp_lag",
    "pts_rolling_5", "pts_rolling_10", "transfers_balance_rolling_3",
    "transfers_out_rolling_3", "prev_minutes", "value_momentum", "pos_code",
]
PPLAY_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
                "min_child_samples": 200, "feature_fraction": 0.9,
                "bagging_fraction": 0.8, "bagging_freq": 5, "lambda_l2": 1.0,
                "verbose": -1, "seed": 42, "num_threads": 4}
MODEL_FILE = "pplay.lgb"
FEATURES_FILE = "pplay_features.json"


def _matrix(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for f in feats:
        if f == "pos_code":
            X[f] = df["position"].map(POS_INT).astype(float)
        else:
            X[f] = pd.to_numeric(df[f], errors="coerce") if f in df.columns else np.nan
    return X


class PlayModel:
    """LightGBM classifier P(minutes > 0) on the point model's feature rows."""

    def __init__(self, booster=None, features: list[str] | None = None):
        self.booster = booster
        self.features = features or list(PPLAY_FEATURES)

    @classmethod
    def train(cls, df: pd.DataFrame, played: np.ndarray, rounds: int = 400) -> "PlayModel":
        import lightgbm as lgb

        X = _matrix(df, PPLAY_FEATURES)
        booster = lgb.train(PPLAY_PARAMS, lgb.Dataset(X, label=np.asarray(played).astype(int)),
                            num_boost_round=rounds)
        return cls(booster, list(PPLAY_FEATURES))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(_matrix(df, self.features))

    def save(self, model_dir: Path) -> None:
        d = Path(model_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(d / MODEL_FILE))
        (d / FEATURES_FILE).write_text(json.dumps(self.features))

    @classmethod
    def load(cls, model_dir: Path) -> "PlayModel | None":
        """The saved model, or None when ``model_dir`` has none."""
        import lightgbm as lgb

        d = Path(model_dir)
        if not (d / MODEL_FILE).exists():
            return None
        return cls(lgb.Booster(model_file=str(d / MODEL_FILE)),
                   json.loads((d / FEATURES_FILE).read_text()))


def load_played_labels(data_dir: Path, seasons: list[str]) -> pd.DataFrame:
    """(season, element, GW) -> played (minutes > 0) from raw merged_gw."""
    out = []
    for s in seasons:
        p = Path(data_dir) / "raw" / s / "gws" / "merged_gw.csv"
        if not p.exists():
            continue
        try:
            m = pd.read_csv(p, usecols=["element", "GW", "minutes"], encoding="utf-8")
        except UnicodeDecodeError:
            m = pd.read_csv(p, usecols=["element", "GW", "minutes"], encoding="latin-1")
        m["minutes"] = pd.to_numeric(m["minutes"], errors="coerce").fillna(0)
        g = m.groupby(["element", "GW"], as_index=False)["minutes"].sum()
        g["season"] = s
        out.append(g)
    if not out:
        return pd.DataFrame(columns=["season", "element", "GW", "played"])
    lab = pd.concat(out, ignore_index=True)
    lab["played"] = lab["minutes"] > 0
    return lab[["season", "element", "GW", "played"]]


def apply_play_model(model: PlayModel, features: pd.DataFrame, preds: pd.DataFrame,
                     t: int) -> pd.DataFrame:
    """Replace ``preds.p_play`` with the classifier's as-of-t P(play).

    ``preds``: ``element, GW, k, pred, p_play`` (horizon predictions);
    ``features``: the season's pipeline rows.  P(play) at lead k =
    P_model(as-of row) x lookup(k) / lookup(0) (the lookup's decay with k,
    :func:`~fpl_optimizer.prediction.horizon.p_play_from_playing_prob`).
    Elements without an as-of row keep their lookup value.
    """
    from fpl_optimizer.prediction.horizon import asof_rows, p_play_from_playing_prob

    base = asof_rows(features, t)
    if base.empty:
        return preds
    p0 = pd.Series(np.clip(model.predict(base), 0.001, 0.999),
                   index=base["element"].astype(int).to_numpy())
    pp = pd.Series(pd.to_numeric(base.get("playing_prob"), errors="coerce").to_numpy(),
                   index=p0.index)
    out = preds.copy()
    vals = []
    for eid, k, old in zip(out["element"].astype(int), out["k"].astype(int), out["p_play"]):
        if eid not in p0.index:
            vals.append(float(old))
            continue
        ppv = pp.get(eid, np.nan)
        decay = p_play_from_playing_prob(ppv, k) / p_play_from_playing_prob(ppv, 0)
        vals.append(float(np.clip(p0[eid] * decay, 0.001, 0.999)))
    out["p_play"] = vals
    return out
