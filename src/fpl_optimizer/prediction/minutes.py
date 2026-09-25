"""Minutes model + minutes-aware point predictor.

``MinutesModel``
    LightGBM multiclass P(0 min) / P(1-59) / P(60+) from pre-match features
    only (:data:`MM_BASE` + ``minutes_history.MH_FEATURES`` + position).

``MinutesBlendPredictor`` (the 2026-09-24 experiment winner "blend_fact_mh+stack")
    ``w * [P60 * E[pts|60+] + P(1-59) * E[pts|1-59]]``   factorized: two
    conditional PointPredictors on points features + minutes history, trained
    on rows selected by the REALISED minutes class (target structure only; at
    predict time they are weighted by PREDICTED probabilities);
    ``+ (1 - w) * PointPredictor(points features + mm_pplay, mm_p60)`` stacked
    (no minutes-history features).
    The stacked model's TRAINING rows get temporally nested minutes
    predictions (:func:`expanding_minutes_oof`: season S predicted by a
    MinutesModel fit on seasons < S only), so no training row ever sees its own
    label; serve-time probabilities come from the MinutesModel fit on every
    training row. An optional monotone calibration (``model.fit_calibration``)
    maps the FINAL blended output to points.

Same interface as :class:`~fpl_optimizer.prediction.model.PointPredictor`
(``train`` / ``predict`` / ``save`` / ``load`` / ``feature_names``);
:func:`load_predictor` loads either kind from a model dir.

Serve-time availability: keep ``pool.build_live_candidates(availability_scaling=True)``
— multiplying the final E[pts] by chance_of_playing was the best-calibrated
rule on the 2026-27 flagged players (the model's P(play) and FPL's flag are
close to independent).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_optimizer.prediction.features.minutes_history import MH_FEATURES, MINUTES_LABELS
from fpl_optimizer.prediction.model import (
    _NON_FEATURE_COLS,
    PointPredictor,
    apply_calibration,
    fit_calibration,
)

logger = logging.getLogger(__name__)

POS_CODE = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
# Minutes-relevant subset of the point features (kept small so the 10 nested
# fits stay cheap; a full-feature minutes model was no better in 2017-19 folds).
MM_BASE = [
    "pts_rolling_3", "pts_rolling_5", "pts_rolling_10", "mins_rolling_3", "mins_rolling_5",
    "mins_std_5", "starts_rolling_5", "season_avg_pts", "season_total_mins", "games_played",
    "value", "selected_norm", "value_momentum", "selected_momentum", "minutes_share_5",
    "transfers_balance_rolling_3", "transfers_in_rolling_3", "transfers_out_rolling_3",
    "transfer_ratio_3", "ict_rolling_5", "bps_rolling_5", "yellows_rolling_5", "reds_rolling_10",
    "prev_minutes", "was_home", "is_dgw", "odds_team_strength", "opp_strength", "fpl_xp_lag",
    "synthetic_ep", "playing_prob", "gw_phase", "nailedness", "set_piece_order_sum",
    "is_penalty_taker",
]
MM_PARAMS = {
    "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
    "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "lambda_l2": 1.0, "max_bin": 127, "verbose": -1, "seed": 42, "num_threads": 4,
}
# Early-stopped once (train <= 2022-23, val 2023-24): point-in-time for every
# holdout used since, so a fixed count needs no validation split.
MM_ROUNDS = 223
STACK = ["mm_pplay", "mm_p60"]
OOF_KEY = ["season", "element", "GW"]

if not set(MINUTES_LABELS) <= _NON_FEATURE_COLS:  # pragma: no cover - import guard
    raise ImportError("model._NON_FEATURE_COLS must contain the minutes labels")


class MinutesModel:
    """Multiclass minutes model: P(0), P(1-59), P(60+) per row."""

    def __init__(self, n_rounds: int = MM_ROUNDS, params: dict | None = None) -> None:
        self.n_rounds = int(n_rounds)
        self.params = {**MM_PARAMS, **(params or {})}
        self.features = MM_BASE + MH_FEATURES + ["pos_code"]
        self._booster = None

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.reindex(columns=[c for c in self.features if c != "pos_code"])
        X = X.assign(pos_code=df["position"].map(POS_CODE))
        return X[self.features].apply(pd.to_numeric, errors="coerce").astype("float32")

    def train(self, df: pd.DataFrame) -> "MinutesModel":
        import lightgbm as lgb

        # Only columns the frame has (a recipe may exclude some features)
        self.features = [c for c in MM_BASE + MH_FEATURES if c in df.columns] + ["pos_code"]
        lab = df["mcls"].notna().to_numpy()
        ds = lgb.Dataset(self._X(df[lab]), label=df.loc[lab, "mcls"].astype(int))
        params = {k: v for k, v in self.params.items() if k != "n_estimators"}
        self._booster = lgb.train({**params, "num_iterations": self.n_rounds}, ds)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """(n, 3) array: P(0), P(1-59), P(60+)."""
        if self._booster is None:
            raise RuntimeError("MinutesModel not trained")
        return self._booster.predict(self._X(df))

    def save(self, d: Path) -> None:
        d = Path(d)
        self._booster.save_model(str(d / "minutes.lgb"))
        (d / "minutes_features.json").write_text(json.dumps(self.features), encoding="utf-8")

    @classmethod
    def load(cls, d: Path, n_rounds: int = MM_ROUNDS, params: dict | None = None) -> "MinutesModel":
        import lightgbm as lgb

        d = Path(d)
        m = cls(n_rounds, params)
        m._booster = lgb.Booster(model_file=str(d / "minutes.lgb"))
        m.features = json.loads((d / "minutes_features.json").read_text(encoding="utf-8"))
        return m


def _probs_frame(p: np.ndarray, index) -> pd.DataFrame:
    return pd.DataFrame({"mm_pplay": 1.0 - p[:, 0], "mm_p60": p[:, 2]}, index=index)


def expanding_minutes_oof(
    df: pd.DataFrame, n_rounds: int = MM_ROUNDS, params: dict | None = None,
) -> pd.DataFrame:
    """Temporally nested minutes probabilities for every row of *df*.

    Rows of season S come from a MinutesModel trained on seasons < S only
    (the first season -> NaN). Returns ``mm_pplay`` / ``mm_p60`` aligned with
    ``df.index``. The result for season S depends only on rows of seasons
    <= S, so one call on a superset frame serves every training phase.
    """
    out = pd.DataFrame({"mm_pplay": np.nan, "mm_p60": np.nan}, index=df.index)
    seasons = sorted(df["season"].unique())
    for i, s in enumerate(seasons[1:], start=1):
        mm = MinutesModel(n_rounds, params).train(df[df["season"].isin(seasons[:i])])
        m = (df["season"] == s).to_numpy()
        out.loc[m, ["mm_pplay", "mm_p60"]] = _probs_frame(
            mm.predict_proba(df[m]), df.index[m]).to_numpy()
        logger.info("minutes OOF: %s from %d earlier seasons", s, i)
    return out


def _cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """Columns handed to PointPredictor.train (it trains on every non-meta column)."""
    return [c for c in df.columns if c not in exclude]


class MinutesBlendPredictor:
    """``w_fact * factorized + (1 - w_fact) * stacked`` (see module docstring)."""

    kind = "minutes_blend"
    _COMPONENTS = ("e60", "esub", "stack")

    def __init__(
        self,
        params: dict | None = None,
        w_fact: float = 0.5,
        early_stopping_rounds: int = 50,
        mm_rounds: int = MM_ROUNDS,
        mm_params: dict | None = None,
    ) -> None:
        self.params = dict(params or {})
        self.w_fact = float(w_fact)
        self.early_stopping_rounds = early_stopping_rounds
        self.mm_rounds = int(mm_rounds)
        self.mm_params = dict(mm_params or {})
        self.minutes: MinutesModel | None = None
        self.e60: PointPredictor | None = None
        self.esub: PointPredictor | None = None
        self.stack: PointPredictor | None = None
        self.calibration: dict | None = None
        # val rows' nested minutes probabilities, kept from train() for
        # fit_calibration (transient; not saved)
        self._val_oof: pd.DataFrame | None = None

    # -- training -------------------------------------------------------------

    def _oof(self, full: pd.DataFrame, minutes_oof: pd.DataFrame | None) -> pd.DataFrame:
        if minutes_oof is None:
            return expanding_minutes_oof(full, self.mm_rounds, self.mm_params)
        got = full[OOF_KEY].merge(minutes_oof[OOF_KEY + STACK], on=OOF_KEY,
                                  how="left", indicator=True)
        missing = int((got["_merge"] != "both").sum())
        if missing:
            raise ValueError(f"minutes_oof lacks {missing} training rows")
        return pd.DataFrame(got[STACK].to_numpy(), index=full.index, columns=STACK)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        minutes_oof: pd.DataFrame | None = None,
    ) -> dict:
        """Fit the minutes model, the two conditionals and the stacked model.

        *minutes_oof* (optional): ``season, element, GW, mm_pplay, mm_p60``
        from :func:`expanding_minutes_oof` on a superset of the rows —
        computed once and reused across training phases. Season S's values
        only depend on the rows of seasons < S, so they are the same nested
        predictions (bit-identical when those rows come in the same order).
        """
        full = pd.concat([train_df, val_df]) if val_df is not None else train_df
        n_tr = len(train_df)
        # 1. serve-time minutes model: every labelled training row
        self.minutes = MinutesModel(self.mm_rounds, self.mm_params).train(full)
        # 2. factorized conditionals (minutes labels only select rows)
        report = {}
        for cls, attr in [(2, "e60"), (1, "esub")]:
            tr = train_df[train_df["mcls"] == cls]
            va = None if val_df is None else val_df[val_df["mcls"] == cls]
            m = PointPredictor(params=self.params, early_stopping_rounds=self.early_stopping_rounds)
            report[attr] = m.train(tr[_cols(tr, set(STACK))],
                                   None if va is None else va[_cols(va, set(STACK))])
            setattr(self, attr, m)
        # 3. stacked model: nested minutes probabilities, no minutes-history features
        oof = self._oof(full, minutes_oof)
        stacked = full.drop(columns=[c for c in MH_FEATURES + STACK if c in full.columns])
        stacked = stacked.assign(mm_pplay=oof["mm_pplay"].to_numpy(),
                                 mm_p60=oof["mm_p60"].to_numpy())
        self.stack = PointPredictor(params=self.params,
                                    early_stopping_rounds=self.early_stopping_rounds)
        report["stack"] = self.stack.train(
            stacked.iloc[:n_tr], None if val_df is None else stacked.iloc[n_tr:])
        self._val_oof = None if val_df is None else oof.iloc[n_tr:]
        self.calibration = None
        return report

    def best_iterations(self) -> dict[str, dict[str, int]]:
        return {name: getattr(self, name).best_iterations() for name in self._COMPONENTS}

    def refit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        minutes_oof: pd.DataFrame | None = None,
    ) -> dict[str, dict[str, int]]:
        """Retrain the three point models on train+val at their early-stopped
        best iterations (the minutes model already saw every row). Keeps the
        calibration fitted before the refit."""
        best = self.best_iterations()
        full = pd.concat([train_df, val_df])
        for cls, attr in [(2, "e60"), (1, "esub")]:
            sub = full[full["mcls"] == cls]
            getattr(self, attr).train_fixed_rounds(sub[_cols(sub, set(STACK))], best[attr])
        oof = self._oof(full, minutes_oof)
        stacked = full.drop(columns=[c for c in MH_FEATURES + STACK if c in full.columns])
        stacked = stacked.assign(mm_pplay=oof["mm_pplay"].to_numpy(),
                                 mm_p60=oof["mm_p60"].to_numpy())
        self.stack.train_fixed_rounds(stacked, best["stack"])
        return best

    def fit_calibration(self, val_df: pd.DataFrame) -> dict:
        """Monotone map of the FINAL blended output -> points, fit on *val_df*.

        The serve-time minutes model was trained on the val rows, so its val
        probabilities are in-sample; the val rows' nested (out-of-sample)
        probabilities from train() are used instead when available, which is
        how every test/live row is predicted.
        """
        p = None
        if self._val_oof is not None and len(self._val_oof) == len(val_df):
            pplay = self._val_oof["mm_pplay"].to_numpy(float)
            p60 = self._val_oof["mm_p60"].to_numpy(float)
            p = np.column_stack([1.0 - pplay, pplay - p60, p60])
        raw = self._blend(val_df, p)
        self.calibration = fit_calibration(raw, val_df["target"].to_numpy(float))
        return self.calibration

    # -- prediction -----------------------------------------------------------

    def _blend(self, df: pd.DataFrame, p: np.ndarray | None = None) -> np.ndarray:
        if p is None:
            p = self.minutes.predict_proba(df)
        fact = p[:, 2] * self.e60.predict(df) + p[:, 1] * self.esub.predict(df)
        stack = self.stack.predict(df.assign(mm_pplay=1.0 - p[:, 0], mm_p60=p[:, 2]))
        return self.w_fact * fact + (1.0 - self.w_fact) * stack

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Uncalibrated blend."""
        if self.minutes is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self._blend(df)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted points per row (blend through the calibration, if any)."""
        return apply_calibration(self.predict_raw(df), self.calibration)

    def predict_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-row P(play) / P(60+) / E[pts|60+] / E[pts|1-59] (raw scale)."""
        p = self.minutes.predict_proba(df)
        return pd.DataFrame({"p_play": 1 - p[:, 0], "p60": p[:, 2],
                             "e60": self.e60.predict(df), "esub": self.esub.predict(df)},
                            index=df.index)

    @property
    def is_trained(self) -> bool:
        return self.minutes is not None and self.stack is not None

    @property
    def feature_names(self) -> list[str]:
        """Every pipeline column any component reads (mm_* are computed inside)."""
        names: list[str] = []
        comps = [getattr(self, c) for c in self._COMPONENTS if getattr(self, c) is not None]
        for comp in comps:
            names += [f for f in comp.feature_names if f not in STACK]
        if self.minutes is not None:
            names += [f for f in self.minutes.features if f != "pos_code"]
        return list(dict.fromkeys(names))

    # PointPredictor-compatible alias used by older scripts
    _feature_names = feature_names

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Mean gain across the three point models (share-normalised)."""
        frames = []
        for name in self._COMPONENTS:
            fi = getattr(self, name).feature_importance(importance_type)
            if not fi.empty:
                fi = fi.assign(importance=fi["importance"] / fi["importance"].sum())
                frames.append(fi)
        if not frames:
            return pd.DataFrame()
        fi = pd.concat(frames).groupby("feature", as_index=False)["importance"].mean()
        return fi.sort_values("importance", ascending=False).reset_index(drop=True)

    # -- persistence ----------------------------------------------------------

    def save(self, d: Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        self.minutes.save(d)
        for name in self._COMPONENTS:
            getattr(self, name).save(d / name)
        (d / "feature_names.json").write_text(json.dumps(self.feature_names), encoding="utf-8")
        meta = {
            "kind": self.kind,
            "w_fact": self.w_fact,
            "mm_rounds": self.mm_rounds,
            "mm_params": self.minutes.params,
            "params": self.params,
            "early_stopping_rounds": self.early_stopping_rounds,
            "calibration": self.calibration,
            "components": list(self._COMPONENTS),
            "n_features": len(self.feature_names),
        }
        (d / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, d: Path) -> "MinutesBlendPredictor":
        d = Path(d)
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        m = cls(params=meta.get("params"), w_fact=meta["w_fact"],
                early_stopping_rounds=meta.get("early_stopping_rounds", 50),
                mm_rounds=meta.get("mm_rounds", MM_ROUNDS), mm_params=meta.get("mm_params"))
        m.minutes = MinutesModel.load(d, m.mm_rounds, meta.get("mm_params"))
        for name in cls._COMPONENTS:
            setattr(m, name, PointPredictor.load(d / name))
        m.calibration = meta.get("calibration")
        return m


def load_predictor(model_dir: Path):
    """Load a saved predictor of either kind.

    Dispatches on ``metadata.json["kind"]``: ``"minutes_blend"`` ->
    :class:`MinutesBlendPredictor`; ``"point"`` or no kind (every model saved
    before kinds existed, e.g. models/prod_2026-27) -> PointPredictor.
    """
    meta = json.loads((Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))
    kind = meta.get("kind", "point")
    if kind == MinutesBlendPredictor.kind:
        return MinutesBlendPredictor.load(model_dir)
    if kind == PointPredictor.kind:
        return PointPredictor.load(model_dir)
    raise ValueError(f"unknown predictor kind {kind!r} in {model_dir}")
