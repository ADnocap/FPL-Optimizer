"""LightGBM point prediction model — one model per position."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

POSITIONS = ["GK", "DEF", "MID", "FWD"]

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "n_estimators": 500,
    "verbose": -1,
}

# Columns that are NOT features (metadata / target / labels). The minutes
# labels (realised minutes of the row's own GW, from features/minutes_history)
# would be a direct target leak as features.
_NON_FEATURE_COLS = {
    "code", "element", "season", "GW", "position", "target", "total_points",
    "min_total", "min_max", "mcls",
}

# LightGBM aliases of the boosting-round count. In LightGBM 4.x an alias in
# params OVERRIDES lgb.train(num_boost_round=...), so fixed-round refits must
# strip them and pass num_iterations explicitly.
_NUM_ITER_ALIASES = {
    "n_estimators", "num_iterations", "num_iteration", "n_iter", "num_tree",
    "num_trees", "num_round", "num_rounds", "nrounds", "num_boost_round", "max_iter",
}


def fit_calibration(raw: np.ndarray, y: np.ndarray) -> dict:
    """Monotone quadratic map raw prediction -> points, fitted by least squares.

    Robust losses (huber) predict near the conditional median, well below the
    mean of the right-skewed FPL points; the MILP needs points on the real
    scale (hits cost 4, bench EV). ``y = c0 + c1*x + c2*x^2`` is kept strictly
    increasing (ranking unchanged): the quadratic is used only when it curves
    up with its vertex more than 1 point below the smallest raw prediction
    (predictions below the vertex are clamped to it at apply time); otherwise
    a linear fit. Fit on the early-stopping validation split, never on test
    rows.
    """
    raw = np.asarray(raw, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(raw) & np.isfinite(y)
    raw, y = raw[ok], y[ok]
    if len(raw) < 10 or np.ptp(raw) <= 0:
        raise ValueError("calibration needs >= 10 finite, non-constant predictions")
    c2, c1, c0 = np.polyfit(raw, y, 2)
    vertex = None
    if c2 > 0 and -c1 / (2 * c2) < raw.min() - 1.0:
        vertex = float(-c1 / (2 * c2))
    else:
        c2 = 0.0
        c1, c0 = np.polyfit(raw, y, 1)
        if c1 <= 0:
            raise ValueError(f"calibration is not increasing (slope {c1:.3f})")
    return {
        "type": "quad",
        "coef": [float(c2), float(c1), float(c0)],
        "vertex": vertex,
        "n": int(len(raw)),
        "raw_range": [float(raw.min()), float(raw.max())],
    }


def apply_calibration(raw: np.ndarray, calibration: dict | None) -> np.ndarray:
    """Apply a :func:`fit_calibration` map; identity when *calibration* is None."""
    raw = np.asarray(raw, dtype=float)
    if not calibration:
        return raw
    c2, c1, c0 = calibration["coef"]
    vertex = calibration.get("vertex")
    # Below the vertex the parabola would turn back up: clamp (monotone guard)
    x = raw if vertex is None else np.maximum(raw, vertex)
    return c0 + c1 * x + c2 * x * x


class PointPredictor:
    """LightGBM regressor that trains one model per position.

    Parameters
    ----------
    params : dict | None
        LightGBM parameters, merged over :data:`DEFAULT_PARAMS`. Every
        default key — including ``objective`` and ``metric`` — can be
        overridden per instance (e.g. ``{"objective": "tweedie",
        "tweedie_variance_power": 1.3}``).
    early_stopping_rounds : int
        Early stopping patience.
    """

    def __init__(
        self,
        params: dict | None = None,
        early_stopping_rounds: int = 50,
    ) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self._models: dict[str, object] = {}  # position -> lgb.Booster
        self._feature_names: list[str] = []
        # Optional monotone raw -> points map (fit_calibration); None = identity
        self.calibration: dict | None = None

    kind = "point"

    @property
    def is_trained(self) -> bool:
        return len(self._models) > 0

    @property
    def feature_names(self) -> list[str]:
        """Pipeline columns the model reads (serving-health checks)."""
        return list(self._feature_names)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> dict[str, float]:
        """Train one LightGBM model per position.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with feature columns, ``position``, and ``target``.
        val_df : pd.DataFrame | None
            Validation data for early stopping. If None, no early stopping.
        sample_weight : np.ndarray | pd.Series | None
            Optional per-row training weights, aligned **positionally** with
            ``train_df`` (not by index). Sliced per position and passed to
            LightGBM. Validation data stays unweighted so early stopping
            selects on the unweighted metric. If None (default), behavior is
            identical to before this parameter existed.

        Returns
        -------
        dict[str, float]
            Per-position training MAE: ``{"GK": 1.5, "DEF": 1.8, ...}``
        """
        import lightgbm as lgb

        if sample_weight is not None and len(sample_weight) != len(train_df):
            raise ValueError(
                f"sample_weight length {len(sample_weight)} != "
                f"train_df length {len(train_df)}"
            )
        weight_arr = (
            None if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )

        # Determine feature columns
        self._feature_names = [
            c for c in train_df.columns if c not in _NON_FEATURE_COLS
        ]
        logger.info("Training with %d features: %s", len(self._feature_names), self._feature_names[:10])

        results: dict[str, float] = {}

        for pos in POSITIONS:
            pos_mask = (train_df["position"] == pos).to_numpy()
            pos_train = train_df[pos_mask]
            if pos_train.empty:
                logger.warning("No training data for position %s", pos)
                continue

            X_train = pos_train[self._feature_names]
            y_train = pos_train["target"]
            pos_weight = weight_arr[pos_mask] if weight_arr is not None else None

            train_set = lgb.Dataset(X_train, label=y_train, weight=pos_weight)

            callbacks = [lgb.log_evaluation(period=0)]  # suppress output
            valid_sets = [train_set]
            valid_names = ["train"]

            if val_df is not None:
                pos_val = val_df[val_df["position"] == pos]
                if not pos_val.empty:
                    X_val = pos_val[self._feature_names]
                    y_val = pos_val["target"]
                    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
                    valid_sets.append(val_set)
                    valid_names.append("valid")
                    callbacks.append(
                        lgb.early_stopping(self.early_stopping_rounds, verbose=False)
                    )

            model = lgb.train(
                self.params,
                train_set,
                num_boost_round=self.params.get("n_estimators", 500),
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )

            self._models[pos] = model

            # Training MAE
            preds = model.predict(X_train)
            mae = float(np.mean(np.abs(preds - y_train)))
            results[pos] = mae
            logger.info("  %s: %d samples, train MAE=%.3f", pos, len(pos_train), mae)

        return results

    def best_iterations(self) -> dict[str, int]:
        """Per-position boosting rounds in use (the early-stopped best)."""
        return {
            pos: int(b.best_iteration or b.current_iteration())
            for pos, b in self._models.items()
        }

    def train_fixed_rounds(
        self,
        train_df: pd.DataFrame,
        rounds: dict[str, int],
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> None:
        """Retrain each position booster for exactly ``rounds[pos]`` iterations.

        No validation / early stopping: the optional refit of the in-season
        recipe (train+val at the early-stopped best iteration). Keeps the
        current :attr:`calibration`.
        """
        import lightgbm as lgb

        self._feature_names = [c for c in train_df.columns if c not in _NON_FEATURE_COLS]
        base = {k: v for k, v in self.params.items() if k not in _NUM_ITER_ALIASES}
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        self._models = {}
        for pos in POSITIONS:
            m = (train_df["position"] == pos).to_numpy()
            if not m.any() or pos not in rounds:
                continue
            ds = lgb.Dataset(
                train_df.loc[m, self._feature_names], label=train_df.loc[m, "target"],
                weight=None if w is None else w[m],
            )
            self._models[pos] = lgb.train(
                {**base, "num_iterations": max(1, int(rounds[pos]))}, ds,
            )

    def refit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict[str, int]:
        """Retrain on train+val at the early-stopped best iterations."""
        best = self.best_iterations()
        self.train_fixed_rounds(pd.concat([train_df, val_df]), best)
        return best

    def fit_calibration(self, val_df: pd.DataFrame) -> dict:
        """Fit :attr:`calibration` on held-out rows (the early-stopping split)."""
        raw = self.predict_raw(val_df)
        self.calibration = fit_calibration(raw, val_df["target"].to_numpy(float))
        return self.calibration

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted points: the raw model output through :attr:`calibration`
        (identity when there is none, e.g. every model saved before it existed).
        """
        return apply_calibration(self.predict_raw(df), self.calibration)

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Predict total_points for each row, uncalibrated.

        Routes each row to the position-specific model. Rows with unknown
        position get prediction 2.0 (league average).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain feature columns and ``position``.

        Returns
        -------
        np.ndarray
            Predicted points, shape ``(len(df),)``.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        preds = np.full(len(df), 2.0, dtype=np.float64)

        for pos in POSITIONS:
            if pos not in self._models:
                continue
            mask = df["position"] == pos
            if not mask.any():
                continue
            X = df.loc[mask, self._feature_names]
            preds[mask.values] = self._models[pos].predict(X)

        return preds

    def save(self, model_dir: Path) -> None:
        """Save all position models and metadata.

        Creates:
            {model_dir}/{position}.lgb
            {model_dir}/feature_names.json
            {model_dir}/metadata.json
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        for pos, model in self._models.items():
            model.save_model(str(model_dir / f"{pos}.lgb"))

        with open(model_dir / "feature_names.json", "w") as f:
            json.dump(self._feature_names, f)

        metadata = {
            "kind": self.kind,
            "positions": list(self._models.keys()),
            "params": self.params,
            "n_features": len(self._feature_names),
            "calibration": self.calibration,
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved %d models to %s", len(self._models), model_dir)

    @classmethod
    def load(cls, model_dir: Path) -> PointPredictor:
        """Load a saved PointPredictor from disk.

        Parameters
        ----------
        model_dir : Path
            Directory containing saved model files.

        Returns
        -------
        PointPredictor
            Loaded predictor ready for inference.
        """
        import lightgbm as lgb

        model_dir = Path(model_dir)

        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)

        with open(model_dir / "feature_names.json") as f:
            feature_names = json.load(f)

        predictor = cls(params=metadata.get("params", {}))
        predictor._feature_names = feature_names
        # Models saved before calibration existed have no key -> identity
        predictor.calibration = metadata.get("calibration")

        for pos in metadata.get("positions", []):
            model_path = model_dir / f"{pos}.lgb"
            if model_path.exists():
                predictor._models[pos] = lgb.Booster(model_file=str(model_path))

        logger.info(
            "Loaded %d models from %s (%d features)",
            len(predictor._models), model_dir, len(feature_names),
        )
        return predictor

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Get feature importance across all position models.

        Returns
        -------
        pd.DataFrame
            Columns: feature, importance, position.
        """
        rows = []
        for pos, model in self._models.items():
            importances = model.feature_importance(importance_type=importance_type)
            for name, imp in zip(self._feature_names, importances):
                rows.append({"feature": name, "importance": imp, "position": pos})

        df = pd.DataFrame(rows)
        if not df.empty:
            # Average across positions
            avg = df.groupby("feature", as_index=False)["importance"].mean()
            avg = avg.sort_values("importance", ascending=False)
            return avg
        return df
