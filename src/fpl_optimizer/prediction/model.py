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

# Columns that are NOT features (metadata / target)
_NON_FEATURE_COLS = {"code", "element", "season", "GW", "position", "target", "total_points"}


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

    @property
    def is_trained(self) -> bool:
        return len(self._models) > 0

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

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict total_points for each row.

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
            "positions": list(self._models.keys()),
            "params": self.params,
            "n_features": len(self._feature_names),
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
