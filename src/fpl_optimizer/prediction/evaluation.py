"""Temporal cross-validation and evaluation metrics for point prediction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fpl_optimizer.prediction.model import PointPredictor, POSITIONS

logger = logging.getLogger(__name__)

# Seasons in chronological order
ALL_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25",
]

# Holdout season — never used during development
HOLDOUT_SEASON = "2024-25"

# Number of GWs at the end of training data used for early-stopping validation
VALIDATION_GWS = 8


class TemporalCV:
    """Expanding-window temporal cross-validation.

    Fold structure (training always starts from the earliest season):
      Fold 1: train 2016-18, test 2018-19
      Fold 2: train 2016-19, test 2019-20
      ...
      Fold 6: train 2016-23, test 2023-24

    2024-25 is held out and never used.
    """

    def __init__(
        self,
        min_train_seasons: int = 2,
        holdout: str = HOLDOUT_SEASON,
    ) -> None:
        self.min_train_seasons = min_train_seasons
        self.holdout = holdout

    def generate_folds(
        self, df: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Generate (train, val, test) splits.

        Parameters
        ----------
        df : pd.DataFrame
            Full feature DataFrame with ``season`` and ``GW`` columns.

        Returns
        -------
        list of (train_df, val_df, test_df)
            Each tuple is one fold. ``val_df`` is the last
            :data:`VALIDATION_GWS` of the training period (for early stopping).
        """
        available = [s for s in ALL_SEASONS if s != self.holdout and s in df["season"].unique()]
        available.sort(key=ALL_SEASONS.index)

        folds = []
        for i in range(self.min_train_seasons, len(available)):
            train_seasons = available[:i]
            test_season = available[i]

            train_full = df[df["season"].isin(train_seasons)]
            test_df = df[df["season"] == test_season]

            if train_full.empty or test_df.empty:
                continue

            # Split last VALIDATION_GWS from last training season for val
            last_train_season = train_seasons[-1]
            last_season_data = train_full[train_full["season"] == last_train_season]
            max_gw = last_season_data["GW"].max()
            val_cutoff = max_gw - VALIDATION_GWS

            val_mask = (
                (train_full["season"] == last_train_season)
                & (train_full["GW"] > val_cutoff)
            )
            val_df = train_full[val_mask]
            train_df = train_full[~val_mask]

            folds.append((train_df, val_df, test_df))
            logger.info(
                "Fold %d: train=%s (%d rows), val=%d rows, test=%s (%d rows)",
                len(folds), train_seasons, len(train_df), len(val_df),
                test_season, len(test_df),
            )

        return folds

    def evaluate(
        self,
        df: pd.DataFrame,
        params: dict | None = None,
    ) -> dict:
        """Run temporal CV and return aggregated metrics.

        Parameters
        ----------
        df : pd.DataFrame
            Full feature DataFrame.
        params : dict | None
            LightGBM parameters passed to PointPredictor.

        Returns
        -------
        dict
            Keys: ``mae``, ``rmse``, ``per_position_mae``, ``fold_results``.
        """
        folds = self.generate_folds(df)
        if not folds:
            logger.warning("No folds generated")
            return {"mae": float("nan"), "rmse": float("nan"), "per_position_mae": {}, "fold_results": []}

        fold_results = []
        all_errors = []
        per_pos_errors: dict[str, list[float]] = {p: [] for p in POSITIONS}

        for i, (train_df, val_df, test_df) in enumerate(folds):
            predictor = PointPredictor(params=params)
            predictor.train(train_df, val_df if not val_df.empty else None)

            preds = predictor.predict(test_df)
            actuals = test_df["target"].values
            errors = np.abs(preds - actuals)

            fold_mae = float(errors.mean())
            fold_rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))

            all_errors.extend(errors.tolist())

            # Per-position breakdown
            pos_mae = {}
            for pos in POSITIONS:
                mask = test_df["position"].values == pos
                if mask.any():
                    pos_errors = errors[mask]
                    pos_mae[pos] = float(pos_errors.mean())
                    per_pos_errors[pos].extend(pos_errors.tolist())

            test_season = test_df["season"].iloc[0]
            fold_results.append({
                "fold": i + 1,
                "test_season": test_season,
                "mae": fold_mae,
                "rmse": fold_rmse,
                "per_position_mae": pos_mae,
                "n_test": len(test_df),
            })

            logger.info(
                "  Fold %d (%s): MAE=%.3f, RMSE=%.3f",
                i + 1, test_season, fold_mae, fold_rmse,
            )

        overall_mae = float(np.mean(all_errors))
        overall_rmse = float(np.sqrt(np.mean(np.array(all_errors) ** 2)))

        overall_pos_mae = {}
        for pos in POSITIONS:
            if per_pos_errors[pos]:
                overall_pos_mae[pos] = float(np.mean(per_pos_errors[pos]))

        return {
            "mae": overall_mae,
            "rmse": overall_rmse,
            "per_position_mae": overall_pos_mae,
            "fold_results": fold_results,
        }
