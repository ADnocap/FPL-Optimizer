"""Tests for temporal cross-validation and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_optimizer.prediction.evaluation import TemporalCV, ALL_SEASONS, HOLDOUT_SEASON


def _make_cv_df(seasons: list[str], n_per_gw: int = 10, n_gws: int = 20) -> pd.DataFrame:
    """Create synthetic feature DataFrame for CV testing."""
    rng = np.random.RandomState(42)
    rows = []
    positions = ["GK", "DEF", "MID", "FWD"]
    for season in seasons:
        for gw in range(1, n_gws + 1):
            for i in range(n_per_gw):
                row = {
                    "season": season,
                    "GW": gw,
                    "position": positions[i % 4],
                    "target": rng.poisson(3.0),
                    "feature_0": rng.normal(0, 1),
                    "feature_1": rng.normal(0, 1),
                    "feature_2": rng.normal(0, 1),
                }
                rows.append(row)
    return pd.DataFrame(rows)


class TestTemporalCVFolds:
    def test_fold_count(self) -> None:
        """With 5 seasons and min_train=2, expect 3 folds."""
        seasons = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21"]
        df = _make_cv_df(seasons)
        cv = TemporalCV(min_train_seasons=2)
        folds = cv.generate_folds(df)

        assert len(folds) == 3

    def test_no_future_leak(self) -> None:
        """Test season in each fold must come after all training seasons."""
        seasons = ALL_SEASONS[:-1]  # exclude holdout
        df = _make_cv_df(seasons)
        cv = TemporalCV(min_train_seasons=2)
        folds = cv.generate_folds(df)

        for train_df, val_df, test_df in folds:
            train_seasons = set(train_df["season"].unique())
            val_seasons = set(val_df["season"].unique())
            test_season = test_df["season"].unique()

            assert len(test_season) == 1
            test_s = test_season[0]
            test_idx = ALL_SEASONS.index(test_s)

            # All training seasons must precede test season
            for s in train_seasons:
                assert ALL_SEASONS.index(s) < test_idx

            # Validation comes from the training period
            for s in val_seasons:
                assert ALL_SEASONS.index(s) < test_idx

    def test_holdout_excluded(self) -> None:
        """Holdout season should never appear in folds."""
        df = _make_cv_df(ALL_SEASONS)
        cv = TemporalCV()
        folds = cv.generate_folds(df)

        for train_df, val_df, test_df in folds:
            assert HOLDOUT_SEASON not in train_df["season"].values
            assert HOLDOUT_SEASON not in val_df["season"].values
            assert HOLDOUT_SEASON not in test_df["season"].values

    def test_validation_from_end_of_training(self) -> None:
        """Validation set should come from last GWs of last training season."""
        seasons = ["2016-17", "2017-18", "2018-19"]
        df = _make_cv_df(seasons, n_gws=20)
        cv = TemporalCV(min_train_seasons=2)
        folds = cv.generate_folds(df)

        assert len(folds) == 1
        train_df, val_df, test_df = folds[0]

        # Val should be from 2017-18 (last train season), last 8 GWs
        assert set(val_df["season"].unique()) == {"2017-18"}
        assert val_df["GW"].min() > 12  # GWs 13-20 for 20 GW season

    def test_expanding_window(self) -> None:
        """Each successive fold should have more training data."""
        seasons = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21"]
        df = _make_cv_df(seasons)
        cv = TemporalCV(min_train_seasons=2)
        folds = cv.generate_folds(df)

        prev_size = 0
        for train_df, val_df, _ in folds:
            total = len(train_df) + len(val_df)
            assert total > prev_size
            prev_size = total

    def test_empty_folds_when_too_few_seasons(self) -> None:
        """With only 1 season and min_train=2, no folds possible."""
        df = _make_cv_df(["2016-17"])
        cv = TemporalCV(min_train_seasons=2)
        folds = cv.generate_folds(df)

        assert len(folds) == 0


class TestEvaluation:
    def test_evaluate_returns_metrics(self) -> None:
        """Smoke test: evaluate produces expected metric keys."""
        seasons = ["2016-17", "2017-18", "2018-19"]
        df = _make_cv_df(seasons, n_per_gw=10, n_gws=15)
        cv = TemporalCV(min_train_seasons=2)

        results = cv.evaluate(df, params={"n_estimators": 5, "verbose": -1})

        assert "mae" in results
        assert "rmse" in results
        assert "per_position_mae" in results
        assert "fold_results" in results
        assert not np.isnan(results["mae"])
        assert results["mae"] >= 0
        assert len(results["fold_results"]) == 1  # 3 seasons, min_train=2 -> 1 fold

    def test_evaluate_per_position_breakdown(self) -> None:
        seasons = ["2016-17", "2017-18", "2018-19"]
        df = _make_cv_df(seasons, n_per_gw=20, n_gws=15)
        cv = TemporalCV(min_train_seasons=2)

        results = cv.evaluate(df, params={"n_estimators": 5, "verbose": -1})

        for pos in results["per_position_mae"]:
            assert pos in ["GK", "DEF", "MID", "FWD"]
            assert results["per_position_mae"][pos] >= 0
