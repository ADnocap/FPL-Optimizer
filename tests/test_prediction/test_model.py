"""Tests for LightGBM point prediction model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl_optimizer.prediction.model import PointPredictor, POSITIONS


def _make_train_df(n_per_pos: int = 50, n_features: int = 10) -> pd.DataFrame:
    """Create synthetic training data with known patterns."""
    rng = np.random.RandomState(42)
    rows = []
    for pos in POSITIONS:
        for i in range(n_per_pos):
            row = {"position": pos, "target": rng.poisson(3.0)}
            for f in range(n_features):
                row[f"feature_{f}"] = rng.normal(0, 1)
            rows.append(row)
    return pd.DataFrame(rows)


class TestPointPredictor:
    def test_train_returns_per_position_mae(self) -> None:
        df = _make_train_df()
        predictor = PointPredictor(params={"n_estimators": 10, "verbose": -1})
        results = predictor.train(df)

        assert isinstance(results, dict)
        for pos in POSITIONS:
            assert pos in results
            assert results[pos] >= 0.0

    def test_predict_returns_correct_shape(self) -> None:
        df = _make_train_df()
        predictor = PointPredictor(params={"n_estimators": 10, "verbose": -1})
        predictor.train(df)
        preds = predictor.predict(df)

        assert preds.shape == (len(df),)
        assert not np.isnan(preds).any()

    def test_predict_before_train_raises(self) -> None:
        predictor = PointPredictor()
        df = _make_train_df(n_per_pos=5)
        with pytest.raises(RuntimeError, match="not trained"):
            predictor.predict(df)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        df = _make_train_df()
        predictor = PointPredictor(params={"n_estimators": 10, "verbose": -1})
        predictor.train(df)

        model_dir = tmp_path / "model"
        predictor.save(model_dir)

        # Verify files exist
        assert (model_dir / "feature_names.json").exists()
        assert (model_dir / "metadata.json").exists()
        for pos in POSITIONS:
            assert (model_dir / f"{pos}.lgb").exists()

        # Load and predict
        loaded = PointPredictor.load(model_dir)
        preds_original = predictor.predict(df)
        preds_loaded = loaded.predict(df)

        np.testing.assert_allclose(preds_original, preds_loaded, rtol=1e-6)

    def test_train_with_validation(self) -> None:
        train_df = _make_train_df(n_per_pos=100)
        val_df = _make_train_df(n_per_pos=20)
        predictor = PointPredictor(
            params={"n_estimators": 100, "verbose": -1},
            early_stopping_rounds=5,
        )
        results = predictor.train(train_df, val_df)

        assert len(results) == 4

    def test_is_trained_flag(self) -> None:
        predictor = PointPredictor()
        assert not predictor.is_trained

        df = _make_train_df(n_per_pos=20)
        predictor.train(df)
        assert predictor.is_trained

    def test_feature_importance(self) -> None:
        df = _make_train_df()
        predictor = PointPredictor(params={"n_estimators": 10, "verbose": -1})
        predictor.train(df)

        fi = predictor.feature_importance()
        assert not fi.empty
        assert "feature" in fi.columns
        assert "importance" in fi.columns

    def test_predict_unknown_position_gets_default(self) -> None:
        df = _make_train_df(n_per_pos=20)
        predictor = PointPredictor(params={"n_estimators": 10, "verbose": -1})
        predictor.train(df)

        # Add a row with unknown position
        test_row = df.iloc[[0]].copy()
        test_row["position"] = "UNKNOWN"
        preds = predictor.predict(test_row)
        assert preds[0] == pytest.approx(2.0)  # default prediction

    def test_missing_position_in_training(self) -> None:
        """Training with only some positions should still work."""
        df = _make_train_df(n_per_pos=20)
        # Remove GK data
        df = df[df["position"] != "GK"]
        predictor = PointPredictor(params={"n_estimators": 10, "verbose": -1})
        results = predictor.train(df)

        assert "GK" not in results
        assert "DEF" in results
