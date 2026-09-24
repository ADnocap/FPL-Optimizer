"""Tests for PredictionIntegrator."""

from __future__ import annotations

import pytest

from fpl_optimizer.prediction.integration import PredictionIntegrator


class TestPredictionIntegrator:
    def test_lookup_existing_key(self) -> None:
        predictions = {(10, 1): 5.5, (10, 2): 3.2, (20, 1): 2.1}
        integrator = PredictionIntegrator(predictions)

        assert integrator.get_predicted_points(10, 1) == pytest.approx(5.5)
        assert integrator.get_predicted_points(20, 1) == pytest.approx(2.1)

    def test_lookup_missing_key_returns_zero(self) -> None:
        predictions = {(10, 1): 5.5}
        integrator = PredictionIntegrator(predictions)

        assert integrator.get_predicted_points(99, 1) == 0.0
        assert integrator.get_predicted_points(10, 99) == 0.0

    def test_len(self) -> None:
        predictions = {(10, 1): 5.5, (10, 2): 3.2}
        integrator = PredictionIntegrator(predictions)
        assert len(integrator) == 2

    def test_empty_predictions(self) -> None:
        integrator = PredictionIntegrator({})
        assert integrator.get_predicted_points(1, 1) == 0.0
        assert len(integrator) == 0

