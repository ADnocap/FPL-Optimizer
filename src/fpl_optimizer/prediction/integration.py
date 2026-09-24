"""Season-wide (element_id, gw) -> predicted points lookup (used by the backtest)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fpl_optimizer.prediction.id_resolver import IDResolver
from fpl_optimizer.prediction.model import PointPredictor
from fpl_optimizer.prediction.feature_pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


class PredictionIntegrator:
    """Pre-computes point predictions for all (element_id, gw) in a season.

    Stores predictions as a dict for O(1) lookup.

    Parameters
    ----------
    predictions : dict[tuple[int, int], float]
        Mapping of ``(element_id, gw) -> predicted_points``.
    """

    def __init__(self, predictions: dict[tuple[int, int], float]) -> None:
        self._predictions = predictions

    def get_predicted_points(self, element_id: int, gw: int) -> float:
        """Look up predicted points for a player in a gameweek.

        Returns 0.0 if no prediction is available.
        """
        return self._predictions.get((element_id, gw), 0.0)

    @classmethod
    def from_model(
        cls,
        model_dir: Path,
        data_dir: Path,
        season: str,
    ) -> PredictionIntegrator:
        """Build integrator by running the model on a full season.

        Parameters
        ----------
        model_dir : Path
            Directory containing saved PointPredictor model files.
        data_dir : Path
            Root data directory.
        season : str
            Season to generate predictions for.

        Returns
        -------
        PredictionIntegrator
            Ready for lookups by the backtest.
        """
        predictor = PointPredictor.load(model_dir)
        id_resolver = IDResolver(data_dir)

        pipeline = FeaturePipeline(data_dir, id_resolver, [season])
        df = pipeline.build()

        if df.empty:
            logger.warning("No feature data for season %s", season)
            return cls({})

        preds = predictor.predict(df)

        predictions: dict[tuple[int, int], float] = {}
        for i, (_, row) in enumerate(df.iterrows()):
            eid = id_resolver.element_id_from_code(int(row["code"]), season)
            if eid is not None:
                predictions[(eid, int(row["GW"]))] = float(preds[i])

        logger.info(
            "PredictionIntegrator: %d predictions for season %s",
            len(predictions), season,
        )
        return cls(predictions)

    def __len__(self) -> int:
        return len(self._predictions)
