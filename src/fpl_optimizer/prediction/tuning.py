"""Random search hyperparameter tuning for LightGBM point prediction model."""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import pandas as pd

from fpl_optimizer.prediction.evaluation import TemporalCV
from fpl_optimizer.prediction.model import DEFAULT_PARAMS

logger = logging.getLogger(__name__)

# Search space for random search
SEARCH_SPACE: dict[str, list[Any]] = {
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "min_child_samples": [10, 20, 50, 100],
    "feature_fraction": [0.6, 0.8, 1.0],
    "lambda_l1": [0.0, 0.1, 1.0],
    "lambda_l2": [0.0, 0.1, 1.0],
    "n_estimators": [500, 1000, 2000],
}


def _sample_params(search_space: dict[str, list], rng: random.Random) -> dict:
    """Sample one random configuration from the search space."""
    return {key: rng.choice(values) for key, values in search_space.items()}


def random_search(
    df: pd.DataFrame,
    n_trials: int = 50,
    last_n_folds: int = 3,
    search_space: dict[str, list] | None = None,
    seed: int = 42,
) -> dict:
    """Run random search hyperparameter tuning using temporal CV.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature DataFrame with ``season``, ``position``, ``target``.
    n_trials : int
        Number of random parameter configurations to try.
    last_n_folds : int
        Use only the last N folds for speed. Set to 0 to use all folds.
    search_space : dict | None
        Custom search space. Defaults to :data:`SEARCH_SPACE`.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``best_params``, ``best_mae``, ``all_trials``.
    """
    if search_space is None:
        search_space = SEARCH_SPACE

    rng = random.Random(seed)

    # Pre-generate folds once
    cv = TemporalCV()
    all_folds = cv.generate_folds(df)

    if not all_folds:
        logger.warning("No CV folds generated — cannot tune")
        return {"best_params": {}, "best_mae": float("nan"), "all_trials": []}

    if last_n_folds > 0:
        folds_to_use = all_folds[-last_n_folds:]
    else:
        folds_to_use = all_folds

    logger.info(
        "Tuning: %d trials, %d folds (of %d total)",
        n_trials, len(folds_to_use), len(all_folds),
    )

    best_mae = float("inf")
    best_params: dict = {}
    all_trials: list[dict] = []

    for trial_idx in range(n_trials):
        sampled = _sample_params(search_space, rng)
        params = {**DEFAULT_PARAMS, **sampled}

        fold_maes: list[float] = []

        for train_df, val_df, test_df in folds_to_use:
            from fpl_optimizer.prediction.model import PointPredictor

            predictor = PointPredictor(params=params)
            predictor.train(train_df, val_df if not val_df.empty else None)
            preds = predictor.predict(test_df)
            actuals = test_df["target"].values
            fold_mae = float(np.mean(np.abs(preds - actuals)))
            fold_maes.append(fold_mae)

        mean_mae = float(np.mean(fold_maes))

        trial_result = {
            "trial": trial_idx + 1,
            "params": sampled,
            "mae": mean_mae,
            "fold_maes": fold_maes,
        }
        all_trials.append(trial_result)

        if mean_mae < best_mae:
            best_mae = mean_mae
            best_params = sampled
            logger.info(
                "  Trial %d/%d: MAE=%.4f (NEW BEST) — %s",
                trial_idx + 1, n_trials, mean_mae, sampled,
            )
        else:
            logger.debug(
                "  Trial %d/%d: MAE=%.4f — %s",
                trial_idx + 1, n_trials, mean_mae, sampled,
            )

    logger.info("Best MAE=%.4f with params: %s", best_mae, best_params)

    return {
        "best_params": best_params,
        "best_mae": best_mae,
        "all_trials": all_trials,
    }
