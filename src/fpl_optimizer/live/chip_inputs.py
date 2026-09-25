"""Live inputs for the chip evaluation (``scripts/chip_eval.py``).

Glue between the live season files / entry API and
:mod:`fpl_optimizer.optimizer.chip_eval`: the predictor (same loading
interface as the weekly run), the leak guard, horizon predictions with the
DGW blend, the optional P(play) classifier, and an offline entry state.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Double GWs: the horizon predictor's "blend" (mean of the aggregated-row and
# per-fixture-sum predictions; best MAE on 2024-25/2025-26 DGWs).  Summing
# single-fixture predictions over-states DGW players (~+20%).
DGW_MODE = "blend"

# Same-GW xP (vaastav's post-deadline ep_this) is a lookahead leak; the
# leak-fixed pipeline serves only its previous-GW value ``fpl_xp_lag``.
LEAKY_FEATURES = ("fpl_xp",)

UNDERSTAT_FEATURES = (
    "xg_rolling_3", "xg_rolling_5", "xg_rolling_10", "xa_rolling_3", "xa_rolling_5",
    "xa_rolling_10", "npxg_rolling_5", "npxg_rolling_10", "shots_rolling_5",
    "key_passes_rolling_5", "xgchain_rolling_5", "xgchain_rolling_10", "xgbuildup_rolling_5",
)


def load_predictor(model_dir: Path):
    """The point predictor, loaded the way the weekly run loads it.

    Callers only use ``.predict(df)`` and ``._feature_names``.
    """
    from fpl_optimizer.prediction.model import PointPredictor

    return PointPredictor.load(model_dir)


def leaky_features(predictor) -> list[str]:
    """Model features that leak the GW being predicted (exact name match:
    ``fpl_xp`` is leaky, ``fpl_xp_lag`` is not)."""
    return [f for f in getattr(predictor, "_feature_names", []) if f in LEAKY_FEATURES]


def horizon_predictions(data_dir: Path, model_dir: Path, season: str, t: int,
                        horizon: int, predictor=None, play_model=None,
                        health: list[str] | None = None):
    """Horizon predictions GW t..t+horizon-1 (DGW blend) + the feature rows.

    Wraps :func:`fpl_optimizer.live.predict.predict_horizon_live`; with a
    ``play_model`` (:class:`~fpl_optimizer.prediction.play_model.PlayModel`)
    its P(play) replaces the lookup's.  Returns ``(preds, features)``.
    """
    from fpl_optimizer.live.predict import predict_horizon_live

    preds, feats = predict_horizon_live(
        data_dir, model_dir, season, t, horizon, dgw_mode=DGW_MODE, health=health,
        predictor=predictor, return_features=True,
    )
    if play_model is not None and not preds.empty:
        from fpl_optimizer.prediction.play_model import apply_play_model

        preds = apply_play_model(play_model, feats, preds, t)
    return preds, feats


def understat_gap(feature_names: list[str], features: pd.DataFrame, t: int) -> list[str]:
    """Understat features the model uses that are all-NaN in the GW-t rows."""
    if features is None or features.empty or "GW" not in features.columns:
        return []
    rows = features[features["GW"] == t]
    cols = [c for c in UNDERSTAT_FEATURES if c in feature_names]
    return [c for c in cols if c not in rows.columns or rows[c].isna().all()]


def offline_entry_state(entry_dir: Path, bootstrap: dict, team_id: int | None = None):
    """:func:`fetch_entry_state` from saved public-API JSONs (no network)."""
    import json

    from fpl_optimizer.live.entry import fetch_entry_state, offline_fetcher

    if team_id is None:
        entry = json.loads((Path(entry_dir) / "entry.json").read_text(encoding="utf-8"))
        team_id = int(entry.get("id", 0))
    return fetch_entry_state(team_id, bootstrap, fetcher=offline_fetcher(entry_dir))
