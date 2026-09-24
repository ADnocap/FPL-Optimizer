"""Pre-deadline predictions for the upcoming GW of the live season.

Runs the trained LightGBM PointPredictor over the live-built season files
(which include synthetic rows for the upcoming GW — see
``fpl_optimizer.data.collectors.fpl_live``).  DGW players get their per-row
prediction summed; BGW players simply have no row and fall back to 0.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from fpl_optimizer.prediction.feature_pipeline import FeaturePipeline
from fpl_optimizer.prediction.id_resolver import IDResolver
from fpl_optimizer.prediction.model import PointPredictor

logger = logging.getLogger(__name__)


def predict_upcoming_gw(
    data_dir: Path,
    model_dir: Path,
    season: str,
    gw: int,
) -> dict[int, float]:
    """Return element_id -> predicted points for the given upcoming GW."""
    predictor = PointPredictor.load(model_dir)
    id_resolver = IDResolver(data_dir)

    pipeline = FeaturePipeline(data_dir, id_resolver, [season])
    df = pipeline.build()
    if df.empty:
        logger.warning("Live predict: no feature rows for %s", season)
        return {}

    gw_df = df[df["GW"] == gw]
    if gw_df.empty:
        logger.warning(
            "Live predict: no rows for GW%d — was the season rebuilt with "
            "include_upcoming=True?",
            gw,
        )
        return {}

    preds = predictor.predict(gw_df)
    out: dict[int, float] = defaultdict(float)
    for pred, (_, row) in zip(preds, gw_df.iterrows()):
        eid = id_resolver.element_id_from_code(int(row["code"]), season)
        if eid is not None:
            out[eid] += float(pred)  # sums DGW rows if the pipeline emits them

    logger.info("Live predict: %d predictions for GW%d", len(out), gw)
    return dict(out)


def predict_horizon_live(
    data_dir: Path,
    model_dir: Path,
    season: str,
    gw: int,
    horizon: int,
    dgw_mode: str = "blend",
):
    """Predictions for GWs gw..gw+horizon-1 as of the GW ``gw`` deadline.

    Uses the live season files (whose synthetic upcoming-GW rows are the
    as-of rows) and swaps fixture features for later GWs — see
    :mod:`fpl_optimizer.prediction.horizon`.  Returns a DataFrame
    ``element, GW, k, pred, p_play`` (blank GWs absent = 0 points).
    """
    import pandas as pd

    from fpl_optimizer.prediction.horizon import FixtureContext, predict_horizon

    predictor = PointPredictor.load(model_dir)
    id_resolver = IDResolver(data_dir)
    df = FeaturePipeline(data_dir, id_resolver, [season]).build()
    if df.empty or not (df["GW"] == gw).any():
        logger.warning("Live horizon: no feature rows for GW%d", gw)
        return pd.DataFrame(columns=["element", "GW", "k", "pred", "p_play"])
    missing = [c for c in predictor._feature_names if c not in df.columns]
    if missing:
        logger.warning(
            "Live horizon: model features missing from the pipeline output, "
            "served as NaN: %s", missing,
        )
        for c in missing:
            df[c] = float("nan")
    ctx = FixtureContext.from_raw_dir(data_dir / "raw" / season)
    preds = predict_horizon(predictor, df, ctx, gw, horizon, dgw_mode)
    logger.info("Live horizon: %d (element, GW) predictions for GW%d-%d",
                len(preds), gw, gw + horizon - 1)
    return preds


def ep_reference(bootstrap: dict, gw_is_next: bool = True) -> dict[int, float]:
    """FPL's own EP (ep_next pre-deadline) — used for sanity comparison only."""
    field = "ep_next" if gw_is_next else "ep_this"
    out = {}
    for el in bootstrap["elements"]:
        try:
            out[el["id"]] = float(el.get(field) or 0.0)
        except (TypeError, ValueError):
            out[el["id"]] = 0.0
    return out
