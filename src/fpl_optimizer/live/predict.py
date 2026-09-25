"""Pre-deadline predictions for the upcoming GW of the live season.

Runs the trained predictor (any kind — ``minutes.load_predictor``: a
PointPredictor or the MinutesBlendPredictor) over the live-built season files
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
from fpl_optimizer.prediction.minutes import load_predictor

logger = logging.getLogger(__name__)


def _fill_missing_features(df, feature_names: list[str], where: str):
    """Serve model features the pipeline no longer emits as NaN (logged).

    Keeps an older model (e.g. a ``.prev`` rollback trained on a since-renamed
    feature) servable instead of crashing; serving_health reports the gap.
    """
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        logger.warning(
            "%s: model features missing from the pipeline output, served as NaN: %s",
            where, missing,
        )
        df = df.copy()
        for c in missing:
            df[c] = float("nan")
    return df

# Written next to the model by scripts/train_predictor.py: per-feature
# non-null rate and std on recent training rows of players who are playing.
SERVING_REFERENCE_FILE = "serving_reference.json"


def _regular_rows(df):
    """Rows of players who have been playing (the rows decisions depend on)."""
    if "mins_rolling_3" in df.columns:
        reg = df[df["mins_rolling_3"].fillna(0) >= 30]
        if len(reg) >= 50:
            return reg
    return df


EARLY_GW = 7  # GWs whose rolling windows are still filling


def serving_reference(df, feature_names: list[str]) -> dict[str, dict[str, float]]:
    """Per-feature non-null rate / std on *df* (training side of the check).

    With ``season``/``GW`` columns it also records ``const_gw_share``: the
    share of training GWs in which the feature is constant across players
    (1.0 for gw_phase, ~0.9 for is_dgw). A feature that is usually constant
    within a GW is not flagged for being constant at a deadline.
    """
    reg = _regular_rows(df)
    ref = {}
    groups = None
    if {"season", "GW"} <= set(reg.columns):
        groups = reg.groupby(["season", "GW"])
    for f in feature_names:
        if f in reg.columns:
            col = reg[f]
            ref[f] = {
                "nonnull": float(col.notna().mean()),
                "std": float(col.std()) if col.notna().sum() > 1 else 0.0,
            }
            if groups is not None:
                const = groups[f].nunique() <= 1
                ref[f]["const_gw_share"] = float(const.mean())
                # early season separately: windowed deltas (bps_form_delta)
                # are constant until ~GW7 in training too
                early = const[const.index.get_level_values("GW") <= EARLY_GW]
                if len(early):
                    ref[f]["const_gw_share_early"] = float(early.mean())
    return ref


def serving_health(gw_df, feature_names: list[str], reference: dict | None) -> list[str]:
    """Warnings for features the model will see empty/constant at this deadline.

    Silent train/serve gaps (understat never collected, odds file missing,
    team strengths published as 0) degraded the whole 2026-27 GW1-5 run
    without a single error. Compares the upcoming GW's rows with the
    training reference; returns one line per suspicious feature.
    """
    reg = _regular_rows(gw_df)
    warnings = []
    missing = [f for f in feature_names if f not in gw_df.columns]
    if missing:
        warnings.append(f"{len(missing)} model features absent from the pipeline: {missing}")
    early = "GW" in gw_df.columns and len(gw_df) and int(gw_df["GW"].max()) <= EARLY_GW
    for f in feature_names:
        if f not in reg.columns:
            continue
        live_nn = float(reg[f].notna().mean())
        ref = (reference or {}).get(f)
        if ref is None:
            continue
        const_share = ref.get("const_gw_share_early" if early else "const_gw_share",
                              ref.get("const_gw_share", 0.0))
        if ref["nonnull"] >= 0.5 and live_nn < ref["nonnull"] - 0.3:
            warnings.append(
                f"{f}: {live_nn:.0%} populated live vs {ref['nonnull']:.0%} in training"
            )
        elif ref["nonnull"] >= 0.2 and live_nn == 0.0:
            # sparse-in-training sources that vanished entirely (e.g. the
            # player-props snapshot failed: all props_* NaN this GW)
            warnings.append(
                f"{f}: EMPTY live vs {ref['nonnull']:.0%} in training (source missing?)"
            )
        elif (live_nn > 0.5 and ref["std"] > 0 and reg[f].nunique() <= 1
              and const_share < 0.5):
            warnings.append(f"{f}: constant live ({reg[f].dropna().iloc[0]!r}), varies in training")
    return warnings


def _collect_extras(gw_df, id_resolver, season: str, extras: dict[int, dict]) -> None:
    """element -> {"p_goal": bookmaker anytime-scorer probability} (DGW summed xG)."""
    import math

    if "props_xg" not in gw_df.columns:
        return
    for code, xg in zip(gw_df["code"], gw_df["props_xg"]):
        eid = id_resolver.element_id_from_code(int(code), season)
        if eid is None or xg != xg:  # NaN: no bookmaker line
            continue
        rec = extras.setdefault(eid, {"props_xg": 0.0})
        rec["props_xg"] = rec.get("props_xg", 0.0) + float(xg)
        rec["p_goal"] = 1.0 - math.exp(-rec["props_xg"])


def _features_of(predictor) -> list[str]:
    """Input feature names of any predictor kind (duck-typed: ``feature_names``
    on PointPredictor / MinutesBlendPredictor, ``_feature_names`` on stubs)."""
    names = getattr(predictor, "feature_names", None)
    return list(names if names is not None else predictor._feature_names)


def _load_reference(model_dir: Path) -> dict | None:
    import json

    path = Path(model_dir) / SERVING_REFERENCE_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def predict_upcoming_gw(
    data_dir: Path,
    model_dir: Path,
    season: str,
    gw: int,
    health: list[str] | None = None,
    extras: dict[int, dict] | None = None,
) -> dict[int, float]:
    """Return element_id -> predicted points for the given upcoming GW.

    If *health* is a list, serving-health warnings (see :func:`serving_health`)
    are appended to it. If *extras* is a dict, it receives per-player display
    context from the same rows (see :func:`_collect_extras`).
    """
    predictor = load_predictor(model_dir)
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

    warnings = serving_health(gw_df, _features_of(predictor), _load_reference(model_dir))
    for w in warnings:
        logger.warning("Serving health: %s", w)
    if health is not None:
        health.extend(warnings)

    if extras is not None:
        _collect_extras(gw_df, id_resolver, season, extras)

    gw_df = _fill_missing_features(gw_df, _features_of(predictor), "Live predict")
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
    health: list[str] | None = None,
    extras: dict[int, dict] | None = None,
    predictor=None,
    return_features: bool = False,
):
    """Predictions for GWs gw..gw+horizon-1 as of the GW ``gw`` deadline.

    Uses the live season files (whose synthetic upcoming-GW rows are the
    as-of rows) and swaps fixture features for later GWs — see
    :mod:`fpl_optimizer.prediction.horizon`.  Returns a DataFrame
    ``element, GW, k, pred, p_play`` (blank GWs absent = 0 points).

    ``predictor``: an already-loaded predictor (anything with
    ``.predict(df)`` and ``._feature_names``); default
    ``PointPredictor.load(model_dir)``.  ``return_features=True`` returns
    ``(predictions, feature_rows)`` (the season's pipeline rows, used by
    callers that need the as-of rows, e.g. scripts/chip_eval.py).
    """
    import pandas as pd

    from fpl_optimizer.prediction.horizon import FixtureContext, predict_horizon

    if predictor is None:
        predictor = load_predictor(model_dir)
    id_resolver = IDResolver(data_dir)
    df = FeaturePipeline(data_dir, id_resolver, [season]).build()
    if df.empty or not (df["GW"] == gw).any():
        logger.warning("Live horizon: no feature rows for GW%d", gw)
        empty = pd.DataFrame(columns=["element", "GW", "k", "pred", "p_play"])
        return (empty, df) if return_features else empty
    df = _fill_missing_features(df, _features_of(predictor), "Live horizon")
    if extras is not None:
        _collect_extras(df[df["GW"] == gw], id_resolver, season, extras)
    warnings = serving_health(df[df["GW"] == gw], _features_of(predictor),
                              _load_reference(model_dir))
    for w in warnings:
        logger.warning("Serving health: %s", w)
    if health is not None:
        health.extend(warnings)
    ctx = FixtureContext.from_raw_dir(data_dir / "raw" / season)
    preds = predict_horizon(predictor, df, ctx, gw, horizon, dgw_mode)
    logger.info("Live horizon: %d (element, GW) predictions for GW%d-%d",
                len(preds), gw, gw + horizon - 1)
    return (preds, df) if return_features else preds


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
