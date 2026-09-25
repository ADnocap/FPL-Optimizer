"""Point-in-time minutes-history features (and minutes labels).

Feeds the minutes model (``fpl_optimizer.prediction.minutes``): who will play,
and for how long, is the single biggest driver of FPL points.

Every feature for (element, GW=k) uses only

- rows with GW < k of the same player (``shift(1)`` before any rolling /
  expanding / streak / EWM),
- completed previous seasons (``data/raw/<season>/gws/merged_gw.csv``), or
- pre-deadline facts of row k itself (team, price, ``fpl_xp_lag``).

Synthetic upcoming rows written by ``fpl_live`` (stats zeroed) therefore never
leak: their own minutes are never read, only their team.

Two entry points, called from ``FeaturePipeline``:

``compute_minutes_history_features(merged_gw, data_dir, season, id_resolver)``
    (per season, in ``_build_season``) One row per (element, GW): ``mh_*``
    (this season), ``xs_*`` (across seasons) and ``ps_*`` (previous season)
    features, the helper columns ``_mh_team`` / ``_mh_score`` (consumed by
    :func:`add_depth_and_availability`) and the minutes LABELS ``min_total`` /
    ``min_max`` / ``mcls``.  The labels are the realised minutes of the row's
    own GW — they are listed in ``model._NON_FEATURE_COLS`` and must never
    become features.

``add_depth_and_availability(result)``
    (in ``build``, AFTER the cross-season position back-fill — 2016-19 rows
    only get a position there) ``dp_*`` team depth-chart ranks within
    (season, GW, team, position) and ``av_*`` availability proxies; drops the
    helpers.

Validated in the minutes-model experiment (2026-09-24): a truncation test
(scramble every label from (season, GW >= g) on, rebuild) changed 0 features of
rows with GW <= g across 6 cuts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_optimizer.utils.constants import AVAILABLE_SEASONS

logger = logging.getLogger(__name__)

# Starting slots per position in a typical XI (depth-chart gap reference).
SLOT = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}

MH_FEATURES = [
    # this season, strictly earlier GWs
    "mh_lag1", "mh_lag2", "mh_lag3", "mh_p60_r3", "mh_p60_r5", "mh_p60_r10",
    "mh_play_r5", "mh_sub_r5", "mh_starts_r3", "mh_p60_season", "mh_zero_streak",
    "mh_full_streak", "mh_ewm", "mh_rows_prev", "mh_team_changed",
    # across seasons / previous season
    "xs_ewm", "xs_p60_r5", "ps_share", "ps_p60", "ps_end_share", "ps_rows",
    # team depth chart (same GW, team, position)
    "dp_rank", "dp_n", "dp_val_rank", "dp_share_rank", "dp_slot_gap",
    # availability proxy from the previous GW's ep_this
    "av_xp_is0", "av_xp_ratio",
]
# Realised minutes of the row's own GW: LABELS, never features.
MINUTES_LABELS = ["min_total", "min_max", "mcls"]
_HELPERS = ["_mh_team", "_mh_score"]


def _aggregate(merged_gw: pd.DataFrame) -> pd.DataFrame:
    """Per (element, GW): summed/max minutes, fixture count, starts, team, code."""
    m = merged_gw.copy()
    for c in ["element", "GW", "minutes", "starts", "team"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    if "starts" not in m.columns:
        m["starts"] = np.nan
    if "team" not in m.columns:
        m["team"] = np.nan
    if "code" not in m.columns:
        m["code"] = np.nan
    m = m.dropna(subset=["element", "GW"])
    agg = (m.sort_values(["element", "GW"])
           .groupby(["element", "GW"], as_index=False)
           .agg(min_total=("minutes", "sum"), min_max=("minutes", "max"),
                n_fix=("minutes", "size"), starts=("starts", "sum"),
                team=("team", "last"), code=("code", "first")))
    if m["starts"].isna().all():  # pre-2022-23 seasons have no `starts`
        agg["starts"] = np.nan
    agg["element"] = agg["element"].astype(int)
    agg["GW"] = agg["GW"].astype(int)
    return agg


# (data_dir, season) -> aggregated rows of that COMPLETED season. Training builds
# every season in turn and each one re-reads all earlier seasons.
_PREV_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def _previous_season_rows(data_dir: Path, season: str, id_resolver) -> pd.DataFrame:
    """Aggregated rows (code, GW, min_total, min_max, n_fix) of a completed season."""
    key = (str(Path(data_dir).resolve()), season)
    if key in _PREV_CACHE:
        return _PREV_CACHE[key]
    p = Path(data_dir) / "raw" / season / "gws" / "merged_gw.csv"
    if not p.exists():
        _PREV_CACHE[key] = pd.DataFrame()
        return _PREV_CACHE[key]
    cols = ["element", "GW", "minutes", "starts"]
    try:
        m = pd.read_csv(p, encoding="utf-8", on_bad_lines="skip",
                        usecols=lambda c: c in cols)
    except UnicodeDecodeError:
        m = pd.read_csv(p, encoding="latin-1", on_bad_lines="skip",
                        usecols=lambda c: c in cols)
    m["element"] = pd.to_numeric(m["element"], errors="coerce")
    m = m.dropna(subset=["element"])
    m["code"] = [id_resolver.code_from_element_id(season, int(e)) for e in m["element"]]
    m = m.dropna(subset=["code"])
    m["team"] = np.nan  # not needed for cross-season features
    _PREV_CACHE[key] = _aggregate(m)[["code", "GW", "min_total", "min_max", "n_fix"]]
    return _PREV_CACHE[key]


def _streak(flag: pd.Series, keys: pd.Series) -> pd.Series:
    """Length of the current run of True per key (0 on a False row)."""
    f = flag.astype(int)
    brk = (f == 0) | (keys != keys.shift(1))
    return f.groupby(brk.cumsum()).cumsum()


def compute_minutes_history_features(
    merged_gw: pd.DataFrame, data_dir: Path, season: str, id_resolver,
) -> pd.DataFrame:
    """See module docstring. ``merged_gw`` must already carry ``code`` and a
    numeric ``team`` (FeaturePipeline._build_season steps 1-2). Returns the
    helper columns too; the caller must run :func:`add_depth_and_availability`.
    """
    df = _aggregate(merged_gw).sort_values(["element", "GW"]).reset_index(drop=True)
    df["mcls"] = np.select([df["min_total"] <= 0, df["min_total"] < 60], [0, 1], 2)
    per_fix = df["min_total"] / (90.0 * df["n_fix"].clip(lower=1))
    f60 = (df["min_max"] >= 60).astype(float)
    fplay = (df["min_total"] > 0).astype(float)
    fsub = ((df["min_total"] > 0) & (df["min_max"] < 60)).astype(float)
    fstart = (df["starts"] > 0).astype(float).where(df["starts"].notna())
    el = df["element"]

    def sh(s):
        return s.groupby(el, sort=False).shift(1)

    def roll(s, w):
        x = sh(s)
        return (x.groupby(el, sort=False).rolling(w, min_periods=1).mean()
                .reset_index(level=0, drop=True).reindex(df.index))

    g = df.groupby("element", sort=False)
    df["mh_lag1"] = sh(df["min_total"])
    df["mh_lag2"] = g["min_total"].shift(2)
    df["mh_lag3"] = g["min_total"].shift(3)
    df["mh_p60_r3"], df["mh_p60_r5"], df["mh_p60_r10"] = roll(f60, 3), roll(f60, 5), roll(f60, 10)
    df["mh_play_r5"], df["mh_sub_r5"] = roll(fplay, 5), roll(fsub, 5)
    df["mh_starts_r3"] = roll(fstart, 3)
    x = sh(f60)
    df["mh_p60_season"] = (x.groupby(el, sort=False).cumsum()
                           / x.notna().groupby(el, sort=False).cumsum().replace(0, np.nan))
    df["mh_zero_streak"] = sh(_streak(df["min_total"] <= 0, el).astype(float)).fillna(0)
    df["mh_full_streak"] = sh(_streak(df["min_max"] >= 60, el).astype(float)).fillna(0)
    ewm = per_fix.groupby(el, sort=False).transform(lambda s: s.ewm(halflife=2, adjust=True).mean())
    df["mh_ewm"] = sh(ewm)
    df["mh_rows_prev"] = g.cumcount().astype(float)
    prev_team = sh(df["team"])
    df["mh_team_changed"] = (prev_team.notna() & (prev_team != df["team"])).astype(float)

    # ---- cross-season: every completed season before `season` ----
    prior = [s for s in AVAILABLE_SEASONS if s < season]
    idx = len(prior)
    hist = []
    for i, s in enumerate(prior):
        r = _previous_season_rows(data_dir, s, id_resolver)
        if not r.empty:
            hist.append(r.assign(s_idx=i))
    cur = df[["code", "GW", "min_total", "min_max", "n_fix"]].assign(
        s_idx=idx, _cur=True, _row=df.index)
    allr = pd.concat(hist + [cur], ignore_index=True) if hist else cur.reset_index(drop=True)
    allr["_cur"] = allr["_cur"].fillna(False).astype(bool)
    allr = allr.sort_values(["code", "s_idx", "GW"]).reset_index(drop=True)
    pf = allr["min_total"] / (90.0 * allr["n_fix"].clip(lower=1))
    a60 = (allr["min_max"] >= 60).astype(float)
    c = allr["code"]
    xewm = pf.groupby(c, sort=False).transform(lambda s: s.ewm(halflife=3, adjust=True).mean())
    allr["xs_ewm"] = xewm.groupby(c, sort=False).shift(1)
    xa = a60.groupby(c, sort=False).shift(1)
    allr["xs_p60_r5"] = (xa.groupby(c, sort=False).rolling(5, min_periods=1).mean()
                         .reset_index(level=0, drop=True).reindex(allr.index))
    allr["_pf"], allr["_f60"] = pf, a60
    prev = allr[allr["s_idx"] == idx - 1]
    ps = prev.groupby("code").agg(ps_share=("_pf", "mean"), ps_p60=("_f60", "mean"),
                                  ps_rows=("_pf", "size"),
                                  ps_end_share=("_pf", lambda s: s.iloc[-6:].mean()))
    curr = allr[allr["_cur"]].set_index("_row")
    df["xs_ewm"] = curr["xs_ewm"].reindex(df.index)
    df["xs_p60_r5"] = curr["xs_p60_r5"].reindex(df.index)
    df = df.join(ps, on="code")

    df["_mh_team"] = df["team"]
    df["_mh_score"] = df["xs_ewm"].fillna(df["mh_ewm"]).fillna(-1.0)
    keep = ["element", "GW"] + _HELPERS + MINUTES_LABELS + [
        f for f in MH_FEATURES if not f.startswith(("dp_", "av_"))]
    return df[keep]


def add_depth_and_availability(result: pd.DataFrame) -> pd.DataFrame:
    """Team depth-chart ranks + ep-based availability proxy.

    Needs ``position``, ``value``, ``fpl_xp_lag`` (previous-GW ep_this; live =
    the pre-deadline snapshot's ep_this) and ``pts_rolling_3`` already merged
    (missing inputs give NaN features). Drops the helper columns.
    """
    df = result

    def col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series(np.nan, index=df.index, dtype=float)

    team, pos = col("_mh_team"), col("position")
    grp = [df["GW"], team, pos]
    if "season" in df.columns:  # multi-season frame: never rank across seasons
        grp = [df["season"]] + grp
    score = col("_mh_score")
    df["dp_rank"] = score.groupby(grp).rank(ascending=False, method="average")
    df["dp_n"] = score.groupby(grp).transform("size").astype(float)
    df["dp_val_rank"] = col("value").groupby(grp).rank(ascending=False, method="average")
    df["dp_share_rank"] = df["dp_rank"] / df["dp_n"]
    df["dp_slot_gap"] = df["dp_rank"] - pos.map(SLOT)
    miss = team.isna() | pos.isna()
    for c in ["dp_rank", "dp_n", "dp_val_rank", "dp_share_rank", "dp_slot_gap"]:
        df.loc[miss, c] = np.nan
    xp = col("fpl_xp_lag")
    df["av_xp_is0"] = (xp == 0).astype(float).where(xp.notna())
    df["av_xp_ratio"] = xp / (col("pts_rolling_3").clip(lower=0) + 0.5)
    return df.drop(columns=[c for c in _HELPERS if c in df.columns])
