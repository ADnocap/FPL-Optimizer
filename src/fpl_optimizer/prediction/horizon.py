"""Multi-GW (horizon) point predictions, point-in-time as of a GW deadline.

The single-GW model predicts GW ``t`` from a feature row built with data from
GW < t plus pre-match information for GW t.  To plan transfers over GWs
``t .. t+H-1`` we need predictions for future fixtures *as of the GW t
deadline*.  This module builds them by taking each player's as-of-t feature
row and swapping ONLY the fixture-dependent features for each future fixture:

- ``was_home``, ``fdr``, ``is_dgw``, ``gw_phase``
- ``opp_strength`` / ``opp_attack_strength`` / ``opp_defence_strength``
  (teams.csv ratings of the future opponent)
- ``opp_goals_conceded_r5`` / ``opp_pts_conceded_r5`` of the future opponent,
  rolled over the opponent's matches strictly BEFORE GW t (as of t)
- ``fixture_offset`` / ``synthetic_ep`` (own vs opponent team form as of t)
- match-market features (``odds_team_*``, ``props_*``) are not known beyond
  GW t -> NaN ("unknown", the state of every pre-market training row).  NOT
  the "no line" state (``props_has_line=0``): a model trained on seasons with
  props reads "no line" as "fringe player" and marks every future row down
  (seen live 2026-27: bias -0.2, ranking below persistence).

Everything player-level (rolling form, minutes, price, ownership, xP lag...)
stays at its as-of-t value.  Nothing from after the GW t deadline is used:
team stats only read merged_gw rows with GW < t, and the fixture list is the
one published at the deadline (live) — in historical backtests the final
fixtures.csv is used, so later-rescheduled fixtures are a (small) caveat.

Blank GWs give 0 points.  Double GWs follow ``dgw_mode``:
- ``"row"``: one aggregated row per (player, GW) exactly like the training
  pipeline (fixture features averaged, ``is_dgw``=1, ``synthetic_ep`` x n).
- ``"sum"``: one single-fixture row per fixture (``is_dgw``=0), predictions
  summed.
- ``"blend"`` (default): the mean of the two.  On the DGW player-GWs of
  2024-25 / 2025-26 (regulars, n=109/113, leak-free models) "row"
  under-predicts (bias -1.50 / -0.45), "sum" over-predicts in 2025-26
  (+0.03 / +0.95); the blend has the best or near-best MAE (3.81 / 3.28)
  and rank correlation in both seasons.

The same code serves live operation (the live season files contain synthetic
rows for the upcoming GW, which are the as-of-t rows) and historical
backtests (the real feature rows of GW t).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_WAS_HOME_MAP = {True: True, False: False, "True": True, "False": False,
                 1: True, 0: False, "1": True, "0": False}

ODDS_FEATURES = [
    "odds_team_win_prob", "odds_team_draw_prob",
    "odds_team_loss_prob", "odds_team_strength",
]
PROPS_FEATURES = ["props_xg", "props_xa", "props_sot", "props_card_prob",
                  "props_has_line", "props_n_books"]
DGW_MODES = ("row", "sum", "blend")

# P(minutes > 0) in GW t+k given the as-of-t ``playing_prob`` feature
# (= clip(mean minutes over the last 3 GWs / 90, 0, 1)).  Calibrated on
# 2020-21..2023-24 vaastav rows (~100k player-GWs per offset: the share of
# rows with minutes > 0 in GW t+k per playing_prob bin at GW t).
# Columns: k = 0, 1, 2, 3+.
_PPLAY_BINS = [0.0, 1e-9, 0.34, 0.67, 0.9]  # lower edges
_PPLAY_TABLE = np.array([
    [0.060, 0.080, 0.094, 0.104],   # playing_prob == 0
    [0.506, 0.512, 0.514, 0.519],   # (0, 0.34)
    [0.710, 0.696, 0.688, 0.674],   # [0.34, 0.67)
    [0.829, 0.802, 0.780, 0.766],   # [0.67, 0.9)
    [0.921, 0.878, 0.851, 0.831],   # >= 0.9
])
_PPLAY_NAN = np.array([0.425, 0.433, 0.423, 0.418])  # no minutes history yet


def p_play_from_playing_prob(playing_prob: float | None, k: int) -> float:
    """Calibrated P(minutes > 0) for a GW ``k`` weeks after the as-of GW."""
    col = min(max(int(k), 0), 3)
    if playing_prob is None or not np.isfinite(playing_prob):
        return float(_PPLAY_NAN[col])
    row = int(np.searchsorted(_PPLAY_BINS, float(playing_prob), side="right")) - 1
    row = min(max(row, 0), len(_PPLAY_BINS) - 1)
    return float(_PPLAY_TABLE[row, col])


# ---------------------------------------------------------------------------
# Fixture / team context
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", low_memory=False)


@dataclass
class _Fixture:
    fixture_id: int
    opp_id: int
    was_home: bool
    fdr: float  # difficulty / 5 (pipeline scale), NaN if unknown


class FixtureContext:
    """Season fixtures + team-level per-GW stats for as-of-t feature swaps.

    Parameters
    ----------
    merged_gw : pd.DataFrame
        Season merged_gw rows (vaastav format; may include live synthetic
        rows for the upcoming GW — only rows with GW < t are ever used for
        team statistics).
    fixtures : pd.DataFrame
        fixtures.csv (``id``, ``event``, ``team_h``, ``team_a``,
        ``team_h_difficulty``, ``team_a_difficulty``).
    teams : pd.DataFrame
        teams.csv with the strength columns used by the opponent features.
    """

    def __init__(
        self,
        merged_gw: pd.DataFrame,
        fixtures: pd.DataFrame,
        teams: pd.DataFrame | None = None,
    ) -> None:
        fx = fixtures.copy()
        for c in ("id", "event", "team_h", "team_a"):
            fx[c] = pd.to_numeric(fx[c], errors="coerce")
        fx = fx.dropna(subset=["id", "team_h", "team_a"])
        self._fixtures = fx

        # Per-team fixture lists by GW (unscheduled fixtures have NaN event)
        self._team_gw_fixtures: dict[tuple[int, int], list[_Fixture]] = {}
        sched = fx.dropna(subset=["event"]).sort_values(["event", "id"])
        for r in sched.itertuples(index=False):
            gw = int(r.event)
            h_diff = getattr(r, "team_h_difficulty", np.nan)
            a_diff = getattr(r, "team_a_difficulty", np.nan)
            self._team_gw_fixtures.setdefault((int(r.team_h), gw), []).append(
                _Fixture(int(r.id), int(r.team_a), True,
                         float(h_diff) / 5.0 if pd.notna(h_diff) else np.nan)
            )
            self._team_gw_fixtures.setdefault((int(r.team_a), gw), []).append(
                _Fixture(int(r.id), int(r.team_h), False,
                         float(a_diff) / 5.0 if pd.notna(a_diff) else np.nan)
            )

        # Team strengths
        self._strength: dict[int, dict[str, float]] = {}
        if teams is not None and not teams.empty and "id" in teams.columns:
            cols = ["strength", "strength_attack_home", "strength_attack_away",
                    "strength_defence_home", "strength_defence_away"]
            for r in teams.to_dict("records"):
                self._strength[int(r["id"])] = {
                    c: float(r[c]) if c in r and pd.notna(r[c]) else np.nan for c in cols
                }

        # Player rows with the team derived from the FIXTURE (robust to a
        # mislabelled ``team`` column, e.g. mid-season transfers)
        m = merged_gw.copy()
        for c in ("element", "GW", "fixture", "total_points", "goals_conceded"):
            if c in m.columns:
                m[c] = pd.to_numeric(m[c], errors="coerce")
        m = m.dropna(subset=["element", "GW", "fixture"])
        m["was_home"] = m["was_home"].map(_WAS_HOME_MAP).fillna(False).astype(bool)
        m = m.merge(
            fx[["id", "team_h", "team_a"]].rename(columns={"id": "fixture"}),
            on="fixture", how="inner",
        )
        m["team_id"] = np.where(m["was_home"], m["team_h"], m["team_a"]).astype(int)
        m["opp_id"] = np.where(m["was_home"], m["team_a"], m["team_h"]).astype(int)
        m["element"] = m["element"].astype(int)
        m["GW"] = m["GW"].astype(int)
        self._rows = m

        # element -> sorted list of (GW, team_id)
        pt = (m.sort_values(["element", "GW", "fixture"])
              .drop_duplicates(["element", "GW"])[["element", "GW", "team_id"]])
        self._player_team: dict[int, list[tuple[int, int]]] = {}
        for eid, gw, tid in pt.itertuples(index=False):
            self._player_team.setdefault(int(eid), []).append((int(gw), int(tid)))

        self._team_gw_stats = self._build_team_gw_stats(m)
        self._asof_cache: dict[int, pd.DataFrame] = {}

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_raw_dir(cls, raw_season_dir: Path) -> "FixtureContext":
        raw_season_dir = Path(raw_season_dir)
        merged = _read_csv(raw_season_dir / "gws" / "merged_gw.csv")
        fixtures = _read_csv(raw_season_dir / "fixtures.csv")
        teams = _read_csv(raw_season_dir / "teams.csv")
        if fixtures.empty:
            raise FileNotFoundError(f"No fixtures.csv in {raw_season_dir}")
        return cls(merged, fixtures, teams)

    @staticmethod
    def _build_team_gw_stats(m: pd.DataFrame) -> pd.DataFrame:
        """Per (team, GW): goals conceded, points conceded, mean player points.

        Mirrors features/opponent.py and features/players_raw.py, keyed by
        the fixture-derived team.
        """
        gc = (m.groupby(["team_id", "GW", "fixture"])["goals_conceded"].max()
              .groupby(["team_id", "GW"]).sum().rename("gc"))
        pts_fx = m.groupby(["team_id", "GW", "fixture"], as_index=False)["total_points"].sum()
        opp_map = m[["team_id", "opp_id", "GW", "fixture"]].drop_duplicates(
            ["team_id", "GW", "fixture"])
        conc = opp_map.merge(
            pts_fx.rename(columns={"team_id": "opp_id", "total_points": "opp_pts"}),
            on=["opp_id", "GW", "fixture"], how="left",
        )
        conc["opp_pts"] = conc["opp_pts"].fillna(0)
        pc = conc.groupby(["team_id", "GW"])["opp_pts"].sum().rename("pc")
        per_player = (m.sort_values(["element", "GW"])
                      .groupby(["element", "GW"], as_index=False)
                      .agg(total_points=("total_points", "sum"), team_id=("team_id", "first")))
        form = per_player.groupby(["team_id", "GW"])["total_points"].mean().rename("form")
        out = pd.concat([gc, pc, form], axis=1).reset_index()
        return out.sort_values(["team_id", "GW"]).reset_index(drop=True)

    # -- queries --------------------------------------------------------------

    def fixtures_for(self, team_id: int, gw: int) -> list[_Fixture]:
        return self._team_gw_fixtures.get((int(team_id), int(gw)), [])

    def player_team(self, element_id: int, t: int) -> int | None:
        """Team the player belongs to as of GW t (latest row with GW <= t)."""
        hist = self._player_team.get(int(element_id))
        if not hist:
            return None
        team = None
        for gw, tid in hist:
            if gw <= t:
                team = tid
            else:
                break
        return team if team is not None else hist[0][1]

    def team_stats_asof(self, t: int) -> pd.DataFrame:
        """Per team: rolling-5 means of GW < t stats (opp_* / team form)."""
        if t in self._asof_cache:
            return self._asof_cache[t]
        s = self._team_gw_stats[self._team_gw_stats["GW"] < t]
        last5 = s.groupby("team_id").tail(5)
        out = last5.groupby("team_id")[["gc", "pc", "form"]].mean()
        self._asof_cache[t] = out
        return out

    def opponent_features(self, fx: _Fixture, t: int) -> dict[str, float]:
        """Fixture-dependent features for one fixture, as of GW t."""
        st = self.team_stats_asof(t)
        opp = fx.opp_id
        feats = {
            "was_home": 1.0 if fx.was_home else 0.0,
            "fdr": fx.fdr,
            "opp_goals_conceded_r5": float(st.at[opp, "gc"]) if opp in st.index else np.nan,
            "opp_pts_conceded_r5": float(st.at[opp, "pc"]) if opp in st.index else np.nan,
        }
        strength = self._strength.get(opp)
        if strength:
            feats["opp_strength"] = strength["strength"] / 1500.0
            if fx.was_home:  # opponent is away
                feats["opp_attack_strength"] = strength["strength_attack_away"] / 1500.0
                feats["opp_defence_strength"] = strength["strength_defence_away"] / 1500.0
            else:
                feats["opp_attack_strength"] = strength["strength_attack_home"] / 1500.0
                feats["opp_defence_strength"] = strength["strength_defence_home"] / 1500.0
        else:
            feats["opp_strength"] = np.nan
            feats["opp_attack_strength"] = np.nan
            feats["opp_defence_strength"] = np.nan
        return feats

    def fixture_offset(self, own_team: int, opp_team: int, t: int) -> float:
        """Quantised own-vs-opponent team form differential (players_raw.py)."""
        st = self.team_stats_asof(t)
        own = float(st.at[own_team, "form"]) if own_team in st.index else 0.0
        opp = float(st.at[opp_team, "form"]) if opp_team in st.index else 0.0
        own = 0.0 if not np.isfinite(own) else own
        opp = 0.0 if not np.isfinite(opp) else opp
        return float(np.clip(np.round((own - opp) * 2) / 2, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Row construction + prediction
# ---------------------------------------------------------------------------


def asof_rows(features: pd.DataFrame, t: int) -> pd.DataFrame:
    """One feature row per element as of GW t.

    The GW t row when it exists (its features only use data < t plus
    pre-deadline info); otherwise (team blanks in GW t) the latest earlier
    row, which is older but still point-in-time safe.
    """
    f = features[features["GW"] <= t]
    f = f.sort_values(["element", "GW"])
    base = f.groupby("element", as_index=False).tail(1).copy()
    base["asof_gw"] = base["GW"].astype(int)
    return base.reset_index(drop=True)


def _swap_fixture_rows(
    base: pd.DataFrame,
    ctx: FixtureContext,
    t: int,
    gw: int,
    dgw_mode: str,
    keep_market: bool = False,
) -> pd.DataFrame:
    """Future-GW rows: base (as-of-t) rows with fixture features swapped.

    ``keep_market`` keeps the rows' odds/props (only valid for gw == t, where
    the market is known pre-deadline).
    """
    records: list[dict] = []
    idx: list[int] = []
    for i, r in enumerate(base[["element"]].itertuples(index=False)):
        eid = int(r.element)
        team = ctx.player_team(eid, t)
        if team is None:
            continue
        fixtures = ctx.fixtures_for(team, gw)
        if not fixtures:
            continue
        per_fx = []
        for fx in fixtures:
            feats = ctx.opponent_features(fx, t)
            feats["fixture_offset"] = ctx.fixture_offset(team, fx.opp_id, t)
            per_fx.append(feats)
        n = len(per_fx)
        if dgw_mode == "row" or n == 1:
            agg = {k: float(np.nanmean([f[k] for f in per_fx]))
                   if any(np.isfinite(f[k]) for f in per_fx) else np.nan
                   for k in per_fx[0] if k != "fixture_offset"}
            agg["fixture_offset"] = per_fx[0]["fixture_offset"]  # pipeline: first opp
            agg["is_dgw"] = 1.0 if n > 1 else 0.0
            agg["_n_fix"] = float(n)
            records.append(agg)
            idx.append(i)
        else:  # "sum": one single-fixture row per fixture
            for f in per_fx:
                rec = dict(f)
                rec["is_dgw"] = 0.0
                rec["_n_fix"] = 1.0
                records.append(rec)
                idx.append(i)
    if not records:
        return base.iloc[0:0].copy()
    rows = base.iloc[idx].copy().reset_index(drop=True)
    swap = pd.DataFrame(records)
    n_fix = swap.pop("_n_fix").to_numpy()
    for col in swap.columns:
        if col in rows.columns:
            rows[col] = swap[col].to_numpy()
    # synthetic_ep = (form + fixture_offset) * playing_prob * n_fixtures
    if "synthetic_ep" in rows.columns and "fixture_offset" in rows.columns:
        form = rows["pts_rolling_5"].fillna(0.0) if "pts_rolling_5" in rows.columns else 0.0
        pp = rows["playing_prob"] if "playing_prob" in rows.columns else np.nan
        rows["synthetic_ep"] = (form + rows["fixture_offset"]) * pp * n_fix
    if "gw_phase" in rows.columns:
        rows["gw_phase"] = gw / 38.0
    if not keep_market:
        for col in ODDS_FEATURES + PROPS_FEATURES:
            if col in rows.columns:
                rows[col] = np.nan
    rows["GW"] = gw
    return rows


def build_horizon_rows(
    features: pd.DataFrame,
    ctx: FixtureContext,
    t: int,
    horizon: int,
    dgw_mode: str = "row",
    market: str = "keep",
) -> pd.DataFrame:
    """Feature rows for GWs t..t+horizon-1 as of the GW t deadline.

    Returns the rows with extra columns ``k`` (offset from t) and ``asof_gw``.
    The k=0 rows are the real as-of-t rows (incl. odds/props when present,
    unless ``market="drop"``); k>=1 rows have fixture features swapped (see
    module docstring).  ``market="drop"`` blanks odds/props at k=0 too, so
    every horizon GW is predicted on the same footing (no artificial
    now-vs-later gap from the market features, which shift regulars'
    predictions by ~0.1-0.25 pts with ~0.5 sd while barely changing ranking).
    """
    if dgw_mode not in ("row", "sum"):
        raise ValueError(f"dgw_mode must be 'row' or 'sum', got {dgw_mode!r}")
    if market not in ("keep", "drop"):
        raise ValueError(f"market must be 'keep' or 'drop', got {market!r}")
    base = asof_rows(features, t)
    if market == "drop":
        base = base.copy()
        for col in ODDS_FEATURES + PROPS_FEATURES:
            if col in base.columns:
                base[col] = np.nan
    out = []
    cur = base[base["asof_gw"] == t].copy()
    if dgw_mode == "sum" and "is_dgw" in cur.columns and (cur["is_dgw"] == 1).any():
        # split this GW's aggregated DGW rows into single-fixture rows too
        dgw = (cur["is_dgw"] == 1).to_numpy()
        split = _swap_fixture_rows(cur[dgw].reset_index(drop=True), ctx, t, t,
                                   "sum", keep_market=True)
        split["asof_gw"] = t
        cur = pd.concat([cur[~dgw], split], ignore_index=True)
    cur["k"] = 0
    out.append(cur)
    for k in range(1, horizon):
        rows = _swap_fixture_rows(base, ctx, t, t + k, dgw_mode)
        rows["k"] = k
        out.append(rows)
    return pd.concat(out, ignore_index=True)


def predict_horizon(
    predictor,
    features: pd.DataFrame,
    ctx: FixtureContext,
    t: int,
    horizon: int,
    dgw_mode: str = "blend",
    market: str = "keep",
) -> pd.DataFrame:
    """Predicted points per (element, GW) for GWs t..t+horizon-1.

    Returns a DataFrame with columns ``element, GW, k, pred, p_play`` (one row
    per element and GW with a fixture; DGWs per ``dgw_mode``, blank GWs absent
    = 0 points).  ``p_play`` is the calibrated probability of playing (for
    bench/auto-sub value in the planner).
    """
    if dgw_mode not in DGW_MODES:
        raise ValueError(f"dgw_mode must be one of {DGW_MODES}, got {dgw_mode!r}")
    if dgw_mode == "blend":
        a = predict_horizon(predictor, features, ctx, t, horizon, "row", market)
        b = predict_horizon(predictor, features, ctx, t, horizon, "sum", market)
        m = a.merge(b, on=["element", "GW", "k"], how="outer", suffixes=("_r", "_s"))
        m["pred"] = m[["pred_r", "pred_s"]].mean(axis=1)
        m["p_play"] = m[["p_play_r", "p_play_s"]].max(axis=1)
        return m[["element", "GW", "k", "pred", "p_play"]]
    rows = build_horizon_rows(features, ctx, t, horizon, dgw_mode, market)
    if rows.empty:
        return pd.DataFrame(columns=["element", "GW", "k", "pred", "p_play"])
    rows = rows.reset_index(drop=True)
    rows["pred"] = predictor.predict(rows)
    pp = rows["playing_prob"] if "playing_prob" in rows.columns else pd.Series(np.nan, index=rows.index)
    rows["p_play"] = [p_play_from_playing_prob(v, k) for v, k in zip(pp, rows["k"])]
    out = (rows.groupby(["element", "GW", "k"], as_index=False)
           .agg(pred=("pred", "sum"), p_play=("p_play", "max")))
    out["element"] = out["element"].astype(int)
    out["GW"] = out["GW"].astype(int)
    return out
