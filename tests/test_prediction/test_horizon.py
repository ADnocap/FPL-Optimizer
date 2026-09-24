"""Tests for as-of-t multi-GW (horizon) feature rows and predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_optimizer.prediction.horizon import (
    FixtureContext,
    build_horizon_rows,
    p_play_from_playing_prob,
    predict_horizon,
)

# 4 teams. GW1: 1v2, 3v4 | GW2: 2v1, 4v3 | GW3: 1v3 (team 2 & 4 blank)
# GW4: 1v4, 4v2 (team 4 double), 3v2... keep: 1v4, 4v2 -> team 4 DGW, team 3 blank
FIXTURES = pd.DataFrame([
    {"id": 1, "event": 1, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4},
    {"id": 2, "event": 1, "team_h": 3, "team_a": 4, "team_h_difficulty": 3, "team_a_difficulty": 3},
    {"id": 3, "event": 2, "team_h": 2, "team_a": 1, "team_h_difficulty": 4, "team_a_difficulty": 2},
    {"id": 4, "event": 2, "team_h": 4, "team_a": 3, "team_h_difficulty": 3, "team_a_difficulty": 3},
    {"id": 5, "event": 3, "team_h": 1, "team_a": 3, "team_h_difficulty": 2, "team_a_difficulty": 5},
    {"id": 6, "event": 4, "team_h": 1, "team_a": 4, "team_h_difficulty": 2, "team_a_difficulty": 5},
    {"id": 7, "event": 4, "team_h": 4, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3},
])
TEAMS = pd.DataFrame([
    {"id": t, "strength": 3, "strength_attack_home": 1000 + 10 * t,
     "strength_attack_away": 1100 + 10 * t, "strength_defence_home": 1200 + 10 * t,
     "strength_defence_away": 1300 + 10 * t}
    for t in (1, 2, 3, 4)
])


def _merged() -> pd.DataFrame:
    """Players: 11 (team 1), 21 (team 2), 31 (team 3), 41 (team 4); GW1-2 played."""
    rows = []
    for fid, gw, h, a, gh, ga in [(1, 1, 1, 2, 2, 0), (2, 1, 3, 4, 1, 1),
                                  (3, 2, 2, 1, 1, 3), (4, 2, 4, 3, 0, 2)]:
        for eid, team in ((11, 1), (21, 2), (31, 3), (41, 4)):
            if team not in (h, a):
                continue
            home = team == h
            rows.append({
                "element": eid, "GW": gw, "fixture": fid, "was_home": home,
                "team": team, "opponent_team": a if home else h,
                "total_points": 2 + eid % 7, "minutes": 90,
                "goals_conceded": ga if home else gh,
            })
    return pd.DataFrame(rows)


def _features(t: int) -> pd.DataFrame:
    """As-of-t feature rows for the 4 players (fixture features = junk)."""
    return pd.DataFrame([{
        "element": eid, "GW": t, "code": 1000 + eid, "position": "MID",
        "pts_rolling_5": 3.0, "playing_prob": 1.0,
        "was_home": -1.0, "fdr": -1.0, "opp_strength": -1.0,
        "opp_attack_strength": -1.0, "opp_defence_strength": -1.0,
        "opp_goals_conceded_r5": -1.0, "opp_pts_conceded_r5": -1.0,
        "is_dgw": -1.0, "fixture_offset": 0.0, "synthetic_ep": -1.0,
        "gw_phase": t / 38.0, "odds_team_win_prob": 0.5,
        "props_xg": 0.3, "props_has_line": 1.0, "props_n_books": 3.0,
    } for eid in (11, 21, 31, 41)])


@pytest.fixture
def ctx() -> FixtureContext:
    return FixtureContext(_merged(), FIXTURES, TEAMS)


class TestFixtureContext:
    def test_player_team_from_fixture(self, ctx):
        assert ctx.player_team(11, 3) == 1
        assert ctx.player_team(41, 3) == 4

    def test_team_stats_only_use_past_gws(self, ctx):
        st2 = ctx.team_stats_asof(2)  # only GW1
        # team 2 conceded 2 in GW1 (lost 2-0 at team 1)
        assert st2.at[2, "gc"] == pytest.approx(2.0)
        st3 = ctx.team_stats_asof(3)  # GW1-2: conceded 2 then 3
        assert st3.at[2, "gc"] == pytest.approx(2.5)
        assert 4 not in ctx.team_stats_asof(1).index  # nothing before GW1

    def test_opponent_features_swap_home_away_strength(self, ctx):
        fx = ctx.fixtures_for(1, 3)[0]  # team 1 hosts team 3
        f = ctx.opponent_features(fx, 3)
        assert f["was_home"] == 1.0 and f["fdr"] == pytest.approx(2 / 5)
        # opponent (3) plays away -> its *_away strengths
        assert f["opp_attack_strength"] == pytest.approx(1130 / 1500)
        assert f["opp_defence_strength"] == pytest.approx(1330 / 1500)


class TestHorizonRows:
    def test_k0_rows_are_the_asof_rows(self, ctx):
        rows = build_horizon_rows(_features(3), ctx, t=3, horizon=2)
        k0 = rows[rows["k"] == 0]
        assert set(k0["element"]) == {11, 21, 31, 41}
        assert (k0["was_home"] == -1.0).all()  # untouched

    def test_future_rows_swap_fixture_features_only(self, ctx):
        feats = pd.concat([_features(2), _features(3)])
        rows = build_horizon_rows(feats, ctx, t=2, horizon=2)  # GW3
        k1 = rows[rows["k"] == 1].set_index("element")
        # GW3: only 1v3 is played -> teams 2 and 4 blank (no rows)
        assert set(k1.index) == {11, 31}
        assert k1.at[11, "was_home"] == 1.0 and k1.at[31, "was_home"] == 0.0
        assert k1.at[11, "fdr"] == pytest.approx(0.4)
        assert np.isnan(k1.at[11, "odds_team_win_prob"])  # unknown market
        # props unknown (NaN), NOT the "no line" state a props-era model
        # reads as "fringe player"
        assert np.isnan(k1.at[11, "props_xg"])
        assert np.isnan(k1.at[11, "props_has_line"]) and np.isnan(k1.at[11, "props_n_books"])
        assert k1.at[11, "gw_phase"] == pytest.approx(3 / 38)
        assert k1.at[11, "pts_rolling_5"] == 3.0  # player features as of t
        # built from the GW2 (as-of) rows, not the GW3 rows
        assert (rows["asof_gw"] == 2).all()

    def test_double_gameweek_modes(self, ctx):
        feats = _features(2)
        row = build_horizon_rows(feats, ctx, t=2, horizon=3, dgw_mode="row")
        r4 = row[(row["k"] == 2) & (row["element"] == 41)]
        assert len(r4) == 1 and r4["is_dgw"].iloc[0] == 1.0
        assert r4["was_home"].iloc[0] == pytest.approx(0.5)  # away at 1, home v 2
        summed = build_horizon_rows(feats, ctx, t=2, horizon=3, dgw_mode="sum")
        s4 = summed[(summed["k"] == 2) & (summed["element"] == 41)]
        assert len(s4) == 2 and (s4["is_dgw"] == 0.0).all()
        # team 3 blanks in GW4
        assert not ((row["k"] == 2) & (row["element"] == 31)).any()

    def test_sum_mode_splits_current_gw_double_keeping_market(self, ctx):
        feats = _features(4)
        feats.loc[feats["element"] == 41, "is_dgw"] = 1.0  # pipeline DGW row
        rows = build_horizon_rows(feats, ctx, t=4, horizon=1, dgw_mode="sum")
        r41 = rows[rows["element"] == 41]
        assert len(r41) == 2 and (r41["is_dgw"] == 0.0).all()
        assert (r41["odds_team_win_prob"] == 0.5).all()  # known pre-deadline
        assert sorted(r41["was_home"]) == [0.0, 1.0]
        assert len(rows[rows["element"] == 11]) == 1

    def test_predict_horizon_sums_doubles_and_skips_blanks(self, ctx):
        class ConstPredictor:
            def predict(self, df):
                return np.full(len(df), 2.0)

        out = predict_horizon(ConstPredictor(), _features(2), ctx, t=2, horizon=3,
                              dgw_mode="sum")
        got = {(r.element, r.GW): r.pred for r in out.itertuples()}
        assert got[(41, 4)] == pytest.approx(4.0)  # DGW summed
        assert (31, 4) not in got  # blank
        assert got[(11, 2)] == pytest.approx(2.0)
        assert set(out["k"]) == {0, 1, 2}
        assert out["p_play"].between(0, 1).all()

    def test_blend_averages_row_and_sum_for_doubles(self, ctx):
        class DgwAwarePredictor:  # 3 pts for an aggregated DGW row, 2 per fixture
            def predict(self, df):
                return np.where(df["is_dgw"] == 1.0, 3.0, 2.0)

        out = predict_horizon(DgwAwarePredictor(), _features(2), ctx, t=2, horizon=3,
                              dgw_mode="blend")
        got = {(r.element, r.GW): r.pred for r in out.itertuples()}
        assert got[(41, 4)] == pytest.approx((3.0 + 4.0) / 2)
        assert got[(11, 4)] == pytest.approx(2.0)  # single GW unaffected
        with pytest.raises(ValueError):
            predict_horizon(DgwAwarePredictor(), _features(2), ctx, 2, 2, dgw_mode="x")


def test_p_play_calibration_monotone():
    assert p_play_from_playing_prob(1.0, 0) > p_play_from_playing_prob(0.5, 0)
    assert p_play_from_playing_prob(0.5, 0) > p_play_from_playing_prob(0.0, 0)
    assert p_play_from_playing_prob(1.0, 0) > p_play_from_playing_prob(1.0, 3)
    assert 0 < p_play_from_playing_prob(float("nan"), 1) < 1
    assert p_play_from_playing_prob(1.0, 9) == p_play_from_playing_prob(1.0, 3)
