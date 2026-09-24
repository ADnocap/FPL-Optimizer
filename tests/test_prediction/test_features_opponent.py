"""Tests for opponent / fixture context features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_optimizer.prediction.features.opponent import (
    FEATURE_COLS,
    compute_opponent_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_merged_gw(
    rows: list[dict],
) -> pd.DataFrame:
    """Build a minimal merged_gw DataFrame from a list of dicts.

    Each dict should have keys:
        element, GW, team, opponent_team, was_home, goals_conceded,
        total_points, fixture
    """
    return pd.DataFrame(rows)


def _empty_fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "id", "event", "team_h", "team_a",
            "team_h_difficulty", "team_a_difficulty",
        ]
    )


def _empty_teams() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "id", "strength",
            "strength_attack_home", "strength_attack_away",
            "strength_defence_home", "strength_defence_away",
        ]
    )


# ---------------------------------------------------------------------------
# Fixtures: small 2-team, 5-GW scenario
#
# Team 1 players: 101, 102
# Team 2 players: 201, 202
# They play each other every GW (alternating home/away).
# ---------------------------------------------------------------------------

def _two_team_scenario() -> pd.DataFrame:
    """5 GWs, 2 teams (1 & 2), 2 players per team, alternating home/away."""
    rows: list[dict] = []
    fixture_id = 1000
    for gw in range(1, 6):
        home_team = 1 if gw % 2 == 1 else 2
        away_team = 2 if gw % 2 == 1 else 1

        # Goals conceded by each side (deterministic pattern)
        # Team 1 concedes gw goals, Team 2 concedes (gw * 2) goals
        gc_home = gw if home_team == 1 else gw * 2
        gc_away = gw * 2 if away_team == 2 else gw

        # Total points: team 1 players get gw pts, team 2 players get gw+1
        for eid in [101, 102]:
            rows.append({
                "element": eid,
                "GW": gw,
                "team": 1,
                "opponent_team": 2,
                "was_home": home_team == 1,
                "goals_conceded": gc_home if home_team == 1 else gc_away,
                "total_points": gw,
                "fixture": fixture_id,
            })
        for eid in [201, 202]:
            rows.append({
                "element": eid,
                "GW": gw,
                "team": 2,
                "opponent_team": 1,
                "was_home": home_team == 2,
                "goals_conceded": gc_away if home_team == 1 else gc_home,
                "total_points": gw + 1,
                "fixture": fixture_id,
            })
        fixture_id += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWasHome:
    def test_single_fixture_home(self) -> None:
        mgw = _make_merged_gw([
            {"element": 1, "GW": 1, "team": 1, "opponent_team": 2,
             "was_home": True, "goals_conceded": 0, "total_points": 5,
             "fixture": 100},
        ])
        result = compute_opponent_features(mgw, _empty_fixtures(), _empty_teams())
        assert result.loc[0, "was_home"] == 1.0

    def test_single_fixture_away(self) -> None:
        mgw = _make_merged_gw([
            {"element": 1, "GW": 1, "team": 1, "opponent_team": 2,
             "was_home": False, "goals_conceded": 0, "total_points": 5,
             "fixture": 100},
        ])
        result = compute_opponent_features(mgw, _empty_fixtures(), _empty_teams())
        assert result.loc[0, "was_home"] == 0.0


class TestIsDgw:
    def test_single_fixture_not_dgw(self) -> None:
        mgw = _make_merged_gw([
            {"element": 1, "GW": 1, "team": 1, "opponent_team": 2,
             "was_home": True, "goals_conceded": 0, "total_points": 5,
             "fixture": 100},
        ])
        result = compute_opponent_features(mgw, _empty_fixtures(), _empty_teams())
        assert result.loc[0, "is_dgw"] == 0.0

    def test_double_fixture_is_dgw(self) -> None:
        mgw = _make_merged_gw([
            {"element": 1, "GW": 1, "team": 1, "opponent_team": 2,
             "was_home": True, "goals_conceded": 0, "total_points": 5,
             "fixture": 100},
            {"element": 1, "GW": 1, "team": 1, "opponent_team": 3,
             "was_home": False, "goals_conceded": 1, "total_points": 2,
             "fixture": 101},
            # Need team 2 and 3 rows so the stats are computable
            {"element": 2, "GW": 1, "team": 2, "opponent_team": 1,
             "was_home": False, "goals_conceded": 1, "total_points": 3,
             "fixture": 100},
            {"element": 3, "GW": 1, "team": 3, "opponent_team": 1,
             "was_home": True, "goals_conceded": 0, "total_points": 4,
             "fixture": 101},
        ])
        result = compute_opponent_features(mgw, _empty_fixtures(), _empty_teams())
        row = result[result["element"] == 1]
        assert len(row) == 1
        assert row.iloc[0]["is_dgw"] == 1.0

        # Players 2 and 3 each have a single fixture
        for eid in [2, 3]:
            row = result[result["element"] == eid]
            assert row.iloc[0]["is_dgw"] == 0.0


class TestOppGoalsConcededRolling:
    def test_known_values(self) -> None:
        """Verify rolling goals conceded for the 2-team scenario.

        Team 2 concedes gw*2 goals each GW (2, 4, 6, 8, 10).
        The rolling-5 with shift(1) for team 2 at GW g uses
        values from GW 1..g-1.

        For player 101 (team 1, opponent=team 2):
            GW 1: shift -> NaN (no prior data)
            GW 2: shift -> [2] -> mean 2.0
            GW 3: shift -> [2, 4] -> mean 3.0
            GW 4: shift -> [2, 4, 6] -> mean 4.0
            GW 5: shift -> [2, 4, 6, 8] -> mean 5.0
        """
        mgw = _two_team_scenario()
        result = compute_opponent_features(mgw, _empty_fixtures(), _empty_teams())

        p101 = result[result["element"] == 101].sort_values("GW")
        vals = p101["opp_goals_conceded_r5"].tolist()

        assert np.isnan(vals[0])  # GW 1
        assert vals[1] == pytest.approx(2.0)  # GW 2
        assert vals[2] == pytest.approx(3.0)  # GW 3
        assert vals[3] == pytest.approx(4.0)  # GW 4
        assert vals[4] == pytest.approx(5.0)  # GW 5


class TestEmptyFixturesAndTeams:
    def test_nan_fdr_when_fixtures_empty(self) -> None:
        mgw = _make_merged_gw([
            {"element": 1, "GW": 1, "team": 1, "opponent_team": 2,
             "was_home": True, "goals_conceded": 0, "total_points": 5,
             "fixture": 100},
        ])
        result = compute_opponent_features(mgw, _empty_fixtures(), _empty_teams())
        assert np.isnan(result.loc[0, "fdr"])

    def test_nan_strength_when_teams_empty(self) -> None:
        mgw = _make_merged_gw([
            {"element": 1, "GW": 1, "team": 1, "opponent_team": 2,
             "was_home": True, "goals_conceded": 0, "total_points": 5,
             "fixture": 100},
        ])
        result = compute_opponent_features(mgw, _empty_fixtures(), _empty_teams())
        assert np.isnan(result.loc[0, "opp_strength"])
        assert np.isnan(result.loc[0, "opp_attack_strength"])
        assert np.isnan(result.loc[0, "opp_defence_strength"])


class TestDgwAveraging:
    def test_was_home_averaged_for_dgw(self) -> None:
        """Player plays one home and one away fixture in a DGW."""
        mgw = _make_merged_gw([
            {"element": 1, "GW": 1, "team": 1, "opponent_team": 2,
             "was_home": True, "goals_conceded": 0, "total_points": 5,
             "fixture": 100},
            {"element": 1, "GW": 1, "team": 1, "opponent_team": 3,
             "was_home": False, "goals_conceded": 1, "total_points": 2,
             "fixture": 101},
            # opponent teams must exist
            {"element": 2, "GW": 1, "team": 2, "opponent_team": 1,
             "was_home": False, "goals_conceded": 1, "total_points": 3,
             "fixture": 100},
            {"element": 3, "GW": 1, "team": 3, "opponent_team": 1,
             "was_home": True, "goals_conceded": 0, "total_points": 4,
             "fixture": 101},
        ])
        result = compute_opponent_features(mgw, _empty_fixtures(), _empty_teams())
        row = result[result["element"] == 1]
        # Average of True (1.0) and False (0.0)
        assert row.iloc[0]["was_home"] == pytest.approx(0.5)

    def test_fdr_averaged_for_dgw(self) -> None:
        """FDR should be averaged across two fixtures in a DGW."""
        mgw = _make_merged_gw([
            {"element": 1, "GW": 1, "team": 1, "opponent_team": 2,
             "was_home": True, "goals_conceded": 0, "total_points": 5,
             "fixture": 100},
            {"element": 1, "GW": 1, "team": 1, "opponent_team": 3,
             "was_home": False, "goals_conceded": 1, "total_points": 2,
             "fixture": 101},
            {"element": 2, "GW": 1, "team": 2, "opponent_team": 1,
             "was_home": False, "goals_conceded": 1, "total_points": 3,
             "fixture": 100},
            {"element": 3, "GW": 1, "team": 3, "opponent_team": 1,
             "was_home": True, "goals_conceded": 0, "total_points": 4,
             "fixture": 101},
        ])
        fixtures = pd.DataFrame([
            {"id": 100, "event": 1, "team_h": 1, "team_a": 2,
             "team_h_difficulty": 2, "team_a_difficulty": 4},
            {"id": 101, "event": 1, "team_h": 3, "team_a": 1,
             "team_h_difficulty": 5, "team_a_difficulty": 3},
        ])
        result = compute_opponent_features(mgw, fixtures, _empty_teams())
        row = result[result["element"] == 1]
        # Fixture 100: player is home -> team_h_difficulty = 2 -> 2/5 = 0.4
        # Fixture 101: player is away -> team_a_difficulty = 3 -> 3/5 = 0.6
        # Average: (0.4 + 0.6) / 2 = 0.5
        assert row.iloc[0]["fdr"] == pytest.approx(0.5)


class TestNoLookahead:
    def test_rolling_uses_only_past_data(self) -> None:
        """GW g rolling features must not include data from GW g itself."""
        mgw = _two_team_scenario()
        result = compute_opponent_features(mgw, _empty_fixtures(), _empty_teams())

        # For player 101 at GW 1, there should be no prior data -> NaN
        p101_gw1 = result[
            (result["element"] == 101) & (result["GW"] == 1)
        ]
        assert np.isnan(p101_gw1.iloc[0]["opp_goals_conceded_r5"])
        assert np.isnan(p101_gw1.iloc[0]["opp_pts_conceded_r5"])

        # For player 101 at GW 2, rolling should only use GW 1 data
        p101_gw2 = result[
            (result["element"] == 101) & (result["GW"] == 2)
        ]
        # Team 2 conceded 2*1=2 goals in GW 1
        assert p101_gw2.iloc[0]["opp_goals_conceded_r5"] == pytest.approx(2.0)

    def test_pts_conceded_rolling_no_lookahead(self) -> None:
        """Points conceded rolling also should not include current GW."""
        mgw = _two_team_scenario()
        result = compute_opponent_features(mgw, _empty_fixtures(), _empty_teams())

        # For player 101 (team 1, opponent=team 2):
        # Points conceded by team 2 in GW g = sum of pts of all opponent
        # players they faced = team 1's total pts = 2*gw (two players, each gw pts)
        # GW 1: team 2 conceded 2*1 = 2 pts
        # GW 2: team 2 conceded 2*2 = 4 pts
        #
        # At GW 2: shift -> [2] -> mean = 2.0
        p101_gw2 = result[
            (result["element"] == 101) & (result["GW"] == 2)
        ]
        assert p101_gw2.iloc[0]["opp_pts_conceded_r5"] == pytest.approx(2.0)

        # At GW 3: shift -> [2, 4] -> mean = 3.0
        p101_gw3 = result[
            (result["element"] == 101) & (result["GW"] == 3)
        ]
        assert p101_gw3.iloc[0]["opp_pts_conceded_r5"] == pytest.approx(3.0)


class TestOutputShape:
    def test_all_feature_columns_present(self) -> None:
        mgw = _two_team_scenario()
        result = compute_opponent_features(mgw, _empty_fixtures(), _empty_teams())
        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing column: {col}"
        assert "element" in result.columns
        assert "GW" in result.columns

    def test_one_row_per_element_gw(self) -> None:
        mgw = _two_team_scenario()
        result = compute_opponent_features(mgw, _empty_fixtures(), _empty_teams())
        # 4 players x 5 GWs = 20 rows
        assert len(result) == 20
        assert result.duplicated(subset=["element", "GW"]).sum() == 0

    def test_empty_input(self) -> None:
        mgw = pd.DataFrame(
            columns=[
                "element", "GW", "team", "opponent_team", "was_home",
                "goals_conceded", "total_points", "fixture",
            ]
        )
        result = compute_opponent_features(mgw, _empty_fixtures(), _empty_teams())
        assert len(result) == 0
        for col in FEATURE_COLS:
            assert col in result.columns
