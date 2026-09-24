"""Tests for odds-based prediction features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl_optimizer.prediction.features.odds import (
    FEATURE_COLS,
    compute_odds_features,
    _match_odds_to_teams,
)
from fpl_optimizer.data.collectors.odds import odds_team_to_fpl_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_merged_gw(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _make_teams_df(teams: dict[int, str]) -> pd.DataFrame:
    return pd.DataFrame([{"id": tid, "name": name} for tid, name in teams.items()])


def _write_odds_json(data_dir: Path, season: str, odds_data: dict) -> None:
    odds_dir = data_dir / "odds"
    odds_dir.mkdir(parents=True, exist_ok=True)
    (odds_dir / f"{season}.json").write_text(
        json.dumps(odds_data), encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Team name mapping tests
# ---------------------------------------------------------------------------

class TestOddsTeamNameMapping:
    def test_exact_match_passthrough(self):
        assert odds_team_to_fpl_name("Arsenal") == "Arsenal"
        assert odds_team_to_fpl_name("Everton") == "Everton"

    def test_known_aliases(self):
        assert odds_team_to_fpl_name("Manchester City") == "Man City"
        assert odds_team_to_fpl_name("Manchester United") == "Man Utd"
        assert odds_team_to_fpl_name("Tottenham Hotspur") == "Spurs"
        assert odds_team_to_fpl_name("Wolverhampton Wanderers") == "Wolves"
        assert odds_team_to_fpl_name("Brighton and Hove Albion") == "Brighton"
        assert odds_team_to_fpl_name("West Ham United") == "West Ham"
        assert odds_team_to_fpl_name("Nottingham Forest") == "Nott'm Forest"
        assert odds_team_to_fpl_name("Newcastle United") == "Newcastle"
        assert odds_team_to_fpl_name("Sheffield United") == "Sheffield Utd"
        assert odds_team_to_fpl_name("Leicester City") == "Leicester"

    def test_unknown_name_returned_as_is(self):
        assert odds_team_to_fpl_name("Unknown FC") == "Unknown FC"


# ---------------------------------------------------------------------------
# Implied probability tests
# ---------------------------------------------------------------------------

class TestMatchOddsToTeams:
    def test_basic_probability_normalisation(self):
        matches = [{
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_odds": 2.0,   # raw implied = 0.5
            "draw_odds": 3.5,   # raw implied ≈ 0.286
            "away_odds": 4.0,   # raw implied = 0.25
        }]
        name_to_id = {"Arsenal": 1, "Chelsea": 7}

        rows = _match_odds_to_teams(matches, name_to_id)
        assert len(rows) == 2

        home_row = [r for r in rows if r["team"] == 1][0]
        away_row = [r for r in rows if r["team"] == 7][0]

        # Probabilities should sum to ~1.0 for each team
        assert abs(home_row["win_prob"] + home_row["draw_prob"] + home_row["loss_prob"] - 1.0) < 1e-6
        assert abs(away_row["win_prob"] + away_row["draw_prob"] + away_row["loss_prob"] - 1.0) < 1e-6

        # Home team should be favourite (higher win prob)
        assert home_row["win_prob"] > away_row["win_prob"]

        # Home's win prob == Away's loss prob
        assert abs(home_row["win_prob"] - away_row["loss_prob"]) < 1e-6

    def test_unmapped_team_skipped(self):
        matches = [{
            "home_team": "Arsenal",
            "away_team": "Unknown FC",
            "home_odds": 1.5,
            "draw_odds": 4.0,
            "away_odds": 6.0,
        }]
        name_to_id = {"Arsenal": 1}  # no "Unknown FC"

        rows = _match_odds_to_teams(matches, name_to_id)
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# Full feature computation tests
# ---------------------------------------------------------------------------

class TestComputeOddsFeatures:
    def test_basic_features(self, tmp_path):
        """Odds features map correctly to players via team."""
        teams = {1: "Arsenal", 7: "Chelsea"}
        teams_df = _make_teams_df(teams)

        merged_gw = _make_merged_gw([
            {"element": 10, "GW": 1, "team": 1},
            {"element": 20, "GW": 1, "team": 7},
        ])

        _write_odds_json(tmp_path, "2023-24", {
            "1": [{
                "event_id": "e1",
                "commence_time": "2023-08-12T14:00:00Z",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_odds": 1.8,
                "draw_odds": 3.5,
                "away_odds": 5.0,
                "last_update": "2023-08-12T12:00:00Z",
            }],
        })

        result = compute_odds_features(tmp_path, "2023-24", merged_gw, teams_df)

        assert len(result) == 2
        for col in FEATURE_COLS:
            assert col in result.columns

        # Arsenal (home favourite): higher win prob
        ars = result[result["element"] == 10].iloc[0]
        che = result[result["element"] == 20].iloc[0]

        assert ars["odds_team_win_prob"] > che["odds_team_win_prob"]
        assert ars["odds_team_strength"] > 0  # favourite
        assert che["odds_team_strength"] < 0  # underdog

    def test_no_odds_file_returns_nan(self, tmp_path):
        """Missing odds file produces NaN features (not an error)."""
        teams_df = _make_teams_df({1: "Arsenal"})
        merged_gw = _make_merged_gw([
            {"element": 10, "GW": 1, "team": 1},
        ])

        result = compute_odds_features(tmp_path, "2023-24", merged_gw, teams_df)

        assert len(result) == 1
        assert np.isnan(result.iloc[0]["odds_team_win_prob"])

    def test_dgw_averages(self, tmp_path):
        """DGW odds are averaged across both matches."""
        teams = {1: "Arsenal", 7: "Chelsea", 8: "Crystal Palace"}
        teams_df = _make_teams_df(teams)

        # Player on Arsenal faces Chelsea AND Crystal Palace in GW5 (DGW)
        merged_gw = _make_merged_gw([
            {"element": 10, "GW": 5, "team": 1},
        ])

        _write_odds_json(tmp_path, "2023-24", {
            "5": [
                {
                    "event_id": "e1",
                    "commence_time": "2023-10-01T14:00:00Z",
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "home_odds": 2.0,
                    "draw_odds": 3.5,
                    "away_odds": 4.0,
                    "last_update": "",
                },
                {
                    "event_id": "e2",
                    "commence_time": "2023-10-04T19:45:00Z",
                    "home_team": "Crystal Palace",
                    "away_team": "Arsenal",
                    "home_odds": 3.0,
                    "draw_odds": 3.2,
                    "away_odds": 2.5,
                    "last_update": "",
                },
            ],
        })

        result = compute_odds_features(tmp_path, "2023-24", merged_gw, teams_df)

        assert len(result) == 1
        row = result.iloc[0]
        # Should be average of home-favourite and away-favourite probabilities
        assert not np.isnan(row["odds_team_win_prob"])
        # Strength should be moderate (strong at home, decent away)
        assert row["odds_team_strength"] > 0

    def test_all_feature_cols_present(self, tmp_path):
        """All expected feature columns are in the output."""
        teams_df = _make_teams_df({1: "Arsenal", 2: "Aston Villa"})
        merged_gw = _make_merged_gw([
            {"element": 10, "GW": 1, "team": 1},
        ])

        _write_odds_json(tmp_path, "2023-24", {
            "1": [{
                "event_id": "e1",
                "commence_time": "2023-08-12T14:00:00Z",
                "home_team": "Arsenal",
                "away_team": "Aston Villa",
                "home_odds": 1.5,
                "draw_odds": 4.0,
                "away_odds": 7.0,
                "last_update": "",
            }],
        })

        result = compute_odds_features(tmp_path, "2023-24", merged_gw, teams_df)
        assert set(FEATURE_COLS).issubset(set(result.columns))
        assert "element" in result.columns
        assert "GW" in result.columns

    def test_empty_merged_gw(self, tmp_path):
        """Empty merged_gw returns empty DataFrame."""
        teams_df = _make_teams_df({1: "Arsenal"})
        merged_gw = pd.DataFrame(columns=["element", "GW", "team"])

        result = compute_odds_features(tmp_path, "2023-24", merged_gw, teams_df)
        assert result.empty
