"""Upcoming-GW h2h odds from The Odds API when football-data lacks them (offline)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "collect_football_data_odds", REPO / "scripts" / "collect_football_data_odds.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _h2h(home, away, ph, pd_, pa):
    return {"key": "h2h", "outcomes": [
        {"name": home, "price": ph}, {"name": "Draw", "price": pd_}, {"name": away, "price": pa}]}


@pytest.fixture
def data_dir(tmp_path):
    raw = tmp_path / "raw" / "2026-27"
    raw.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3, 4], "name": ["Man City", "Ipswich Town", "Spurs", "Hull City"]}
                 ).to_csv(raw / "teams.csv", index=False)
    pd.DataFrame({"id": [60, 61, 70], "event": [6, 6, 7], "team_h": [1, 3, 2], "team_a": [2, 4, 1],
                  "kickoff_time": ["2026-10-10T14:00:00Z"] * 2 + ["2026-10-17T14:00:00Z"]}
                 ).to_csv(raw / "fixtures.csv", index=False)
    return tmp_path


def test_matches_gw_fixtures_prefers_pinnacle_else_averages(data_dir):
    mod = _load_script()
    events = [
        {"id": "a", "home_team": "Manchester City", "away_team": "Ipswich Town",
         "commence_time": "2026-10-10T14:00:00Z", "bookmakers": [
             {"key": "pinnacle", "markets": [_h2h("Manchester City", "Ipswich Town", 1.3, 6.0, 11.0)]},
             {"key": "other", "markets": [_h2h("Manchester City", "Ipswich Town", 1.2, 7.0, 15.0)]}]},
        {"id": "b", "home_team": "Tottenham Hotspur", "away_team": "Hull City",
         "commence_time": "2026-10-10T14:00:00Z", "bookmakers": [
             {"key": "b1", "markets": [_h2h("Tottenham Hotspur", "Hull City", 1.6, 4.0, 5.0)]},
             {"key": "b2", "markets": [_h2h("Tottenham Hotspur", "Hull City", 1.8, 4.2, 5.4)]}]},
        # a GW7 match must be ignored for GW6
        {"id": "c", "home_team": "Ipswich Town", "away_team": "Manchester City",
         "commence_time": "2026-10-17T14:00:00Z", "bookmakers": [
             {"key": "pinnacle", "markets": [_h2h("Ipswich Town", "Manchester City", 9.0, 5.5, 1.35)]}]},
    ]
    out = mod.match_odds_api_events(events, "2026-27", data_dir, gw=6)
    by = {(m["home_team"], m["away_team"]): m for m in out}
    assert set(by) == {("Man City", "Ipswich Town"), ("Spurs", "Hull City")}
    assert by[("Man City", "Ipswich Town")]["home_odds"] == pytest.approx(1.3)  # Pinnacle only
    assert by[("Spurs", "Hull City")]["home_odds"] == pytest.approx(1.7)  # mean of 2 books
    assert by[("Spurs", "Hull City")]["away_odds"] == pytest.approx(5.2)
