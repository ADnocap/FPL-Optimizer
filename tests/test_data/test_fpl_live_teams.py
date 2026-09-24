"""LiveFPLCollector must label each merged_gw row with the club of THAT fixture."""

from __future__ import annotations

import json

import pandas as pd

from fpl_optimizer.data.collectors.fpl_live import LiveFPLCollector


def _history(element: int, gw: int, fixture: int, was_home: bool, opp: int) -> dict:
    return {
        "element": element, "round": gw, "fixture": fixture, "was_home": was_home,
        "opponent_team": opp, "total_points": 2, "minutes": 90, "value": 50,
        "kickoff_time": f"2026-08-{14 + 7 * gw}T15:00:00Z",
    }


def test_transferred_player_keeps_old_club_on_old_rows(tmp_path):
    season = "2026-27"
    teams = [{"id": i, "name": f"Team{i}", "short_name": f"T{i}",
              "strength_overall_home": 3, "strength_overall_away": 3} for i in (1, 2, 3, 4)]
    # player 10 played GW1 for Team1, then moved to Team2 (current club)
    elements = [{
        "id": 10, "code": 1010, "first_name": "A", "second_name": "Mover",
        "web_name": "Mover", "element_type": 3, "team": 2, "now_cost": 50,
        "selected_by_percent": "1.0", "status": "a",
    }]
    fixtures = [
        {"id": 1, "event": 1, "team_h": 1, "team_a": 3},  # Team1 home GW1
        {"id": 2, "event": 1, "team_h": 4, "team_a": 2},  # Team2 away GW1
        {"id": 3, "event": 2, "team_h": 2, "team_a": 1},  # Team2 home GW2
    ]
    summaries = tmp_path / "fpl_api" / "element_summaries" / season
    summaries.mkdir(parents=True)
    (summaries / "10.json").write_text(json.dumps({"history": [
        _history(10, 1, 1, True, 3),   # for Team1
        _history(10, 2, 3, True, 1),   # for Team2
    ]}), encoding="utf-8")
    bootstrap = {"elements": elements, "teams": teams, "events": [], "total_players": 100}

    collector = LiveFPLCollector(data_dir=tmp_path, season=season)
    raw = collector.build_season_files(bootstrap, fixtures, include_upcoming=False)
    merged = pd.read_csv(raw / "gws" / "merged_gw.csv")
    by_gw = dict(zip(merged["GW"], merged["team"]))
    assert by_gw[1] == "Team1"  # not the current club
    assert by_gw[2] == "Team2"
    # exactly one fixture per (team, GW): no phantom double gameweek
    assert merged.groupby(["team", "GW"])["fixture"].nunique().max() == 1
