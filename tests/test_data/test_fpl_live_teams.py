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


def test_upcoming_row_ownership_is_reconstructed_not_rounded(tmp_path):
    """selected at the deadline = last exact count + transfers in - out."""
    season = "2026-27"
    teams = [{"id": i, "name": f"Team{i}", "short_name": f"T{i}",
              "strength_overall_home": 3, "strength_overall_away": 3} for i in (1, 2)]
    elements = [
        {"id": 10, "code": 1010, "first_name": "A", "second_name": "Owned",
         "web_name": "Owned", "element_type": 3, "team": 1, "now_cost": 50,
         "selected_by_percent": "0.1", "status": "a",
         "transfers_in_event": 700, "transfers_out_event": 200},
        {"id": 11, "code": 1011, "first_name": "B", "second_name": "New",
         "web_name": "New", "element_type": 3, "team": 2, "now_cost": 45,
         "selected_by_percent": "0.2", "status": "a",
         "transfers_in_event": 5, "transfers_out_event": 0},
    ]
    fixtures = [
        {"id": 1, "event": 1, "team_h": 1, "team_a": 2},
        {"id": 2, "event": 2, "team_h": 2, "team_a": 1},
    ]
    summaries = tmp_path / "fpl_api" / "element_summaries" / season
    summaries.mkdir(parents=True)
    hist = _history(10, 1, 1, True, 2)
    hist["selected"] = 12_345
    (summaries / "10.json").write_text(json.dumps({"history": [hist]}), encoding="utf-8")
    events = [{"id": 1, "is_current": True, "is_next": False, "finished": True,
               "deadline_time": "2026-08-21T17:30:00Z"},
              {"id": 2, "is_current": False, "is_next": True, "finished": False,
               "deadline_time": "2099-08-28T17:30:00Z"}]
    bootstrap = {"elements": elements, "teams": teams, "events": events,
                 "total_players": 11_000_000}
    collector = LiveFPLCollector(data_dir=tmp_path, season=season)

    # without the GW1 deadline's manager count: rounded percentage (old rule)
    raw = collector.build_season_files(bootstrap, fixtures, include_upcoming=True)
    up = pd.read_csv(raw / "gws" / "merged_gw.csv").query("GW == 2").set_index("element")
    assert up.loc[10, "selected"] == int(0.1 / 100 * 11_000_000)

    # with it: last exact count + transfers in - out + new managers x share
    collector.snapshot_dir.mkdir(parents=True)
    (collector.snapshot_dir / "gw1_bootstrap.json").write_text(
        json.dumps({"total_players": 10_000_000, "elements": [], "events": []}),
        encoding="utf-8")
    raw = collector.build_season_files(bootstrap, fixtures, include_upcoming=True)
    up = pd.read_csv(raw / "gws" / "merged_gw.csv").query("GW == 2").set_index("element")
    assert up.loc[10, "selected"] == 12_345 + 700 - 200 + round(0.001 * 1_000_000)
    assert up.loc[11, "selected"] == int(0.2 / 100 * 11_000_000)  # no history


def test_popular_players_keep_the_rounded_percentage():
    """>= 0.75% owned: transfers miss part of the drift -> rounded % is closer."""
    from fpl_optimizer.data.collectors.fpl_live import _deadline_selected

    el = {"selected_by_percent": "12.4", "transfers_in_event": 90_000,
          "transfers_out_event": 10_000}
    assert _deadline_selected(el, (4, 1_200_000), 11_000_000, 10_900_000) == int(
        0.124 * 11_000_000)
