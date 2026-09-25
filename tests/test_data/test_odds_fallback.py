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


def test_partially_listed_gw_is_completed_by_the_api_without_duplicates(data_dir, monkeypatch):
    """football-data lists one of GW6's two matches: the API adds only the other."""
    mod = _load_script()
    played = pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "AvgH", "AvgD", "AvgA"])
    upcoming = pd.DataFrame([{"Div": "E0", "Date": "10/10/2026", "HomeTeam": "Man City",
                              "AwayTeam": "Ipswich", "AvgH": 1.3, "AvgD": 6.0, "AvgA": 10.0}])
    monkeypatch.setattr(mod, "_download_csv",
                        lambda url: upcoming if "fixtures" in url else played)
    api = [  # the API returns BOTH GW6 matches, in teams.csv names
        {"event_id": "x1", "commence_time": "", "home_team": "Man City", "away_team": "Ipswich Town",
         "home_odds": 1.31, "draw_odds": 6.1, "away_odds": 10.5, "last_update": ""},
        {"event_id": "x2", "commence_time": "", "home_team": "Spurs", "away_team": "Hull City",
         "home_odds": 1.7, "draw_odds": 4.1, "away_odds": 5.2, "last_update": ""},
    ]
    calls = []
    monkeypatch.setattr(mod, "odds_api_upcoming",
                        lambda season, dd, gw: calls.append(gw) or api)
    out = mod.build_season_odds("2026-27", data_dir, include_upcoming=True, gw=6)
    import json
    gw6 = json.loads(out.read_text(encoding="utf-8"))["6"]
    assert calls == [6]                      # the fallback ran for the int GW
    assert len(gw6) == 2                     # no duplicate of the listed match
    assert {m["home_team"] for m in gw6} == {"Man City", "Spurs"}


def test_summary_swap_falls_back_to_copying_when_rename_is_denied(tmp_path, monkeypatch):
    """Windows/OneDrive can deny the directory rename: copy instead of crashing."""
    import json
    from pathlib import Path

    from fpl_optimizer.data.collectors.fpl_live import LiveFPLCollector

    col = LiveFPLCollector(data_dir=tmp_path, season="2026-27")
    base = tmp_path / "fpl_api" / "element_summaries"
    (base / "2026-27").mkdir(parents=True)
    (base / "2026-27" / "1.json").write_text("old", encoding="utf-8")

    def fake_download(tag, max_workers=4):
        d = base / tag
        d.mkdir(parents=True, exist_ok=True)
        (d / "1.json").write_text(json.dumps({"history": []}), encoding="utf-8")
        return True

    monkeypatch.setattr(col.api, "_collect_element_summaries", fake_download)
    real_rename = Path.rename

    def deny_live_dir(self, target):
        if self.name == "2026-27":
            raise PermissionError("in use")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", deny_live_dir)
    col.refresh_element_summaries({"elements": [{"id": 1}]})
    assert json.loads((base / "2026-27" / "1.json").read_text(encoding="utf-8")) == {"history": []}
    assert not (base / "2026-27.staging").exists()
