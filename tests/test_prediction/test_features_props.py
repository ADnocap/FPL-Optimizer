"""Tests for player-prop odds prediction features."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl_optimizer.prediction.features.props import (
    FEATURE_COLS,
    _SOT_OVERROUND,
    compute_props_features,
    _solve_sot_lambda,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubResolver:
    """Minimal IDResolver stand-in: code -> full name / web name."""

    def __init__(self, full: dict[int, str], web: dict[int, str] | None = None):
        self._full = full
        self._web = web or {}

    def player_full_name(self, code: int) -> str | None:
        return self._full.get(code)

    def player_name(self, code: int) -> str:
        return self._web.get(code, "Unknown")


def _make_teams_df(teams: dict[int, str]) -> pd.DataFrame:
    return pd.DataFrame([{"id": tid, "name": name} for tid, name in teams.items()])


def _write_props_match(
    data_dir: Path, season: str, gw: int, event_id: str, match: dict
) -> None:
    props_dir = data_dir / "props" / season
    props_dir.mkdir(parents=True, exist_ok=True)
    (props_dir / f"gw{gw}_{event_id}.json").write_text(
        json.dumps(match), encoding="utf-8",
    )


def _match_json(gw: int, home: str, away: str, bookmakers: list[dict]) -> dict:
    return {
        "gw": gw,
        "event": {
            "id": "e1",
            "home_team": home,
            "away_team": away,
            "commence_time": "2025-08-16T14:00:00Z",
        },
        "snapshot_requested": "2025-08-16T12:00:00Z",
        "odds": {"data": {"bookmakers": bookmakers}},
    }


# ---------------------------------------------------------------------------
# SOT Poisson inversion
# ---------------------------------------------------------------------------

class TestSolveSotLambda:
    def test_half_line_is_survival_inversion(self):
        # Over 0.5 wins iff X >= 1: P(X>=1) = 1 - exp(-lam)
        p = 0.6
        lam = _solve_sot_lambda(0.5, p)
        assert lam == pytest.approx(-math.log(1 - p), abs=1e-4)

    def test_whole_line_push_aware(self):
        # Over 1.0: win X>=2, push X=1 — implied lambda must exceed the
        # naive P(X>=2) inversion because pushes shrink the denominator
        lam = _solve_sot_lambda(1.0, 0.5)
        assert lam is not None
        # verify round trip: P(X>=2)/(1-P(X=1)) == 0.5
        p_win = 1 - math.exp(-lam) * (1 + lam)
        p_push = math.exp(-lam) * lam
        assert p_win / (1 - p_push) == pytest.approx(0.5, abs=1e-4)

    def test_monotone_in_probability(self):
        lams = [_solve_sot_lambda(1.5, p) for p in (0.2, 0.4, 0.6, 0.8)]
        assert all(a < b for a, b in zip(lams, lams[1:]))

    def test_degenerate_probabilities_rejected(self):
        assert _solve_sot_lambda(0.5, 0.9999) is None
        assert _solve_sot_lambda(0.5, 0.0) is None


# ---------------------------------------------------------------------------
# Full feature computation
# ---------------------------------------------------------------------------

class TestComputePropsFeatures:
    SEASON = "2025-26"

    def _base_inputs(self):
        teams_df = _make_teams_df({1: "Arsenal", 7: "Chelsea"})
        merged_gw = pd.DataFrame([
            {"element": 10, "GW": 1, "team": 1, "code": 100, "name": "Bukayo Saka"},
            {"element": 20, "GW": 1, "team": 7, "code": 200, "name": "Cole Palmer"},
            {"element": 30, "GW": 1, "team": 1, "code": 300, "name": "David Raya"},
        ])
        resolver = _StubResolver(
            full={100: "Bukayo Saka", 200: "Cole Palmer", 300: "David Raya"},
            web={100: "Saka", 200: "Palmer", 300: "Raya"},
        )
        return merged_gw, resolver, teams_df

    def test_basic_features(self, tmp_path):
        merged_gw, resolver, teams_df = self._base_inputs()
        _write_props_match(tmp_path, self.SEASON, 1, "e1", _match_json(
            1, "Arsenal", "Chelsea", [
                {"key": "bookA", "markets": [
                    {"key": "player_goal_scorer_anytime", "outcomes": [
                        {"name": "Yes", "description": "Bukayo Saka", "price": 2.0},
                        {"name": "Yes", "description": "Cole Palmer", "price": 3.0},
                        {"name": "Yes", "description": "No Scorer", "price": 8.0},
                    ]},
                    {"key": "player_assists", "outcomes": [
                        {"name": "Over", "description": "Bukayo Saka",
                         "price": 3.0, "point": 0.5},
                    ]},
                    {"key": "player_shots_on_target", "outcomes": [
                        {"name": "Over", "description": "Bukayo Saka",
                         "price": 1.5, "point": 0.5},
                        {"name": "Over", "description": "Bukayo Saka",
                         "price": 3.0, "point": 1.5},
                    ]},
                    {"key": "player_to_receive_card", "outcomes": [
                        {"name": "Yes", "description": "Cole Palmer", "price": 5.0},
                    ]},
                ]},
                {"key": "bookB", "markets": [
                    {"key": "player_goal_scorer_anytime", "outcomes": [
                        {"name": "Yes", "description": "Bukayo Saka", "price": 2.2},
                    ]},
                ]},
            ],
        ))

        result = compute_props_features(
            merged_gw, tmp_path, self.SEASON, resolver, teams_df,
        )
        assert set(FEATURE_COLS).issubset(result.columns)
        assert len(result) == 3

        saka = result[result["element"] == 10].iloc[0]
        palmer = result[result["element"] == 20].iloc[0]
        raya = result[result["element"] == 30].iloc[0]

        # Shorter AGS price -> higher implied xG
        assert saka["props_xg"] > palmer["props_xg"] > 0
        assert saka["props_n_books"] == 2
        assert palmer["props_n_books"] == 1
        assert saka["props_has_line"] == 1
        assert saka["props_xa"] > 0
        assert palmer["props_card_prob"] > 0

        # SOT from bookA's lowest line (Over 0.5 @ 1.5)
        expected_lam = _solve_sot_lambda(0.5, (1 / 1.5) / _SOT_OVERROUND)
        assert saka["props_sot"] == pytest.approx(expected_lam, abs=1e-6)

        # Unquoted player in a covered GW: the absence IS the signal
        assert raya["props_has_line"] == 0
        assert raya["props_n_books"] == 0
        assert np.isnan(raya["props_xg"])

    def test_accent_and_subset_name_matching(self, tmp_path):
        """Books strip accents and shorten names vs FPL full names."""
        teams_df = _make_teams_df({4: "Newcastle", 7: "Chelsea"})
        merged_gw = pd.DataFrame([
            {"element": 10, "GW": 1, "team": 4, "code": 100,
             "name": "Bruno Guimarães Rodriguez Moura"},
        ])
        resolver = _StubResolver(full={100: "Bruno Guimarães Rodriguez Moura"})
        _write_props_match(tmp_path, self.SEASON, 1, "e1", _match_json(
            1, "Newcastle United", "Chelsea", [
                {"key": "bookA", "markets": [
                    {"key": "player_goal_scorer_anytime", "outcomes": [
                        {"name": "Yes", "description": "Bruno Guimaraes",
                         "price": 4.0},
                    ]},
                ]},
            ],
        ))

        result = compute_props_features(
            merged_gw, tmp_path, self.SEASON, resolver, teams_df,
        )
        assert result[result["element"] == 10].iloc[0]["props_xg"] > 0

    def test_no_props_dir_returns_nan(self, tmp_path):
        merged_gw, resolver, teams_df = self._base_inputs()
        result = compute_props_features(
            merged_gw, tmp_path, self.SEASON, resolver, teams_df,
        )
        assert len(result) == 3
        for col in FEATURE_COLS:
            assert result[col].isna().all()

    def test_empty_merged_gw(self, tmp_path):
        teams_df = _make_teams_df({1: "Arsenal"})
        result = compute_props_features(
            pd.DataFrame(columns=["element", "GW", "team"]),
            tmp_path, self.SEASON, _StubResolver({}), teams_df,
        )
        assert result.empty
