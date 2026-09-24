"""Tests for understat per-match rolling features."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fpl_optimizer.prediction.id_resolver import IDResolver
from fpl_optimizer.prediction.features.understat import (
    FEATURE_COLUMNS,
    compute_understat_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_match(date: str, xG: float, xA: float, npxG: float,
                shots: int, key_passes: int,
                xGChain: float = 0.0, xGBuildup: float = 0.0) -> dict:
    """Create a single understat match dict with string-typed values."""
    return {
        "date": date,
        "xG": str(xG),
        "xA": str(xA),
        "npxG": str(npxG),
        "shots": str(shots),
        "key_passes": str(key_passes),
        "goals": "0",
        "time": "90",
        "position": "FW",
        "h_team": "TeamA",
        "a_team": "TeamB",
        "h_goals": "1",
        "a_goals": "0",
        "id": "99999",
        "season": "2023",
        "roster_id": "12345",
        "assists": "0",
        "npg": "0",
        "xGChain": str(xGChain),
        "xGBuildup": str(xGBuildup),
    }


def _write_understat_json(
    data_dir: Path, season: str, understat_id: int, matches: list[dict]
) -> None:
    """Write a list of match dicts to the expected understat JSON path."""
    us_dir = data_dir / "understat" / "players" / season
    us_dir.mkdir(parents=True, exist_ok=True)
    json_path = us_dir / f"{understat_id}.json"
    json_path.write_text(json.dumps(matches), encoding="utf-8")


def _gw_dates_series(dates: dict[int, str]) -> pd.Series:
    """Build a GW-indexed Series from {gw: 'YYYY-MM-DD'} dict."""
    return pd.Series(
        {gw: pd.Timestamp(d) for gw, d in dates.items()}
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pred_data_dir_understat(pred_data_dir: Path) -> Path:
    """Extend pred_data_dir with understat JSON files for Kane (code=100).

    Kane has understat_id=1234 in the conftest ID map.
    Creates 5 matches for the 2023-24 season.
    """
    matches = [
        _make_match("2023-08-12", xG=0.5, xA=0.1, npxG=0.4, shots=3, key_passes=1, xGChain=0.7, xGBuildup=0.2),
        _make_match("2023-08-19", xG=1.0, xA=0.2, npxG=0.8, shots=5, key_passes=2, xGChain=1.3, xGBuildup=0.4),
        _make_match("2023-08-26", xG=0.3, xA=0.0, npxG=0.2, shots=2, key_passes=0, xGChain=0.4, xGBuildup=0.1),
        _make_match("2023-09-02", xG=0.8, xA=0.3, npxG=0.7, shots=4, key_passes=3, xGChain=1.0, xGBuildup=0.3),
        _make_match("2023-09-16", xG=0.6, xA=0.1, npxG=0.5, shots=3, key_passes=1, xGChain=0.8, xGBuildup=0.2),
    ]
    _write_understat_json(pred_data_dir, "2023-24", 1234, matches)

    return pred_data_dir


@pytest.fixture
def resolver(pred_data_dir_understat: Path) -> IDResolver:
    return IDResolver(pred_data_dir_understat)


@pytest.fixture
def gw_dates() -> pd.Series:
    """GW dates that sit between the test match dates."""
    return _gw_dates_series({
        1: "2023-08-15",   # after match 1 (Aug 12)
        2: "2023-08-22",   # after match 2 (Aug 19)
        3: "2023-08-29",   # after match 3 (Aug 26)
        4: "2023-09-05",   # after match 4 (Sep 2)
        5: "2023-09-19",   # after match 5 (Sep 16)
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicRolling:
    """Test that rolling computations produce correct values."""

    def test_output_shape_and_columns(
        self, pred_data_dir_understat: Path, resolver: IDResolver, gw_dates: pd.Series
    ) -> None:
        df = compute_understat_features(
            pred_data_dir_understat, "2023-24", resolver, gw_dates,
        )
        # 4 players in the ID map, 5 GWs
        assert len(df) == 4 * 5
        assert list(df.columns[:2]) == ["code", "GW"]
        for col in FEATURE_COLUMNS:
            assert col in df.columns

    def test_xg_rolling_3_values(
        self, pred_data_dir_understat: Path, resolver: IDResolver, gw_dates: pd.Series
    ) -> None:
        """Verify xg_rolling_3 for Kane (code=100) with known xG values.

        Matches (xG): 0.5, 1.0, 0.3, 0.8, 0.6
        GW1: 1 match eligible (0.5) -> rolling_3 mean = 0.5
        GW2: 2 matches (0.5, 1.0) -> rolling_3 mean = 0.75
        GW3: 3 matches (0.5, 1.0, 0.3) -> rolling_3 mean = 0.6
        GW4: 4 matches -> last 3 = (1.0, 0.3, 0.8) -> mean = 0.7
        GW5: 5 matches -> last 3 = (0.3, 0.8, 0.6) -> mean ≈ 0.5667
        """
        df = compute_understat_features(
            pred_data_dir_understat, "2023-24", resolver, gw_dates,
        )
        kane = df[df["code"] == 100].sort_values("GW")
        xg3 = kane["xg_rolling_3"].tolist()

        assert xg3[0] == pytest.approx(0.5, abs=1e-6)           # GW1
        assert xg3[1] == pytest.approx(0.75, abs=1e-6)          # GW2
        assert xg3[2] == pytest.approx(0.6, abs=1e-6)           # GW3
        assert xg3[3] == pytest.approx(0.7, abs=1e-6)           # GW4
        assert xg3[4] == pytest.approx(0.5667, abs=1e-3)        # GW5

    def test_xa_rolling_5_values(
        self, pred_data_dir_understat: Path, resolver: IDResolver, gw_dates: pd.Series
    ) -> None:
        """Verify xa_rolling_5 for Kane (code=100).

        Matches (xA): 0.1, 0.2, 0.0, 0.3, 0.1
        GW5: all 5 matches -> mean = 0.14
        """
        df = compute_understat_features(
            pred_data_dir_understat, "2023-24", resolver, gw_dates,
        )
        kane = df[df["code"] == 100].sort_values("GW")
        xa5_gw5 = kane[kane["GW"] == 5]["xa_rolling_5"].iloc[0]
        assert xa5_gw5 == pytest.approx(0.14, abs=1e-6)

    def test_shots_rolling_5_values(
        self, pred_data_dir_understat: Path, resolver: IDResolver, gw_dates: pd.Series
    ) -> None:
        """Verify shots_rolling_5 for Kane (code=100).

        Matches (shots): 3, 5, 2, 4, 3
        GW5: all 5 matches -> mean = 3.4
        """
        df = compute_understat_features(
            pred_data_dir_understat, "2023-24", resolver, gw_dates,
        )
        kane = df[df["code"] == 100].sort_values("GW")
        shots5_gw5 = kane[kane["GW"] == 5]["shots_rolling_5"].iloc[0]
        assert shots5_gw5 == pytest.approx(3.4, abs=1e-6)


class TestDateAlignment:
    """Test that temporal alignment correctly filters matches."""

    def test_future_matches_excluded(
        self, pred_data_dir_understat: Path, resolver: IDResolver
    ) -> None:
        """GW date set before all matches -> all features NaN."""
        early_dates = _gw_dates_series({1: "2023-01-01"})
        df = compute_understat_features(
            pred_data_dir_understat, "2023-24", resolver, early_dates,
        )
        kane = df[df["code"] == 100]
        for col in FEATURE_COLUMNS:
            assert kane[col].isna().all(), f"{col} should be NaN when no matches available"

    def test_partial_date_alignment(
        self, pred_data_dir_understat: Path, resolver: IDResolver
    ) -> None:
        """GW date between match 2 and match 3 -> only 2 matches used."""
        partial_dates = _gw_dates_series({1: "2023-08-25"})
        df = compute_understat_features(
            pred_data_dir_understat, "2023-24", resolver, partial_dates,
        )
        kane = df[df["code"] == 100]
        # xG matches: 0.5 (Aug 12), 1.0 (Aug 19) => rolling_3 mean = 0.75
        xg3 = kane["xg_rolling_3"].iloc[0]
        assert xg3 == pytest.approx(0.75, abs=1e-6)

    def test_exact_date_boundary(
        self, pred_data_dir_understat: Path, resolver: IDResolver
    ) -> None:
        """GW date exactly equal to a match date -> that match is excluded (strict <)."""
        # Match on Aug 19 should NOT be included when GW date is Aug 19
        boundary_dates = _gw_dates_series({1: "2023-08-19"})
        df = compute_understat_features(
            pred_data_dir_understat, "2023-24", resolver, boundary_dates,
        )
        kane = df[df["code"] == 100]
        # Only match from Aug 12 (xG=0.5) should be included
        xg3 = kane["xg_rolling_3"].iloc[0]
        assert xg3 == pytest.approx(0.5, abs=1e-6)


class TestMissingData:
    """Test graceful handling of missing understat data."""

    def test_missing_understat_id(self, pred_data_dir: Path) -> None:
        """Player without understat mapping -> NaN features.

        Create a player in the ID map without an understat ID.
        """
        # Modify the ID map to add a player without understat ID
        id_map_csv = (
            "code,first_name,second_name,web_name,"
            "16-17,17-18,18-19,19-20,20-21,21-22,22-23,23-24,24-25,"
            "fbref,understat,transfermarkt\n"
            "500,No,Understat,NoUS,,,,50,51,52,53,54,55,xyz,,\n"
        )
        (pred_data_dir / "id_maps" / "master_id_map.csv").write_text(
            id_map_csv, encoding="utf-8"
        )
        resolver = IDResolver(pred_data_dir)
        gw_dates = _gw_dates_series({1: "2023-09-01"})

        df = compute_understat_features(
            pred_data_dir, "2023-24", resolver, gw_dates,
        )
        player = df[df["code"] == 500]
        assert len(player) == 1
        for col in FEATURE_COLUMNS:
            assert pd.isna(player[col].iloc[0]), f"{col} should be NaN"

    def test_missing_json_file(self, pred_data_dir: Path) -> None:
        """Player with understat ID but no JSON file -> NaN features."""
        resolver = IDResolver(pred_data_dir)
        gw_dates = _gw_dates_series({1: "2023-09-01"})

        # Code 100 (Kane) has understat_id=1234 but no JSON file exists
        df = compute_understat_features(
            pred_data_dir, "2023-24", resolver, gw_dates,
        )
        kane = df[df["code"] == 100]
        assert len(kane) == 1
        for col in FEATURE_COLUMNS:
            assert pd.isna(kane[col].iloc[0]), f"{col} should be NaN"

    def test_empty_json_file(self, pred_data_dir: Path) -> None:
        """Player with empty match list -> NaN features."""
        _write_understat_json(pred_data_dir, "2023-24", 1234, [])
        resolver = IDResolver(pred_data_dir)
        gw_dates = _gw_dates_series({1: "2023-09-01"})

        df = compute_understat_features(
            pred_data_dir, "2023-24", resolver, gw_dates,
        )
        kane = df[df["code"] == 100]
        assert len(kane) == 1
        for col in FEATURE_COLUMNS:
            assert pd.isna(kane[col].iloc[0]), f"{col} should be NaN"


class TestThreeMatchSequence:
    """Detailed test of a simple 3-match sequence for all features."""

    def test_all_features_for_3_matches(self, pred_data_dir: Path) -> None:
        """Create exactly 3 matches and verify all 10 features at GW after all 3."""
        matches = [
            _make_match("2023-08-12", xG=0.3, xA=0.1, npxG=0.2, shots=2, key_passes=1),
            _make_match("2023-08-19", xG=0.6, xA=0.4, npxG=0.5, shots=4, key_passes=3),
            _make_match("2023-08-26", xG=0.9, xA=0.2, npxG=0.8, shots=6, key_passes=2),
        ]
        _write_understat_json(pred_data_dir, "2023-24", 1234, matches)
        resolver = IDResolver(pred_data_dir)
        gw_dates = _gw_dates_series({1: "2023-09-01"})  # after all 3 matches

        df = compute_understat_features(
            pred_data_dir, "2023-24", resolver, gw_dates,
        )
        kane = df[df["code"] == 100]
        assert len(kane) == 1

        row = kane.iloc[0]

        # xG values: 0.3, 0.6, 0.9
        assert row["xg_rolling_3"] == pytest.approx(0.6, abs=1e-6)      # mean(0.3,0.6,0.9)
        assert row["xg_rolling_5"] == pytest.approx(0.6, abs=1e-6)      # min_periods=1
        assert row["xg_rolling_10"] == pytest.approx(0.6, abs=1e-6)     # min_periods=1

        # xA values: 0.1, 0.4, 0.2
        assert row["xa_rolling_3"] == pytest.approx(0.2333, abs=1e-3)
        assert row["xa_rolling_5"] == pytest.approx(0.2333, abs=1e-3)
        assert row["xa_rolling_10"] == pytest.approx(0.2333, abs=1e-3)

        # npxG values: 0.2, 0.5, 0.8
        assert row["npxg_rolling_5"] == pytest.approx(0.5, abs=1e-6)
        assert row["npxg_rolling_10"] == pytest.approx(0.5, abs=1e-6)

        # shots values: 2, 4, 6
        assert row["shots_rolling_5"] == pytest.approx(4.0, abs=1e-6)

        # key_passes values: 1, 3, 2
        assert row["key_passes_rolling_5"] == pytest.approx(2.0, abs=1e-6)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_season(self, pred_data_dir: Path) -> None:
        """Season with no codes -> empty DataFrame."""
        resolver = IDResolver(pred_data_dir)
        gw_dates = _gw_dates_series({1: "2023-09-01"})

        df = compute_understat_features(
            pred_data_dir, "2016-17", resolver, gw_dates,
        )
        # No players have element_ids for 2016-17 in the test ID map
        assert len(df) == 0
        assert list(df.columns[:2]) == ["code", "GW"]

    def test_no_gw_dates(self, pred_data_dir: Path) -> None:
        """Empty gw_dates -> empty DataFrame."""
        resolver = IDResolver(pred_data_dir)
        empty_dates = pd.Series(dtype="datetime64[ns]")

        df = compute_understat_features(
            pred_data_dir, "2023-24", resolver, empty_dates,
        )
        assert len(df) == 0

    def test_multiple_gws_accumulate(
        self, pred_data_dir_understat: Path, resolver: IDResolver
    ) -> None:
        """Features should accumulate as more matches become available across GWs."""
        gw_dates = _gw_dates_series({
            1: "2023-08-15",  # 1 match available
            2: "2023-08-22",  # 2 matches available
        })
        df = compute_understat_features(
            pred_data_dir_understat, "2023-24", resolver, gw_dates,
        )
        kane = df[df["code"] == 100].sort_values("GW")

        # GW1: 1 match (xG=0.5), GW2: 2 matches (xG=0.5, 1.0)
        assert kane.iloc[0]["xg_rolling_5"] == pytest.approx(0.5, abs=1e-6)
        assert kane.iloc[1]["xg_rolling_5"] == pytest.approx(0.75, abs=1e-6)


class TestXGChainAndBuildup:
    """Test xGChain and xGBuildup rolling features."""

    def test_xgchain_columns_in_feature_columns(self) -> None:
        """New columns should be in FEATURE_COLUMNS."""
        assert "xgchain_rolling_5" in FEATURE_COLUMNS
        assert "xgchain_rolling_10" in FEATURE_COLUMNS
        assert "xgbuildup_rolling_5" in FEATURE_COLUMNS

    def test_xgchain_rolling_5_values(
        self, pred_data_dir_understat: Path, resolver: IDResolver, gw_dates: pd.Series
    ) -> None:
        """Verify xgchain_rolling_5 for Kane (code=100).

        Matches (xGChain): 0.7, 1.3, 0.4, 1.0, 0.8
        GW5: all 5 matches -> mean = 0.84
        """
        df = compute_understat_features(
            pred_data_dir_understat, "2023-24", resolver, gw_dates,
        )
        kane = df[df["code"] == 100].sort_values("GW")
        xgc5_gw5 = kane[kane["GW"] == 5]["xgchain_rolling_5"].iloc[0]
        assert xgc5_gw5 == pytest.approx(0.84, abs=1e-6)

    def test_xgbuildup_rolling_5_values(
        self, pred_data_dir_understat: Path, resolver: IDResolver, gw_dates: pd.Series
    ) -> None:
        """Verify xgbuildup_rolling_5 for Kane (code=100).

        Matches (xGBuildup): 0.2, 0.4, 0.1, 0.3, 0.2
        GW5: all 5 matches -> mean = 0.24
        """
        df = compute_understat_features(
            pred_data_dir_understat, "2023-24", resolver, gw_dates,
        )
        kane = df[df["code"] == 100].sort_values("GW")
        xgb5_gw5 = kane[kane["GW"] == 5]["xgbuildup_rolling_5"].iloc[0]
        assert xgb5_gw5 == pytest.approx(0.24, abs=1e-6)

    def test_xgchain_rolling_3_accumulates(
        self, pred_data_dir_understat: Path, resolver: IDResolver, gw_dates: pd.Series
    ) -> None:
        """xgchain_rolling_10 should use available matches (min_periods=1)."""
        df = compute_understat_features(
            pred_data_dir_understat, "2023-24", resolver, gw_dates,
        )
        kane = df[df["code"] == 100].sort_values("GW")

        # GW1: 1 match (xGChain=0.7)
        assert kane.iloc[0]["xgchain_rolling_10"] == pytest.approx(0.7, abs=1e-6)
        # GW3: 3 matches (0.7, 1.3, 0.4) -> mean = 0.8
        assert kane.iloc[2]["xgchain_rolling_10"] == pytest.approx(0.8, abs=1e-6)
