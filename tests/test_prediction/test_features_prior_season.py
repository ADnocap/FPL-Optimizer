"""Tests for prior-season features from Understat and FBref data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl_optimizer.prediction.features.prior_season import (
    PRIOR_FEATURE_COLUMNS,
    PREV_SEASON,
    compute_prior_season_features,
)
from fpl_optimizer.prediction.id_resolver import IDResolver


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _make_understat_league(
    data_dir: Path, season: str, players: list[dict]
) -> None:
    """Write a synthetic understat league JSON file."""
    league_dir = data_dir / "understat" / "league"
    league_dir.mkdir(parents=True, exist_ok=True)
    with open(league_dir / f"{season}.json", "w", encoding="utf-8") as f:
        json.dump(players, f)


def _make_fbref_parquet(
    data_dir: Path, season: str, stat_type: str, df: pd.DataFrame
) -> None:
    """Write a synthetic FBref parquet file."""
    fbref_dir = data_dir / "fbref"
    fbref_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(fbref_dir / f"{season}_{stat_type}.parquet", index=False)


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


class TestUnderstatFeatures:
    """Tests for understat-sourced prior-season features."""

    def test_basic_understat_features(self, pred_data_dir: Path) -> None:
        """Known understat values produce correct per-90 features."""
        # Season 2020-21 uses prev season 2019-20
        # Players in test ID map: code=100 -> understat=1234, code=200 -> understat=5678
        _make_understat_league(
            pred_data_dir,
            "2019-20",
            [
                {
                    "id": "1234",
                    "player_name": "Harry Kane",
                    "games": "30",
                    "time": "2700",  # 30 * 90
                    "goals": "18",
                    "xG": "15.0",
                    "assists": "3",
                    "xA": "6.0",
                    "shots": "120",
                    "key_passes": "30",
                    "npg": "15",
                    "npxG": "12.0",
                    "position": "F",
                    "team_title": "Tottenham",
                },
                {
                    "id": "5678",
                    "player_name": "Mohamed Salah",
                    "games": "34",
                    "time": "3060",  # 34 * 90
                    "goals": "22",
                    "xG": "20.4",
                    "assists": "10",
                    "xA": "10.2",
                    "shots": "136",
                    "key_passes": "68",
                    "npg": "20",
                    "npxG": "17.0",
                    "position": "F",
                    "team_title": "Liverpool",
                },
            ],
        )

        resolver = IDResolver(pred_data_dir)
        result = compute_prior_season_features(pred_data_dir, "2020-21", resolver)

        assert "code" in result.columns
        for col in PRIOR_FEATURE_COLUMNS:
            assert col in result.columns

        # Check Kane (code=100)
        kane = result[result["code"] == 100].iloc[0]
        nineties_kane = 2700 / 90.0  # = 30.0
        assert kane["prev_minutes"] == 2700.0
        assert kane["prev_xg_per90"] == pytest.approx(15.0 / nineties_kane)
        assert kane["prev_xa_per90"] == pytest.approx(6.0 / nineties_kane)
        assert kane["prev_npxg_per90"] == pytest.approx(12.0 / nineties_kane)
        assert kane["prev_shots_per90"] == pytest.approx(120.0 / nineties_kane)
        assert kane["prev_key_passes_per90"] == pytest.approx(30.0 / nineties_kane)

        # Check Salah (code=200)
        salah = result[result["code"] == 200].iloc[0]
        nineties_salah = 3060 / 90.0  # = 34.0
        assert salah["prev_minutes"] == 3060.0
        assert salah["prev_xg_per90"] == pytest.approx(20.4 / nineties_salah)
        assert salah["prev_xa_per90"] == pytest.approx(10.2 / nineties_salah)

    def test_low_minutes_gives_nan_per90(self, pred_data_dir: Path) -> None:
        """Players with < 90 minutes get NaN for per-90 stats."""
        _make_understat_league(
            pred_data_dir,
            "2019-20",
            [
                {
                    "id": "1234",
                    "player_name": "Harry Kane",
                    "games": "1",
                    "time": "45",  # < 90
                    "goals": "1",
                    "xG": "0.8",
                    "assists": "0",
                    "xA": "0.1",
                    "shots": "3",
                    "key_passes": "1",
                    "npg": "1",
                    "npxG": "0.7",
                    "position": "F",
                    "team_title": "Tottenham",
                },
            ],
        )

        resolver = IDResolver(pred_data_dir)
        result = compute_prior_season_features(pred_data_dir, "2020-21", resolver)

        kane = result[result["code"] == 100].iloc[0]
        assert kane["prev_minutes"] == 45.0
        assert np.isnan(kane["prev_xg_per90"])
        assert np.isnan(kane["prev_xa_per90"])
        assert np.isnan(kane["prev_npxg_per90"])
        assert np.isnan(kane["prev_shots_per90"])
        assert np.isnan(kane["prev_key_passes_per90"])

    def test_per90_calculation_correctness(self, pred_data_dir: Path) -> None:
        """xG=5.0, time=900 should give per90=0.5."""
        _make_understat_league(
            pred_data_dir,
            "2019-20",
            [
                {
                    "id": "1234",
                    "player_name": "Harry Kane",
                    "games": "10",
                    "time": "900",  # 10 * 90
                    "goals": "5",
                    "xG": "5.0",
                    "assists": "2",
                    "xA": "2.0",
                    "shots": "40",
                    "key_passes": "10",
                    "npg": "4",
                    "npxG": "4.0",
                    "position": "F",
                    "team_title": "Tottenham",
                },
            ],
        )

        resolver = IDResolver(pred_data_dir)
        result = compute_prior_season_features(pred_data_dir, "2020-21", resolver)

        kane = result[result["code"] == 100].iloc[0]
        # 900 / 90 = 10 nineties, 5.0 / 10 = 0.5
        assert kane["prev_xg_per90"] == pytest.approx(0.5)
        assert kane["prev_xa_per90"] == pytest.approx(0.2)
        assert kane["prev_npxg_per90"] == pytest.approx(0.4)
        assert kane["prev_shots_per90"] == pytest.approx(4.0)
        assert kane["prev_key_passes_per90"] == pytest.approx(1.0)


class TestFirstSeason:
    """Test that seasons with no prior return NaN."""

    def test_first_season_returns_nan(self, pred_data_dir: Path) -> None:
        """Season 2016-17 has no prior season → all features are NaN."""
        resolver = IDResolver(pred_data_dir)
        result = compute_prior_season_features(pred_data_dir, "2016-17", resolver)

        # 2016-17 has no element_ids in the test data (columns 19-20 onwards)
        # so result should be empty or have all NaN features
        for col in PRIOR_FEATURE_COLUMNS:
            assert col in result.columns

        # All feature values should be NaN
        for col in PRIOR_FEATURE_COLUMNS:
            assert result[col].isna().all(), f"Expected all NaN for {col}"

    def test_season_not_in_prev_map(self, pred_data_dir: Path) -> None:
        """A season without PREV_SEASON entry returns all NaN."""
        resolver = IDResolver(pred_data_dir)
        result = compute_prior_season_features(pred_data_dir, "2015-16", resolver)

        for col in PRIOR_FEATURE_COLUMNS:
            assert col in result.columns
        assert len(result) == 0  # no codes for unknown season


class TestFBrefFeatures:
    """Tests for FBref-sourced prior-season features."""

    def test_fbref_features_from_parquets(self, pred_data_dir: Path) -> None:
        """FBref parquets with known values produce correct features."""
        prev_season = "2019-20"

        # Create shooting parquet
        shooting_df = pd.DataFrame(
            {
                "Unnamed: 1_level_0_Player": ["Kane", "Salah", "Other Player"],
                "Unnamed: 7_level_0_90s": [30.0, 34.0, 20.0],
                "Standard_SoT": [60, 68, 40],
                "Standard_SoT/90": [2.0, 2.0, 2.0],
            }
        )
        _make_fbref_parquet(pred_data_dir, prev_season, "shooting", shooting_df)

        # Create passing parquet
        passing_df = pd.DataFrame(
            {
                "Unnamed: 1_level_0_Player": ["Kane", "Salah", "Other Player"],
                "Unnamed: 7_level_0_90s": [30.0, 34.0, 20.0],
                "Total_Cmp%": [75.0, 80.0, 70.0],
                "Total_PrgDist": [3000, 6800, 2000],
            }
        )
        _make_fbref_parquet(pred_data_dir, prev_season, "passing", passing_df)

        # Create defense parquet
        defense_df = pd.DataFrame(
            {
                "Unnamed: 1_level_0_Player": ["Kane", "Salah", "Van Dijk"],
                "Unnamed: 7_level_0_90s": [30.0, 34.0, 35.0],
                "Unnamed: 21_level_0_Tkl+Int": [30, 34, 105],
                "Blocks_Blocks": [15, 17, 70],
            }
        )
        _make_fbref_parquet(pred_data_dir, prev_season, "defense", defense_df)

        # Create standard parquet
        standard_df = pd.DataFrame(
            {
                "Unnamed: 1_level_0_Player": ["Kane", "Salah", "Van Dijk"],
                "Playing Time_90s": [30.0, 34.0, 35.0],
                "Per 90 Minutes_Gls": [0.6, 0.65, 0.1],
            }
        )
        _make_fbref_parquet(pred_data_dir, prev_season, "standard", standard_df)

        resolver = IDResolver(pred_data_dir)
        result = compute_prior_season_features(pred_data_dir, "2020-21", resolver)

        # Kane (code=100)
        kane = result[result["code"] == 100].iloc[0]
        assert kane["prev_sot_per90"] == pytest.approx(2.0)
        assert kane["prev_pass_cmp_pct"] == pytest.approx(75.0)
        assert kane["prev_prog_dist_per90"] == pytest.approx(3000 / 30.0)
        assert kane["prev_tkl_int_per90"] == pytest.approx(30 / 30.0)
        assert kane["prev_blocks_per90"] == pytest.approx(15 / 30.0)
        assert kane["prev_gls_per90"] == pytest.approx(0.6)

        # Salah (code=200)
        salah = result[result["code"] == 200].iloc[0]
        assert salah["prev_sot_per90"] == pytest.approx(2.0)
        assert salah["prev_pass_cmp_pct"] == pytest.approx(80.0)
        assert salah["prev_prog_dist_per90"] == pytest.approx(6800 / 34.0)
        assert salah["prev_gls_per90"] == pytest.approx(0.65)

        # Van Dijk (code=300)
        vvd = result[result["code"] == 300].iloc[0]
        assert vvd["prev_tkl_int_per90"] == pytest.approx(105 / 35.0)
        assert vvd["prev_blocks_per90"] == pytest.approx(70 / 35.0)
        assert vvd["prev_gls_per90"] == pytest.approx(0.1)


class TestMissingData:
    """Tests for graceful handling of missing data files."""

    def test_missing_understat_file(self, pred_data_dir: Path) -> None:
        """Missing understat JSON produces NaN for understat features."""
        # Don't create any understat file — only FBref
        prev_season = "2019-20"

        shooting_df = pd.DataFrame(
            {
                "Unnamed: 1_level_0_Player": ["Kane"],
                "Unnamed: 7_level_0_90s": [30.0],
                "Standard_SoT/90": [2.0],
            }
        )
        _make_fbref_parquet(pred_data_dir, prev_season, "shooting", shooting_df)

        # Create empty parquets for other stat types so they don't error
        for stat_type in ["passing", "defense", "standard"]:
            empty_df = pd.DataFrame({"Unnamed: 1_level_0_Player": []})
            _make_fbref_parquet(pred_data_dir, prev_season, stat_type, empty_df)

        resolver = IDResolver(pred_data_dir)
        result = compute_prior_season_features(pred_data_dir, "2020-21", resolver)

        # Understat features should all be NaN
        kane = result[result["code"] == 100].iloc[0]
        assert np.isnan(kane["prev_xg_per90"])
        assert np.isnan(kane["prev_xa_per90"])
        assert np.isnan(kane["prev_npxg_per90"])
        assert np.isnan(kane["prev_shots_per90"])
        assert np.isnan(kane["prev_key_passes_per90"])
        assert np.isnan(kane["prev_minutes"])

        # FBref feature that was provided should still work
        assert kane["prev_sot_per90"] == pytest.approx(2.0)

    def test_missing_fbref_files(self, pred_data_dir: Path) -> None:
        """Missing FBref parquets produce NaN for FBref features."""
        _make_understat_league(
            pred_data_dir,
            "2019-20",
            [
                {
                    "id": "1234",
                    "player_name": "Harry Kane",
                    "games": "30",
                    "time": "2700",
                    "goals": "18",
                    "xG": "15.0",
                    "assists": "3",
                    "xA": "6.0",
                    "shots": "120",
                    "key_passes": "30",
                    "npg": "15",
                    "npxG": "12.0",
                    "position": "F",
                    "team_title": "Tottenham",
                },
            ],
        )

        resolver = IDResolver(pred_data_dir)
        result = compute_prior_season_features(pred_data_dir, "2020-21", resolver)

        kane = result[result["code"] == 100].iloc[0]
        # Understat features should be populated
        assert kane["prev_minutes"] == 2700.0
        assert kane["prev_xg_per90"] == pytest.approx(15.0 / 30.0)

        # FBref features should all be NaN
        assert np.isnan(kane["prev_sot_per90"])
        assert np.isnan(kane["prev_pass_cmp_pct"])
        assert np.isnan(kane["prev_prog_dist_per90"])
        assert np.isnan(kane["prev_tkl_int_per90"])
        assert np.isnan(kane["prev_blocks_per90"])
        assert np.isnan(kane["prev_gls_per90"])

    def test_unmatched_player_gets_nan(self, pred_data_dir: Path) -> None:
        """A player in the ID map but not in understat/fbref data gets NaN."""
        # Only include data for code=100 (understat 1234), not code=200 (5678)
        _make_understat_league(
            pred_data_dir,
            "2019-20",
            [
                {
                    "id": "1234",
                    "player_name": "Harry Kane",
                    "games": "30",
                    "time": "2700",
                    "goals": "18",
                    "xG": "15.0",
                    "assists": "3",
                    "xA": "6.0",
                    "shots": "120",
                    "key_passes": "30",
                    "npg": "15",
                    "npxG": "12.0",
                    "position": "F",
                    "team_title": "Tottenham",
                },
            ],
        )

        resolver = IDResolver(pred_data_dir)
        result = compute_prior_season_features(pred_data_dir, "2020-21", resolver)

        # Salah (code=200) should have all NaN understat features
        salah = result[result["code"] == 200].iloc[0]
        assert np.isnan(salah["prev_xg_per90"])
        assert np.isnan(salah["prev_minutes"])


class TestOutputFormat:
    """Tests for correct output shape and columns."""

    def test_output_columns(self, pred_data_dir: Path) -> None:
        """Output DataFrame has code + all 12 feature columns."""
        resolver = IDResolver(pred_data_dir)
        result = compute_prior_season_features(pred_data_dir, "2020-21", resolver)

        expected_cols = ["code"] + PRIOR_FEATURE_COLUMNS
        assert list(result.columns) == expected_cols

    def test_one_row_per_code(self, pred_data_dir: Path) -> None:
        """Output has exactly one row per code in the season."""
        _make_understat_league(
            pred_data_dir,
            "2019-20",
            [
                {
                    "id": "1234",
                    "player_name": "Harry Kane",
                    "games": "30",
                    "time": "2700",
                    "goals": "18",
                    "xG": "15.0",
                    "assists": "3",
                    "xA": "6.0",
                    "shots": "120",
                    "key_passes": "30",
                    "npg": "15",
                    "npxG": "12.0",
                    "position": "F",
                    "team_title": "Tottenham",
                },
            ],
        )

        resolver = IDResolver(pred_data_dir)
        result = compute_prior_season_features(pred_data_dir, "2020-21", resolver)

        codes_in_season = resolver.all_codes_for_season("2020-21")
        assert len(result) == len(codes_in_season)
        assert set(result["code"].tolist()) == set(codes_in_season)

    def test_prev_season_mapping(self) -> None:
        """PREV_SEASON dict covers the expected seasons."""
        assert PREV_SEASON["2024-25"] == "2023-24"
        assert PREV_SEASON["2017-18"] == "2016-17"
        assert "2016-17" not in PREV_SEASON


# -------------------------------------------------------------------------
# FotMob features
# -------------------------------------------------------------------------


def _make_fotmob_json(
    data_dir: Path, season: str, data: dict[str, list[dict]]
) -> None:
    """Write a synthetic FotMob JSON file."""
    fotmob_dir = data_dir / "fotmob"
    fotmob_dir.mkdir(parents=True, exist_ok=True)
    with open(fotmob_dir / f"{season}.json", "w", encoding="utf-8") as f:
        json.dump(data, f)


class TestFotMobFeatures:
    """Tests for FotMob-sourced prior-season features."""

    def test_fotmob_features_from_json(self, pred_data_dir: Path) -> None:
        """Known FotMob values produce correct features."""
        _make_fotmob_json(
            pred_data_dir,
            "2019-20",
            {
                "accurate_pass": [
                    {"player_name": "Harry Kane", "stat_value": "35.2", "sub_stat_value": "75.3%"},
                    {"player_name": "Mohamed Salah", "stat_value": "42.1", "sub_stat_value": "82.0%"},
                ],
                "outfielder_block": [
                    {"player_name": "Virgil van Dijk", "stat_value": "1.5"},
                    {"player_name": "Harry Kane", "stat_value": "0.3"},
                ],
                "accurate_long_balls": [
                    {"player_name": "Harry Kane", "stat_value": "2.1"},
                    {"player_name": "Virgil van Dijk", "stat_value": "3.8"},
                ],
            },
        )

        resolver = IDResolver(pred_data_dir)
        result = compute_prior_season_features(pred_data_dir, "2020-21", resolver)

        # Kane (code=100): pass_cmp_pct=75.3, blocks=0.3, prog_dist=2.1
        kane = result[result["code"] == 100].iloc[0]
        assert kane["prev_pass_cmp_pct"] == pytest.approx(75.3)
        assert kane["prev_blocks_per90"] == pytest.approx(0.3)
        assert kane["prev_prog_dist_per90"] == pytest.approx(2.1)

        # Salah (code=200): pass_cmp_pct=82.0
        salah = result[result["code"] == 200].iloc[0]
        assert salah["prev_pass_cmp_pct"] == pytest.approx(82.0)

        # Van Dijk (code=300): blocks=1.5, prog_dist=3.8
        vvd = result[result["code"] == 300].iloc[0]
        assert vvd["prev_blocks_per90"] == pytest.approx(1.5)
        assert vvd["prev_prog_dist_per90"] == pytest.approx(3.8)

    def test_fotmob_fallback_for_missing_fbref(self, pred_data_dir: Path) -> None:
        """FotMob fills NaN where FBref data is missing."""
        # Create FBref shooting + standard (no passing or defense data)
        prev_season = "2019-20"
        shooting_df = pd.DataFrame({
            "Unnamed: 1_level_0_Player": ["Kane"],
            "Unnamed: 7_level_0_90s": [30.0],
            "Standard_SoT/90": [2.0],
        })
        _make_fbref_parquet(pred_data_dir, prev_season, "shooting", shooting_df)

        standard_df = pd.DataFrame({
            "Unnamed: 1_level_0_Player": ["Kane"],
            "Playing Time_90s": [30.0],
            "Per 90 Minutes_Gls": [0.6],
        })
        _make_fbref_parquet(pred_data_dir, prev_season, "standard", standard_df)

        # No passing/defense parquets → those FBref features will be NaN

        # FotMob provides pass_cmp_pct and blocks_per90
        _make_fotmob_json(
            pred_data_dir,
            prev_season,
            {
                "accurate_pass": [
                    {"player_name": "Harry Kane", "stat_value": "30", "sub_stat_value": "78.5%"},
                ],
                "outfielder_block": [
                    {"player_name": "Harry Kane", "stat_value": "0.4"},
                ],
                "accurate_long_balls": [
                    {"player_name": "Harry Kane", "stat_value": "1.9"},
                ],
            },
        )

        # Also provide understat so the merge works cleanly
        _make_understat_league(
            pred_data_dir, prev_season,
            [{
                "id": "1234", "player_name": "Harry Kane",
                "games": "30", "time": "2700", "goals": "18",
                "xG": "15.0", "assists": "3", "xA": "6.0",
                "shots": "120", "key_passes": "30",
                "npg": "15", "npxG": "12.0",
                "position": "F", "team_title": "Tottenham",
            }],
        )

        resolver = IDResolver(pred_data_dir)
        result = compute_prior_season_features(pred_data_dir, "2020-21", resolver)

        kane = result[result["code"] == 100].iloc[0]
        # FBref features that are present should still come from FBref
        assert kane["prev_sot_per90"] == pytest.approx(2.0)
        assert kane["prev_gls_per90"] == pytest.approx(0.6)

        # FBref features that are missing should be filled by FotMob
        assert kane["prev_pass_cmp_pct"] == pytest.approx(78.5)
        assert kane["prev_blocks_per90"] == pytest.approx(0.4)
        assert kane["prev_prog_dist_per90"] == pytest.approx(1.9)

    def test_fotmob_does_not_overwrite_fbref(self, pred_data_dir: Path) -> None:
        """When FBref has data, FotMob does not overwrite it."""
        prev_season = "2019-20"

        # FBref passing with data
        passing_df = pd.DataFrame({
            "Unnamed: 1_level_0_Player": ["Kane"],
            "Unnamed: 7_level_0_90s": [30.0],
            "Total_Cmp%": [80.0],
            "Total_PrgDist": [3000],
        })
        _make_fbref_parquet(pred_data_dir, prev_season, "passing", passing_df)

        # FBref defense with data
        defense_df = pd.DataFrame({
            "Unnamed: 1_level_0_Player": ["Kane"],
            "Unnamed: 7_level_0_90s": [30.0],
            "Unnamed: 21_level_0_Tkl+Int": [30],
            "Blocks_Blocks": [15],
        })
        _make_fbref_parquet(pred_data_dir, prev_season, "defense", defense_df)

        # FotMob with different values
        _make_fotmob_json(
            pred_data_dir,
            prev_season,
            {
                "accurate_pass": [
                    {"player_name": "Harry Kane", "stat_value": "30", "sub_stat_value": "99.9%"},
                ],
                "outfielder_block": [
                    {"player_name": "Harry Kane", "stat_value": "9.9"},
                ],
                "accurate_long_balls": [
                    {"player_name": "Harry Kane", "stat_value": "9.9"},
                ],
            },
        )

        resolver = IDResolver(pred_data_dir)
        result = compute_prior_season_features(pred_data_dir, "2020-21", resolver)

        kane = result[result["code"] == 100].iloc[0]
        # FBref values should be preserved, NOT overwritten by FotMob
        assert kane["prev_pass_cmp_pct"] == pytest.approx(80.0)
        assert kane["prev_blocks_per90"] == pytest.approx(15 / 30.0)
        assert kane["prev_prog_dist_per90"] == pytest.approx(3000 / 30.0)

    def test_fotmob_name_matching(self, pred_data_dir: Path) -> None:
        """Normalized names match correctly (accents, case)."""
        _make_fotmob_json(
            pred_data_dir,
            "2019-20",
            {
                "accurate_pass": [
                    # Different casing
                    {"player_name": "harry kane", "stat_value": "30", "sub_stat_value": "75%"},
                    # Full name with different order shouldn't match — but exact match should
                    {"player_name": "Mohamed Salah", "stat_value": "40", "sub_stat_value": "82%"},
                ],
                "outfielder_block": [],
                "accurate_long_balls": [],
            },
        )

        resolver = IDResolver(pred_data_dir)
        result = compute_prior_season_features(pred_data_dir, "2020-21", resolver)

        kane = result[result["code"] == 100].iloc[0]
        assert kane["prev_pass_cmp_pct"] == pytest.approx(75.0)

        salah = result[result["code"] == 200].iloc[0]
        assert salah["prev_pass_cmp_pct"] == pytest.approx(82.0)
