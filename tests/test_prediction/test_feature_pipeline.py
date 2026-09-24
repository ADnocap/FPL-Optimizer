"""Tests for the feature pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fpl_optimizer.prediction.id_resolver import IDResolver
from fpl_optimizer.prediction.feature_pipeline import FeaturePipeline


def _make_merged_gw(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal merged_gw DataFrame."""
    defaults = {
        "name": "Player",
        "total_points": 2,
        "minutes": 90,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 0,
        "own_goals": 0,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "saves": 0,
        "bonus": 0,
        "bps": 10,
        "influence": 5.0,
        "creativity": 3.0,
        "threat": 2.0,
        "ict_index": 10.0,
        "value": 50,
        "transfers_balance": 0,
        "selected": 500000,
        "transfers_in": 0,
        "transfers_out": 0,
        "was_home": True,
        "opponent_team": 2,
        "fixture": 1,
        "kickoff_time": "2023-08-12 15:00:00",
        "round": 1,
        "position": "MID",
        "team": 1,
        "expected_goals": 0.0,
        "expected_assists": 0.0,
        "expected_goal_involvements": 0.0,
        "expected_goals_conceded": 0.0,
        "xP": 2.0,
    }
    full_rows = []
    for r in rows:
        row = {**defaults, **r}
        full_rows.append(row)
    return pd.DataFrame(full_rows)


@pytest.fixture
def pipeline_data_dir(pred_data_dir: Path) -> Path:
    """Extend pred_data_dir with minimal raw season data."""
    season = "2023-24"
    raw_dir = pred_data_dir / "raw" / season
    gws_dir = raw_dir / "gws"
    gws_dir.mkdir(parents=True)

    # Create merged_gw.csv with 2 players, 3 GWs
    # Kane (code=100, element=14 in 2023-24) and Salah (code=200, element=24 in 2023-24)
    rows = []
    for gw in range(1, 4):
        rows.append({
            "name": "Kane", "element": 14, "GW": gw,
            "total_points": gw * 2, "minutes": 90,
            "goals_scored": 1 if gw == 2 else 0,
            "team": 1, "opponent_team": 2, "fixture": gw,
            "was_home": gw % 2 == 1, "position": "FWD",
            "kickoff_time": f"2023-08-{10 + gw * 7} 15:00:00",
        })
        rows.append({
            "name": "Salah", "element": 24, "GW": gw,
            "total_points": gw * 3, "minutes": 90,
            "goals_scored": 1 if gw == 1 else 0,
            "team": 2, "opponent_team": 1, "fixture": gw,
            "was_home": gw % 2 == 0, "position": "MID",
            "kickoff_time": f"2023-08-{10 + gw * 7} 15:00:00",
        })

    merged_gw = _make_merged_gw(rows)
    merged_gw.to_csv(gws_dir / "merged_gw.csv", index=False)

    # Create cleaned_players.csv
    cp = pd.DataFrame({
        "id": [14, 24],
        "first_name": ["Harry", "Mohamed"],
        "second_name": ["Kane", "Salah"],
        "web_name": ["Kane", "Salah"],
        "element_type": [4, 3],  # FWD, MID
        "team": [1, 2],
    })
    cp.to_csv(raw_dir / "cleaned_players.csv", index=False)

    # Empty fixtures and teams (simulate pre-2019 season)
    pd.DataFrame().to_csv(raw_dir / "fixtures.csv", index=False)
    pd.DataFrame().to_csv(raw_dir / "teams.csv", index=False)

    return pred_data_dir


class TestFeaturePipeline:
    def test_builds_dataframe_with_expected_columns(self, pipeline_data_dir: Path) -> None:
        resolver = IDResolver(pipeline_data_dir)
        pipeline = FeaturePipeline(pipeline_data_dir, resolver, ["2023-24"])
        df = pipeline.build()

        assert not df.empty
        assert "code" in df.columns
        assert "season" in df.columns
        assert "GW" in df.columns
        assert "position" in df.columns
        assert "target" in df.columns
        # Check some vaastav features
        assert "pts_rolling_3" in df.columns
        assert "value" in df.columns

    def test_correct_number_of_rows(self, pipeline_data_dir: Path) -> None:
        resolver = IDResolver(pipeline_data_dir)
        pipeline = FeaturePipeline(pipeline_data_dir, resolver, ["2023-24"])
        df = pipeline.build()

        # 2 players x 3 GWs = 6 rows
        assert len(df) == 6

    def test_season_column_populated(self, pipeline_data_dir: Path) -> None:
        resolver = IDResolver(pipeline_data_dir)
        pipeline = FeaturePipeline(pipeline_data_dir, resolver, ["2023-24"])
        df = pipeline.build()

        assert (df["season"] == "2023-24").all()

    def test_target_is_total_points(self, pipeline_data_dir: Path) -> None:
        resolver = IDResolver(pipeline_data_dir)
        pipeline = FeaturePipeline(pipeline_data_dir, resolver, ["2023-24"])
        df = pipeline.build()

        # Kane GW1: 2 pts, GW2: 4 pts, GW3: 6 pts
        kane = df[df["code"] == 100].sort_values("GW")
        assert list(kane["target"]) == [2, 4, 6]

    def test_position_mapped_correctly(self, pipeline_data_dir: Path) -> None:
        resolver = IDResolver(pipeline_data_dir)
        pipeline = FeaturePipeline(pipeline_data_dir, resolver, ["2023-24"])
        df = pipeline.build()

        kane = df[df["code"] == 100]
        assert (kane["position"] == "FWD").all()

        salah = df[df["code"] == 200]
        assert (salah["position"] == "MID").all()

    def test_empty_season_returns_empty(self, pipeline_data_dir: Path) -> None:
        resolver = IDResolver(pipeline_data_dir)
        pipeline = FeaturePipeline(pipeline_data_dir, resolver, ["2016-17"])
        df = pipeline.build()

        # 2016-17 has no raw data in our test dir
        assert df.empty

    def test_multiple_seasons(self, pipeline_data_dir: Path) -> None:
        """Pipeline with mix of valid and missing seasons."""
        resolver = IDResolver(pipeline_data_dir)
        pipeline = FeaturePipeline(
            pipeline_data_dir, resolver, ["2016-17", "2023-24"]
        )
        df = pipeline.build()

        # Only 2023-24 has data
        assert len(df) == 6
        assert (df["season"] == "2023-24").all()

    def test_vaastav_features_no_lookahead(self, pipeline_data_dir: Path) -> None:
        """Verify rolling features at GW1 are NaN (no prior data)."""
        resolver = IDResolver(pipeline_data_dir)
        pipeline = FeaturePipeline(pipeline_data_dir, resolver, ["2023-24"])
        df = pipeline.build()

        gw1 = df[df["GW"] == 1]
        assert gw1["pts_rolling_3"].isna().all()


class TestDerivedFeatures:
    """Test derived interaction features added by _add_derived_features()."""

    def test_derived_columns_exist(self, pipeline_data_dir: Path) -> None:
        """Derived features should be present in the output."""
        resolver = IDResolver(pipeline_data_dir)
        pipeline = FeaturePipeline(pipeline_data_dir, resolver, ["2023-24"])
        df = pipeline.build()

        for col in ["pts_per_min_5", "pts_form_delta", "gw_phase"]:
            assert col in df.columns, f"Missing derived column: {col}"

    def test_gw_phase_values(self, pipeline_data_dir: Path) -> None:
        """gw_phase should be GW / 38."""
        resolver = IDResolver(pipeline_data_dir)
        pipeline = FeaturePipeline(pipeline_data_dir, resolver, ["2023-24"])
        df = pipeline.build()

        gw1 = df[df["GW"] == 1].iloc[0]
        assert gw1["gw_phase"] == pytest.approx(1 / 38.0)

        gw3 = df[df["GW"] == 3].iloc[0]
        assert gw3["gw_phase"] == pytest.approx(3 / 38.0)

    def test_pts_form_delta(self, pipeline_data_dir: Path) -> None:
        """pts_form_delta = pts_rolling_3 - pts_rolling_10."""
        resolver = IDResolver(pipeline_data_dir)
        pipeline = FeaturePipeline(pipeline_data_dir, resolver, ["2023-24"])
        df = pipeline.build()

        # At GW3, both pts_rolling_3 and pts_rolling_10 should exist
        gw3 = df[(df["GW"] == 3) & (df["code"] == 100)].iloc[0]
        expected = gw3["pts_rolling_3"] - gw3["pts_rolling_10"]
        assert gw3["pts_form_delta"] == pytest.approx(expected)
