"""Tests for vaastav rolling features module."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fpl_optimizer.prediction.features.vaastav import (
    FEATURE_COLUMNS,
    compute_vaastav_features,
)


def _make_player_rows(
    element: int,
    gw_data: list[dict],
) -> pd.DataFrame:
    """Build a DataFrame of raw merged_gw rows for a single player.

    Each entry in gw_data should have at minimum ``GW`` and ``total_points``.
    Missing columns get sensible defaults.
    """
    defaults = {
        "total_points": 0,
        "minutes": 90,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 0,
        "bonus": 0,
        "bps": 0,
        "influence": 0.0,
        "creativity": 0.0,
        "threat": 0.0,
        "ict_index": 0.0,
        "value": 100,
        "selected": 5_000_000,
        "saves": 0,
        "goals_conceded": 0,
        "transfers_balance": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "starts": 0,
        "expected_goals": 0.0,
        "expected_assists": 0.0,
        "expected_goal_involvements": 0.0,
        "expected_goals_conceded": 0.0,
    }
    rows = []
    for gw in gw_data:
        row = {**defaults, **gw, "element": element}
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Basic rolling computation on a 5-GW sequence for one player
# ---------------------------------------------------------------------------
class TestBasicRolling:
    """Verify rolling windows produce correct values on a simple sequence."""

    @pytest.fixture()
    def single_player(self) -> pd.DataFrame:
        """One player, 5 consecutive GWs, known total_points = [2, 4, 6, 8, 10]."""
        return _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "total_points": 2, "minutes": 90},
                {"GW": 2, "total_points": 4, "minutes": 80},
                {"GW": 3, "total_points": 6, "minutes": 70},
                {"GW": 4, "total_points": 8, "minutes": 60},
                {"GW": 5, "total_points": 10, "minutes": 50},
            ],
        )

    def test_pts_rolling_3(self, single_player: pd.DataFrame) -> None:
        result = compute_vaastav_features(single_player)
        pts3 = result.set_index("GW")["pts_rolling_3"]

        # GW1: shift means no prior data -> NaN
        assert math.isnan(pts3[1])
        # GW2: shift -> only GW1 data (2). rolling(3, min_periods=1) -> mean([2]) = 2.0
        assert pts3[2] == pytest.approx(2.0)
        # GW3: shift -> GW1,GW2 data (2,4). rolling(3, min_periods=1) -> mean([2,4]) = 3.0
        assert pts3[3] == pytest.approx(3.0)
        # GW4: shift -> GW1,GW2,GW3 (2,4,6). rolling(3) -> mean([2,4,6]) = 4.0
        assert pts3[4] == pytest.approx(4.0)
        # GW5: shift -> GW2,GW3,GW4 (4,6,8). rolling(3) -> mean([4,6,8]) = 6.0
        assert pts3[5] == pytest.approx(6.0)

    def test_pts_rolling_5(self, single_player: pd.DataFrame) -> None:
        result = compute_vaastav_features(single_player)
        pts5 = result.set_index("GW")["pts_rolling_5"]

        # GW5: shift -> GW1..4 (2,4,6,8). rolling(5, min_periods=1) -> mean = 5.0
        assert pts5[5] == pytest.approx(5.0)

    def test_mins_rolling_3(self, single_player: pd.DataFrame) -> None:
        result = compute_vaastav_features(single_player)
        mins3 = result.set_index("GW")["mins_rolling_3"]

        # GW4: shift -> GW1,GW2,GW3 mins = (90,80,70). rolling(3) -> mean = 80.0
        assert mins3[4] == pytest.approx(80.0)

    def test_goals_rolling_sum(self) -> None:
        """goals_rolling_3 should be a *sum*, not a mean."""
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "goals_scored": 1},
                {"GW": 2, "goals_scored": 2},
                {"GW": 3, "goals_scored": 3},
                {"GW": 4, "goals_scored": 4},
            ],
        )
        result = compute_vaastav_features(df)
        g3 = result.set_index("GW")["goals_rolling_3"]

        # GW4: shift -> GW1,GW2,GW3 = (1,2,3). rolling(3).sum() = 6
        assert g3[4] == pytest.approx(6.0)

    def test_output_shape(self, single_player: pd.DataFrame) -> None:
        result = compute_vaastav_features(single_player)
        # One row per GW
        assert len(result) == 5
        # All feature columns plus element and GW
        assert "element" in result.columns
        assert "GW" in result.columns
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# 2. DGW aggregation
# ---------------------------------------------------------------------------
class TestDGWAggregation:
    """Multiple fixture rows for the same (element, GW) get summed."""

    def test_dgw_points_summed(self) -> None:
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "total_points": 3, "minutes": 90},
                {"GW": 2, "total_points": 5, "minutes": 90},  # fixture 1
                {"GW": 2, "total_points": 7, "minutes": 90},  # fixture 2 (DGW)
                {"GW": 3, "total_points": 4, "minutes": 90},
            ],
        )
        result = compute_vaastav_features(df)

        # After DGW aggregation, GW2 total_points = 5 + 7 = 12
        # GW3: shift -> only GW1(3), GW2(12) available for rolling.
        # pts_rolling_3 at GW3 = mean([3, 12]) = 7.5  (min_periods=1, only 2 values)
        pts3 = result.set_index("GW")["pts_rolling_3"]
        assert pts3[3] == pytest.approx(7.5)

    def test_dgw_produces_one_row_per_gw(self) -> None:
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "total_points": 3},
                {"GW": 2, "total_points": 5},
                {"GW": 2, "total_points": 7},  # DGW
            ],
        )
        result = compute_vaastav_features(df)
        assert len(result) == 2
        assert list(result["GW"]) == [1, 2]


# ---------------------------------------------------------------------------
# 3. No-lookahead test
# ---------------------------------------------------------------------------
class TestNoLookahead:
    """Rolling features for GW=k must use only data from GW < k."""

    def test_gw3_uses_only_gw1_gw2(self) -> None:
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "total_points": 10},
                {"GW": 2, "total_points": 20},
                {"GW": 3, "total_points": 100},  # should NOT affect GW3 features
            ],
        )
        result = compute_vaastav_features(df)
        row_gw3 = result[result["GW"] == 3].iloc[0]

        # pts_rolling_3 at GW3: shift -> (10, 20), rolling(3, min_periods=1) = mean(10,20) = 15
        assert row_gw3["pts_rolling_3"] == pytest.approx(15.0)
        # season_avg_pts at GW3: expanding mean of shifted (10, 20) = 15
        assert row_gw3["season_avg_pts"] == pytest.approx(15.0)

    def test_gw1_has_nan_rolling(self) -> None:
        """GW1 cannot have any prior data, so rolling features are NaN."""
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "total_points": 99},
                {"GW": 2, "total_points": 1},
            ],
        )
        result = compute_vaastav_features(df)
        row_gw1 = result[result["GW"] == 1].iloc[0]

        assert math.isnan(row_gw1["pts_rolling_3"])
        assert math.isnan(row_gw1["pts_rolling_5"])
        assert math.isnan(row_gw1["season_avg_pts"])


# ---------------------------------------------------------------------------
# 4. Multiple players don't cross-contaminate
# ---------------------------------------------------------------------------
class TestMultiplePlayersIsolation:
    """Features for player A must not depend on player B's data."""

    def test_two_players_independent(self) -> None:
        player_a = _make_player_rows(
            element=10,
            gw_data=[
                {"GW": 1, "total_points": 2},
                {"GW": 2, "total_points": 4},
                {"GW": 3, "total_points": 6},
            ],
        )
        player_b = _make_player_rows(
            element=20,
            gw_data=[
                {"GW": 1, "total_points": 100},
                {"GW": 2, "total_points": 200},
                {"GW": 3, "total_points": 300},
            ],
        )
        combined = pd.concat([player_a, player_b], ignore_index=True)
        result = compute_vaastav_features(combined)

        # Player A, GW3: shift -> (2, 4). rolling(3, min_periods=1) = mean(2,4) = 3.0
        a_gw3 = result[(result["element"] == 10) & (result["GW"] == 3)].iloc[0]
        assert a_gw3["pts_rolling_3"] == pytest.approx(3.0)

        # Player B, GW3: shift -> (100, 200). rolling(3, min_periods=1) = mean(100,200) = 150.0
        b_gw3 = result[(result["element"] == 20) & (result["GW"] == 3)].iloc[0]
        assert b_gw3["pts_rolling_3"] == pytest.approx(150.0)

    def test_output_has_all_players(self) -> None:
        player_a = _make_player_rows(element=10, gw_data=[{"GW": 1, "total_points": 5}])
        player_b = _make_player_rows(element=20, gw_data=[{"GW": 1, "total_points": 9}])
        combined = pd.concat([player_a, player_b], ignore_index=True)
        result = compute_vaastav_features(combined)

        assert set(result["element"]) == {10, 20}
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 5. First GW gets NaN for rolling features
# ---------------------------------------------------------------------------
class TestFirstGWNaN:
    """The very first GW for a player should have NaN rolling features."""

    def test_all_rolling_nan_at_gw1(self) -> None:
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "total_points": 5, "minutes": 90, "goals_scored": 1},
                {"GW": 2, "total_points": 3, "minutes": 60, "goals_scored": 0},
            ],
        )
        result = compute_vaastav_features(df)
        row_gw1 = result[result["GW"] == 1].iloc[0]

        # All rolling and expanding features should be NaN at GW1
        rolling_cols = [
            "pts_rolling_3", "pts_rolling_5", "pts_rolling_10",
            "mins_rolling_3", "mins_rolling_5",
            "goals_rolling_3", "goals_rolling_5",
            "assists_rolling_3", "assists_rolling_5",
            "cs_rolling_5", "cs_rolling_10",
            "bonus_rolling_5", "bonus_rolling_10",
            "bps_rolling_5", "bps_rolling_10",
            "ict_rolling_3", "ict_rolling_5", "ict_rolling_10",
            "influence_rolling_5", "creativity_rolling_5", "threat_rolling_5",
            "season_avg_pts", "season_total_mins",
        ]
        for col in rolling_cols:
            assert math.isnan(row_gw1[col]), f"{col} should be NaN at GW1"

    def test_value_and_selected_norm_not_nan_at_gw1(self) -> None:
        """value and selected_norm come from the current row, so never NaN."""
        df = _make_player_rows(
            element=1,
            gw_data=[{"GW": 1, "total_points": 5, "value": 100, "selected": 5_000_000}],
        )
        result = compute_vaastav_features(df)
        row_gw1 = result[result["GW"] == 1].iloc[0]

        assert row_gw1["value"] == 100
        assert row_gw1["selected_norm"] == pytest.approx(0.5)

    def test_games_played_zero_at_gw1(self) -> None:
        """games_played at GW1 should be 0 (no prior GWs)."""
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "total_points": 5, "minutes": 90},
                {"GW": 2, "total_points": 3, "minutes": 60},
            ],
        )
        result = compute_vaastav_features(df)
        row_gw1 = result[result["GW"] == 1].iloc[0]
        assert row_gw1["games_played"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. season_avg_pts expanding correctness
# ---------------------------------------------------------------------------
class TestSeasonAvgPts:
    """season_avg_pts should be the expanding mean of prior-GW total_points."""

    def test_expanding_mean_sequence(self) -> None:
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "total_points": 10},
                {"GW": 2, "total_points": 20},
                {"GW": 3, "total_points": 30},
                {"GW": 4, "total_points": 40},
                {"GW": 5, "total_points": 50},
            ],
        )
        result = compute_vaastav_features(df)
        avg = result.set_index("GW")["season_avg_pts"]

        # GW1: NaN (no prior data)
        assert math.isnan(avg[1])
        # GW2: expanding mean of shifted [10] = 10.0
        assert avg[2] == pytest.approx(10.0)
        # GW3: expanding mean of shifted [10, 20] = 15.0
        assert avg[3] == pytest.approx(15.0)
        # GW4: expanding mean of shifted [10, 20, 30] = 20.0
        assert avg[4] == pytest.approx(20.0)
        # GW5: expanding mean of shifted [10, 20, 30, 40] = 25.0
        assert avg[5] == pytest.approx(25.0)

    def test_season_total_mins(self) -> None:
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "minutes": 90},
                {"GW": 2, "minutes": 60},
                {"GW": 3, "minutes": 45},
            ],
        )
        result = compute_vaastav_features(df)
        total = result.set_index("GW")["season_total_mins"]

        # GW1: NaN
        assert math.isnan(total[1])
        # GW2: sum of shifted [90] = 90
        assert total[2] == pytest.approx(90.0)
        # GW3: sum of shifted [90, 60] = 150
        assert total[3] == pytest.approx(150.0)

    def test_games_played_counts_prior(self) -> None:
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "minutes": 90},
                {"GW": 2, "minutes": 0},   # did not play
                {"GW": 3, "minutes": 60},
                {"GW": 4, "minutes": 45},
            ],
        )
        result = compute_vaastav_features(df)
        gp = result.set_index("GW")["games_played"]

        # GW1: 0 prior games
        assert gp[1] == pytest.approx(0.0)
        # GW2: 1 prior game (GW1 had 90 > 0)
        assert gp[2] == pytest.approx(1.0)
        # GW3: 1 prior game (GW1 played, GW2 did not)
        assert gp[3] == pytest.approx(1.0)
        # GW4: 2 prior games (GW1 and GW3 played)
        assert gp[4] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 7. mins_std_5 sanity check
# ---------------------------------------------------------------------------
class TestMinsStd:
    """mins_std_5 should be the rolling std of shifted minutes."""

    def test_constant_minutes_std_is_zero(self) -> None:
        """If all prior minutes are the same, std should be 0."""
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": gw, "minutes": 90}
                for gw in range(1, 8)
            ],
        )
        result = compute_vaastav_features(df)
        std5 = result.set_index("GW")["mins_std_5"]

        # GW7: shift -> GW1..6 all 90. rolling(5) window is [90,90,90,90,90] -> std = 0
        assert std5[7] == pytest.approx(0.0)

    def test_varied_minutes_std(self) -> None:
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "minutes": 90},
                {"GW": 2, "minutes": 0},
                {"GW": 3, "minutes": 90},
            ],
        )
        result = compute_vaastav_features(df)
        std5 = result.set_index("GW")["mins_std_5"]

        # GW3: shift -> [90, 0]. rolling(5, min_periods=1).std()
        # pandas std with ddof=1: std([90, 0]) = 63.639...
        expected = pd.Series([90.0, 0.0]).std()  # ddof=1 default
        assert std5[3] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 8. New rolling features (saves, goals_conceded, etc.)
# ---------------------------------------------------------------------------
class TestNewRollingFeatures:
    """Test the 10 new rolling features added for model improvement."""

    def test_saves_rolling_5(self) -> None:
        """saves_rolling_5 should be the rolling mean of saves."""
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": gw, "saves": gw}
                for gw in range(1, 7)
            ],
        )
        result = compute_vaastav_features(df)
        saves5 = result.set_index("GW")["saves_rolling_5"]

        # GW6: shift -> GW1..5 saves = (1,2,3,4,5). rolling(5) mean = 3.0
        assert saves5[6] == pytest.approx(3.0)

    def test_goals_conceded_rolling_5(self) -> None:
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "goals_conceded": 2},
                {"GW": 2, "goals_conceded": 0},
                {"GW": 3, "goals_conceded": 1},
            ],
        )
        result = compute_vaastav_features(df)
        gc5 = result.set_index("GW")["goals_conceded_rolling_5"]

        # GW3: shift -> (2, 0). rolling(5, min_periods=1) mean = 1.0
        assert gc5[3] == pytest.approx(1.0)

    def test_transfers_balance_rolling_3(self) -> None:
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "transfers_balance": 1000},
                {"GW": 2, "transfers_balance": -500},
                {"GW": 3, "transfers_balance": 200},
                {"GW": 4, "transfers_balance": 300},
            ],
        )
        result = compute_vaastav_features(df)
        tb3 = result.set_index("GW")["transfers_balance_rolling_3"]

        # GW4: shift -> (1000, -500, 200). rolling(3) mean ≈ 233.33
        assert tb3[4] == pytest.approx(700 / 3, abs=0.01)

    def test_yellows_rolling_5_is_sum(self) -> None:
        """yellows_rolling_5 should be a sum, not a mean."""
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "yellow_cards": 1},
                {"GW": 2, "yellow_cards": 0},
                {"GW": 3, "yellow_cards": 1},
                {"GW": 4, "yellow_cards": 1},
            ],
        )
        result = compute_vaastav_features(df)
        yel5 = result.set_index("GW")["yellows_rolling_5"]

        # GW4: shift -> (1, 0, 1). rolling(5, min_periods=1).sum() = 2
        assert yel5[4] == pytest.approx(2.0)

    def test_fpl_xg_rolling_5(self) -> None:
        """FPL expected_goals rolling feature."""
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": gw, "expected_goals": 0.5 * gw}
                for gw in range(1, 7)
            ],
        )
        result = compute_vaastav_features(df)
        fxg5 = result.set_index("GW")["fpl_xg_rolling_5"]

        # GW6: shift -> GW1..5 xG = (0.5,1.0,1.5,2.0,2.5). mean = 1.5
        assert fxg5[6] == pytest.approx(1.5)

    def test_new_features_in_feature_columns(self) -> None:
        """All 10 new feature columns should be in FEATURE_COLUMNS."""
        new_cols = [
            "saves_rolling_5", "goals_conceded_rolling_5",
            "transfers_balance_rolling_3", "yellows_rolling_5",
            "reds_rolling_10", "starts_rolling_5",
            "fpl_xg_rolling_5", "fpl_xa_rolling_5",
            "fpl_xgi_rolling_5", "fpl_xgc_rolling_5",
        ]
        for col in new_cols:
            assert col in FEATURE_COLUMNS, f"{col} missing from FEATURE_COLUMNS"


# ---------------------------------------------------------------------------
# 9. Venue-split rolling form
# ---------------------------------------------------------------------------
class TestVenueSplitRolling:
    """pts_rolling_5_home/away roll over that venue's past matches only."""

    @pytest.fixture()
    def alternating_venues(self) -> pd.DataFrame:
        """H(5), A(2), H(8), H(3), A(7) — hand-checkable venue sequences."""
        return _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "total_points": 5, "was_home": True},
                {"GW": 2, "total_points": 2, "was_home": False},
                {"GW": 3, "total_points": 8, "was_home": True},
                {"GW": 4, "total_points": 3, "was_home": True},
                {"GW": 5, "total_points": 7, "was_home": False},
            ],
        )

    def test_home_rolling_hand_check(self, alternating_venues: pd.DataFrame) -> None:
        result = compute_vaastav_features(alternating_venues)
        home = result.set_index("GW")["pts_rolling_5_home"]

        # GW1: no prior home matches -> NaN
        assert math.isnan(home[1])
        # GW2 (away): last home matches = {GW1} -> mean(5) = 5
        assert home[2] == pytest.approx(5.0)
        # GW3 (home): prior home = {GW1} -> 5 (GW3 itself excluded)
        assert home[3] == pytest.approx(5.0)
        # GW4: prior home = {GW1, GW3} -> mean(5, 8) = 6.5
        assert home[4] == pytest.approx(6.5)
        # GW5: prior home = {GW1, GW3, GW4} -> mean(5, 8, 3)
        assert home[5] == pytest.approx(16 / 3)

    def test_away_rolling_hand_check(self, alternating_venues: pd.DataFrame) -> None:
        result = compute_vaastav_features(alternating_venues)
        away = result.set_index("GW")["pts_rolling_5_away"]

        # GW1, GW2: no prior away matches -> NaN
        assert math.isnan(away[1])
        assert math.isnan(away[2])
        # GW3..5: prior away = {GW2} -> 2
        assert away[3] == pytest.approx(2.0)
        assert away[4] == pytest.approx(2.0)
        assert away[5] == pytest.approx(2.0)

    def test_missing_was_home_gives_nan(self) -> None:
        """Without a was_home column the venue features are NaN but present."""
        df = _make_player_rows(
            element=1,
            gw_data=[{"GW": 1, "total_points": 5}, {"GW": 2, "total_points": 3}],
        )
        result = compute_vaastav_features(df)
        assert result["pts_rolling_5_home"].isna().all()
        assert result["pts_rolling_5_away"].isna().all()

    def test_dgw_same_venue_counts_once(self) -> None:
        """Two home fixtures in one GW = one observation with summed points."""
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "total_points": 4, "was_home": True},
                {"GW": 2, "total_points": 3, "was_home": True},  # DGW fixture 1
                {"GW": 2, "total_points": 5, "was_home": True},  # DGW fixture 2
                {"GW": 3, "total_points": 1, "was_home": False},
            ],
        )
        result = compute_vaastav_features(df)
        home = result.set_index("GW")["pts_rolling_5_home"]

        # GW3: prior home obs = {GW1: 4, GW2: 3+5=8} -> mean = 6
        assert home[3] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# 10. Minutes share
# ---------------------------------------------------------------------------
class TestMinutesShare:
    """minutes_share_5 = rolling-5 sum of prior minutes / 450."""

    def test_full_minutes_share(self) -> None:
        df = _make_player_rows(
            element=1,
            gw_data=[{"GW": gw, "minutes": 90} for gw in range(1, 8)],
        )
        result = compute_vaastav_features(df)
        share = result.set_index("GW")["minutes_share_5"]

        # GW1: NaN (no prior data)
        assert math.isnan(share[1])
        # GW2: 90 / 450 = 0.2 (denominator is always the full 450)
        assert share[2] == pytest.approx(0.2)
        # GW6: prior 5 GWs all 90 -> 450/450 = 1.0
        assert share[6] == pytest.approx(1.0)
        assert share[7] == pytest.approx(1.0)

    def test_partial_minutes_share(self) -> None:
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "minutes": 90},
                {"GW": 2, "minutes": 0},
                {"GW": 3, "minutes": 45},
                {"GW": 4, "minutes": 90},
            ],
        )
        result = compute_vaastav_features(df)
        share = result.set_index("GW")["minutes_share_5"]

        # GW4: (90 + 0 + 45) / 450 = 0.3
        assert share[4] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 11. Team xGI share
# ---------------------------------------------------------------------------
class TestTeamXgiShare:
    """team_xgi_share_5 = player rolling-5 xGI sum / team rolling-5 xGI sum."""

    def test_share_hand_check(self) -> None:
        """Two teammates: shares reflect each player's slice of team xGI."""
        player_a = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "expected_goal_involvements": 0.6, "team": 1},
                {"GW": 2, "expected_goal_involvements": 0.4, "team": 1},
                {"GW": 3, "expected_goal_involvements": 0.0, "team": 1},
            ],
        )
        player_b = _make_player_rows(
            element=2,
            gw_data=[
                {"GW": 1, "expected_goal_involvements": 0.2, "team": 1},
                {"GW": 2, "expected_goal_involvements": 0.6, "team": 1},
                {"GW": 3, "expected_goal_involvements": 0.0, "team": 1},
            ],
        )
        combined = pd.concat([player_a, player_b], ignore_index=True)
        result = compute_vaastav_features(combined)
        share = result.set_index(["element", "GW"])["team_xgi_share_5"]

        # GW1: no prior data -> NaN
        assert math.isnan(share[(1, 1)])
        # GW2: A = 0.6 / (0.6 + 0.2) = 0.75; B = 0.2 / 0.8 = 0.25
        assert share[(1, 2)] == pytest.approx(0.75)
        assert share[(2, 2)] == pytest.approx(0.25)
        # GW3: A = (0.6+0.4) / 1.8; B = (0.2+0.6) / 1.8
        assert share[(1, 3)] == pytest.approx(1.0 / 1.8)
        assert share[(2, 3)] == pytest.approx(0.8 / 1.8)

    def test_zero_team_xgi_gives_nan(self) -> None:
        """A zero denominator must produce NaN, not inf."""
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "expected_goal_involvements": 0.0, "team": 1},
                {"GW": 2, "expected_goal_involvements": 0.5, "team": 1},
            ],
        )
        result = compute_vaastav_features(df)
        assert result["team_xgi_share_5"].isna().all()

    def test_missing_team_column_gives_nan(self) -> None:
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "expected_goal_involvements": 0.5},
                {"GW": 2, "expected_goal_involvements": 0.5},
            ],
        )
        result = compute_vaastav_features(df)
        assert "team_xgi_share_5" in result.columns
        assert result["team_xgi_share_5"].isna().all()


# ---------------------------------------------------------------------------
# 12. Missing-column guard
# ---------------------------------------------------------------------------
class TestMissingColumnGuard:
    """When source columns don't exist, output should be NaN."""

    def test_missing_source_col_produces_nan(self) -> None:
        """If a source column (e.g. 'starts') is absent, the output is NaN."""
        # Build a DataFrame WITHOUT the 'starts' column
        df = _make_player_rows(
            element=1,
            gw_data=[
                {"GW": 1, "total_points": 5},
                {"GW": 2, "total_points": 3},
            ],
        )
        # Explicitly remove columns that some specs reference
        cols_to_drop = [c for c in ["starts", "expected_goals", "expected_assists",
                                      "expected_goal_involvements",
                                      "expected_goals_conceded"]
                        if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        result = compute_vaastav_features(df)

        # Features depending on missing cols should be NaN
        for col in ["starts_rolling_5", "fpl_xg_rolling_5", "fpl_xa_rolling_5",
                     "fpl_xgi_rolling_5", "fpl_xgc_rolling_5"]:
            assert col in result.columns, f"{col} should exist even when source is missing"
            assert result[col].isna().all(), f"{col} should be all NaN when source col is missing"

        # Features depending on existing cols should still work
        assert not result["pts_rolling_3"].isna().all()  # GW2 should have a value
