"""Tests for point lookup, captain doubling, and triple captain."""

import pytest

from fpl_optimizer.engine.scoring import (
    calculate_captain_points,
    calculate_squad_points,
    did_player_play,
    get_player_minutes,
    get_player_points,
)


class TestGetPlayerPoints:
    def test_player_with_points(self, loader):
        # Player_MID1 (element=8) scored 12 points in GW1
        assert get_player_points(loader, 8, 1) == 12

    def test_player_no_data(self, loader):
        # Non-existent player
        assert get_player_points(loader, 999, 1) == 0

    def test_player_zero_minutes(self, loader):
        # Player_DEF5 (element=7) had 0 minutes in GW1
        assert get_player_points(loader, 7, 1) == 0

    def test_gw2_points(self, loader):
        # Player_FWD2 (element=14) scored 8 points in GW2
        assert get_player_points(loader, 14, 2) == 8


class TestGetPlayerMinutes:
    def test_full_90(self, loader):
        assert get_player_minutes(loader, 1, 1) == 90

    def test_zero_minutes(self, loader):
        assert get_player_minutes(loader, 7, 1) == 0

    def test_partial_minutes(self, loader):
        assert get_player_minutes(loader, 6, 1) == 45


class TestDidPlayerPlay:
    def test_played_90_minutes(self, loader):
        assert did_player_play(loader, 1, 1)

    def test_didnt_play(self, loader):
        assert not did_player_play(loader, 7, 1)

    def test_played_partial(self, loader):
        assert did_player_play(loader, 6, 1)

    def test_no_data(self, loader):
        assert not did_player_play(loader, 999, 1)


class TestCaptainPoints:
    def test_captain_played_double(self, loader, sample_squad):
        # Captain is MID1 (idx=7, element=8), who scored 12 in GW1
        # Normal captain = 2x, so bonus = 12 * 1 = 12
        bonus, failover = calculate_captain_points(loader, sample_squad, 1)
        assert bonus == 12
        assert not failover

    def test_triple_captain(self, loader, sample_squad):
        # Triple captain: 3x, so bonus = 12 * 2 = 24
        bonus, failover = calculate_captain_points(
            loader, sample_squad, 1, triple_captain=True
        )
        assert bonus == 24
        assert not failover

    def test_captain_failover(self, loader, sample_squad):
        # Set captain to player who didn't play (element 7, idx 6)
        sample_squad.captain_idx = 6  # DEF5, 0 minutes in GW1
        sample_squad.vice_captain_idx = 12  # FWD1 (element 13), 10 pts in GW1
        bonus, failover = calculate_captain_points(loader, sample_squad, 1)
        assert bonus == 10  # Vice captain's points * 1
        assert failover

    def test_both_didnt_play(self, loader, sample_squad):
        # Both captain and vice didn't play
        sample_squad.captain_idx = 6  # DEF5, 0 min
        sample_squad.vice_captain_idx = 1  # GK2, 0 min
        bonus, failover = calculate_captain_points(loader, sample_squad, 1)
        assert bonus == 0
        assert failover


class TestSquadPoints:
    def test_lineup_and_bench_points(self, loader, sample_squad):
        lineup_pts, bench_pts = calculate_squad_points(loader, sample_squad, 1)
        # Lineup: GK1(6) + DEF1(8) + DEF2(6) + DEF3(2) + DEF4(1) +
        #         MID1(12) + MID2(5) + MID3(3) + MID4(2) + FWD1(10) + FWD2(5) = 60
        assert lineup_pts == 60
        # Bench: GK2(1) + DEF5(0) + MID5(1) + FWD3(1) = 3
        assert bench_pts == 3
