"""Tests for squad validity and formation checks."""

import pytest

from fpl_optimizer.engine.constraints import (
    can_substitute,
    check_club_limits,
    get_formation,
    is_valid_formation,
    is_valid_squad,
    is_valid_squad_composition,
)
from fpl_optimizer.engine.state import PlayerSlot, Squad
from fpl_optimizer.utils.constants import VALID_FORMATIONS, Position


def _make_player(eid: int, pos: Position) -> PlayerSlot:
    return PlayerSlot(element_id=eid, position=pos, purchase_price=50, selling_price=50)


class TestFormations:
    """Test all 8 valid formations."""

    @pytest.mark.parametrize(
        "defs,mids,fwds",
        VALID_FORMATIONS,
        ids=[f"{d}-{m}-{f}" for d, m, f in VALID_FORMATIONS],
    )
    def test_valid_formations(self, defs: int, mids: int, fwds: int):
        players = [_make_player(0, Position.GK)]
        for i in range(defs):
            players.append(_make_player(i + 1, Position.DEF))
        for i in range(mids):
            players.append(_make_player(i + 20, Position.MID))
        for i in range(fwds):
            players.append(_make_player(i + 40, Position.FWD))
        assert len(players) == 11
        assert is_valid_formation(players)

    def test_invalid_formation_no_gk(self):
        players = [_make_player(i, Position.DEF) for i in range(4)]
        players += [_make_player(i + 10, Position.MID) for i in range(4)]
        players += [_make_player(i + 20, Position.FWD) for i in range(3)]
        assert len(players) == 11
        assert not is_valid_formation(players)

    def test_invalid_formation_two_gk(self):
        players = [_make_player(0, Position.GK), _make_player(1, Position.GK)]
        players += [_make_player(i + 2, Position.DEF) for i in range(3)]
        players += [_make_player(i + 10, Position.MID) for i in range(4)]
        players += [_make_player(i + 20, Position.FWD) for i in range(2)]
        assert len(players) == 11
        assert not is_valid_formation(players)

    def test_invalid_formation_2_def(self):
        """2-5-3 is not valid."""
        players = [_make_player(0, Position.GK)]
        players += [_make_player(i + 1, Position.DEF) for i in range(2)]
        players += [_make_player(i + 10, Position.MID) for i in range(5)]
        players += [_make_player(i + 20, Position.FWD) for i in range(3)]
        assert not is_valid_formation(players)

    def test_invalid_formation_wrong_count(self):
        players = [_make_player(0, Position.GK)]
        players += [_make_player(i + 1, Position.DEF) for i in range(3)]
        assert not is_valid_formation(players)  # Only 4 players

    def test_get_formation(self):
        players = [_make_player(0, Position.GK)]
        players += [_make_player(i + 1, Position.DEF) for i in range(4)]
        players += [_make_player(i + 10, Position.MID) for i in range(4)]
        players += [_make_player(i + 20, Position.FWD) for i in range(2)]
        assert get_formation(players) == (4, 4, 2)


class TestSquadComposition:
    def test_valid_15_players(self, sample_squad):
        assert is_valid_squad_composition(sample_squad.players)

    def test_invalid_14_players(self):
        players = [_make_player(i, Position.GK) for i in range(2)]
        players += [_make_player(i + 10, Position.DEF) for i in range(5)]
        players += [_make_player(i + 20, Position.MID) for i in range(5)]
        players += [_make_player(i + 30, Position.FWD) for i in range(2)]
        assert not is_valid_squad_composition(players)

    def test_invalid_too_many_gk(self):
        players = [_make_player(i, Position.GK) for i in range(3)]
        players += [_make_player(i + 10, Position.DEF) for i in range(5)]
        players += [_make_player(i + 20, Position.MID) for i in range(5)]
        players += [_make_player(i + 30, Position.FWD) for i in range(2)]
        assert not is_valid_squad_composition(players)


class TestClubLimits:
    def test_valid_club_limits(self):
        team_map = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3,
                    10: 4, 11: 5, 12: 6, 13: 7, 14: 8, 15: 9}
        players = [_make_player(i + 1, Position.GK) for i in range(15)]
        assert check_club_limits(players, team_map)

    def test_invalid_4_from_same_club(self):
        team_map = {1: 1, 2: 1, 3: 1, 4: 1}
        players = [_make_player(i + 1, Position.GK) for i in range(4)]
        assert not check_club_limits(players, team_map)

    def test_exactly_3_from_club(self):
        team_map = {1: 1, 2: 1, 3: 1, 4: 2}
        players = [_make_player(i + 1, Position.GK) for i in range(4)]
        assert check_club_limits(players, team_map)


class TestIsValidSquad:
    def test_valid_squad(self, sample_squad):
        team_map = {i: i for i in range(1, 16)}  # All different teams
        assert is_valid_squad(sample_squad, team_map)


class TestCanSubstitute:
    def test_sub_maintains_formation(self):
        # 4-4-2: replace a MID with a MID
        lineup = [_make_player(0, Position.GK)]
        lineup += [_make_player(i + 1, Position.DEF) for i in range(4)]
        lineup += [_make_player(i + 10, Position.MID) for i in range(4)]
        lineup += [_make_player(i + 20, Position.FWD) for i in range(2)]

        sub = _make_player(99, Position.MID)
        assert can_substitute(lineup, 5, sub)  # Replace first MID

    def test_sub_breaks_formation(self):
        # 4-4-2: replace a DEF with a FWD -> 3-4-3
        lineup = [_make_player(0, Position.GK)]
        lineup += [_make_player(i + 1, Position.DEF) for i in range(4)]
        lineup += [_make_player(i + 10, Position.MID) for i in range(4)]
        lineup += [_make_player(i + 20, Position.FWD) for i in range(2)]

        sub = _make_player(99, Position.FWD)
        # Replace first DEF with FWD -> 3-4-3 which IS valid
        assert can_substitute(lineup, 1, sub)

    def test_sub_would_create_invalid_formation(self):
        # 3-4-3: replace a DEF with a FWD -> 2-4-4 which is INVALID
        lineup = [_make_player(0, Position.GK)]
        lineup += [_make_player(i + 1, Position.DEF) for i in range(3)]
        lineup += [_make_player(i + 10, Position.MID) for i in range(4)]
        lineup += [_make_player(i + 20, Position.FWD) for i in range(3)]

        sub = _make_player(99, Position.FWD)
        assert not can_substitute(lineup, 1, sub)  # 2 DEF is invalid
