"""Shared fixtures for optimizer tests."""

from __future__ import annotations

import pytest

from fpl_optimizer.engine.state import (
    ChipState,
    GameState,
    PlayerSlot,
    Squad,
)
from fpl_optimizer.optimizer.types import PlayerCandidate
from fpl_optimizer.utils.constants import Position, STARTING_BUDGET


def _make_candidate(
    eid: int, pos: Position, price: int, team: int, xp: float
) -> PlayerCandidate:
    return PlayerCandidate(
        element_id=eid, position=pos, price=price, team_id=team, predicted_points=xp,
    )


@pytest.fixture
def squad_15_candidates() -> list[PlayerCandidate]:
    """15-player squad as PlayerCandidates (2 GK, 5 DEF, 5 MID, 3 FWD).

    Teams spread across 1-10 so club limits aren't violated.
    Total cost = 845 (within 1000 budget).
    """
    return [
        # GK (2) — teams 1, 2
        _make_candidate(1, Position.GK, 50, 1, xp=6.0),
        _make_candidate(2, Position.GK, 40, 2, xp=1.0),
        # DEF (5) — teams 1, 3, 4, 5, 6
        _make_candidate(3, Position.DEF, 55, 1, xp=8.0),
        _make_candidate(4, Position.DEF, 50, 3, xp=6.0),
        _make_candidate(5, Position.DEF, 45, 4, xp=2.0),
        _make_candidate(6, Position.DEF, 43, 5, xp=1.0),
        _make_candidate(7, Position.DEF, 42, 6, xp=0.0),
        # MID (5) — teams 1, 3, 7, 8, 9
        _make_candidate(8, Position.MID, 80, 1, xp=12.0),
        _make_candidate(9, Position.MID, 65, 3, xp=5.0),
        _make_candidate(10, Position.MID, 55, 7, xp=3.0),
        _make_candidate(11, Position.MID, 50, 8, xp=2.0),
        _make_candidate(12, Position.MID, 45, 9, xp=1.0),
        # FWD (3) — teams 2, 4, 10
        _make_candidate(13, Position.FWD, 100, 2, xp=10.0),
        _make_candidate(14, Position.FWD, 70, 4, xp=5.0),
        _make_candidate(15, Position.FWD, 55, 10, xp=1.0),
    ]


@pytest.fixture
def large_candidate_pool() -> list[PlayerCandidate]:
    """A larger pool of ~40 candidates for squad selection tests.

    Includes enough diversity in positions and teams.
    """
    cands: list[PlayerCandidate] = []
    eid = 1

    # 4 GKs across 4 teams
    for i, (price, team, xp) in enumerate([
        (50, 1, 6.0), (45, 2, 4.0), (40, 3, 2.0), (40, 4, 1.0),
    ]):
        cands.append(_make_candidate(eid, Position.GK, price, team, xp))
        eid += 1

    # 12 DEFs across 8 teams
    for i, (price, team, xp) in enumerate([
        (60, 1, 8.0), (55, 2, 7.0), (55, 3, 6.0), (50, 4, 5.0),
        (50, 5, 5.0), (45, 6, 4.0), (45, 7, 3.0), (43, 8, 2.0),
        (42, 1, 2.0), (42, 5, 1.0), (40, 9, 1.0), (40, 10, 0.5),
    ]):
        cands.append(_make_candidate(eid, Position.DEF, price, team, xp))
        eid += 1

    # 12 MIDs across 8 teams
    for i, (price, team, xp) in enumerate([
        (100, 1, 12.0), (85, 2, 10.0), (75, 3, 8.0), (70, 4, 7.0),
        (65, 5, 6.0), (60, 6, 5.0), (55, 7, 4.0), (50, 8, 3.0),
        (50, 9, 3.0), (45, 10, 2.0), (45, 2, 2.0), (45, 3, 1.5),
    ]):
        cands.append(_make_candidate(eid, Position.MID, price, team, xp))
        eid += 1

    # 10 FWDs across 6 teams
    for i, (price, team, xp) in enumerate([
        (110, 2, 10.0), (95, 4, 8.0), (80, 6, 7.0), (75, 8, 6.0),
        (70, 1, 5.0), (65, 3, 4.0), (60, 5, 3.0), (55, 7, 2.0),
        (50, 9, 1.5), (45, 10, 1.0),
    ]):
        cands.append(_make_candidate(eid, Position.FWD, price, team, xp))
        eid += 1

    return cands


@pytest.fixture
def optimizer_game_state(squad_15_candidates: list[PlayerCandidate]) -> GameState:
    """A valid GameState at GW2 built from squad_15_candidates."""
    players = [
        PlayerSlot(
            element_id=c.element_id,
            position=c.position,
            purchase_price=c.price,
            selling_price=c.price,
        )
        for c in squad_15_candidates
    ]

    # 4-4-2: GK + 4 DEF + 4 MID + 2 FWD
    lineup = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]
    bench = [1, 6, 11, 14]

    squad = Squad(
        players=players,
        lineup=lineup,
        bench=bench,
        captain_idx=7,
        vice_captain_idx=12,
    )

    total_cost = sum(c.price for c in squad_15_candidates)
    return GameState(
        squad=squad,
        bank=STARTING_BUDGET - total_cost,
        free_transfers=1,
        chips=ChipState(),
        current_gw=2,
        total_points=50,
    )
