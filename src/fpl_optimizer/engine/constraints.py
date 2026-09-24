"""Squad validity and formation checks."""

from __future__ import annotations

from collections import Counter

from fpl_optimizer.engine.state import PlayerSlot, Squad
from fpl_optimizer.utils.constants import (
    MAX_PER_CLUB,
    POSITION_LIMITS,
    SQUAD_SIZE,
    STARTING_XI,
    VALID_FORMATIONS,
    Position,
)


def get_formation(players: list[PlayerSlot]) -> tuple[int, int, int] | None:
    """Get the (DEF, MID, FWD) formation from a list of outfield players.

    Returns None if there's not exactly 1 GK or if formation is invalid.
    """
    counts = Counter(p.position for p in players)
    if counts.get(Position.GK, 0) != 1:
        return None
    formation = (
        counts.get(Position.DEF, 0),
        counts.get(Position.MID, 0),
        counts.get(Position.FWD, 0),
    )
    return formation


def is_valid_formation(players: list[PlayerSlot]) -> bool:
    """Check if a list of 11 players forms a valid formation."""
    if len(players) != STARTING_XI:
        return False
    formation = get_formation(players)
    return formation is not None and formation in VALID_FORMATIONS


def is_valid_squad_composition(players: list[PlayerSlot]) -> bool:
    """Check if 15 players have the correct position distribution."""
    if len(players) != SQUAD_SIZE:
        return False
    counts = Counter(p.position for p in players)
    return all(counts.get(pos, 0) == limit for pos, limit in POSITION_LIMITS.items())


def check_club_limits(
    players: list[PlayerSlot],
    team_map: dict[int, int],
    max_per_club: int = MAX_PER_CLUB,
) -> bool:
    """Check that no more than max_per_club players are from the same team."""
    club_counts: Counter[int] = Counter()
    for p in players:
        team_id = team_map.get(p.element_id)
        if team_id is not None:
            club_counts[team_id] += 1
            if club_counts[team_id] > max_per_club:
                return False
    return True


def is_valid_squad(
    squad: Squad, team_map: dict[int, int]
) -> bool:
    """Full squad validity check: composition, formation, club limits."""
    if not is_valid_squad_composition(squad.players):
        return False
    lineup_players = squad.get_lineup_players()
    if not is_valid_formation(lineup_players):
        return False
    if not check_club_limits(squad.players, team_map):
        return False
    # Check lineup + bench = all players
    all_indices = set(squad.lineup) | set(squad.bench)
    if len(all_indices) != SQUAD_SIZE or len(squad.lineup) != STARTING_XI:
        return False
    return True


def can_substitute(
    lineup_players: list[PlayerSlot],
    starter_idx_in_lineup: int,
    sub: PlayerSlot,
) -> bool:
    """Check if replacing a starter with a sub maintains a valid formation.

    Args:
        lineup_players: Current starting XI as PlayerSlot list.
        starter_idx_in_lineup: Index within lineup_players to remove.
        sub: The bench player to bring in.
    """
    new_lineup = list(lineup_players)
    new_lineup[starter_idx_in_lineup] = sub
    return is_valid_formation(new_lineup)
