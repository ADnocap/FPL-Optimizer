"""Point lookup from historical data and captain/chip multipliers."""

from __future__ import annotations

from fpl_optimizer.data.loader import SeasonDataLoader
from fpl_optimizer.engine.state import Squad


def get_player_points(
    loader: SeasonDataLoader, element_id: int, gw: int
) -> int:
    """Get a player's total points for a GW from historical data.

    Handles DGW automatically (loader sums across fixtures).
    Returns 0 if player has no data for that GW.
    """
    data = loader.get_player_gw(element_id, gw)
    if data is None:
        return 0
    return int(data.get("total_points", 0))


def get_player_minutes(
    loader: SeasonDataLoader, element_id: int, gw: int
) -> int:
    """Get a player's total minutes for a GW. Returns 0 if not found."""
    data = loader.get_player_gw(element_id, gw)
    if data is None:
        return 0
    return int(data.get("minutes", 0))


def did_player_play(
    loader: SeasonDataLoader, element_id: int, gw: int
) -> bool:
    """Check if a player 'played' (1+ minutes OR received a card).

    Per 2025/26 rules, a player counts as having played if they
    played 1+ minutes or received a yellow/red card.
    """
    data = loader.get_player_gw(element_id, gw)
    if data is None:
        return False
    minutes = int(data.get("minutes", 0))
    if minutes > 0:
        return True
    yellow = int(data.get("yellow_cards", 0))
    red = int(data.get("red_cards", 0))
    return (yellow + red) > 0


def calculate_captain_points(
    loader: SeasonDataLoader,
    squad: Squad,
    gw: int,
    triple_captain: bool = False,
) -> tuple[int, bool]:
    """Calculate captain bonus points and handle failover.

    Captain bonus is only awarded if the captain (or vice-captain) is in
    the active lineup after auto-substitutions.

    Returns:
        (bonus_points, failover_used): bonus points from captain multiplier
        and whether vice-captain failover was triggered.
    """
    captain_id = squad.players[squad.captain_idx].element_id
    vice_id = squad.players[squad.vice_captain_idx].element_id
    lineup_ids = {squad.players[i].element_id for i in squad.lineup}

    captain_played = did_player_play(loader, captain_id, gw)
    captain_in_lineup = captain_id in lineup_ids
    multiplier = 3 if triple_captain else 2

    if captain_played and captain_in_lineup:
        base_points = get_player_points(loader, captain_id, gw)
        # Bonus is (multiplier - 1) * points since base points already counted
        return base_points * (multiplier - 1), False

    # Captain didn't play or isn't in lineup — try vice-captain
    vice_played = did_player_play(loader, vice_id, gw)
    vice_in_lineup = vice_id in lineup_ids
    if vice_played and vice_in_lineup:
        base_points = get_player_points(loader, vice_id, gw)
        return base_points * (multiplier - 1), True

    # Neither played/in lineup — no bonus
    return 0, True


def calculate_squad_points(
    loader: SeasonDataLoader,
    squad: Squad,
    gw: int,
) -> tuple[int, int]:
    """Calculate raw points for lineup and bench separately.

    Returns (lineup_points, bench_points).
    Does NOT apply captain multiplier — that's handled separately.
    """
    lineup_points = sum(
        get_player_points(loader, squad.players[i].element_id, gw)
        for i in squad.lineup
    )
    bench_points = sum(
        get_player_points(loader, squad.players[i].element_id, gw)
        for i in squad.bench
    )
    return lineup_points, bench_points
