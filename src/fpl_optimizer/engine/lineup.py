"""Auto-substitution and captain failover logic."""

from __future__ import annotations

from fpl_optimizer.data.loader import SeasonDataLoader
from fpl_optimizer.engine.constraints import can_substitute
from fpl_optimizer.engine.scoring import did_player_play
from fpl_optimizer.engine.state import Squad


def perform_auto_subs(
    squad: Squad, loader: SeasonDataLoader, gw: int
) -> tuple[Squad, list[tuple[int, int]]]:
    """Perform auto-substitution for starters who didn't play.

    Walks bench in priority order. For each bench player, checks if they
    can replace a non-playing starter while maintaining a valid formation.

    Returns a new Squad with subs applied and a list of (out_id, in_id) pairs.
    """
    squad = squad.copy()
    subs_made: list[tuple[int, int]] = []

    # Find starters who didn't play
    non_playing_starters: list[int] = []
    for lineup_pos, player_idx in enumerate(squad.lineup):
        element_id = squad.players[player_idx].element_id
        if not did_player_play(loader, element_id, gw):
            non_playing_starters.append(lineup_pos)

    if not non_playing_starters:
        return squad, subs_made

    # Walk bench in priority order
    used_bench: set[int] = set()
    for bench_pos, bench_player_idx in enumerate(squad.bench):
        bench_element_id = squad.players[bench_player_idx].element_id

        # Skip bench players who didn't play
        if not did_player_play(loader, bench_element_id, gw):
            continue

        # Try to sub in for each non-playing starter
        for lineup_pos in non_playing_starters:
            if lineup_pos in used_bench:
                continue  # Already substituted this position

            starter_idx = squad.lineup[lineup_pos]
            lineup_players = squad.get_lineup_players()

            # Find the starter's index within the lineup_players list
            starter_pos_in_list = squad.lineup.index(starter_idx)

            if can_substitute(
                lineup_players, starter_pos_in_list, squad.players[bench_player_idx]
            ):
                # Make the substitution
                out_id = squad.players[starter_idx].element_id
                squad.lineup[lineup_pos] = bench_player_idx
                used_bench.add(lineup_pos)
                subs_made.append((out_id, bench_element_id))
                break  # This bench player is now used

    # Remove subbed-in players from bench, keep order
    subbed_in_ids = {s[1] for s in subs_made}
    squad.bench = [
        b for b in squad.bench
        if squad.players[b].element_id not in subbed_in_ids
    ]

    return squad, subs_made
