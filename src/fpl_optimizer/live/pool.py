"""Live candidate pool built from bootstrap-static (not historical data).

Prices come from ``now_cost`` (what you would actually pay today) and
availability comes from ``status`` / ``chance_of_playing_next_round`` —
signals that only exist pre-deadline and are absent from historical replays.
"""

from __future__ import annotations

import logging

from fpl_optimizer.optimizer.types import PlayerCandidate
from fpl_optimizer.utils.constants import Position

logger = logging.getLogger(__name__)

# status: a=available, d=doubtful, i=injured, s=suspended, u=unavailable, n=NA
_EXCLUDED_STATUS = {"i", "s", "u", "n"}


def build_live_candidates(
    bootstrap: dict,
    predicted_points: dict[int, float],
    *,
    min_chance: int = 75,
    always_include: set[int] | None = None,
    availability_scaling: bool = True,
) -> list[PlayerCandidate]:
    """Build optimizer candidates from live bootstrap data.

    Parameters
    ----------
    bootstrap : dict
        Current bootstrap-static JSON.
    predicted_points : dict[int, float]
        element_id -> predicted points for the upcoming GW.
    min_chance : int
        Exclude players whose chance_of_playing_next_round is below this
        (unless they are in *always_include* — you can still sell them).
    always_include : set[int] | None
        Element ids that must appear in the pool regardless of availability
        (the current squad — the optimizer needs them to price transfers out).
    availability_scaling : bool
        Scale predictions by chance_of_playing (75% chance -> 0.75x points).
    """
    always_include = always_include or set()
    candidates: list[PlayerCandidate] = []
    for el in bootstrap["elements"]:
        eid = el["id"]
        in_squad = eid in always_include
        chance = el.get("chance_of_playing_next_round")
        if not in_squad:
            if el["status"] in _EXCLUDED_STATUS:
                continue
            if chance is not None and chance < min_chance:
                continue
        pts = predicted_points.get(eid, 0.0)
        if availability_scaling and chance is not None:
            pts *= max(0, min(100, chance)) / 100.0
        if el["status"] in _EXCLUDED_STATUS:
            pts = 0.0  # in-squad but out injured: never expect points
        candidates.append(
            PlayerCandidate(
                element_id=eid,
                position=Position(el["element_type"]),
                price=el["now_cost"],
                team_id=el["team"],
                predicted_points=pts,
            )
        )
    logger.info("Live pool: %d candidates", len(candidates))
    return candidates
