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


# P(plays | flag) relative to an unflagged player, for the UPCOMING GW.
# 2026-27 GW1-5 audit: 75%-flagged players played 0 minutes 40/53 times
# (~0.25 played vs ~0.8 for unflagged players with similar minutes history),
# so a 75% flag is worth far less than 0.75x.  Lower flags scale down further.
FLAG_MULTIPLIER = {100: 1.0, 75: 0.35, 50: 0.2, 25: 0.08, 0: 0.0}
# Fraction of the gap to full availability recovered k GWs later
# (doubtful players mostly return within 1-2 GWs; injured/suspended slower).
_RECOVERY = {
    "d": (0.0, 0.6, 0.8, 0.9),
    "i": (0.0, 0.25, 0.5, 0.65),
    "s": (0.0, 0.3, 0.6, 0.8),
}


def availability_multipliers(el: dict, horizon: int) -> list[float]:
    """Per-GW multipliers for a bootstrap element over the horizon."""
    status = el.get("status", "a")
    chance = el.get("chance_of_playing_next_round")
    if status in ("u", "n"):
        return [0.0] * horizon  # left the club / not in the game
    if chance is None:
        m0 = 0.0 if status in ("i", "s") else 1.0
    else:
        c = max(0, min(100, int(chance)))
        keys = sorted(FLAG_MULTIPLIER)
        lo = max(k for k in keys if k <= c)
        hi = min(k for k in keys if k >= c)
        m0 = FLAG_MULTIPLIER[lo] if hi == lo else (
            FLAG_MULTIPLIER[lo]
            + (FLAG_MULTIPLIER[hi] - FLAG_MULTIPLIER[lo]) * (c - lo) / (hi - lo)
        )
    if m0 >= 1.0:
        return [1.0] * horizon
    rec = _RECOVERY.get(status, _RECOVERY["d"])
    return [m0 + (1.0 - m0) * rec[min(k, len(rec) - 1)] for k in range(horizon)]


def build_live_horizon_candidates(
    bootstrap: dict,
    horizon_preds,
    gws: list[int],
    *,
    min_chance: int = 75,
    always_include: set[int] | None = None,
):
    """HorizonCandidates from bootstrap + per-GW horizon predictions.

    ``horizon_preds``: DataFrame ``element, GW, pred, p_play`` (from
    :func:`fpl_optimizer.live.predict.predict_horizon_live`).  Availability
    flags scale the upcoming GW by :data:`FLAG_MULTIPLIER` and recover over
    the horizon; unavailable non-squad players are excluded as in
    :func:`build_live_candidates`.
    """
    from fpl_optimizer.optimizer.horizon_optimizer import HorizonCandidate

    always_include = always_include or set()
    H = len(gws)
    gw_pos = {g: k for k, g in enumerate(gws)}
    xp: dict[int, list[float]] = {}
    pp: dict[int, list[float]] = {}
    for eid, gw, pred, p_play in horizon_preds[["element", "GW", "pred", "p_play"]].itertuples(
            index=False):
        k = gw_pos.get(int(gw))
        if k is None:
            continue
        xp.setdefault(int(eid), [0.0] * H)[k] = float(pred)
        pp.setdefault(int(eid), [0.0] * H)[k] = float(p_play)
    out = []
    for el in bootstrap["elements"]:
        eid = el["id"]
        chance = el.get("chance_of_playing_next_round")
        if eid not in always_include:
            if el["status"] in _EXCLUDED_STATUS:
                continue
            if chance is not None and chance < min_chance:
                continue
        mult = availability_multipliers(el, H)
        x = xp.get(eid, [0.0] * H)
        p = pp.get(eid, [0.0] * H)
        out.append(HorizonCandidate(
            element_id=eid,
            position=Position(el["element_type"]),
            price=el["now_cost"],
            team_id=el["team"],
            xpts=tuple(v * m for v, m in zip(x, mult)),
            p_play=tuple(v * m for v, m in zip(p, mult)),
        ))
    logger.info("Live horizon pool: %d candidates over GW%d-%d", len(out), gws[0], gws[-1])
    return out
