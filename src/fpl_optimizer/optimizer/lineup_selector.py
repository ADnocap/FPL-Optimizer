"""Lineup selector: given a fixed 15-player squad, pick optimal XI + captain."""

from __future__ import annotations

from fpl_optimizer.optimizer.types import OptimizerResult, PlayerCandidate
from fpl_optimizer.utils.constants import Position

# Community-calibrated EV proxies (FPL analytics consensus, e.g. fplreview-style
# effective-ownership models): expected fraction of a bench player's points
# realised through auto-subs (per bench slot), and of the vice-captain's points
# realised through captain failover.
BENCH_GK_WEIGHT = 0.03
BENCH_OUTFIELD_WEIGHTS = (0.21, 0.06, 0.002)  # outfield bench slots 1..3
VICE_CAPTAIN_WEIGHT = 0.1


def select_lineup(
    squad: list[PlayerCandidate],
) -> OptimizerResult:
    """Select optimal starting XI, captain, vice-captain, and bench order.

    Solves a binary MILP:
        max  sum(xP_i * y_i) + sum(xP_i * c_i) + 0.1 * sum(xP_i * v_i)
             + bench auto-sub EV (weighted bench GK + slot-weighted outfield)
    where y_i=1 means player i starts, c_i=1 means captain, v_i=1 means
    vice-captain. Bench slot assignment is part of the MILP so bench order
    is optimized, not a post-hoc sort.

    Parameters
    ----------
    squad : list[PlayerCandidate]
        Exactly 15 players (2 GK, 5 DEF, 5 MID, 3 FWD).

    Returns
    -------
    OptimizerResult
    """
    import pulp

    n = len(squad)
    if n != 15:
        raise ValueError(f"Squad must have exactly 15 players, got {n}")

    # Position index lists
    gk_idx = [i for i, p in enumerate(squad) if p.position == Position.GK]
    def_idx = [i for i, p in enumerate(squad) if p.position == Position.DEF]
    mid_idx = [i for i, p in enumerate(squad) if p.position == Position.MID]
    fwd_idx = [i for i, p in enumerate(squad) if p.position == Position.FWD]
    outfield_idx = [i for i, p in enumerate(squad) if p.position != Position.GK]
    n_slots = len(BENCH_OUTFIELD_WEIGHTS)

    # Decision variables
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]
    c = [pulp.LpVariable(f"c_{i}", cat="Binary") for i in range(n)]
    v = [pulp.LpVariable(f"v_{i}", cat="Binary") for i in range(n)]
    # b[i, s] = 1 if outfield player i sits in bench slot s (after the GK)
    b = {
        (i, s): pulp.LpVariable(f"b_{i}_{s}", cat="Binary")
        for i in outfield_idx
        for s in range(n_slots)
    }

    prob = pulp.LpProblem("lineup_selection", pulp.LpMaximize)

    # Objective: expected points + captain bonus + failover/auto-sub EV
    xp = [p.predicted_points for p in squad]
    prob += (
        pulp.lpSum(xp[i] * y[i] for i in range(n))
        + pulp.lpSum(xp[i] * c[i] for i in range(n))
        + VICE_CAPTAIN_WEIGHT * pulp.lpSum(xp[i] * v[i] for i in range(n))
        + BENCH_GK_WEIGHT * pulp.lpSum(xp[i] * (1 - y[i]) for i in gk_idx)
        + pulp.lpSum(
            BENCH_OUTFIELD_WEIGHTS[s] * xp[i] * b[i, s]
            for i in outfield_idx
            for s in range(n_slots)
        )
    )

    # --- constraints ---
    # Exactly 11 starters
    prob += pulp.lpSum(y) == 11

    # Position constraints (on starters)
    prob += pulp.lpSum(y[i] for i in gk_idx) == 1
    prob += pulp.lpSum(y[i] for i in def_idx) >= 3
    prob += pulp.lpSum(y[i] for i in def_idx) <= 5
    prob += pulp.lpSum(y[i] for i in mid_idx) >= 2
    prob += pulp.lpSum(y[i] for i in mid_idx) <= 5
    prob += pulp.lpSum(y[i] for i in fwd_idx) >= 1
    prob += pulp.lpSum(y[i] for i in fwd_idx) <= 3

    # Captain / vice-captain must be in starting XI and distinct
    for i in range(n):
        prob += c[i] <= y[i]
        prob += v[i] <= y[i]
        prob += c[i] + v[i] <= 1
    prob += pulp.lpSum(c) == 1
    prob += pulp.lpSum(v) == 1

    # Bench slot assignment: each benched outfielder fills exactly one slot,
    # each slot holds exactly one player
    for i in outfield_idx:
        prob += pulp.lpSum(b[i, s] for s in range(n_slots)) == 1 - y[i]
    for s in range(n_slots):
        prob += pulp.lpSum(b[i, s] for i in outfield_idx) == 1

    # Solve (use default available solver)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.constants.LpStatusOptimal:
        raise RuntimeError(f"Lineup LP not optimal, status={prob.status}")

    # Extract solution
    lineup_ids: list[int] = []
    captain_id = squad[0].element_id
    vice_captain_id = squad[1].element_id

    for i in range(n):
        if pulp.value(y[i]) > 0.5:
            lineup_ids.append(squad[i].element_id)
        if pulp.value(c[i]) > 0.5:
            captain_id = squad[i].element_id
        if pulp.value(v[i]) > 0.5:
            vice_captain_id = squad[i].element_id

    # Bench order: backup GK first, then outfield by optimized slot assignment
    bench_gk = [squad[i].element_id for i in gk_idx if pulp.value(y[i]) < 0.5]
    slot_to_eid: dict[int, int] = {}
    for i in outfield_idx:
        for s in range(n_slots):
            if pulp.value(b[i, s]) > 0.5:
                slot_to_eid[s] = squad[i].element_id
    bench_ids = bench_gk + [slot_to_eid[s] for s in range(n_slots)]

    all_ids = lineup_ids + bench_ids
    return OptimizerResult(
        squad_element_ids=all_ids,
        lineup_element_ids=lineup_ids,
        bench_element_ids=bench_ids,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        objective_value=pulp.value(prob.objective),
    )
