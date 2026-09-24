"""Initial squad selection: pick 15 players + lineup + captain from full pool."""

from __future__ import annotations

from collections import defaultdict

from fpl_optimizer.optimizer.types import OptimizerResult, PlayerCandidate
from fpl_optimizer.utils.constants import (
    MAX_PER_CLUB,
    POSITION_LIMITS,
    SQUAD_SIZE,
    STARTING_BUDGET,
    Position,
)


def select_squad(
    candidates: list[PlayerCandidate],
    budget: int = STARTING_BUDGET,
) -> OptimizerResult:
    """Select an optimal 15-player squad with lineup and captain.

    Integrated MILP that simultaneously picks squad, starting XI, and captain.

    Parameters
    ----------
    candidates : list[PlayerCandidate]
        Full candidate pool.
    budget : int
        Total budget in tenths (default 1000 = £100m).

    Returns
    -------
    OptimizerResult
    """
    import pulp

    n = len(candidates)
    if n == 0:
        raise ValueError("Empty candidate pool")

    # Decision variables
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]  # in squad
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]  # in lineup
    c = [pulp.LpVariable(f"c_{i}", cat="Binary") for i in range(n)]  # captain
    v = [pulp.LpVariable(f"v_{i}", cat="Binary") for i in range(n)]  # vice-captain

    prob = pulp.LpProblem("squad_selection", pulp.LpMaximize)

    xp = [p.predicted_points for p in candidates]

    # Objective: maximise lineup points + captain bonus
    prob += pulp.lpSum(xp[i] * y[i] for i in range(n)) + pulp.lpSum(
        xp[i] * c[i] for i in range(n)
    )

    # --- Squad constraints ---
    prob += pulp.lpSum(x) == SQUAD_SIZE
    prob += pulp.lpSum(candidates[i].price * x[i] for i in range(n)) <= budget

    # Position limits for squad
    pos_indices: dict[Position, list[int]] = defaultdict(list)
    for i, p in enumerate(candidates):
        pos_indices[p.position].append(i)

    for pos, limit in POSITION_LIMITS.items():
        prob += pulp.lpSum(x[i] for i in pos_indices[pos]) == limit

    # Club limit
    team_indices: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(candidates):
        team_indices[p.team_id].append(i)
    for team_id, indices in team_indices.items():
        prob += pulp.lpSum(x[i] for i in indices) <= MAX_PER_CLUB

    # --- Lineup constraints ---
    # Can only start players in squad
    for i in range(n):
        prob += y[i] <= x[i]

    # Exactly 11 starters
    prob += pulp.lpSum(y) == 11

    # Formation constraints on starters
    prob += pulp.lpSum(y[i] for i in pos_indices[Position.GK]) == 1
    prob += pulp.lpSum(y[i] for i in pos_indices[Position.DEF]) >= 3
    prob += pulp.lpSum(y[i] for i in pos_indices[Position.DEF]) <= 5
    prob += pulp.lpSum(y[i] for i in pos_indices[Position.MID]) >= 2
    prob += pulp.lpSum(y[i] for i in pos_indices[Position.MID]) <= 5
    prob += pulp.lpSum(y[i] for i in pos_indices[Position.FWD]) >= 1
    prob += pulp.lpSum(y[i] for i in pos_indices[Position.FWD]) <= 3

    # Captain / vice-captain
    for i in range(n):
        prob += c[i] <= y[i]
        prob += v[i] <= y[i]
        prob += c[i] + v[i] <= 1
    prob += pulp.lpSum(c) == 1
    prob += pulp.lpSum(v) == 1

    # Solve (use default available solver)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.constants.LpStatusOptimal:
        raise RuntimeError(f"Squad selection LP not optimal, status={prob.status}")

    # Extract solution
    squad_ids: list[int] = []
    lineup_ids: list[int] = []
    bench_candidates: list[tuple[float, Position, int]] = []
    captain_id = candidates[0].element_id
    vice_captain_id = candidates[1].element_id
    total_cost = 0

    for i in range(n):
        if pulp.value(x[i]) > 0.5:
            squad_ids.append(candidates[i].element_id)
            total_cost += candidates[i].price
            if pulp.value(y[i]) > 0.5:
                lineup_ids.append(candidates[i].element_id)
            else:
                bench_candidates.append(
                    (candidates[i].predicted_points, candidates[i].position, candidates[i].element_id)
                )
        if pulp.value(c[i]) > 0.5:
            captain_id = candidates[i].element_id
        if pulp.value(v[i]) > 0.5:
            vice_captain_id = candidates[i].element_id

    # Bench order: backup GK first, then by predicted points descending
    bench_gk = [eid for _, pos, eid in bench_candidates if pos == Position.GK]
    bench_outfield = sorted(
        [(xp_val, eid) for xp_val, pos, eid in bench_candidates if pos != Position.GK],
        key=lambda t: t[0],
        reverse=True,
    )
    bench_ids = bench_gk + [eid for _, eid in bench_outfield]

    return OptimizerResult(
        squad_element_ids=squad_ids,
        lineup_element_ids=lineup_ids,
        bench_element_ids=bench_ids,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        objective_value=pulp.value(prob.objective),
        total_cost=total_cost,
    )
