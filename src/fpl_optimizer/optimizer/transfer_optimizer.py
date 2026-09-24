"""Transfer optimizer: decide which players to buy/sell each GW."""

from __future__ import annotations

from collections import defaultdict

from fpl_optimizer.engine.state import GameState
from fpl_optimizer.optimizer.lineup_selector import (
    BENCH_GK_WEIGHT,
    BENCH_OUTFIELD_WEIGHTS,
    VICE_CAPTAIN_WEIGHT,
)
from fpl_optimizer.optimizer.types import OptimizerResult, PlayerCandidate
from fpl_optimizer.utils.constants import (
    MAX_PER_CLUB,
    POSITION_LIMITS,
    SQUAD_SIZE,
    TRANSFER_HIT_COST,
    Position,
)

# Pre-filter: keep top N candidates per position (+ all current squad)
_TOP_PER_POSITION = 20


def _pre_filter(
    candidates: list[PlayerCandidate],
    current_eids: set[int],
    top_per_pos: int = _TOP_PER_POSITION,
) -> list[PlayerCandidate]:
    """Keep current squad members + top predicted-points candidates per position."""
    by_pos: dict[Position, list[PlayerCandidate]] = defaultdict(list)
    current: list[PlayerCandidate] = []

    for cand in candidates:
        if cand.element_id in current_eids:
            current.append(cand)
        else:
            by_pos[cand.position].append(cand)

    filtered = list(current)
    for pos in Position:
        sorted_pos = sorted(by_pos[pos], key=lambda p: p.predicted_points, reverse=True)
        filtered.extend(sorted_pos[:top_per_pos])

    # Deduplicate (current squad member may also appear in top-N)
    seen: set[int] = set()
    deduped: list[PlayerCandidate] = []
    for c in filtered:
        if c.element_id not in seen:
            seen.add(c.element_id)
            deduped.append(c)
    return deduped


def optimize_transfers(
    state: GameState,
    candidates: list[PlayerCandidate],
    chip: str | None = None,
    top_per_pos: int = _TOP_PER_POSITION,
    max_transfers: int | None = None,
) -> OptimizerResult:
    """Decide optimal transfers for one gameweek.

    Parameters
    ----------
    state : GameState
        Current game state (squad, bank, free_transfers).
    candidates : list[PlayerCandidate]
        Full candidate pool for this GW.
    chip : str | None
        If ``"wildcard"`` or ``"free_hit"``, all transfers are free.
    top_per_pos : int
        Pre-filter to top N candidates per position (plus current squad).
    max_transfers : int | None
        Upper bound on the number of transfers. When set (and no WC/FH),
        the optimizer may make at most this many transfers. The optimizer
        can make fewer if not profitable. Default ``None`` = unconstrained.

    Returns
    -------
    OptimizerResult
    """
    import pulp

    current_squad = state.squad.players
    current_eids = {p.element_id for p in current_squad}
    free_transfers = state.free_transfers

    # Build selling price map for current squad
    sell_price: dict[int, int] = {p.element_id: p.selling_price for p in current_squad}

    # Pre-filter candidates
    pool = _pre_filter(candidates, current_eids, top_per_pos)

    # Build lookup
    cand_map: dict[int, PlayerCandidate] = {c.element_id: c for c in pool}

    # Ensure all current squad members are in the pool
    # (they may not appear in GW data if they didn't play — add with 0 xP)
    for p in current_squad:
        if p.element_id not in cand_map:
            cand_map[p.element_id] = PlayerCandidate(
                element_id=p.element_id,
                position=p.position,
                price=p.selling_price,
                team_id=0,  # placeholder — will be constrained via squad membership
                predicted_points=0.0,
            )

    all_cands = list(cand_map.values())
    n = len(all_cands)
    eid_to_idx = {c.element_id: i for i, c in enumerate(all_cands)}

    free_hit = chip == "free_hit"
    wildcard = chip == "wildcard"
    free_chip = free_hit or wildcard
    bench_boost = chip == "bench_boost"

    gk_idx = [i for i, p in enumerate(all_cands) if p.position == Position.GK]
    outfield_idx = [i for i, p in enumerate(all_cands) if p.position != Position.GK]
    n_slots = len(BENCH_OUTFIELD_WEIGHTS)

    # --- Variables ---
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]  # in new squad
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]  # in lineup
    c_var = [pulp.LpVariable(f"c_{i}", cat="Binary") for i in range(n)]  # captain
    v_var = [pulp.LpVariable(f"v_{i}", cat="Binary") for i in range(n)]  # vice-captain
    # b[i, s] = 1 if outfield player i sits in bench slot s (after the GK).
    # Not needed under Bench Boost: all bench players score in full there.
    b: dict[tuple[int, int], object] = {}
    if not bench_boost:
        b = {
            (i, s): pulp.LpVariable(f"b_{i}_{s}", cat="Binary")
            for i in outfield_idx
            for s in range(n_slots)
        }

    prob = pulp.LpProblem("transfer_optimizer", pulp.LpMaximize)

    xp = [p.predicted_points for p in all_cands]

    # --- Linearised transfer count ---
    # s_out_j = 1 if current squad player j is sold (not in new squad)
    current_indices = [eid_to_idx[p.element_id] for p in current_squad]
    s_out = {j: 1 - x[j] for j in current_indices}
    n_transfers = pulp.lpSum(s_out[j] for j in current_indices)

    # Optional upper bound on transfers (e.g. 1/GW in the conservative backtest)
    if max_transfers is not None and not free_chip:
        prob += n_transfers <= max_transfers

    # Hit cost linearisation: ordered binary vars t_k, k = 1..max_possible_transfers
    max_possible = SQUAD_SIZE  # at most 15 transfers
    if free_chip:
        # No hit cost
        hit_expr = 0
    else:
        t = [pulp.LpVariable(f"t_{k}", cat="Binary") for k in range(max_possible)]
        # t_k >= 0, sum(t_k) = n_transfers, t_k >= t_{k+1} (ordered)
        prob += pulp.lpSum(t) == n_transfers
        for k in range(max_possible - 1):
            prob += t[k] >= t[k + 1]
        # Hit = 4 * sum of t_k for k >= free_transfers
        hit_expr = TRANSFER_HIT_COST * pulp.lpSum(
            t[k] for k in range(free_transfers, max_possible)
        )

    # Objective: XI points + captain double + failover/auto-sub EV - hits,
    # with chip-aware terms.
    objective = (
        pulp.lpSum(xp[i] * y[i] for i in range(n))
        + pulp.lpSum(xp[i] * c_var[i] for i in range(n))
        + VICE_CAPTAIN_WEIGHT * pulp.lpSum(xp[i] * v_var[i] for i in range(n))
        - hit_expr
    )
    if bench_boost:
        # Bench players score in full — value the whole 15, not just the XI.
        # This replaces (not stacks with) the weighted auto-sub EV proxies.
        objective += pulp.lpSum(xp[i] * (x[i] - y[i]) for i in range(n))
    else:
        # Auto-sub EV: benched GK + slot-weighted benched outfielders
        objective += BENCH_GK_WEIGHT * pulp.lpSum(
            xp[i] * (x[i] - y[i]) for i in gk_idx
        )
        objective += pulp.lpSum(
            BENCH_OUTFIELD_WEIGHTS[s] * xp[i] * b[i, s]
            for i in outfield_idx
            for s in range(n_slots)
        )
    if chip == "triple_captain":
        # Captain counts 3x instead of 2x
        objective += pulp.lpSum(xp[i] * c_var[i] for i in range(n))
    prob += objective

    # --- Squad constraints ---
    prob += pulp.lpSum(x) == SQUAD_SIZE

    pos_indices: dict[Position, list[int]] = defaultdict(list)
    for i, p in enumerate(all_cands):
        pos_indices[p.position].append(i)

    for pos, limit in POSITION_LIMITS.items():
        prob += pulp.lpSum(x[i] for i in pos_indices[pos]) == limit

    # Club limit
    team_indices: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(all_cands):
        team_indices[p.team_id].append(i)
    for team_id, indices in team_indices.items():
        prob += pulp.lpSum(x[i] for i in indices) <= MAX_PER_CLUB

    # Budget: bank + sell revenue - buy cost >= 0
    # For current squad: selling frees sell_price, keeping costs 0 (already owned)
    # For new players: buying costs candidate.price
    budget_expr = state.bank
    for j in current_indices:
        eid = all_cands[j].element_id
        sp = sell_price[eid]
        # selling revenue when not kept: sp * (1 - x[j])
        budget_expr += sp * (1 - x[j])
    for i in range(n):
        eid = all_cands[i].element_id
        if eid not in current_eids:
            # buying cost when selected
            budget_expr -= all_cands[i].price * x[i]
    prob += budget_expr >= 0

    # --- Lineup constraints ---
    for i in range(n):
        prob += y[i] <= x[i]
    prob += pulp.lpSum(y) == 11

    prob += pulp.lpSum(y[i] for i in pos_indices[Position.GK]) == 1
    prob += pulp.lpSum(y[i] for i in pos_indices[Position.DEF]) >= 3
    prob += pulp.lpSum(y[i] for i in pos_indices[Position.DEF]) <= 5
    prob += pulp.lpSum(y[i] for i in pos_indices[Position.MID]) >= 2
    prob += pulp.lpSum(y[i] for i in pos_indices[Position.MID]) <= 5
    prob += pulp.lpSum(y[i] for i in pos_indices[Position.FWD]) >= 1
    prob += pulp.lpSum(y[i] for i in pos_indices[Position.FWD]) <= 3

    # Captain / vice-captain
    for i in range(n):
        prob += c_var[i] <= y[i]
        prob += v_var[i] <= y[i]
        prob += c_var[i] + v_var[i] <= 1
    prob += pulp.lpSum(c_var) == 1
    prob += pulp.lpSum(v_var) == 1

    # Bench slot assignment: each benched outfielder fills exactly one slot,
    # each slot holds exactly one player
    if not bench_boost:
        for i in outfield_idx:
            prob += pulp.lpSum(b[i, s] for s in range(n_slots)) == x[i] - y[i]
        for s in range(n_slots):
            prob += pulp.lpSum(b[i, s] for i in outfield_idx) == 1

    # --- Solve (use default available solver) ---
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.constants.LpStatusOptimal:
        raise RuntimeError(f"Transfer optimizer LP not optimal, status={prob.status}")

    # --- Extract solution ---
    new_squad_ids: list[int] = []
    lineup_ids: list[int] = []
    bench_candidates: list[tuple[float, Position, int]] = []
    captain_id = all_cands[0].element_id
    vice_captain_id = all_cands[1].element_id
    total_cost = 0

    for i in range(n):
        if pulp.value(x[i]) > 0.5:
            eid = all_cands[i].element_id
            new_squad_ids.append(eid)
            # Cost: for new players it's their price; for kept players it's 0 (already paid)
            if eid not in current_eids:
                total_cost += all_cands[i].price
            if pulp.value(y[i]) > 0.5:
                lineup_ids.append(eid)
            else:
                bench_candidates.append(
                    (all_cands[i].predicted_points, all_cands[i].position, eid)
                )
            if pulp.value(c_var[i]) > 0.5:
                captain_id = eid
            if pulp.value(v_var[i]) > 0.5:
                vice_captain_id = eid

    # Bench order: backup GK first, then outfield by optimized slot assignment
    # (under Bench Boost slot order is irrelevant — sort by xP descending)
    bench_gk = [eid for _, pos, eid in bench_candidates if pos == Position.GK]
    if bench_boost:
        bench_outfield = sorted(
            [(xp_val, eid) for xp_val, pos, eid in bench_candidates if pos != Position.GK],
            key=lambda t: t[0],
            reverse=True,
        )
        bench_ids = bench_gk + [eid for _, eid in bench_outfield]
    else:
        slot_to_eid: dict[int, int] = {}
        for i in outfield_idx:
            for s in range(n_slots):
                if pulp.value(b[i, s]) > 0.5:
                    slot_to_eid[s] = all_cands[i].element_id
        bench_ids = bench_gk + [slot_to_eid[s] for s in range(n_slots)]

    # Determine transfers in/out
    new_squad_set = set(new_squad_ids)
    transfers_out = [eid for eid in current_eids if eid not in new_squad_set]
    transfers_in = [eid for eid in new_squad_ids if eid not in current_eids]

    num_transfers = len(transfers_out)
    if free_chip:
        hit_cost = 0
    else:
        hit_cost = TRANSFER_HIT_COST * max(0, num_transfers - free_transfers)

    return OptimizerResult(
        squad_element_ids=new_squad_ids,
        lineup_element_ids=lineup_ids,
        bench_element_ids=bench_ids,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        transfers_in=transfers_in,
        transfers_out=transfers_out,
        chip=chip,
        objective_value=pulp.value(prob.objective),
        total_cost=total_cost,
        hit_cost=hit_cost,
    )
