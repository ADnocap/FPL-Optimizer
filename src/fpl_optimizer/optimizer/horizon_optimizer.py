"""Multi-period (receding-horizon) transfer planner.

Plans squad, lineup, captain/vice and transfers jointly for GWs
``t .. t+H-1`` and returns the GW t decisions for execution (receding
horizon: re-solve every GW with fresh predictions).  Unlike the single-GW
:func:`~fpl_optimizer.optimizer.transfer_optimizer.optimize_transfers`, it can
roll a free transfer when next week's move is better, spread a double move
over two free transfers instead of taking a hit, and plan around fixture swings,
blanks/doubles and a given chip schedule.

Model (per horizon GW k, discount ``d_k = discount**k``):

- squad ``x[i,k]`` (binary), lineup ``y[i,k]`` (binary), captain/vice/bench
  slots continuous (their sub-problems are transportation polytopes, integral
  for integral ``x``/``y``), transfers in/out continuous linked to ``x``.
- FT dynamics exactly as the engine/FPL: 1 new FT per GW, bank capped at 5
  (``MAX_FREE_TRANSFERS``), unused FTs carry, ``-4`` per transfer beyond the
  FTs held (no hit may be taken while FTs are left unused); Wildcard/Free Hit
  weeks are free and leave the FT count unchanged for the next GW.
- budget: bank ``>= 0`` every GW; current players sell at their SELLING price,
  buys at current price (prices assumed flat over the horizon).
- 15-man squad 2/5/5/3, max 3 per club, valid formations (1 GK, 3-5 DEF,
  2-5 MID, 1-3 FWD).
- chips for a GIVEN schedule ``{gw: chip}``: wildcard (unlimited free
  transfers, permanent), free_hit (a separate one-week squad; the real squad
  is unchanged and "reverts"), bench_boost (bench counts in full),
  triple_captain (captain x3).
- bench value: benched players are worth ``w * xPts`` where ``w`` is the
  probability they are needed, derived from the DNP risk of the XI
  (``1 - p_play``; Poisson-binomial over the 10 outfield starters for bench
  slots 1-3, the starting GK's DNP risk for the bench GK).  Bench Boost weeks
  count the bench in full.
- objective: sum_k d_k * (XI + captain (x2/x3) + vice EV + bench EV
  - (4 + hit_margin) * hits) + ft_value * FTs banked after the horizon.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

from fpl_optimizer.engine.state import GameState
from fpl_optimizer.optimizer.types import OptimizerResult
from fpl_optimizer.utils.constants import (
    MAX_FREE_TRANSFERS,
    MAX_PER_CLUB,
    POSITION_LIMITS,
    SQUAD_SIZE,
    TRANSFER_HIT_COST,
    Position,
)

logger = logging.getLogger(__name__)

CHIP_WILDCARD = "wildcard"
CHIP_FREE_HIT = "free_hit"
CHIP_BENCH_BOOST = "bench_boost"
CHIP_TRIPLE_CAPTAIN = "triple_captain"
_CHIPS = {CHIP_WILDCARD, CHIP_FREE_HIT, CHIP_BENCH_BOOST, CHIP_TRIPLE_CAPTAIN}
_CHIP_ALIASES = {
    "wc": CHIP_WILDCARD, "wildcard": CHIP_WILDCARD,
    "fh": CHIP_FREE_HIT, "free_hit": CHIP_FREE_HIT, "freehit": CHIP_FREE_HIT,
    "bb": CHIP_BENCH_BOOST, "bench_boost": CHIP_BENCH_BOOST, "bboost": CHIP_BENCH_BOOST,
    "tc": CHIP_TRIPLE_CAPTAIN, "triple_captain": CHIP_TRIPLE_CAPTAIN,
    "3xc": CHIP_TRIPLE_CAPTAIN,
}
TRANSFERS_CAP_PER_GW = 20  # game_settings.transfers_cap (2026-27)
_N_BENCH_SLOTS = 3


@dataclass(frozen=True)
class HorizonCandidate:
    """A player the planner can hold, with per-GW predictions.

    ``xpts[k]`` / ``p_play[k]`` refer to horizon GW ``gws[k]``; a blank GW is
    ``xpts=0`` (and ``p_play=0``).  ``price`` is today's buy price.
    """

    element_id: int
    position: Position
    price: int
    team_id: int
    xpts: tuple[float, ...]
    p_play: tuple[float, ...] | None = None


@dataclass
class HorizonConfig:
    """Planner knobs (defaults = recommended live settings)."""

    discount: float = 0.85
    # Regularisation against the optimiser's curse (as-of-t prediction errors
    # repeat across the horizon, so planned gains are over-stated): backtests
    # 2023-24..2025-26 — unregularised H=4 took ~370 pts of hits/season and
    # lost ~160 pts/season vs the single-GW MILP; H=3 with a 4-pt margin (a hit
    # must promise > 8 planned pts) was the most consistent config: +19/season vs
    # the single-GW MILP over 9 replays (6 wins), +24/season with chip plans.
    # See scripts/backtest_horizon.py.
    hit_margin: float = 4.0
    ft_value: float = 0.0
    bench_mode: str = "dnp"  # "dnp" | "fixed" (legacy lineup_selector weights)
    default_p_play: float = 0.9
    vice_weight: float = 0.1
    max_transfers_per_gw: int | None = None
    max_hits_per_gw: int | None = None  # 0 = never take a -4 (FT-only planning)
    top_per_pos: dict = field(default_factory=lambda: {
        Position.GK: 10, Position.DEF: 30, Position.MID: 35, Position.FWD: 20})
    value_per_pos: dict = field(default_factory=lambda: {
        Position.GK: 4, Position.DEF: 8, Position.MID: 8, Position.FWD: 5})
    time_limit: float = 60.0
    gap_rel: float = 0.0005
    locked: frozenset = frozenset()
    banned: frozenset = frozenset()


@dataclass
class GWPlan:
    """The planned decisions for one GW of the horizon."""

    gw: int
    chip: str | None
    free_transfers: int  # FTs available before this GW's transfers
    transfers_in: list[int]
    transfers_out: list[int]
    hits: int  # number of -4 hits
    squad: list[int]
    lineup: list[int]
    bench: list[int]
    captain_id: int
    vice_captain_id: int
    expected_points: float  # XI + captain (+ bench under BB), undiscounted
    bank_after: int


@dataclass
class HorizonResult:
    first: OptimizerResult  # the executable GW t decisions
    plan: list[GWPlan]
    objective_value: float
    status: str
    solve_seconds: float
    n_candidates: int
    bench_weights: list[tuple[float, tuple[float, ...]]]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def parse_chip_plan(spec: str | None) -> dict[int, str]:
    """Parse ``"tc:7,wc:11,bb:12"`` (or ``"tc7,wc11"``) into ``{7: 'triple_captain', ...}``."""
    import re

    plan: dict[int, str] = {}
    if not spec:
        return plan
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(3xc|[a-z_]+)\s*:?\s*(\d+)", part.lower())
        chip = _CHIP_ALIASES.get(m.group(1)) if m else None
        if chip is None:
            raise ValueError(f"Bad chip-plan entry {part!r} (use e.g. 'tc:7,wc:11')")
        g = int(m.group(2))
        if g in plan:
            raise ValueError(f"Two chips planned for GW{g}: only one chip per GW")
        plan[g] = chip
    return plan


def _poisson_binomial_tail(qs: list[float], max_s: int) -> list[float]:
    """P(N >= s) for s = 1..max_s, N = number of successes with probs qs."""
    dist = [1.0]
    for q in qs:
        q = min(max(q, 0.0), 1.0)
        new = [0.0] * (len(dist) + 1)
        for n, p in enumerate(dist):
            new[n] += p * (1 - q)
            new[n + 1] += p * q
        dist = new
    return [sum(dist[s:]) for s in range(1, max_s + 1)]


def dnp_bench_weights(
    outfield_q: list[float], gk_q: float
) -> tuple[float, tuple[float, ...]]:
    """Bench weights from the XI's did-not-play probabilities.

    Bench outfield slot s (1-based) comes on when at least s outfield
    starters don't play; the bench GK when the starting GK doesn't.
    """
    tail = _poisson_binomial_tail(outfield_q, _N_BENCH_SLOTS)
    return float(min(max(gk_q, 0.0), 1.0)), tuple(float(w) for w in tail)


def _likely_xi(cands: list[HorizonCandidate], k: int) -> tuple[list[HorizonCandidate], HorizonCandidate | None]:
    """Greedy formation-valid XI by xPts (for estimating DNP risk)."""
    by_pos: dict[Position, list[HorizonCandidate]] = defaultdict(list)
    for c in cands:
        by_pos[c.position].append(c)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda c: c.xpts[k], reverse=True)
    gk = by_pos[Position.GK][0] if by_pos[Position.GK] else None
    mins = {Position.DEF: 3, Position.MID: 2, Position.FWD: 1}
    maxs = {Position.DEF: 5, Position.MID: 5, Position.FWD: 3}
    chosen: list[HorizonCandidate] = []
    for pos, n in mins.items():
        chosen += by_pos[pos][:n]
    rest = sorted(
        (c for pos in mins for c in by_pos[pos][mins[pos]:maxs[pos]]),
        key=lambda c: c.xpts[k], reverse=True,
    )
    chosen += rest[: 10 - len(chosen)]
    return chosen, gk


def _p_play(c: HorizonCandidate, k: int, default: float) -> float:
    if c.p_play is not None:
        return float(c.p_play[k])
    return 0.0 if c.xpts[k] <= 0 else default


def _pre_filter(
    cands: list[HorizonCandidate], keep: set[int], cfg: HorizonConfig, disc: list[float]
) -> list[HorizonCandidate]:
    score = {c.element_id: sum(d * x for d, x in zip(disc, c.xpts)) for c in cands}
    by_pos: dict[Position, list[HorizonCandidate]] = defaultdict(list)
    out: dict[int, HorizonCandidate] = {}
    for c in cands:
        if c.element_id in keep:
            out[c.element_id] = c
        elif c.element_id not in cfg.banned:
            by_pos[c.position].append(c)
    for pos, lst in by_pos.items():
        top = sorted(lst, key=lambda c: score[c.element_id], reverse=True)
        for c in top[: cfg.top_per_pos.get(pos, 20)]:
            out.setdefault(c.element_id, c)
        val = sorted(
            (c for c in lst if score[c.element_id] > 0 and c.price > 0),
            key=lambda c: score[c.element_id] / c.price, reverse=True,
        )
        for c in val[: cfg.value_per_pos.get(pos, 5)]:
            out.setdefault(c.element_id, c)
    return list(out.values())


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


def optimize_horizon(
    state: GameState,
    candidates: list[HorizonCandidate],
    gws: list[int],
    chip_plan: dict[int, str] | None = None,
    config: HorizonConfig | None = None,
) -> HorizonResult:
    """Solve the multi-period transfer plan; return GW ``gws[0]`` decisions.

    Parameters
    ----------
    state : GameState
        Current squad (with selling prices), bank and free transfers for
        ``gws[0]``.  ``state.chips`` is used to drop unavailable chips.
    candidates : list[HorizonCandidate]
        Pool with ``len(xpts) == len(gws)``.  Current squad members missing
        from the pool are added with 0 xPts (they can still be sold).
    gws : list[int]
        Consecutive horizon GWs, e.g. ``[6, 7, 8, 9]``.
    chip_plan : dict[int, str] | None
        ``{gw: chip}`` — chips to play in horizon GWs (a human decision; the
        planner optimises transfers/lineups around it).
    """
    import pulp

    cfg = config or HorizonConfig()
    H = len(gws)
    if H == 0:
        raise ValueError("Empty horizon")
    for c in candidates:
        if len(c.xpts) != H:
            raise ValueError(f"Candidate {c.element_id}: {len(c.xpts)} xPts for {H} GWs")

    # ---- chips in the horizon (validated against availability) ----------
    chips: dict[int, str] = {}
    avail = state.chips.copy()  # consumed in GW order: one use per chip per half
    for gw, chip in sorted((chip_plan or {}).items()):
        chip = _CHIP_ALIASES.get(str(chip).lower(), chip)
        if chip not in _CHIPS:
            raise ValueError(f"Unknown chip {chip!r}")
        if gw not in gws:
            continue
        if not avail.is_available(chip, gw):  # used/expired/FH GW19+20 rule
            logger.warning("Chip %s not available for GW%d — ignored", chip, gw)
            continue
        avail.use_chip(chip, gw)
        chips[gw] = chip
    chip_k = [chips.get(gw) for gw in gws]

    disc = [cfg.discount ** k for k in range(H)]
    squad0 = {p.element_id: p for p in state.squad.players}
    keep = set(squad0) | set(cfg.locked)

    # ---- candidate pool ---------------------------------------------------
    pool = _pre_filter(candidates, keep, cfg, disc)
    pool_ids = {c.element_id for c in pool}
    placeholder_team = -1
    for eid, slot in squad0.items():
        if eid not in pool_ids:
            pool.append(HorizonCandidate(
                element_id=eid, position=slot.position, price=slot.selling_price,
                team_id=placeholder_team, xpts=(0.0,) * H, p_play=(0.0,) * H,
            ))
            placeholder_team -= 1
    n = len(pool)
    idx = {c.element_id: i for i, c in enumerate(pool)}
    in0 = [1 if c.element_id in squad0 else 0 for c in pool]
    # selling value: current players at their selling price, others at price
    sv = [squad0[c.element_id].selling_price if c.element_id in squad0 else c.price
          for c in pool]
    price = [c.price for c in pool]
    xp = [[c.xpts[k] for k in range(H)] for c in pool]

    pos_idx: dict[Position, list[int]] = defaultdict(list)
    team_idx: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(pool):
        pos_idx[c.position].append(i)
        team_idx[c.team_id].append(i)
    gk_idx = pos_idx[Position.GK]
    of_idx = [i for i, c in enumerate(pool) if c.position != Position.GK]

    # ---- bench weights (per GW) ------------------------------------------
    bench_w: list[tuple[float, tuple[float, ...]]] = []
    current = [pool[idx[e]] for e in squad0]
    for k in range(H):
        if cfg.bench_mode == "fixed":
            from fpl_optimizer.optimizer.lineup_selector import (
                BENCH_GK_WEIGHT,
                BENCH_OUTFIELD_WEIGHTS,
            )
            bench_w.append((BENCH_GK_WEIGHT, tuple(BENCH_OUTFIELD_WEIGHTS)))
            continue
        # A WC/FH (or a squad that will change) — estimate from the best
        # affordable-ish XI: the current squad is a good proxy for normal
        # weeks; for WC/FH use the pool's top players.
        src = pool if chip_k[k] in (CHIP_WILDCARD, CHIP_FREE_HIT) else current
        xi_of, xi_gk = _likely_xi(src, k)
        q_of = [1.0 - _p_play(c, k, cfg.default_p_play) for c in xi_of]
        q_gk = 1.0 - _p_play(xi_gk, k, cfg.default_p_play) if xi_gk else 0.1
        bench_w.append(dnp_bench_weights(q_of, q_gk))

    # ---- variables ----------------------------------------------------------
    prob = pulp.LpProblem("horizon_planner", pulp.LpMaximize)
    x = {(i, k): pulp.LpVariable(f"x_{i}_{k}", cat="Binary") for i in range(n) for k in range(H)}
    y = {(i, k): pulp.LpVariable(f"y_{i}_{k}", cat="Binary") for i in range(n) for k in range(H)}
    cap = {(i, k): pulp.LpVariable(f"c_{i}_{k}", 0, 1) for i in range(n) for k in range(H)}
    vc = {(i, k): pulp.LpVariable(f"v_{i}_{k}", 0, 1) for i in range(n) for k in range(H)}
    fh_k = [k for k in range(H) if chip_k[k] == CHIP_FREE_HIT]
    f = {(i, k): pulp.LpVariable(f"f_{i}_{k}", cat="Binary") for i in range(n) for k in fh_k}
    tr_k = [k for k in range(H) if chip_k[k] != CHIP_FREE_HIT]
    tin = {(i, k): pulp.LpVariable(f"tin_{i}_{k}", 0, 1) for i in range(n) for k in tr_k}
    tout = {(i, k): pulp.LpVariable(f"tout_{i}_{k}", 0, 1) for i in range(n) for k in tr_k}
    bank = [pulp.LpVariable(f"bank_{k}", lowBound=0) for k in range(H)]
    ft = [None] + [pulp.LpVariable(f"ft_{k}", 0, MAX_FREE_TRANSFERS, cat="Integer")
                   for k in range(1, H + 1)]
    hit_k = [k for k in range(H) if chip_k[k] is None
             or chip_k[k] in (CHIP_BENCH_BOOST, CHIP_TRIPLE_CAPTAIN)]
    pt = {k: pulp.LpVariable(f"pt_{k}", 0, TRANSFERS_CAP_PER_GW, cat="Integer") for k in hit_k}
    hflag = {k: pulp.LpVariable(f"h_{k}", cat="Binary") for k in hit_k}
    bb_k = {k for k in range(H) if chip_k[k] == CHIP_BENCH_BOOST}
    bslot = {(i, k, s): pulp.LpVariable(f"b_{i}_{k}_{s}", 0, 1)
             for i in of_idx for k in range(H) if k not in bb_k
             for s in range(_N_BENCH_SLOTS)}

    def sq(i: int, k: int):
        """The squad that plays GW k (the Free Hit squad in an FH week)."""
        return f[i, k] if (i, k) in f else x[i, k]

    def prev(i: int, k: int):
        return in0[i] if k == 0 else x[i, k - 1]

    # ---- objective ----------------------------------------------------------
    obj = []
    for k in range(H):
        d = disc[k]
        mult = 2.0 if chip_k[k] == CHIP_TRIPLE_CAPTAIN else 1.0
        terms = [xp[i][k] * y[i, k] + mult * xp[i][k] * cap[i, k]
                 + cfg.vice_weight * xp[i][k] * vc[i, k] for i in range(n)]
        if k in bb_k:
            terms += [xp[i][k] * (sq(i, k) - y[i, k]) for i in range(n)]
        else:
            w_gk, w_of = bench_w[k]
            terms += [w_gk * xp[i][k] * (sq(i, k) - y[i, k]) for i in gk_idx]
            terms += [w_of[s] * xp[i][k] * bslot[i, k, s]
                      for i in of_idx for s in range(_N_BENCH_SLOTS)]
        obj.append(d * pulp.lpSum(terms))
        if k in pt:
            obj.append(-d * (TRANSFER_HIT_COST + cfg.hit_margin) * pt[k])
    if cfg.ft_value:
        obj.append(cfg.ft_value * ft[H])
    prob += pulp.lpSum(obj)

    # ---- constraints ----------------------------------------------------------
    for k in range(H):
        chip = chip_k[k]
        ft_k = state.free_transfers if k == 0 else ft[k]
        bank_prev = state.bank if k == 0 else bank[k - 1]

        if chip == CHIP_FREE_HIT:
            for i in range(n):
                prob += x[i, k] == prev(i, k)
            prob += bank[k] == bank_prev
            # FH budget: sell the whole squad at selling value, buy at price;
            # a kept current player nets out (g = f AND held).
            g = {}
            for i in range(n):
                if in0[i] and price[i] > sv[i]:
                    g[i] = pulp.LpVariable(f"g_{i}_{k}", 0, 1)
                    prob += g[i] <= f[i, k]
                    prob += g[i] <= prev(i, k)
            prob += (
                pulp.lpSum(price[i] * f[i, k] for i in range(n))
                - pulp.lpSum((price[i] - sv[i]) * g[i] for i in g)
                <= bank_prev + pulp.lpSum(sv[i] * prev(i, k) for i in range(n))
            )
            for i in range(n):
                if pool[i].element_id in cfg.banned:
                    prob += f[i, k] <= prev(i, k)
            prob += ft[k + 1] <= ft_k
        else:
            for i in range(n):
                prob += x[i, k] - prev(i, k) == tin[i, k] - tout[i, k]
                prob += tin[i, k] + tout[i, k] <= 1
                if pool[i].element_id in cfg.banned:
                    prob += tin[i, k] == 0
            n_tr = pulp.lpSum(tin[i, k] for i in range(n))
            prob += bank[k] == bank_prev + pulp.lpSum(
                sv[i] * tout[i, k] - price[i] * tin[i, k] for i in range(n))
            if chip == CHIP_WILDCARD:
                prob += ft[k + 1] <= ft_k
            else:
                cap_tr = TRANSFERS_CAP_PER_GW
                if cfg.max_transfers_per_gw is not None:
                    cap_tr = min(cap_tr, cfg.max_transfers_per_gw)
                prob += n_tr <= cap_tr
                prob += pt[k] >= n_tr - ft_k
                prob += pt[k] <= n_tr
                if cfg.max_hits_per_gw is not None:
                    prob += pt[k] <= cfg.max_hits_per_gw
                # no hit while free transfers are left unused
                prob += pt[k] <= TRANSFERS_CAP_PER_GW * hflag[k]
                prob += ft_k - n_tr + pt[k] <= MAX_FREE_TRANSFERS * (1 - hflag[k])
                prob += ft[k + 1] <= ft_k - n_tr + pt[k] + 1
            prob += ft[k + 1] <= MAX_FREE_TRANSFERS

        for e in cfg.locked:
            if e in idx:
                prob += x[idx[e], k] == 1

        # squad composition / club limit on the squad that plays GW k
        for pos, limit in POSITION_LIMITS.items():
            prob += pulp.lpSum(sq(i, k) for i in pos_idx[pos]) == limit
        for tid, members in team_idx.items():
            if len(members) > MAX_PER_CLUB:
                prob += pulp.lpSum(sq(i, k) for i in members) <= MAX_PER_CLUB

        # lineup / formation
        for i in range(n):
            prob += y[i, k] <= sq(i, k)
            prob += cap[i, k] + vc[i, k] <= y[i, k]
        prob += pulp.lpSum(y[i, k] for i in range(n)) == 11
        prob += pulp.lpSum(y[i, k] for i in pos_idx[Position.GK]) == 1
        prob += pulp.lpSum(y[i, k] for i in pos_idx[Position.DEF]) >= 3
        prob += pulp.lpSum(y[i, k] for i in pos_idx[Position.DEF]) <= 5
        prob += pulp.lpSum(y[i, k] for i in pos_idx[Position.MID]) >= 2
        prob += pulp.lpSum(y[i, k] for i in pos_idx[Position.MID]) <= 5
        prob += pulp.lpSum(y[i, k] for i in pos_idx[Position.FWD]) >= 1
        prob += pulp.lpSum(y[i, k] for i in pos_idx[Position.FWD]) <= 3
        prob += pulp.lpSum(cap[i, k] for i in range(n)) == 1
        prob += pulp.lpSum(vc[i, k] for i in range(n)) == 1

        # bench slots
        if k not in bb_k:
            for i in of_idx:
                prob += pulp.lpSum(bslot[i, k, s] for s in range(_N_BENCH_SLOTS)) \
                    == sq(i, k) - y[i, k]
            for s in range(_N_BENCH_SLOTS):
                prob += pulp.lpSum(bslot[i, k, s] for i in of_idx) == 1

    # ---- solve ----------------------------------------------------------------
    t0 = time.time()
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=cfg.time_limit, gapRel=cfg.gap_rel)
    prob.solve(solver)
    secs = time.time() - t0
    sol_status = getattr(prob, "sol_status", None)
    ok = prob.status == pulp.constants.LpStatusOptimal or sol_status in (1, 2)
    if not ok:
        raise RuntimeError(
            f"Horizon planner found no solution (status={pulp.LpStatus[prob.status]})")
    status = "optimal" if sol_status in (None, 1) else "time_limit_feasible"

    def val(v) -> float:
        return float(pulp.value(v) or 0.0)

    # ---- extract ------------------------------------------------------------
    plan: list[GWPlan] = []
    held_prev = {pool[i].element_id for i in range(n) if in0[i]}
    ft_now = state.free_transfers
    bank_now = state.bank
    for k in range(H):
        chip = chip_k[k]
        playing = [i for i in range(n) if val(sq(i, k)) > 0.5]
        held = {pool[i].element_id for i in range(n) if val(x[i, k]) > 0.5}
        playing_ids = {pool[i].element_id for i in playing}
        if chip == CHIP_FREE_HIT:
            t_in = sorted(playing_ids - held_prev)
            t_out = sorted(held_prev - playing_ids)
        else:
            t_in = sorted(held - held_prev)
            t_out = sorted(held_prev - held)
        lineup_i = [i for i in playing if val(y[i, k]) > 0.5]
        captain_i = max(lineup_i, key=lambda i: (val(cap[i, k]), xp[i][k]))
        vice_i = max((i for i in lineup_i if i != captain_i),
                     key=lambda i: (val(vc[i, k]), xp[i][k]))
        bench_i = [i for i in playing if i not in lineup_i]
        bench_gk = [i for i in bench_i if pool[i].position == Position.GK]
        bench_of = [i for i in bench_i if pool[i].position != Position.GK]
        if k in bb_k:
            bench_of.sort(key=lambda i: xp[i][k], reverse=True)
        else:  # the MILP's slot assignment (ties -> higher xPts first)
            bench_of.sort(key=lambda i: (
                max(range(_N_BENCH_SLOTS), key=lambda s: val(bslot[i, k, s])), -xp[i][k]))
        n_tr = len(t_in)
        if chip in (CHIP_WILDCARD, CHIP_FREE_HIT):
            hits = 0
            next_ft = ft_now
        else:
            hits = max(0, n_tr - ft_now)
            next_ft = min(MAX_FREE_TRANSFERS, max(ft_now - n_tr, 0) + 1)
        mult = 3 if chip == CHIP_TRIPLE_CAPTAIN else 2
        exp_pts = sum(xp[i][k] for i in lineup_i) + (mult - 1) * xp[captain_i][k]
        if k in bb_k:
            exp_pts += sum(xp[i][k] for i in bench_i)
        if chip != CHIP_FREE_HIT:
            bank_now = bank_now + sum(sv[idx[e]] for e in t_out) - sum(price[idx[e]] for e in t_in)
        plan.append(GWPlan(
            gw=gws[k], chip=chip, free_transfers=ft_now,
            transfers_in=t_in, transfers_out=t_out, hits=hits,
            squad=sorted(playing_ids),
            lineup=[pool[i].element_id for i in lineup_i],
            bench=[pool[i].element_id for i in bench_gk + bench_of],
            captain_id=pool[captain_i].element_id,
            vice_captain_id=pool[vice_i].element_id,
            expected_points=exp_pts,
            bank_after=bank_now,
        ))
        ft_now = next_ft
        held_prev = held

    p0 = plan[0]
    first = OptimizerResult(
        squad_element_ids=p0.lineup + p0.bench,
        lineup_element_ids=p0.lineup,
        bench_element_ids=p0.bench,
        captain_id=p0.captain_id,
        vice_captain_id=p0.vice_captain_id,
        transfers_in=p0.transfers_in,
        transfers_out=p0.transfers_out,
        chip=p0.chip,
        objective_value=val(prob.objective),
        total_cost=sum(price[idx[e]] for e in p0.transfers_in),
        hit_cost=TRANSFER_HIT_COST * p0.hits,
    )
    return HorizonResult(
        first=first, plan=plan, objective_value=val(prob.objective),
        status=status, solve_seconds=secs, n_candidates=n, bench_weights=bench_w,
    )
