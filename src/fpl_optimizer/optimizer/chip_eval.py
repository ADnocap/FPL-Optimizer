"""Chip decision support: the expected gain of every remaining chip in every GW.

Numbers only: the chip plan is a human decision (SEASON_GUIDE.md) and nothing
here changes or submits it.  ``scripts/chip_eval.py`` is the CLI.

Method (every piece reuses the repo's verified components)
----------------------------------------------------------
Inputs: the live :class:`GameState` at the upcoming GW ``t``
(:func:`fpl_optimizer.live.entry.fetch_entry_state`), horizon predictions for
GWs ``t..T`` as of the GW-t deadline (:mod:`fpl_optimizer.prediction.horizon`,
DGW "blend"), and the planner pool
(:func:`fpl_optimizer.live.pool.build_live_horizon_candidates`, calibrated
availability multipliers on xPts and P(play)).

1. **Base trajectory** (no chip): the weekly receding-horizon run of
   ``gameweek.py --horizon 3`` replayed on the as-of-t predictions — at each
   GW g solve :func:`optimize_horizon` over ``g..g+2`` from the running state,
   execute GW g, advance the FT bank / bank / squad (bought players sell at
   their price: prices are assumed flat).  It gives the planned squad of every
   GW and the state entering it.
2. **TC / BB at g** — Monte Carlo on the planned GW-g squad, XI, bench order,
   captain and vice (:func:`simulate`): who plays ~ independent
   Bernoulli(P(play)); FPL auto-subs walk the bench in priority order within
   formation limits (bench GK only for the GK); captain -> vice failover;
   E[pts | plays] = xP / P(play), so E[pts] stays the model's xP.
   TC = E[points of the effective captain] (the armband's third share);
   BB = E[bench points counted under BB] - E[auto-sub points] (net).
3. **FH / WC at g** — :func:`optimize_horizon` from the state entering g over
   ``g..g+H-1`` with ``{g: chip}`` vs without (same config, horizon and pool;
   FH H = the base step, WC H = 6 by default).  Both plans are re-scored by the
   same Monte Carlo (undiscounted, -4 per hit): "gain" columns.  The MILP
   objective difference (discounted, bench EV, hit margin) is reported too.
   FH is also given vs *holding* the GW-g squad (best XI, no transfers) — the
   definition the 2025-26 calibration was measured on.
   Note: a WC/FH week carries the FT count unchanged (no +1), so a chip can be
   worth slightly < 0 against a plan that would have rolled that week.

Also: remaining chips and their playable GWs (official windows + the entry's
ChipState: WC/FH from GW2, halves GW1-19 / GW20-38, FH in GW19 blocks FH in
GW20, chips already used, first-half expiry), the BGW/DGW/unscheduled map and
the plan's early-WC trigger (>= 3 starters flagged/out).

Calibration ("cal" columns)
---------------------------
Raw gains are model expectations; the optimiser picks whatever the model
over-rates (selection bias) and horizon accuracy decays with the lead
``k = g - t``, so they are optimistic.  ``cal`` = raw x realised/predicted
ratio from a 2025-26 replay (fast leak-fixed model trained <= 2024-25, a
1-FT/week squad trajectory, gains re-scored with actual minutes/points;
prototype outputs ``val_tcbb.csv`` / ``val_fh.csv`` / ``val_wc.csv``).  The
ratios differ by season half — pooled over the season they hid e.g. TC at
k=0: 0.93 in GW1-19 vs 0.56 in GW20-38 — so :data:`CALIB` is split by the
half of the chip GW g.  Ratio of means, 95% bootstrap CI, n = player-GW
evaluations:

======  ====  =========  ====================================  ====================================
chip    k     half 1 (GW<=19)                                 half 2 (GW>=20)
======  ====  =========  ====================================  ====================================
TC      0     0.93 [0.67, 1.23] n=19                          0.56 [0.39, 0.76] n=19
TC      1-2   0.66 [0.51, 0.82] n=35                          0.68 [0.52, 0.85] n=38
TC      3+    0.54 [0.43, 0.65] n=58  (k 3-6 measured)        0.51 [0.42, 0.61] n=76
BB      0     0.72 [0.26, 1.27] n=19                          1.60 [0.75, 2.62] n=19
BB      1+    0.30 [0.13, 0.49] n=93                          1.35 [1.04, 1.70] n=114
FH      0     0.58 [0.19, 0.92] n=19                          0.97 [0.65, 1.26] n=19
FH      3     0.56 [0.32, 0.80] n=16                          0.34 [-0.10, 0.70] n=19
FH      6+    0.46 [0.06, 0.80] n=13                          0.29 [-0.11, 0.67] n=19
WC      any   0.22 [-0.50, 0.70] n=5                          0.06 [-0.66, 0.80] n=3
======  ====  =========  ====================================  ====================================

FH is measured at k = 0, 3, 6 only (linear in between, flat beyond 6); TC/BB
are step functions of the k bucket (flat beyond the measured k <= 6).  FH's
ratio is for "FH vs hold"; applied to it only.  One season, small n (WC:
realised gains -68..+53, CIs include 0): indicative only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fpl_optimizer.engine.state import ChipState, GameState, PlayerSlot, Squad
from fpl_optimizer.optimizer.horizon_optimizer import (
    CHIP_BENCH_BOOST,
    CHIP_FREE_HIT,
    CHIP_TRIPLE_CAPTAIN,
    CHIP_WILDCARD,
    GWPlan,
    HorizonCandidate,
    HorizonConfig,
    HorizonResult,
    optimize_horizon,
)
from fpl_optimizer.utils.constants import (
    FIRST_HALF_END,
    MAX_FREE_TRANSFERS,
    TRANSFER_HIT_COST,
    Position,
)

logger = logging.getLogger(__name__)

CHIP_API_TO_ENGINE = {
    "wildcard": CHIP_WILDCARD, "freehit": CHIP_FREE_HIT,
    "bboost": CHIP_BENCH_BOOST, "3xc": CHIP_TRIPLE_CAPTAIN,
}
CHIP_SHORT = {CHIP_TRIPLE_CAPTAIN: "TC", CHIP_BENCH_BOOST: "BB",
              CHIP_FREE_HIT: "FH", CHIP_WILDCARD: "WC"}
_CHIP_ORDER = [CHIP_TRIPLE_CAPTAIN, CHIP_BENCH_BOOST, CHIP_FREE_HIT, CHIP_WILDCARD]
SHORT_TO_CHIP = {"tc": CHIP_TRIPLE_CAPTAIN, "bb": CHIP_BENCH_BOOST,
                 "fh": CHIP_FREE_HIT, "wc": CHIP_WILDCARD}

N_SIMS = 4000
BASE_STEP_H = 3          # gameweek.py --horizon 3: the backtested planner horizon
WC_HORIZON = 6
EARLY_WC_STARTERS = 3    # SEASON_GUIDE: pull WC1 forward if >= 3 starters flagged/out

# realised/predicted ratios, 2025-26 replay, by half of the chip GW
# (1 = GW1-19, 2 = GW20-38) -> ((k_anchor, ratio), ...); see module docstring.
CALIB: dict[str, dict[int, tuple[tuple[int, float], ...]]] = {
    "tc": {1: ((0, 0.93), (1, 0.66), (3, 0.54)),
           2: ((0, 0.56), (1, 0.68), (3, 0.51))},
    "bb": {1: ((0, 0.72), (1, 0.30)),
           2: ((0, 1.60), (1, 1.35))},
    "fh": {1: ((0, 0.58), (3, 0.56), (6, 0.46)),
           2: ((0, 0.97), (3, 0.34), (6, 0.29))},
    "wc": {1: ((0, 0.22),),
           2: ((0, 0.06),)},
}
_CALIB_INTERPOLATE = {"fh"}


def season_half(gw: int) -> int:
    return 1 if gw <= FIRST_HALF_END else 2


def calib(chip: str, k: int, gw: int) -> float:
    """Realised/predicted ratio for ``chip`` played in GW ``gw`` at lead ``k``.

    The half is the chip GW's (``gw``), not the as-of GW's.
    """
    anchors = CALIB[chip][season_half(gw)]
    k = max(int(k), 0)
    if chip in _CALIB_INTERPOLATE:
        ks = [a for a, _ in anchors]
        rs = [r for _, r in anchors]
        return float(np.interp(k, ks, rs))
    ratio = anchors[0][1]
    for k_a, r in anchors:
        if k >= k_a:
            ratio = r
    return float(ratio)


# ---------------------------------------------------------------------------
# chip windows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChipWindow:
    """One chip instance (a half-season copy) and its official GW window."""

    chip: str   # engine name
    half: int   # 1 = GW1-19, 2 = GW20-38
    start: int
    stop: int


@dataclass(frozen=True)
class RemainingChip:
    window: ChipWindow
    gws: tuple[int, ...]  # playable GWs from the upcoming GW on

    @property
    def chip(self) -> str:
        return self.window.chip

    @property
    def label(self) -> str:
        return f"{CHIP_SHORT[self.chip]}{self.window.half}"


def official_chip_windows(chips_meta: list[dict]) -> list[ChipWindow]:
    """bootstrap-static ``chips`` (or the official rules JSON) -> windows."""
    out = []
    for c in chips_meta:
        name = CHIP_API_TO_ENGINE.get(c.get("name"))
        if name is None:
            continue
        start, stop = int(c["start_event"]), int(c["stop_event"])
        out.append(ChipWindow(name, season_half(stop), start, stop))
    return sorted(out, key=lambda w: (w.half, _CHIP_ORDER.index(w.chip)))


def remaining_chips(chip_state: ChipState, windows: list[ChipWindow],
                    upcoming_gw: int) -> list[RemainingChip]:
    """Chips still playable from ``upcoming_gw`` on, with their GWs.

    A GW is playable when it is inside the official window and the entry's
    :class:`ChipState` allows it: that half's chip unused, the chip's start
    event (WC/FH from GW2), and the Free Hit GW19/GW20 exclusion.  First-half
    chips are gone from GW20 (their window ends at GW19).
    """
    out = []
    for w in windows:
        gws = tuple(g for g in range(max(w.start, upcoming_gw), w.stop + 1)
                    if chip_state.is_available(w.chip, g))
        if gws:
            out.append(RemainingChip(w, gws))
    return out


# ---------------------------------------------------------------------------
# fixtures / availability
# ---------------------------------------------------------------------------


def fixture_calendar(fixtures, team_ids: list[int], gws: list[int]) -> dict:
    """BGW / DGW map for ``gws`` and the unscheduled fixtures (event = null).

    ``fixtures``: the API fixture list (or a DataFrame with ``event``,
    ``team_h``, ``team_a``).  Returns ``per_gw[g] = {n_fixtures, blank_teams,
    double_teams}``, ``unscheduled`` [(home, away)] and ``unscheduled_by_team``.
    """
    fx = fixtures.copy() if isinstance(fixtures, pd.DataFrame) else pd.DataFrame(list(fixtures))
    for c in ("event", "team_h", "team_a"):
        if c not in fx.columns:
            fx[c] = np.nan
    fx["event"] = pd.to_numeric(fx["event"], errors="coerce")
    fx = fx.dropna(subset=["team_h", "team_a"])
    sched = fx[fx["event"].notna()]
    long = pd.concat([
        pd.DataFrame({"event": sched["event"].astype(int), "team": sched["team_h"].astype(int)}),
        pd.DataFrame({"event": sched["event"].astype(int), "team": sched["team_a"].astype(int)}),
    ])
    counts = (long.groupby(["event", "team"]).size().unstack(fill_value=0)
              .reindex(index=gws, columns=team_ids, fill_value=0))
    uns = fx[fx["event"].isna()]
    by_team: dict[int, int] = {}
    for h, a in zip(uns["team_h"].astype(int), uns["team_a"].astype(int)):
        by_team[h] = by_team.get(h, 0) + 1
        by_team[a] = by_team.get(a, 0) + 1
    per_gw = {}
    for g in gws:
        row = counts.loc[g]
        per_gw[int(g)] = {
            "n_fixtures": int(row.sum() // 2),
            "blank_teams": [int(t) for t in row.index[row == 0]],
            "double_teams": [int(t) for t in row.index[row >= 2]],
        }
    return {"per_gw": per_gw,
            "unscheduled": [(int(h), int(a)) for h, a in
                            zip(uns["team_h"].astype(int), uns["team_a"].astype(int))],
            "unscheduled_by_team": by_team}


_STATUS_LABEL = {"d": "doubtful", "i": "injured", "s": "suspended",
                 "u": "unavailable", "n": "not available"}


def availability_flags(bootstrap: dict, element_ids) -> dict[int, str]:
    """element -> reason, for players flagged in bootstrap (status/chance)."""
    el = {e["id"]: e for e in bootstrap["elements"]}
    out = {}
    for eid in element_ids:
        e = el.get(eid)
        if e is None:
            out[eid] = "not in bootstrap"
            continue
        st = e.get("status", "a")
        ch = e.get("chance_of_playing_next_round")
        if st == "a" and (ch is None or ch >= 100):
            continue
        label = _STATUS_LABEL.get(st, st)
        if ch is not None and ch < 100:
            label = f"{label} {ch}%"
        news = (e.get("news") or "").strip()
        out[eid] = label + (f" - {news}" if news else "")
    return out


def early_wc_trigger(flags: dict[int, str], starters,
                     threshold: int = EARLY_WC_STARTERS) -> tuple[int, bool]:
    """(number of flagged starters, trigger met?)."""
    n = sum(1 for e in set(starters) if e in flags)
    return n, n >= threshold


# ---------------------------------------------------------------------------
# Monte Carlo of one GW for a fixed 15 (auto-subs + captain failover)
# ---------------------------------------------------------------------------


@dataclass
class Squad15:
    """A 15-man squad for one GW; ``xi``/``bench``/``cap``/``vice`` index
    into ``elements``; ``bench`` in sub priority (the GK's place is moot)."""

    elements: list[int]
    pos: np.ndarray   # 1..4
    xp: np.ndarray    # E[points] (0 for a blank)
    p: np.ndarray     # P(plays)
    xi: list[int] = field(default_factory=list)
    bench: list[int] = field(default_factory=list)
    cap: int = -1
    vice: int = -1


def cond_means(xp: np.ndarray, p: np.ndarray) -> np.ndarray:
    """E[points | plays] = xP / P(play), guarded for tiny P."""
    m = np.where(p > 1e-3, xp / np.maximum(p, 1e-3), 0.0)
    return np.minimum(m, np.maximum(xp, 15.0))


_LIMITS = {2: (3, 5), 3: (2, 5), 4: (1, 3)}  # DEF / MID / FWD in the XI


def simulate(sq: Squad15, n_sims: int = N_SIMS, seed: int = 0,
             played: np.ndarray | None = None, points: np.ndarray | None = None) -> dict:
    """Expected (Monte Carlo) or, with ``played``/``points``, realised points.

    Returns ``normal`` (no chip: XI + auto-subs + effective captain once
    more), ``tc_gain`` (effective captain's points: the TC extra),
    ``bb_gain`` (bench points counted under BB minus those auto-subs already
    brought in), ``bench_sum`` and ``autosub``.  Auto-subs walk the bench in
    priority order (as the engine does); a bench outfielder replaces the first
    non-playing XI outfielder whose swap keeps DEF 3-5 / MID 2-5 / FWD 1-3;
    the bench GK only replaces the GK.  ``played`` (n, 15) bool and
    ``points`` (15,) give a realised evaluation.
    """
    if played is None:
        rng = np.random.default_rng(seed)
        played = rng.random((n_sims, 15)) < sq.p[None, :]
        m = cond_means(sq.xp, sq.p)
    else:
        played = np.atleast_2d(played).astype(bool)
        m = np.asarray(points, dtype=float)
    n = played.shape[0]
    xi = np.zeros(15, bool)
    xi[sq.xi] = True
    counted = played & xi[None, :]
    need = (~played) & xi[None, :]
    comp = {q: np.full(n, int(np.sum(sq.pos[sq.xi] == q))) for q in (2, 3, 4)}
    gk_start = [i for i in sq.xi if sq.pos[i] == 1][0]
    out_order = [i for i in sq.xi if sq.pos[i] != 1]
    for b in sq.bench:
        pb = played[:, b]
        if sq.pos[b] == 1:
            do = pb & need[:, gk_start]
            counted[do, b] = True
            need[do, gk_start] = False
            continue
        done = np.zeros(n, bool)
        qi = int(sq.pos[b])
        for s in out_order:
            qo = int(sq.pos[s])
            if qo == qi:
                valid = np.ones(n, bool)
            else:
                lo_o, _ = _LIMITS[qo]
                _, hi_i = _LIMITS[qi]
                valid = (comp[qo] - 1 >= lo_o) & (comp[qi] + 1 <= hi_i)
            do = pb & ~done & need[:, s] & valid
            if not do.any():
                continue
            counted[do, b] = True
            need[do, s] = False
            comp[qo] = comp[qo] - do
            comp[qi] = comp[qi] + do
            done |= do
    eff_cap = np.where(played[:, sq.cap], m[sq.cap],
                       np.where(played[:, sq.vice], m[sq.vice], 0.0))
    base_xi = (counted * m[None, :]).sum(1)
    normal = base_xi + eff_cap
    bb = (played * m[None, :]).sum(1) + eff_cap
    bench_idx = np.array(sq.bench)
    autosub = (counted[:, bench_idx] * m[None, bench_idx]).sum(1)
    return {
        "normal": float(normal.mean()),
        "tc_gain": float(eff_cap.mean()),
        "bb_gain": float((bb - normal).mean()),
        "bench_sum": float((played[:, bench_idx] * m[None, bench_idx]).sum(1).mean()),
        "autosub": float(autosub.mean()),
    }


def make_squad15(elements, lineup, bench, captain: int, vice: int,
                 pos_of: dict, xp_of: dict, p_of: dict) -> Squad15:
    """Squad15 from element ids (missing xP/P -> 0: no fixture / not in pool)."""
    elements = [int(e) for e in elements]
    idx = {e: i for i, e in enumerate(elements)}
    sq = Squad15(
        elements=elements,
        pos=np.array([int(pos_of[e]) for e in elements]),
        xp=np.array([float(xp_of.get(e, 0.0)) for e in elements]),
        p=np.array([float(p_of.get(e, 0.0)) for e in elements]),
    )
    sq.xi = [idx[e] for e in lineup]
    bench_i = [idx[e] for e in bench]
    # GK first (only it can replace the GK), outfield keep their priority
    sq.bench = [i for i in bench_i if sq.pos[i] == 1] + [i for i in bench_i if sq.pos[i] != 1]
    sq.cap, sq.vice = idx[captain], idx[vice]
    return sq


def best_lineup(elements, pos_of: dict, xp_of: dict) -> tuple[list[int], list[int], int, int]:
    """XI / bench order / captain / vice for a fixed 15 (repo lineup MILP)."""
    from fpl_optimizer.optimizer.lineup_selector import select_lineup
    from fpl_optimizer.optimizer.types import PlayerCandidate

    r = select_lineup([PlayerCandidate(int(e), Position(int(pos_of[e])), 0, 0,
                                       float(xp_of.get(e, 0.0))) for e in elements])
    return (list(r.lineup_element_ids), list(r.bench_element_ids),
            r.captain_id, r.vice_captain_id)


# ---------------------------------------------------------------------------
# planning context
# ---------------------------------------------------------------------------


@dataclass
class ChipContext:
    """Everything a chip evaluation needs.

    ``candidates``: the planner pool with ``xpts``/``p_play`` over ``gws``
    (consecutive, starting at the upcoming GW ``state.current_gw``).
    """

    state: GameState
    candidates: list[HorizonCandidate]
    gws: list[int]
    config: HorizonConfig = field(default_factory=HorizonConfig)
    n_sims: int = N_SIMS

    def __post_init__(self) -> None:
        if list(self.gws) != list(range(self.gws[0], self.gws[0] + len(self.gws))):
            raise ValueError("gws must be consecutive")
        if self.gws[0] != self.state.current_gw:
            raise ValueError(f"gws start at GW{self.gws[0]} but the state is GW{self.state.current_gw}")
        self._k = {g: k for k, g in enumerate(self.gws)}
        self.pos_of = {c.element_id: int(c.position) for c in self.candidates}
        for p in self.state.squad.players:
            self.pos_of.setdefault(p.element_id, int(p.position))
        self.price_of = {c.element_id: c.price for c in self.candidates}
        self._maps: dict[int, tuple[dict, dict]] = {}
        self._windows: dict[tuple, list[HorizonCandidate]] = {}

    @property
    def t(self) -> int:
        return self.gws[0]

    def maps(self, g: int) -> tuple[dict, dict]:
        """(xP by element, P(play) by element) for GW g."""
        if g not in self._maps:
            k = self._k[g]
            xp = {c.element_id: float(c.xpts[k]) for c in self.candidates}
            pp = {c.element_id: (float(c.p_play[k]) if c.p_play is not None
                                 else (0.9 if c.xpts[k] > 0 else 0.0))
                  for c in self.candidates}
            self._maps[g] = (xp, pp)
        return self._maps[g]

    def window(self, g: int, h: int) -> list[int]:
        return [x for x in range(g, g + h) if x in self._k]

    def window_candidates(self, window: list[int]) -> list[HorizonCandidate]:
        key = tuple(window)
        if key not in self._windows:
            ks = [self._k[g] for g in window]
            self._windows[key] = [
                HorizonCandidate(
                    element_id=c.element_id, position=c.position, price=c.price,
                    team_id=c.team_id, xpts=tuple(c.xpts[k] for k in ks),
                    p_play=None if c.p_play is None else tuple(c.p_play[k] for k in ks),
                )
                for c in self.candidates
            ]
        return self._windows[key]

    def squad_from_plan(self, plan: GWPlan) -> Squad15:
        xp, pp = self.maps(plan.gw)
        return make_squad15(plan.lineup + plan.bench, plan.lineup, plan.bench,
                            plan.captain_id, plan.vice_captain_id, self.pos_of, xp, pp)

    def hold_squad(self, state: GameState, g: int) -> Squad15:
        """``state``'s 15 in GW g with the lineup MILP's XI/captain (no moves)."""
        xp, pp = self.maps(g)
        ids = [p.element_id for p in state.squad.players]
        lineup, bench, cap, vice = best_lineup(ids, self.pos_of, xp)
        return make_squad15(ids, lineup, bench, cap, vice, self.pos_of, xp, pp)


def advance_state(state: GameState, plan: GWPlan, price_of: dict[int, int],
                  pos_of: dict[int, int]) -> GameState:
    """The state entering GW ``plan.gw + 1`` after executing ``plan``.

    FT: min(5, max(FT - moves, 0) + 1); WC/FH weeks carry the count unchanged.
    A Free Hit squad reverts.  Kept players keep their purchase/selling
    prices; bought players' selling price = their price (flat prices).
    """
    new = state.copy()
    new.current_gw = plan.gw + 1
    new.active_chip = None
    if plan.chip:
        new.chips.use_chip(plan.chip, plan.gw)
    if plan.chip == CHIP_FREE_HIT:
        return new
    held = {p.element_id: p for p in state.squad.players}
    order = list(plan.lineup) + list(plan.bench)
    players = [held[e].copy() if e in held
               else PlayerSlot(e, Position(pos_of[e]), price_of[e], price_of[e])
               for e in order]
    idx = {e: i for i, e in enumerate(order)}
    new.squad = Squad(players=players, lineup=list(range(11)), bench=list(range(11, 15)),
                      captain_idx=idx[plan.captain_id], vice_captain_idx=idx[plan.vice_captain_id])
    new.bank = int(round(plan.bank_after))
    n_moves = len(plan.transfers_in)
    if plan.chip != CHIP_WILDCARD:
        new.free_transfers = min(MAX_FREE_TRANSFERS, max(state.free_transfers - n_moves, 0) + 1)
    return new


@dataclass
class TrajectoryStep:
    gw: int
    state: GameState        # entering GW g, before its transfers
    window: list[int]
    result: HorizonResult   # the solve at g over ``window``

    @property
    def plan(self) -> GWPlan:
        return self.result.plan[0]


def base_trajectory(ctx: ChipContext, until_gw: int, step_h: int = BASE_STEP_H,
                    progress=None) -> dict[int, TrajectoryStep]:
    """No-chip receding-horizon trajectory GW t..until_gw (see module doc)."""
    state = ctx.state.copy()
    out: dict[int, TrajectoryStep] = {}
    for g in range(ctx.t, until_gw + 1):
        window = ctx.window(g, step_h)
        res = optimize_horizon(state, ctx.window_candidates(window), window, None, ctx.config)
        out[g] = TrajectoryStep(g, state, window, res)
        if progress:
            progress(f"base plan GW{g}: {len(res.plan[0].transfers_in)} move(s), "
                     f"{res.solve_seconds:.0f}s")
        state = advance_state(state, res.plan[0], ctx.price_of, ctx.pos_of)
    return out


def plan_mc_values(ctx: ChipContext, plans: list[GWPlan], seed: int = 0) -> list[float]:
    """Monte-Carlo E[points] - 4 x hits of each planned GW (chip-aware)."""
    vals = []
    for p in plans:
        r = simulate(ctx.squad_from_plan(p), ctx.n_sims, seed=seed + p.gw)
        v = r["normal"]
        if p.chip == CHIP_TRIPLE_CAPTAIN:
            v += r["tc_gain"]
        elif p.chip == CHIP_BENCH_BOOST:
            v += r["bb_gain"]
        vals.append(v - TRANSFER_HIT_COST * p.hits)
    return vals


@dataclass
class PlanComparison:
    chip_plan: dict[int, str]
    window: list[int]
    applied: bool             # the planner accepted every chip of chip_plan
    mc_gain: float            # sum over window of MC(with) - MC(without)
    milp_gain: float          # objective(with) - objective(without)
    per_gw_gain: list[float]
    with_chip: HorizonResult
    without: HorizonResult


def compare_plans(ctx: ChipContext, state: GameState, window: list[int],
                  chip_plan: dict[int, str], without: HorizonResult | None = None,
                  seed: int = 0) -> PlanComparison:
    """optimize_horizon with ``chip_plan`` vs without: same state/config/window/pool."""
    cands = ctx.window_candidates(window)
    with_chip = optimize_horizon(state, cands, window, chip_plan, ctx.config)
    if without is None:
        without = optimize_horizon(state, cands, window, None, ctx.config)
    got = {p.gw: p.chip for p in with_chip.plan if p.chip}
    applied = all(got.get(g) == c for g, c in chip_plan.items())
    vw = plan_mc_values(ctx, with_chip.plan, seed)
    vo = plan_mc_values(ctx, without.plan, seed)
    return PlanComparison(
        chip_plan=dict(chip_plan), window=list(window), applied=applied,
        mc_gain=float(sum(vw) - sum(vo)),
        milp_gain=float(with_chip.objective_value - without.objective_value),
        per_gw_gain=[a - b for a, b in zip(vw, vo)],
        with_chip=with_chip, without=without,
    )


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate_chips(
    ctx: ChipContext,
    remaining: list[RemainingChip],
    chips: set[str] | None = None,
    last_gw: int = FIRST_HALF_END,
    wc_gws: list[int] | None = None,
    wc_h: int = WC_HORIZON,
    fh_h: int | None = None,
    step_h: int = BASE_STEP_H,
    combos: list[tuple[int, int]] = (),
    progress=None,
) -> dict:
    """Gains of every remaining chip (``chips`` subset) in GWs t..last_gw.

    Returns a dict: ``trajectory`` (per GW: FT, moves, planned xPts),
    ``tc_bb`` / ``fh`` / ``wc`` keyed by GW, ``combos`` keyed "WCg+BBb".
    """
    chips = set(chips or _CHIP_ORDER)
    fh_h = fh_h or step_h
    want: dict[str, list[int]] = {}
    for rc in remaining:
        want.setdefault(rc.chip, []).extend(g for g in rc.gws if g <= last_gw)
    playable = {rc.chip: set() for rc in remaining}
    for rc in remaining:
        playable[rc.chip] |= set(rc.gws)
    want = {c: sorted(set(v)) for c, v in want.items() if c in chips and v}
    if CHIP_WILDCARD in want and wc_gws:
        want[CHIP_WILDCARD] = [g for g in wc_gws if g in set(want[CHIP_WILDCARD])]
    combos = [(w, b) for w, b in combos if w in playable.get(CHIP_WILDCARD, set())]
    need = sorted({g for v in want.values() for g in v} | {w for w, _ in combos})
    res: dict = {"t": ctx.t, "bank": ctx.state.bank, "ft": ctx.state.free_transfers,
                 "want": want, "trajectory": {}, "tc_bb": {}, "fh": {}, "wc": {},
                 "combos": {}}
    if not need:
        return res
    traj = base_trajectory(ctx, max(need), step_h, progress)
    res["_traj"] = traj
    for g, st in traj.items():
        p = st.plan
        res["trajectory"][g] = {"ft": p.free_transfers, "in": p.transfers_in,
                                "out": p.transfers_out, "hits": p.hits,
                                "xpts": p.expected_points, "bank_after": p.bank_after,
                                "captain": p.captain_id}

    # TC / BB on the planned squad of each GW
    for g in sorted(set(want.get(CHIP_TRIPLE_CAPTAIN, [])) | set(want.get(CHIP_BENCH_BOOST, []))
                    | set(want.get(CHIP_FREE_HIT, []))):
        sq = ctx.squad_from_plan(traj[g].plan)
        r = simulate(sq, ctx.n_sims, seed=g)
        k = g - ctx.t
        r.update({
            "k": k,
            "captain": sq.elements[sq.cap], "captain_xp": float(sq.xp[sq.cap]),
            "vice": sq.elements[sq.vice],
            "bench": [sq.elements[i] for i in sq.bench],
            "bench_xp": float(sq.xp[sq.bench].sum()),
            "tc_available": g in want.get(CHIP_TRIPLE_CAPTAIN, []),
            "bb_available": g in want.get(CHIP_BENCH_BOOST, []),
            "tc_cal": r["tc_gain"] * calib("tc", k, g),
            "bb_cal": r["bb_gain"] * calib("bb", k, g),
        })
        res["tc_bb"][g] = r

    # Free Hit: plan with FH at g vs the no-chip plan; also vs holding the squad
    for g in want.get(CHIP_FREE_HIT, []):
        st = traj[g]
        window = ctx.window(g, fh_h)
        reuse = st.result if window == st.window else None
        cmp = compare_plans(ctx, st.state, window, {g: CHIP_FREE_HIT}, reuse, seed=1000)
        hold = simulate(ctx.hold_squad(st.state, g), ctx.n_sims, seed=2000 + g)["normal"]
        fh_gw = simulate(ctx.squad_from_plan(cmp.with_chip.plan[0]), ctx.n_sims,
                         seed=3000 + g)["normal"]
        k = g - ctx.t
        res["fh"][g] = {
            "k": k, "applied": cmp.applied,
            "fh_vs_hold": fh_gw - hold, "fh_vs_plan": cmp.mc_gain,
            "fh_vs_plan_milp": cmp.milp_gain, "window": window,
            "fh_squad": cmp.with_chip.plan[0].squad, "hold_normal": hold,
            "fh_normal": fh_gw,
            "fh_cal": (fh_gw - hold) * calib("fh", k, g),
        }
        if progress:
            progress(f"FH GW{g}: {fh_gw - hold:+.1f} vs hold, {cmp.mc_gain:+.1f} vs plan")

    # Wildcard: plan with WC at g vs without over g..g+wc_h-1
    for g in want.get(CHIP_WILDCARD, []):
        st = traj[g]
        window = ctx.window(g, wc_h)
        reuse = st.result if window == st.window else None
        cmp = compare_plans(ctx, st.state, window, {g: CHIP_WILDCARD}, reuse, seed=4000)
        wplan = cmp.with_chip.plan
        bb_next = {}
        for p in wplan[1:3]:
            bb_next[p.gw] = simulate(ctx.squad_from_plan(p), ctx.n_sims,
                                     seed=5000 + p.gw)["bb_gain"]
        k = g - ctx.t
        res["wc"][g] = {
            "k": k, "applied": cmp.applied, "window": window,
            "wc_gain": cmp.mc_gain, "wc_gain_milp": cmp.milp_gain,
            "per_gw_gain": cmp.per_gw_gain,
            "n_moves": len(wplan[0].transfers_in),
            "wc_in": wplan[0].transfers_in, "wc_out": wplan[0].transfers_out,
            "base_moves": sum(len(p.transfers_in) for p in cmp.without.plan),
            "bb_on_wc_squad": bb_next,
            "wc_cal": cmp.mc_gain * calib("wc", k, g),
        }
        if progress:
            progress(f"WC GW{g}: {cmp.mc_gain:+.1f} (MILP {cmp.milp_gain:+.1f})")

    # WC + BB combos
    for wc_g, bb_g in combos:
        st = traj[wc_g]
        window = ctx.window(wc_g, wc_h)
        if bb_g not in window or bb_g == wc_g:
            res["combos"][f"WC{wc_g}+BB{bb_g}"] = {"error": f"BB GW{bb_g} not in the WC "
                                                   f"horizon GW{window[0]}-{window[-1]}"}
            continue
        cmp = compare_plans(ctx, st.state, window,
                            {wc_g: CHIP_WILDCARD, bb_g: CHIP_BENCH_BOOST}, seed=6000)
        bb_plan = next(p for p in cmp.with_chip.plan if p.gw == bb_g)
        bbg = simulate(ctx.squad_from_plan(bb_plan), ctx.n_sims, seed=7000 + bb_g)["bb_gain"]
        res["combos"][f"WC{wc_g}+BB{bb_g}"] = {
            "applied": cmp.applied, "gain_vs_base": cmp.mc_gain,
            "gain_vs_base_milp": cmp.milp_gain, "bb_gain_on_built_squad": bbg,
        }
    return res


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _nan(x) -> float:
    return float("nan") if x is None else float(x)


def pair_moves(outs, ins, pos_of: dict) -> list[tuple[int, int]]:
    """(out, in) pairs of the same position (FPL pairs transfers by type)."""
    outs, ins = list(outs), list(ins)
    pairs = []
    for o in list(outs):
        i = next((x for x in ins if pos_of.get(x) == pos_of.get(o)), None)
        if i is not None:
            pairs.append((o, i))
            outs.remove(o)
            ins.remove(i)
    return pairs + list(zip(outs, ins))


def format_report(ctx: ChipContext, res: dict, remaining: list[RemainingChip],
                  calendar: dict, names: dict, teams: dict, flags: dict,
                  starters, last_gw: int) -> str:
    """The printed report (plain text)."""
    L: list[str] = []
    t = ctx.t
    L.append(f"=== Chip evaluation as of GW{t} | bank {ctx.state.bank / 10:.1f}m | "
             f"FTs {ctx.state.free_transfers} ===")
    L.append("Chips remaining: " + (", ".join(
        f"{rc.label} (GW{rc.gws[0]}-{rc.gws[-1]}"
        + (", not GW20: FH played in GW19" if rc.chip == CHIP_FREE_HIT and t <= 20
           and rc.window.start <= 20 <= rc.window.stop and 20 not in rc.gws else "") + ")"
        for rc in remaining) or "none"))

    L.append("\nFixture map (BGW = teams with 0 fixtures, DGW = 2+):")
    flagged_any = False
    for g, d in calendar["per_gw"].items():
        if d["blank_teams"] or d["double_teams"]:
            flagged_any = True
            L.append(f"  GW{g}: {d['n_fixtures']} fixtures | BGW: "
                     + (",".join(teams.get(x, str(x)) for x in d["blank_teams"]) or "-")
                     + " | DGW: " + (",".join(teams.get(x, str(x)) for x in d["double_teams"])
                                     or "-"))
    if not flagged_any:
        L.append("  no blank or double GW in the window")
    uns = calendar["unscheduled"]
    L.append(f"  unscheduled fixtures: {len(uns)}"
             + ("" if not uns else " -> " + ", ".join(
                 f"{teams.get(h, h)}-{teams.get(a, a)}" for h, a in uns)))

    n_out, met = early_wc_trigger(flags, starters)
    squad_ids = [p.element_id for p in ctx.state.squad.players]
    fl = [(e, flags[e]) for e in squad_ids if e in flags]
    L.append(f"\nSquad availability GW{t}: " + ("; ".join(
        f"{names.get(e, e)} ({why})" for e, why in fl) or "all available"))
    L.append(f"  early-WC trigger (plan: >={EARLY_WC_STARTERS} starters flagged/out): {n_out} "
             "(starters = last GW's XI + the model's XI) -> " + ("MET" if met else "not met"))

    if res["trajectory"]:
        L.append("\nBase plan (no chips; receding horizon on the as-of predictions):")
        parts = []
        for g, tr in res["trajectory"].items():
            mv = (", ".join(f"{names.get(o, o)}>{names.get(i, i)}"
                            for o, i in pair_moves(tr["out"], tr["in"], ctx.pos_of))
                  if tr["in"] else "roll")
            parts.append(f"GW{g} FT{tr['ft']}: {mv}" + (f" (-{4 * tr['hits']})" if tr["hits"] else ""))
        for i in range(0, len(parts), 3):
            L.append("  " + " | ".join(parts[i:i + 3]))

    if res["tc_bb"]:
        L.append("\n          raw model expectation ....................................  "
                 "x 2025-26 realised/pred (by half)")
        L.append(" GW  captain (xP)          TC gain bench xP autosub BB gain FH/hold FH/plan"
                 "   TC cal  BB cal  FH cal")
        for g in sorted(res["tc_bb"]):
            if g > last_gw:
                continue
            r = res["tc_bb"][g]
            fh = res["fh"].get(g, {})
            mark = "*" if g == 19 and fh else " "

            def col(v, ok=True):
                return f"{v:7.2f}" if ok and v == v else "      -"
            L.append(
                f"{g:>3}  {str(names.get(r['captain'], r['captain']))[:14]:<14} ({r['captain_xp']:4.1f})"
                f"  {col(r['tc_gain'], r['tc_available'])}  {r['bench_xp']:7.2f} {r['autosub']:7.2f}"
                f" {col(r['bb_gain'], r['bb_available'])} {col(_nan(fh.get('fh_vs_hold')))}{mark}"
                f"{col(_nan(fh.get('fh_vs_plan')))}"
                f"  {col(r['tc_cal'], r['tc_available'])} {col(r['bb_cal'], r['bb_available'])}"
                f" {col(_nan(fh.get('fh_cal')))}")
        if any(g == 19 for g in res["fh"]):
            L.append("  * FH in GW19 makes FH2 unplayable in GW20 (FPL rule; plan: never GW19)")
        L.append("  TC = E[armband pts incl. vice failover]; BB = bench xP net of auto-sub EV "
                 "(planned squad of that GW).\n"
                 "  FH/hold = FH squad vs holding the squad that GW; FH/plan = FH plan vs the "
                 "no-chip plan (Monte Carlo, horizon of the base step;\n"
                 "  can be < 0: a chip week carries the FT count without the +1).\n"
                 "  'cal' = raw x the 2025-26 replay's realised/predicted ratio at that lead, "
                 "for the chip GW's season half (CALIB).")
    if res["wc"]:
        L.append("\nWC at GW  gain(MC)  gain(MILP)     cal  moves  BB next GWs on the WC squad")
        for g, r in sorted(res["wc"].items()):
            bbn = ", ".join(f"GW{h} {v:+.1f}" for h, v in r["bb_on_wc_squad"].items())
            L.append(f"{g:>9}  {r['wc_gain']:8.2f}  {r['wc_gain_milp']:10.2f}  {r['wc_cal']:6.2f}"
                     f"  {r['n_moves']:>5}  {bbn}" + ("" if r["applied"] else "  (chip NOT applied)"))
        h = next(iter(res["wc"].values()))["window"]
        L.append(f"  WC gain = sum over {len(h)} GWs of E[plan with WC at g] - E[plan without], "
                 "same state/config/pool, Monte Carlo, -4 per hit; replay ratio "
                 f"{CALIB['wc'][1][0][1]:.2f} (GW1-19) / {CALIB['wc'][2][0][1]:.2f} (GW20-38).")
    for key, v in res.get("combos", {}).items():
        if "error" in v:
            L.append(f"{key}: {v['error']}")
        else:
            L.append(f"{key}: combined gain vs no-chip plan {v['gain_vs_base']:+.1f} "
                     f"(MILP {v['gain_vs_base_milp']:+.1f}; BB on the built squad "
                     f"{v['bb_gain_on_built_squad']:+.1f})")
    first = [rc for rc in remaining if rc.window.half == 1]
    if first:
        left = FIRST_HALF_END - t + 1
        L.append(f"\nFirst-half chips left: {', '.join(CHIP_SHORT[rc.chip] for rc in first)} - "
                 f"they expire after GW{FIRST_HALF_END}; one chip per GW, so they need "
                 f"{len(first)} distinct GWs in GW{t}-{FIRST_HALF_END} ({left} GWs left).")
    L.append("Numbers are expectations under the model; they do not decide the plan.")
    return "\n".join(L)
