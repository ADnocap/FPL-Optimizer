"""Tests for the multi-period (receding-horizon) transfer planner."""

from __future__ import annotations

from collections import Counter

import pytest

from fpl_optimizer.engine.state import ChipState, GameState, PlayerSlot, Squad
from fpl_optimizer.optimizer.horizon_optimizer import (
    HorizonCandidate,
    HorizonConfig,
    dnp_bench_weights,
    optimize_horizon,
    parse_chip_plan,
)
from fpl_optimizer.optimizer.transfer_optimizer import optimize_transfers
from fpl_optimizer.optimizer.types import PlayerCandidate
from fpl_optimizer.utils.constants import VALID_FORMATIONS, Position

GK, DEF, MID, FWD = Position.GK, Position.DEF, Position.MID, Position.FWD

# Base squad: (eid, position, xPts per GW) — prices 50, one club each
_BASE = [
    (1, GK, 4.0), (2, GK, 1.0),
    (3, DEF, 4.0), (4, DEF, 4.0), (5, DEF, 4.0), (6, DEF, 3.5), (7, DEF, 1.0),
    (8, MID, 5.0), (9, MID, 5.0), (10, MID, 4.5), (11, MID, 4.0), (12, MID, 1.0),
    (13, FWD, 5.0), (14, FWD, 4.5), (15, FWD, 1.0),
]


def _cand(eid, pos, xp, h, price=50, team=None, p_play=0.9):
    xp = tuple(xp) if isinstance(xp, (tuple, list)) else (float(xp),) * h
    return HorizonCandidate(eid, pos, price, team if team is not None else eid,
                            xp, (p_play,) * h)


def _base_pool(h: int, overrides: dict | None = None) -> list[HorizonCandidate]:
    overrides = overrides or {}
    return [overrides.get(eid, _cand(eid, pos, xp, h)) for eid, pos, xp in _BASE]


def _state(ft: int = 1, bank: int = 0, selling: dict | None = None,
           chips: ChipState | None = None, gw: int = 6) -> GameState:
    selling = selling or {}
    players = [PlayerSlot(eid, pos, 50, selling.get(eid, 50)) for eid, pos, _ in _BASE]
    squad = Squad(players=players, lineup=list(range(11)), bench=list(range(11, 15)),
                  captain_idx=7, vice_captain_idx=8)
    return GameState(squad=squad, bank=bank, free_transfers=ft,
                     chips=chips or ChipState(), current_gw=gw)


def _solve(state, pool, gws, chip_plan=None, **cfg):
    # mechanics tests: no regularisation unless a test asks for it
    cfg.setdefault("hit_margin", 0.0)
    cfg.setdefault("ft_value", 0.0)
    return optimize_horizon(state, pool, gws, chip_plan, HorizonConfig(**cfg))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_parse_chip_plan(self):
        assert parse_chip_plan("tc:7,wc:11,bb:12") == {
            7: "triple_captain", 11: "wildcard", 12: "bench_boost"}
        assert parse_chip_plan("fh:18") == {18: "free_hit"}
        assert parse_chip_plan("tc7,wc11,3xc:25") == {
            7: "triple_captain", 11: "wildcard", 25: "triple_captain"}
        assert parse_chip_plan(None) == {}

    def test_parse_chip_plan_rejects_bad_input(self):
        with pytest.raises(ValueError):
            parse_chip_plan("xx:7")
        with pytest.raises(ValueError):
            parse_chip_plan("tc:7,bb:7")  # one chip per GW

    def test_dnp_bench_weights(self):
        w_gk, w = dnp_bench_weights([0.0] * 10, 0.0)
        assert w_gk == 0.0 and w == (0.0, 0.0, 0.0)
        w_gk, w = dnp_bench_weights([1.0] * 10, 1.0)
        assert w_gk == 1.0 and w == pytest.approx((1.0, 1.0, 1.0))
        _, w = dnp_bench_weights([0.1] * 10, 0.05)
        assert w[0] == pytest.approx(1 - 0.9 ** 10)
        assert w[0] > w[1] > w[2] > 0


# ---------------------------------------------------------------------------
# free-transfer dynamics
# ---------------------------------------------------------------------------


class TestFreeTransfers:
    def test_rolls_ft_to_make_two_free_moves_next_week(self):
        """Outgoing players are great THIS week, the replacements from next
        week: the planner rolls the FT and makes both moves next GW for free
        (a single-GW optimiser can never see this)."""
        h = 4
        pool = _base_pool(h, {
            8: _cand(8, MID, (6.0, 1.0, 1.0, 1.0), h),
            9: _cand(9, MID, (6.0, 1.0, 1.0, 1.0), h),
        })
        pool += [_cand(101, MID, (0.0, 7.0, 7.0, 7.0), h),
                 _cand(102, MID, (0.0, 7.0, 7.0, 7.0), h)]
        res = _solve(_state(ft=1), pool, [6, 7, 8, 9])
        p0, p1 = res.plan[0], res.plan[1]
        assert p0.transfers_in == [] and p0.hits == 0
        assert p1.free_transfers == 2
        assert sorted(p1.transfers_in) == [101, 102]
        assert sorted(p1.transfers_out) == [8, 9]
        assert p1.hits == 0
        assert res.first.transfers_in == []

    def test_ft_bank_capped_at_five(self):
        """With 5 FTs banked, rolling does not create a 6th: six moves in the
        last GW need a hit, so only five (gain 3 < 4) are made."""
        h = 3
        out_ids = [3, 4, 5, 8, 9, 10]
        overrides = {e: _cand(e, pos, (5.0, 5.0, 0.0), h)
                     for e, pos, _ in _BASE if e in out_ids}
        pool = _base_pool(h, overrides)
        for j, e in enumerate(out_ids):
            pos = DEF if e < 8 else MID
            pool.append(_cand(200 + j, pos, (0.0, 0.0, 3.0 + 0.01 * j), h))
        res = _solve(_state(ft=5), pool, [6, 7, 8], discount=1.0)
        assert [p.free_transfers for p in res.plan] == [5, 5, 5]
        assert len(res.plan[2].transfers_in) == 5
        assert res.plan[2].hits == 0

    def test_hit_taken_only_when_worth_it(self):
        pool = _base_pool(1) + [_cand(101, MID, 10.0, 1), _cand(102, MID, 10.0, 1)]
        # replacing two 1-2 pt players with 10s: +free, then +~5 for -4 -> take
        res = _solve(_state(ft=1), pool, [6])
        assert len(res.first.transfers_in) == 2
        assert res.plan[0].hits == 1 and res.first.hit_cost == 4

    def test_hit_margin_suppresses_marginal_hits(self):
        pool = _base_pool(1) + [_cand(101, MID, 6.0, 1), _cand(102, MID, 6.0, 1)]
        free = _solve(_state(ft=1), pool, [6])
        cautious = _solve(_state(ft=1), pool, [6], hit_margin=3.0)
        assert len(free.first.transfers_in) >= len(cautious.first.transfers_in)
        assert cautious.plan[0].hits == 0

    def test_ft_value_rolls_marginal_free_transfer(self):
        """A +1 pt upgrade is taken with no FT value, rolled with ft_value=1.5.

        Everyone is certain to play (p_play=1), so bench value is 0 and the
        only gain is 101 (5.0) replacing MID 11 (4.0) in the XI.
        """
        sure = [HorizonCandidate(c.element_id, c.position, c.price, c.team_id, c.xpts, (1.0,))
                for c in _base_pool(1)]
        pool = sure + [_cand(101, MID, 5.0, 1, p_play=1.0)]
        assert _solve(_state(ft=1), pool, [6]).first.transfers_in == [101]
        assert _solve(_state(ft=1), pool, [6], ft_value=1.5).first.transfers_in == []

    def test_default_config_is_regularised(self):
        cfg = HorizonConfig()
        assert cfg.hit_margin > 0

    def test_max_hits_zero_is_ft_only(self):
        pool = _base_pool(2) + [_cand(101, MID, 10.0, 2), _cand(102, MID, 10.0, 2)]
        res = _solve(_state(ft=1), pool, [6, 7], max_hits_per_gw=0)
        assert all(p.hits == 0 for p in res.plan)
        assert len(res.plan[0].transfers_in) == 1  # 1 FT now ...
        assert len(res.plan[1].transfers_in) == 1  # ... and the next one later

    def test_max_transfers_per_gw(self):
        pool = _base_pool(1) + [_cand(101, MID, 10.0, 1), _cand(102, MID, 10.0, 1)]
        res = _solve(_state(ft=1), pool, [6], max_transfers_per_gw=1)
        assert len(res.first.transfers_in) == 1


# ---------------------------------------------------------------------------
# budget / club / formation
# ---------------------------------------------------------------------------


class TestSquadRules:
    def test_selling_price_limits_budget(self):
        """Player 12 is worth 80 now but sells for 70: a 75 buy is not
        affordable with an empty bank, but is with 5 in the bank."""
        pool = _base_pool(2, {12: _cand(12, MID, 1.0, 2, price=80)})
        pool.append(_cand(101, MID, 9.0, 2, price=75))
        tight = _solve(_state(ft=1, bank=0, selling={12: 70}), pool, [6, 7])
        assert 101 not in tight.first.transfers_in
        ok = _solve(_state(ft=1, bank=5, selling={12: 70}), pool, [6, 7])
        assert 101 in ok.first.transfers_in
        assert all(p.bank_after >= 0 for p in ok.plan)
        assert ok.plan[0].bank_after == 0

    def test_three_per_club(self):
        h = 2
        pool = _base_pool(h)
        pool += [_cand(300 + j, MID if j < 3 else FWD, 9.0, h, price=40, team=99)
                 for j in range(6)]
        res = _solve(_state(ft=5, bank=100), pool, [6, 7])
        team_of = {c.element_id: c.team_id for c in pool}
        for p in res.plan:
            counts = Counter(team_of[e] for e in p.squad)
            assert counts[99] <= 3

    def test_squad_and_formation_valid(self):
        h = 3
        pool = _base_pool(h) + [
            _cand(400, FWD, 9.0, h), _cand(401, FWD, 8.0, h),
            _cand(402, DEF, 7.0, h), _cand(403, GK, 6.0, h),
        ]
        res = _solve(_state(ft=2, bank=50), pool, [6, 7, 8])
        pos_of = {c.element_id: c.position for c in pool}
        for p in res.plan:
            assert len(p.squad) == 15 and len(p.lineup) == 11 and len(p.bench) == 4
            sq = Counter(pos_of[e] for e in p.squad)
            assert sq == {GK: 2, DEF: 5, MID: 5, FWD: 3}
            xi = Counter(pos_of[e] for e in p.lineup)
            assert xi[GK] == 1
            assert (xi[DEF], xi[MID], xi[FWD]) in VALID_FORMATIONS
            assert pos_of[p.bench[0]] == GK  # backup GK first
            assert p.captain_id in p.lineup and p.vice_captain_id in p.lineup
            assert p.captain_id != p.vice_captain_id
            assert set(p.lineup) | set(p.bench) == set(p.squad)

    def test_missing_current_player_is_handled(self):
        """A squad member absent from the pool (no prediction) is kept at 0
        xPts or sold — the solve must not fail."""
        pool = [c for c in _base_pool(2) if c.element_id != 15]
        pool.append(_cand(500, FWD, 3.0, 2, price=45))
        res = _solve(_state(ft=1), pool, [6, 7])
        assert 15 not in res.plan[0].lineup

    def test_single_gw_matches_transfer_optimizer(self):
        """H=1 with the legacy fixed bench weights reproduces optimize_transfers."""
        pool = _base_pool(1) + [_cand(101, MID, 8.0, 1, price=60),
                                _cand(102, FWD, 7.0, 1, price=55),
                                _cand(103, DEF, 6.0, 1, price=45)]
        state = _state(ft=1, bank=10)
        legacy = optimize_transfers(state, [
            PlayerCandidate(c.element_id, c.position, c.price, c.team_id, c.xpts[0])
            for c in pool])
        res = _solve(state, pool, [6], bench_mode="fixed")
        assert res.objective_value == pytest.approx(legacy.objective_value, abs=1e-4)
        assert sorted(res.first.transfers_in) == sorted(legacy.transfers_in)


# ---------------------------------------------------------------------------
# chips
# ---------------------------------------------------------------------------


class TestChips:
    def test_wildcard_week_is_free_and_keeps_ft_count(self):
        h = 3
        pool = _base_pool(h) + [
            _cand(600 + j, pos, (0.0, 8.0, 8.0), h)
            for j, pos in enumerate([DEF, DEF, MID, MID, FWD])
        ]
        res = _solve(_state(ft=1), pool, [6, 7, 8], {7: "wildcard"})
        wc = res.plan[1]
        assert wc.chip == "wildcard"
        assert len(wc.transfers_in) >= 4
        assert wc.hits == 0
        assert res.plan[2].free_transfers == wc.free_transfers  # no +1 after WC

    def test_free_hit_squad_reverts(self):
        h = 3
        pool = _base_pool(h) + [
            _cand(700 + j, pos, (9.0, 0.0, 0.0), h)
            for j, pos in enumerate([DEF, MID, MID, FWD])
        ]
        res = _solve(_state(ft=1), pool, [6, 7, 8], {6: "free_hit"})
        fh, nxt = res.plan[0], res.plan[1]
        assert fh.chip == "free_hit" and fh.hits == 0
        assert len(fh.transfers_in) == 4
        assert set(nxt.squad) == {e for e, _, _ in _BASE}  # reverted
        assert nxt.transfers_in == []
        assert nxt.free_transfers == fh.free_transfers  # FH: no +1
        assert res.first.chip == "free_hit"
        assert sorted(res.first.transfers_in) == [700, 701, 702, 703]

    def test_triple_captain_counts_three_times(self):
        res = _solve(_state(ft=1), _base_pool(2), [6, 7], {6: "triple_captain"})
        p0 = res.plan[0]
        xp = {e: x for e, _, x in _BASE}
        assert p0.chip == "triple_captain"
        assert p0.expected_points == pytest.approx(
            sum(xp[e] for e in p0.lineup) + 2 * xp[p0.captain_id])
        assert res.plan[1].chip is None

    def test_bench_boost_counts_bench(self):
        res = _solve(_state(ft=1), _base_pool(1), [6], {6: "bench_boost"})
        p0 = res.plan[0]
        xp = {e: x for e, _, x in _BASE}
        assert p0.chip == "bench_boost"
        assert p0.expected_points == pytest.approx(
            sum(xp[e] for e in p0.squad) + xp[p0.captain_id])

    def test_unavailable_chip_is_ignored(self):
        chips = ChipState()
        chips.use_chip("triple_captain", 3)
        res = _solve(_state(ft=1, chips=chips), _base_pool(1), [6], {6: "tc"})
        assert res.plan[0].chip is None and res.first.chip is None

    def test_same_chip_twice_in_one_half_only_first_kept(self):
        res = _solve(_state(ft=1), _base_pool(3), [6, 7, 8], {6: "tc", 8: "tc"})
        assert [p.chip for p in res.plan] == ["triple_captain", None, None]

    def test_free_hit_not_in_both_gw19_and_gw20(self):
        res = _solve(_state(ft=1, gw=19), _base_pool(2), [19, 20],
                     {19: "free_hit", 20: "free_hit"})
        assert [p.chip for p in res.plan] == ["free_hit", None]

    def test_chip_outside_horizon_ignored(self):
        res = _solve(_state(ft=1), _base_pool(2), [6, 7], {11: "wildcard"})
        assert all(p.chip is None for p in res.plan)


# ---------------------------------------------------------------------------
# bench value
# ---------------------------------------------------------------------------


class TestBenchValue:
    def test_dnp_bench_weights_exceed_legacy(self):
        res = _solve(_state(ft=1), _base_pool(1), [6])
        w_gk, w = res.bench_weights[0]
        assert w[0] > 0.21  # legacy slot-1 weight
        assert w[0] > w[1] > w[2]

    def test_risky_starters_raise_bench_value(self):
        safe = _solve(_state(ft=1), _base_pool(1), [6])
        risky_pool = [HorizonCandidate(c.element_id, c.position, c.price, c.team_id,
                                       c.xpts, (0.5,)) for c in _base_pool(1)]
        risky = _solve(_state(ft=1), risky_pool, [6])
        assert risky.bench_weights[0][1][0] > safe.bench_weights[0][1][0]


def test_wildcard_move_cost_prunes_marginal_swaps():
    """A per-move cost in a WC week keeps only swaps whose gain clears it."""
    from fpl_optimizer.engine.state import ChipState, GameState, PlayerSlot, Squad
    from fpl_optimizer.optimizer.horizon_optimizer import (
        HorizonCandidate, HorizonConfig, optimize_horizon,
    )
    from fpl_optimizer.utils.constants import Position

    GK, DEF, MID, FWD = Position.GK, Position.DEF, Position.MID, Position.FWD
    base = [(1, GK, 4.0), (2, GK, 1.0), (3, DEF, 4.0), (4, DEF, 4.0), (5, DEF, 4.0),
            (6, DEF, 3.5), (7, DEF, 1.0), (8, MID, 5.0), (9, MID, 5.0), (10, MID, 4.5),
            (11, MID, 4.0), (12, MID, 1.0), (13, FWD, 5.0), (14, FWD, 4.5), (15, FWD, 1.0)]
    pool = [HorizonCandidate(e, p, 50, e, (x,), (0.95,)) for e, p, x in base]
    # same price; replaces the 1.0 bench MID: +0.5 in the XI plus a stronger
    # bench slot -> a gain of a few points, well under a 10-pt move cost
    pool.append(HorizonCandidate(101, MID, 50, 101, (4.5,), (0.95,)))
    players = [PlayerSlot(e, p, 50, 50) for e, p, _ in base]
    sq = Squad(players, list(range(11)), list(range(11, 15)), 7, 8)
    st = GameState(squad=sq, bank=0, free_transfers=1, chips=ChipState(), current_gw=11)
    free = optimize_horizon(st, pool, [11], {11: "wildcard"}, HorizonConfig())
    costed = optimize_horizon(st, pool, [11], {11: "wildcard"},
                              HorizonConfig(wc_move_cost=10.0))
    assert 101 in free.plan[0].transfers_in       # a free WC takes the upgrade
    assert not costed.plan[0].transfers_in        # a move cost above its gain prunes it
