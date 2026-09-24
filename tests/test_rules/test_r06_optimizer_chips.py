"""R06 — transfer_optimizer.py / lineup_selector.py chip & rule handling.

WC/FH = unlimited free transfers that GW (hit 0, FT/max_transfers ignored);
BB = all 15 count in the objective; TC = captain x3 (vice failover also x3);
hits -4 beyond FTs; squad/club/budget/formation always legal; <= transfers_cap.
"""

from __future__ import annotations

from collections import Counter

import pytest

from fpl_optimizer.engine.state import ChipState, GameState, PlayerSlot, Squad
from fpl_optimizer.optimizer.lineup_selector import (
    BENCH_GK_WEIGHT,
    BENCH_OUTFIELD_WEIGHTS,
    VICE_CAPTAIN_WEIGHT,
    select_lineup,
)
from fpl_optimizer.optimizer.transfer_optimizer import optimize_transfers
from fpl_optimizer.optimizer.types import PlayerCandidate
from fpl_optimizer.utils.constants import VALID_FORMATIONS, Position

from tests.test_rules.rules_helpers import POS, make_state


def cand(eid, xp, price=50, team=None, pos=None):
    return PlayerCandidate(element_id=eid, position=pos or POS[eid], price=price,
                           team_id=team if team is not None else eid,
                           predicted_points=xp)


def squad_pool(xp=None, default=2.0):
    xp = xp or {}
    return [cand(e, xp.get(e, default)) for e in range(1, 16)]


def _formation(res, pool):
    pos = {c.element_id: c.position for c in pool}
    c = Counter(pos[e] for e in res.lineup_element_ids)
    return c[Position.GK], (c[Position.DEF], c[Position.MID], c[Position.FWD])


class TestFreeTransferChips:
    MARKET = [cand(16, 8.0), cand(17, 8.0), cand(18, 8.0)]  # +6 over a 2.0 starter

    def test_no_chip_takes_hits_only_when_worth_it(self):
        st = make_state(ft=1)
        res = optimize_transfers(st, squad_pool() + self.MARKET)
        assert len(res.transfers_in) == 3        # +6 > 4 each
        assert res.hit_cost == 8

    def test_hits_not_taken_for_small_gains(self):
        market = [cand(16, 3.5), cand(17, 3.5), cand(18, 3.5)]  # +1.5 < 4
        res = optimize_transfers(make_state(ft=1), squad_pool() + market)
        assert len(res.transfers_in) == 1 and res.hit_cost == 0

    @pytest.mark.parametrize("chip", ["wildcard", "free_hit"])
    def test_wc_fh_all_transfers_free(self, chip):
        market = [cand(16, 3.5), cand(17, 3.5), cand(18, 3.5)]
        res = optimize_transfers(make_state(ft=1), squad_pool() + market, chip=chip)
        assert len(res.transfers_in) == 3 and res.hit_cost == 0
        assert res.chip == chip

    @pytest.mark.parametrize("chip", ["wildcard", "free_hit"])
    def test_wc_fh_ignore_max_transfers(self, chip):
        market = [cand(16, 3.5), cand(17, 3.5), cand(18, 3.5)]
        res = optimize_transfers(make_state(ft=1), squad_pool() + market, chip=chip,
                                 max_transfers=1)
        assert len(res.transfers_in) == 3

    def test_uses_all_banked_fts(self):
        market = [cand(16, 3.5), cand(17, 3.5), cand(18, 3.5)]
        res = optimize_transfers(make_state(ft=3), squad_pool() + market)
        assert len(res.transfers_in) == 3 and res.hit_cost == 0

    def test_full_rebuild_never_exceeds_transfers_cap(self, rules):
        market = [cand(100 + i, 9.0, pos=POS[e], team=100 + i) for i, e in enumerate(range(1, 16))]
        res = optimize_transfers(make_state(ft=1), squad_pool() + market, chip="wildcard")
        assert len(res.transfers_in) == 15
        assert len(res.transfers_in) <= rules["game_settings"]["transfers_cap"]

    def test_state_not_mutated_by_free_hit_solve(self):
        st = make_state(ft=2, bank=7)
        before = [p.element_id for p in st.squad.players]
        optimize_transfers(st, squad_pool() + self.MARKET, chip="free_hit")
        assert [p.element_id for p in st.squad.players] == before
        assert st.bank == 7 and st.free_transfers == 2


class TestLegalityUnderChips:
    @pytest.mark.parametrize("chip", [None, "wildcard", "free_hit", "bench_boost",
                                      "triple_captain"])
    def test_squad_club_budget_formation(self, chip):
        # 3 strong market players from ONE club (team 50) + a cheap club-mate
        market = [cand(16, 9, team=50), cand(17, 9, team=50), cand(18, 9, team=50),
                  cand(20, 9, team=50), cand(21, 1, price=80)]
        st = make_state(ft=5, bank=0)
        res = optimize_transfers(st, squad_pool() + market, chip=chip)
        teams = Counter((c.team_id) for c in squad_pool() + market
                        if c.element_id in res.squad_element_ids)
        assert max(teams.values()) <= 3
        assert len(res.squad_element_ids) == 15
        gk, form = _formation(res, squad_pool() + market)
        assert gk == 1 and form in VALID_FORMATIONS
        spend = sum(c.price for c in market if c.element_id in res.transfers_in)
        income = 50 * len(res.transfers_out)
        assert spend <= income + st.bank
        assert res.captain_id in res.lineup_element_ids
        assert res.vice_captain_id in res.lineup_element_ids
        assert res.captain_id != res.vice_captain_id
        pos = {c.element_id: c.position for c in squad_pool() + market}
        assert pos[res.bench_element_ids[0]] == Position.GK   # bench GK pinned first


class TestBenchBoostObjective:
    def test_bb_objective_counts_every_bench_player(self):
        xp = {e: 3.0 for e in range(1, 16)}
        xp.update({2: 1.0, 7: 0.5, 12: 0.7, 15: 0.9, 13: 6.0, 14: 5.0})
        res = optimize_transfers(make_state(ft=1), squad_pool(xp), chip="bench_boost",
                                 max_transfers=0)
        # all 15 + captain (13) + vice weight x best other starter (14)
        exp = sum(xp.values()) + 6.0 + VICE_CAPTAIN_WEIGHT * 5.0
        assert res.objective_value == pytest.approx(exp)

    def test_bb_worth_a_hit_for_bench_upgrades(self):
        xp = {e: 3.0 for e in range(1, 16)}
        xp.update({7: 0.0, 12: 0.0, 15: 0.0})     # dead bench
        market = [cand(16, 5.0), cand(17, 5.0)]    # bench-level upgrades (+5)
        no_chip = optimize_transfers(make_state(ft=1), squad_pool(xp) + market)
        bb = optimize_transfers(make_state(ft=1), squad_pool(xp) + market,
                                chip="bench_boost")
        assert len(bb.transfers_in) == 2 and bb.hit_cost == 4
        assert len(no_chip.transfers_in) <= 1 and no_chip.hit_cost == 0


class TestTripleCaptainObjective:
    @pytest.mark.bug("R-07")
    def test_tc_adds_one_more_captain_multiple(self):
        xp = {e: 2.0 for e in range(1, 16)}
        xp.update({13: 8.0, 14: 6.0})
        base = optimize_transfers(make_state(), squad_pool(xp), max_transfers=0)
        tc = optimize_transfers(make_state(), squad_pool(xp), chip="triple_captain",
                                max_transfers=0)
        assert tc.captain_id == 13
        assert tc.objective_value - base.objective_value == pytest.approx(
            8.0 + VICE_CAPTAIN_WEIGHT * 6.0)


class TestClubLimitPlaceholders:
    """Squad members missing from the candidate list (BGW in backtest.py /
    backtest_season.py: build_candidate_pool only has players WITH a GW row)
    become placeholders with team_id=0."""

    @pytest.mark.bug("R-06")
    def test_four_blanking_squad_players_do_not_force_a_sale(self):
        missing = {3, 8, 13, 14}   # four teams blank this GW
        pool = [c for c in squad_pool() if c.element_id not in missing]
        res = optimize_transfers(make_state(ft=1), pool, max_transfers=0)
        assert res.transfers_out == []

    @pytest.mark.bug("R-06")
    def test_blanking_players_still_count_toward_their_club(self):
        # squad DEFs 3, 4, 5 are all club 77 and blank this GW; a club-77 MID star
        # appears in the pool -> buying him (for a MID) makes 4 club-77 players.
        club = 77
        team_of = {e: (club if e in (3, 4, 5) else e) for e in range(1, 16)}
        team_of[21] = club
        pool = [cand(e, 2.0, team=team_of[e]) for e in range(1, 16) if e not in (3, 4, 5)]
        pool.append(cand(21, 12.0, team=club))
        st = make_state(ft=1)
        try:
            res = optimize_transfers(st, pool, squad_teams=team_of)  # patched API
        except TypeError:
            res = optimize_transfers(st, pool)
        final = [team_of[e] for e in res.squad_element_ids]
        assert final.count(club) <= 3


class TestLineupSelector:
    def test_formation_captain_vice_bench_gk(self):
        xp = {e: float(e % 7) for e in range(1, 16)}
        res = select_lineup(squad_pool(xp))
        gk, form = _formation(res, squad_pool(xp))
        assert gk == 1 and form in VALID_FORMATIONS
        assert res.captain_id in res.lineup_element_ids
        assert res.vice_captain_id in res.lineup_element_ids
        assert POS[res.bench_element_ids[0]] == Position.GK
        assert len(res.bench_element_ids) == 4

    def test_captain_is_best_starter(self):
        xp = {e: 1.0 for e in range(1, 16)}
        xp[10] = 9.0
        assert select_lineup(squad_pool(xp)).captain_id == 10
