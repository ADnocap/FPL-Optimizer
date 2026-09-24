"""R03 — engine.step chip & transfer rules (WC, FH, BB, TC, FTs, hits, GW1)."""

from __future__ import annotations

import pytest

from fpl_optimizer.engine.chips import activate_chip
from fpl_optimizer.engine.engine import FPLGameEngine
from fpl_optimizer.engine.state import ChipState, EngineAction, GameState
from fpl_optimizer.utils.constants import INITIAL_FREE_TRANSFERS

from tests.test_rules.rules_helpers import FakeLoader, full_rows, make_squad, make_state, row


def _engine(gws=(1, 2, 6, 7, 8, 19, 20), pts=None, value=50):
    rows = {}
    for gw in gws:
        rows.update(full_rows(gw, pts=pts, value=value))
    return FPLGameEngine(FakeLoader(rows))


class TestFreeTransfersThroughEngine:
    @pytest.mark.bug("R-02")
    def test_gw2_starts_with_exactly_one_ft(self):
        """FPL: unlimited changes before the GW1 deadline, then 1 FT for GW2.

        backtest.py:99 and scripts/backtest_season.py:133 build the GW1 state with
        free_transfers=1; the engine then banks +1 so GW2 gets 2 FTs.
        """
        eng = _engine()
        s = GameState(squad=make_squad(), bank=0,
                      free_transfers=INITIAL_FREE_TRANSFERS, current_gw=1)
        s2, _ = eng.step(s, EngineAction())
        assert s2.current_gw == 2
        assert s2.free_transfers == 1

    def test_roll_then_hit(self):
        eng = _engine()
        s = make_state(gw=6, ft=1)
        s, r = eng.step(s, EngineAction())                  # roll
        assert s.free_transfers == 2 and r.hit_cost == 0
        s, r = eng.step(s, EngineAction(transfers_out=[7, 12, 15],
                                        transfers_in=[16, 17, 18]))  # 3 with 2 FT
        assert r.hit_cost == 4
        assert r.net_points == r.gw_points - 4
        assert s.free_transfers == 1

    def test_cap_at_five(self):
        eng = _engine(gws=range(1, 39))
        s = make_state(gw=2, ft=1)
        for _ in range(8):
            s, _ = eng.step(s, EngineAction())
        assert s.free_transfers == 5

    @pytest.mark.parametrize("ft", [1, 2, 5])
    def test_wildcard_keeps_ft_and_no_hit(self, ft):
        eng = _engine()
        s = make_state(gw=7, ft=ft)
        s2, r = eng.step(s, EngineAction(transfers_out=[7, 12, 15],
                                         transfers_in=[16, 17, 18],
                                         chip="wildcard"))
        assert r.hit_cost == 0
        assert s2.free_transfers == ft          # unchanged, no +1
        assert s2.squad.find_player_idx(16) is not None  # permanent
        assert not s2.chips.is_available("wildcard", 8)

    @pytest.mark.parametrize("ft", [1, 3, 5])
    def test_free_hit_keeps_ft_and_no_hit(self, ft):
        eng = _engine()
        s = make_state(gw=7, ft=ft)
        s2, r = eng.step(s, EngineAction(transfers_out=[7, 12, 15],
                                         transfers_in=[16, 17, 18],
                                         chip="free_hit"))
        assert r.hit_cost == 0
        assert s2.free_transfers == ft


class TestFreeHitRevert:
    def test_squad_reverts(self):
        eng = _engine()
        s = make_state(gw=7, ft=1, bank=10)
        before = [p.element_id for p in s.squad.players]
        s2, _ = eng.step(s, EngineAction(transfers_out=[7], transfers_in=[16],
                                         chip="free_hit"))
        assert [p.element_id for p in s2.squad.players] == before
        assert s2.free_hit_stash is None
        assert s2.active_chip is None

    def test_fh_squad_scores_that_week(self):
        eng = _engine(pts={7: 0, 16: 10})
        s = make_state(gw=7, ft=1)
        # 7 (bench DEF, 0 pts) -> 16 (10 pts) and start 16 instead of DEF 6
        lineup = [1, 3, 4, 5, 16, 8, 9, 10, 11, 13, 14]
        bench = [2, 12, 6, 15]
        _, r = eng.step(s, EngineAction(transfers_out=[7], transfers_in=[16],
                                        chip="free_hit", lineup=lineup, bench=bench))
        # XI: 10 players x 2 + 16's 10 = 30, captain 13 (+2)
        assert r.gw_points == 32

    @pytest.mark.bug("R-03")
    def test_bank_reverts_after_free_hit(self):
        """Official: 'At the next deadline, your original squad and bank are restored'.

        Engine restores the squad but keeps the FH-week bank (free_hit_stash only
        holds the Squad). Selling a 50 and buying a 40 during FH must NOT leave
        +10 in the bank afterwards.
        """
        rows = {}
        for gw in (7, 8):
            rows.update(full_rows(gw))
            rows[(16, gw)] = row(value=40)
        eng = FPLGameEngine(FakeLoader(rows))
        s = make_state(gw=7, ft=1, bank=5)
        s2, _ = eng.step(s, EngineAction(transfers_out=[7], transfers_in=[16],
                                         chip="free_hit"))
        assert s2.bank == 5


class TestChipActivationRules:
    def test_one_chip_per_gw(self):
        s = activate_chip(make_state(gw=7), "bench_boost")
        with pytest.raises(ValueError):
            activate_chip(s, "triple_captain")

    def test_used_chip_rejected_same_half(self):
        eng = _engine()
        s, _ = eng.step(make_state(gw=7), EngineAction(chip="triple_captain"))
        with pytest.raises(ValueError):
            activate_chip(s, "triple_captain")

    @pytest.mark.bug("R-01")
    @pytest.mark.parametrize("chip", ["wildcard", "free_hit"])
    def test_wc_fh_rejected_in_gw1(self, chip):
        with pytest.raises(ValueError):
            activate_chip(make_state(gw=1), chip)

    def test_gw19_expiry_via_engine(self):
        eng = _engine()
        s, _ = eng.step(make_state(gw=19), EngineAction())
        assert s.current_gw == 20
        for chip in ("wildcard", "free_hit", "bench_boost", "triple_captain"):
            assert s.chips.is_available(chip, 20)
            assert s.chips.wildcard[0] is False and s.chips.free_hit[0] is False
            assert s.chips.bench_boost[0] is False and s.chips.triple_captain[0] is False

    def test_fh19_then_fh20_rejected_by_engine(self):
        eng = _engine()
        s, _ = eng.step(make_state(gw=19), EngineAction(chip="free_hit"))
        with pytest.raises(ValueError):
            activate_chip(s, "free_hit")
        # but WC2 in GW20 is fine
        activate_chip(s, "wildcard")


class TestTripleCaptainAndBenchBoost:
    def test_tc_triples_captain(self):
        eng = _engine(pts={13: 10})
        _, r = eng.step(make_state(gw=7), EngineAction(chip="triple_captain"))
        # XI = 10*2 + 10 = 30 ; TC bonus = 2*10
        assert r.captain_points == 20
        assert r.gw_points == 50

    def test_tc_passes_to_vice_when_captain_blanks(self):
        rows = full_rows(7, pts={8: 7})
        rows[(13, 7)] = row(pts=0, minutes=0)
        eng = FPLGameEngine(FakeLoader(rows))
        _, r = eng.step(make_state(gw=7), EngineAction(chip="triple_captain"))
        assert r.captain_failover is True
        assert r.captain_points == 14  # vice 7 x 3 => +14

    def test_bb_counts_all_fifteen(self):
        eng = _engine(pts={2: 3, 12: 4, 7: 5, 15: 6})
        _, r0 = eng.step(make_state(gw=7), EngineAction())
        _, r1 = eng.step(make_state(gw=7), EngineAction(chip="bench_boost"))
        assert r1.gw_points - r0.gw_points == 3 + 4 + 5 + 6

    def test_bb_with_non_playing_starter_counts_each_player_once(self):
        rows = full_rows(7, pts={2: 3, 12: 4, 7: 5, 15: 6})
        rows[(4, 7)] = row(pts=0, minutes=0)   # starting DEF blanks
        eng = FPLGameEngine(FakeLoader(rows))
        _, r = eng.step(make_state(gw=7), EngineAction(chip="bench_boost"))
        # all 15: 10 starters x2 (minus DEF 4 at 0) + bench 3+4+5+6 + captain 13 (+2)
        assert r.gw_points == 2 * 10 + (3 + 4 + 5 + 6) + 2
