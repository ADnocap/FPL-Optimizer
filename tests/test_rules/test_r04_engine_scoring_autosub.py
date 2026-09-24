"""R04 — scoring, auto-subs (incl. GK rules), captain failover, DGW/BGW, and a
replay of our real GW1-5 picks through the engine vs the official history."""

from __future__ import annotations

import itertools
import random

import pandas as pd
import pytest

from fpl_optimizer.data.loader import SeasonDataLoader
from fpl_optimizer.engine.constraints import is_valid_formation
from fpl_optimizer.engine.engine import FPLGameEngine
from fpl_optimizer.engine.lineup import perform_auto_subs
from fpl_optimizer.engine.state import EngineAction, GameState, PlayerSlot, Squad
from fpl_optimizer.utils.constants import Position

from tests.test_rules.rules_helpers import POS, FakeLoader, full_rows, load_gt, make_squad, make_state, row


def _dnp(rows, gw, *eids):
    for e in eids:
        rows[(e, gw)] = row(pts=0, minutes=0)


class TestAutoSubs:
    def test_gk_only_replaced_by_gk(self):
        rows = full_rows(7)
        _dnp(rows, 7, 1)                     # starting GK blanks
        sq, subs = perform_auto_subs(make_squad(), FakeLoader(rows), 7)
        assert subs == [(1, 2)]

    def test_outfield_never_replaces_gk(self):
        rows = full_rows(7)
        _dnp(rows, 7, 1, 2)                  # both GKs blank
        _, subs = perform_auto_subs(make_squad(), FakeLoader(rows), 7)
        assert subs == []

    def test_bench_gk_never_replaces_outfielder(self):
        rows = full_rows(7)
        _dnp(rows, 7, 3, 12, 7, 15)          # DEF 3 out, all outfield bench out
        _, subs = perform_auto_subs(make_squad(), FakeLoader(rows), 7)
        assert subs == []                    # GK 2 played but cannot come on

    def test_priority_order_and_formation_min_3_def(self):
        # 3-4-3: GK1 | 3,4,5 | 8,9,10,11 | 13,14,15 ; bench 2, 12(MID), 6(DEF), 7(DEF)
        sq = make_squad(lineup=(1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15),
                        bench=(2, 12, 6, 7), captain=13, vice=8)
        rows = full_rows(7)
        _dnp(rows, 7, 3)                     # a DEF blanks: MID 12 would give 2 DEF
        _, subs = perform_auto_subs(sq, FakeLoader(rows), 7)
        assert subs == [(3, 6)]              # skips MID 12, takes first DEF

    def test_min_one_fwd(self):
        # 5-4-1 with the lone FWD out; bench 2, 12(MID), 7? -> use 4-5-1
        sq = make_squad(lineup=(1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13),
                        bench=(2, 7, 14, 15), captain=8, vice=9)
        rows = full_rows(7)
        _dnp(rows, 7, 13)
        _, subs = perform_auto_subs(sq, FakeLoader(rows), 7)
        assert subs == [(13, 14)]            # DEF 7 would leave 0 FWD

    def test_non_playing_bench_skipped(self):
        rows = full_rows(7)
        _dnp(rows, 7, 8, 12)                 # MID starter + first outfield sub out
        _, subs = perform_auto_subs(make_squad(), FakeLoader(rows), 7)
        assert subs == [(8, 7)]              # DEF 7 comes on -> 5-3-2

    def test_card_with_zero_minutes_counts_as_played(self):
        rows = full_rows(7)
        rows[(8, 7)] = row(pts=-3, minutes=0, rc=1)
        _, subs = perform_auto_subs(make_squad(), FakeLoader(rows), 7)
        assert subs == []

    def test_multiple_subs_all_valid_formations(self):
        rows = full_rows(7)
        _dnp(rows, 7, 3, 9, 13)
        sq, subs = perform_auto_subs(make_squad(), FakeLoader(rows), 7)
        assert len(subs) == 3
        assert is_valid_formation([sq.players[i] for i in sq.lineup])

    def test_engine_agrees_with_reference_fpl_algorithm(self):
        """Randomised check against a starter-first reference implementation
        (for each non-playing starter in pick order, the highest-priority played
        bench player that keeps a legal formation). Compares realised points."""
        rng = random.Random(2627)
        formations = [(3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1),
                      (5, 2, 3), (5, 3, 2), (5, 4, 1)]
        defs, mids, fwds = [3, 4, 5, 6, 7], [8, 9, 10, 11, 12], [13, 14, 15]
        diffs = 0
        for _ in range(3000):
            d, m, f = rng.choice(formations)
            xi = [1] + defs[:d] + mids[:m] + fwds[:f]
            outf = [e for e in defs + mids + fwds if e not in xi]
            rng.shuffle(outf)
            bench = [2] + outf
            sq = make_squad(lineup=tuple(xi), bench=tuple(bench), captain=xi[1],
                            vice=xi[2])
            rows = {}
            for e in range(1, 16):
                played = rng.random() > 0.35
                rows[(e, 7)] = row(pts=rng.randint(1, 9) if played else 0,
                                   minutes=90 if played else 0)
            ld = FakeLoader(rows)
            new_sq, _ = perform_auto_subs(sq, ld, 7)
            eng_pts = sum(rows[(new_sq.players[i].element_id, 7)]["total_points"]
                          for i in new_sq.lineup)
            # reference
            cur = list(xi)
            used = set()
            for pos_i, e in enumerate(list(cur)):
                if rows[(e, 7)]["minutes"] > 0:
                    continue
                for b in bench:
                    if b in used or rows[(b, 7)]["minutes"] == 0:
                        continue
                    trial = list(cur)
                    trial[pos_i] = b
                    slots = [PlayerSlot(x, POS[x], 50, 50) for x in trial]
                    if is_valid_formation(slots):
                        cur = trial
                        used.add(b)
                        break
            ref_pts = sum(rows[(e, 7)]["total_points"] for e in cur)
            diffs += eng_pts != ref_pts
        assert diffs == 0


class TestCaptaincy:
    def test_captain_doubles(self):
        eng = FPLGameEngine(FakeLoader(full_rows(7, pts={13: 9})))
        _, r = eng.step(make_state(gw=7), EngineAction())
        assert r.captain_points == 9 and not r.captain_failover

    def test_vice_doubles_when_captain_blanks(self):
        rows = full_rows(7, pts={8: 6})
        _dnp(rows, 7, 13)
        _, r = FPLGameEngine(FakeLoader(rows)).step(make_state(gw=7), EngineAction())
        assert r.captain_failover and r.captain_points == 6

    def test_no_multiplier_when_both_blank(self):
        rows = full_rows(7)
        _dnp(rows, 7, 13, 8)
        _, r = FPLGameEngine(FakeLoader(rows)).step(make_state(gw=7), EngineAction())
        assert r.captain_points == 0

    def test_captain_with_card_only_keeps_armband(self):
        rows = full_rows(7, pts={8: 6})
        rows[(13, 7)] = row(pts=-1, minutes=0, yc=1)
        _, r = FPLGameEngine(FakeLoader(rows)).step(make_state(gw=7), EngineAction())
        assert not r.captain_failover and r.captain_points == -1


def _mini_loader(df: pd.DataFrame) -> SeasonDataLoader:
    """Real SeasonDataLoader (DGW aggregation code path) on an in-memory frame."""
    ld = object.__new__(SeasonDataLoader)
    ld.season = "test"
    ld._merged_gw = df.reset_index(drop=True)
    ld._gw_index = {}
    for idx, r in ld._merged_gw.iterrows():
        ld._gw_index.setdefault((int(r["element"]), int(r["GW"])), []).append(idx)
    ld._position_map = dict(POS)
    ld._team_map = {e: e for e in POS}
    return ld


class TestDoubleAndBlankGameweeks:
    def _frame(self, gw=7, dgw=None, bgw=()):
        recs = []
        for e in range(1, 16):
            if e in bgw:
                continue
            fixtures = (dgw or {}).get(e, [(2, 90)])
            for i, (p, m) in enumerate(fixtures):
                recs.append({"element": e, "GW": gw, "fixture": 100 + i,
                             "total_points": p, "minutes": m, "yellow_cards": 0,
                             "red_cards": 0, "value": 50})
        return pd.DataFrame(recs)

    def test_dgw_points_summed_and_captain_doubles_sum(self):
        ld = _mini_loader(self._frame(dgw={13: [(2, 90), (6, 90)]}))
        _, r = FPLGameEngine(ld).step(make_state(gw=7), EngineAction())
        assert r.captain_points == 8
        assert r.gw_points == 10 * 2 + 8 + 8

    def test_dgw_zero_then_ninety_counts_as_played(self):
        ld = _mini_loader(self._frame(dgw={13: [(0, 0), (5, 90)]}))
        _, r = FPLGameEngine(ld).step(make_state(gw=7), EngineAction())
        assert r.auto_subs == [] and r.captain_points == 5

    def test_bgw_player_is_auto_subbed_and_vice_takes_armband(self):
        ld = _mini_loader(self._frame(bgw={13}))
        _, r = FPLGameEngine(ld).step(make_state(gw=7), EngineAction())
        assert r.auto_subs == [(13, 12)]
        assert r.captain_failover and r.captain_points == 2


class TestRealSeasonReplay:
    """Our real picks GW1-5 through the engine vs official entry history."""

    @pytest.mark.parametrize("gw", [1, 2, 3, 4, 5])
    def test_points_and_bench_points_match_history(self, gw):
        live = load_gt(f"live_gw{gw}.json")
        picks = load_gt(f"picks_gw{gw}.json")
        hist = {h["event"]: h for h in load_gt("history.json")["current"]}[gw]
        boot = load_gt("bootstrap_now.json")
        etype = {el["id"]: el["element_type"] for el in boot["elements"]}
        rows = {}
        for el in live["elements"]:
            s = el["stats"]
            rows[(el["id"], gw)] = {"total_points": s["total_points"],
                                    "minutes": s["minutes"],
                                    "yellow_cards": s["yellow_cards"],
                                    "red_cards": s["red_cards"], "value": 50}
        ps = sorted(picks["picks"], key=lambda p: p["position"])
        players = [PlayerSlot(p["element"], Position(etype[p["element"]]), 50, 50)
                   for p in ps]
        squad = Squad(
            players=players,
            lineup=[i for i, p in enumerate(ps) if p["position"] <= 11],
            bench=[i for i, p in enumerate(ps) if p["position"] > 11],
            captain_idx=next(i for i, p in enumerate(ps) if p["is_captain"]),
            vice_captain_idx=next(i for i, p in enumerate(ps) if p["is_vice_captain"]),
        )
        ld = FakeLoader(rows, positions={p.element_id: p.position for p in players})
        state = GameState(squad=squad, bank=0, free_transfers=1, current_gw=gw)
        _, r = FPLGameEngine(ld).step(state, EngineAction())
        assert r.gw_points == hist["points"]
        assert r.bench_points == hist["points_on_bench"]
        assert [(a, b) for a, b in r.auto_subs] == [
            (s["element_out"], s["element_in"]) for s in picks["automatic_subs"]
        ]
