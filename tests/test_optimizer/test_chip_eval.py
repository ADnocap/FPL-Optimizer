"""Tests for the chip evaluation (optimizer/chip_eval.py, live/chip_inputs.py,
scripts/chip_eval.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl_optimizer.engine.state import ChipState, GameState, PlayerSlot, Squad
from fpl_optimizer.optimizer import chip_eval as ce
from fpl_optimizer.optimizer.horizon_optimizer import (
    GWPlan,
    HorizonCandidate,
    HorizonConfig,
)
from fpl_optimizer.utils.constants import Position

GK, DEF, MID, FWD = Position.GK, Position.DEF, Position.MID, Position.FWD
REPO = Path(__file__).resolve().parent.parent.parent
GT = REPO / "tests" / "test_data" / "rules_2026_27"
TC, BB, FH, WC = "triple_captain", "bench_boost", "free_hit", "wildcard"


def _gt(name):
    return json.loads((GT / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# chip windows
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def windows():
    return ce.official_chip_windows(_gt("rules_2026-27.json")["chips"])


def _gws(rem, chip, half):
    got = [rc for rc in rem if rc.chip == chip and rc.window.half == half]
    return got[0].gws if got else None


class TestChipWindows:
    def test_official_windows(self, windows):
        w = {(x.chip, x.half): (x.start, x.stop) for x in windows}
        assert len(windows) == 8
        assert w[(WC, 1)] == (2, 19) and w[(FH, 1)] == (2, 19)
        assert w[(BB, 1)] == (1, 19) and w[(TC, 1)] == (1, 19)
        for chip in (WC, FH, BB, TC):
            assert w[(chip, 2)] == (20, 38)

    def test_all_chips_remaining_at_gw6(self, windows):
        rem = ce.remaining_chips(ChipState(), windows, 6)
        assert len(rem) == 8
        assert _gws(rem, TC, 1) == tuple(range(6, 20))
        assert _gws(rem, WC, 2) == tuple(range(20, 39))

    def test_wc_fh_not_playable_gw1(self, windows):
        rem = ce.remaining_chips(ChipState(), windows, 1)
        assert _gws(rem, WC, 1)[0] == 2 and _gws(rem, FH, 1)[0] == 2
        assert _gws(rem, TC, 1)[0] == 1 and _gws(rem, BB, 1)[0] == 1

    def test_used_chip_consumes_only_its_half(self, windows):
        cs = ChipState()
        cs.use_chip(TC, 7)
        rem = ce.remaining_chips(cs, windows, 8)
        assert _gws(rem, TC, 1) is None
        assert _gws(rem, TC, 2) == tuple(range(20, 39))

    def test_fh19_blocks_fh20(self, windows):
        cs = ChipState()
        cs.use_chip(FH, 19)
        rem = ce.remaining_chips(cs, windows, 20)
        fh2 = _gws(rem, FH, 2)
        assert 20 not in fh2 and fh2[0] == 21
        # the other chips' GW20 is unaffected
        assert _gws(rem, WC, 2)[0] == 20

    def test_fh18_does_not_block_fh20(self, windows):
        cs = ChipState()
        cs.use_chip(FH, 18)
        rem = ce.remaining_chips(cs, windows, 20)
        assert _gws(rem, FH, 2)[0] == 20

    def test_first_half_expiry(self, windows):
        rem19 = ce.remaining_chips(ChipState(), windows, 19)
        assert _gws(rem19, BB, 1) == (19,)
        for cs in (ChipState(), ChipState()):
            rem = ce.remaining_chips(cs, windows, 20)
            assert all(rc.window.half == 2 for rc in rem) and len(rem) == 4
        cs = ChipState()
        cs.expire_first_half()  # what fetch_entry_state does from GW20
        assert all(rc.window.half == 2 for rc in ce.remaining_chips(cs, windows, 20))

    def test_real_entry_gw6_offline(self, windows):
        from fpl_optimizer.live.chip_inputs import offline_entry_state

        es = offline_entry_state(GT, _gt("bootstrap_now.json"))
        assert es.upcoming_gw == 6
        assert es.game_state.free_transfers == 2 and es.game_state.bank == 2
        rem = ce.remaining_chips(es.game_state.chips, windows, es.upcoming_gw)
        assert {(rc.chip, rc.window.half) for rc in rem} == {
            (c, h) for c in (TC, BB, FH, WC) for h in (1, 2)}


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_half_of_the_chip_gw(self):
        assert ce.calib("tc", 0, 7) == pytest.approx(0.93)
        assert ce.calib("tc", 0, 25) == pytest.approx(0.56)
        assert ce.calib("tc", 0, 19) == pytest.approx(0.93)
        assert ce.calib("tc", 0, 20) == pytest.approx(0.56)
        assert ce.calib("wc", 5, 11) == pytest.approx(0.22)
        assert ce.calib("wc", 0, 30) == pytest.approx(0.06)
        assert ce.calib("bb", 2, 12) == pytest.approx(0.30)
        assert ce.calib("bb", 2, 30) == pytest.approx(1.35)

    def test_k_buckets_and_interpolation(self):
        assert ce.calib("tc", 1, 8) == ce.calib("tc", 2, 9) == pytest.approx(0.66)
        assert ce.calib("tc", 3, 10) == ce.calib("tc", 13, 19) == pytest.approx(0.54)
        assert ce.calib("fh", 3, 12) == pytest.approx(0.56)
        assert ce.calib("fh", 1, 12) == pytest.approx(0.58 - 0.02 / 3)  # linear 0 -> 3
        assert ce.calib("fh", 12, 18) == pytest.approx(0.46)  # flat beyond k=6
        assert ce.calib("fh", 0, 20) == pytest.approx(0.97)


# ---------------------------------------------------------------------------
# Monte Carlo: TC failover, BB net of auto-subs
# ---------------------------------------------------------------------------

POS15 = np.array([1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4])
# 4-4-2: GK 0 | DEF 2-5 | MID 7-10 | FWD 12,13; bench GK 1, then 6 (DEF), 11 (MID), 14 (FWD)
XI = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]
BENCH = [1, 6, 11, 14]


def _sq(xp, p, cap=12, vice=7, xi=XI, bench=BENCH):
    return ce.Squad15(list(range(100, 115)), POS15.copy(), np.asarray(xp, float),
                      np.asarray(p, float), xi=list(xi), bench=list(bench), cap=cap, vice=vice)


class TestTripleCaptain:
    def test_realised_failover(self):
        sq = _sq(np.ones(15), np.ones(15))
        pts = np.arange(15, dtype=float) + 1.0  # cap (12) scores 13, vice (7) scores 8
        played = np.ones(15, bool)
        assert ce.simulate(sq, played=played, points=pts)["tc_gain"] == 13.0
        played[12] = False  # captain blanks -> the vice's points are tripled... doubled+1
        r = ce.simulate(sq, played=played, points=pts)
        assert r["tc_gain"] == 8.0
        played[7] = False
        assert ce.simulate(sq, played=played, points=pts)["tc_gain"] == 0.0

    def test_expected_effective_captain(self):
        xp = np.full(15, 2.0)
        p = np.full(15, 0.95)
        xp[12], p[12] = 5.0, 0.5    # captain: 10 pts when he plays, 50% to play
        xp[7], p[7] = 6.0, 1.0      # vice: 6 pts, always plays
        r = ce.simulate(_sq(xp, p), n_sims=200_000, seed=1)
        assert r["tc_gain"] == pytest.approx(0.5 * 10 + 0.5 * 1.0 * 6, abs=0.05)
        # normal = XI + auto-subs + effective captain once more
        assert r["normal"] > r["tc_gain"]


class TestBenchBoost:
    def test_realised_net_of_autosubs(self):
        sq = _sq(np.ones(15), np.ones(15))
        pts = np.zeros(15)
        pts[1], pts[6], pts[11], pts[14] = 1.0, 5.0, 3.0, 2.0
        played = np.ones(15, bool)
        played[2] = False    # a DEF starter misses -> bench DEF 6 comes on
        played[14] = False   # bench FWD did not play
        r = ce.simulate(sq, played=played, points=pts)
        assert r["autosub"] == 5.0
        # BB adds the bench players who played and were NOT auto-subbed: GK 1 + MID 11
        assert r["bb_gain"] == 1.0 + 3.0
        assert r["bench_sum"] == 1.0 + 5.0 + 3.0

    def test_autosub_respects_formation(self):
        # 3-5-2 XI: DEF 2,3,4 | MID 7-11 | FWD 12,13; bench GK 1, MID... first bench is a FWD
        xi = [0, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]
        bench = [1, 14, 5, 6]          # FWD 14 first, then DEF 5, DEF 6
        sq = _sq(np.ones(15), np.ones(15), xi=xi, bench=bench)
        pts = np.zeros(15)
        pts[14], pts[5], pts[6] = 4.0, 3.0, 1.0
        played = np.ones(15, bool)
        played[2] = False              # DEF out: a FWD can't come on (only 2 DEF left)
        r = ce.simulate(sq, played=played, points=pts)
        assert r["autosub"] == 3.0     # DEF 5 came on, FWD 14 stayed on the bench
        assert r["bb_gain"] == 4.0 + 1.0

    def test_expected_bb_is_bench_minus_autosub(self):
        xp = np.linspace(1, 6, 15)
        p = np.full(15, 0.8)
        r = ce.simulate(_sq(xp, p), n_sims=100_000, seed=3)
        assert r["bb_gain"] == pytest.approx(r["bench_sum"] - r["autosub"], abs=1e-9)
        # everyone plays for sure -> no auto-subs: BB = the bench's xP
        r1 = ce.simulate(_sq(xp, np.ones(15)), n_sims=2000, seed=3)
        assert r1["autosub"] == 0.0
        assert r1["bb_gain"] == pytest.approx(xp[BENCH].sum())


# ---------------------------------------------------------------------------
# fixtures / availability
# ---------------------------------------------------------------------------


class TestFixtureMap:
    def test_bgw_dgw_unscheduled(self):
        fx = [
            {"id": 1, "event": 10, "team_h": 1, "team_a": 2},
            {"id": 2, "event": 10, "team_h": 3, "team_a": 1},   # team 1 DGW
            {"id": 3, "event": 11, "team_h": 2, "team_a": 3},
            {"id": 4, "event": 11, "team_h": 4, "team_a": 1},
            {"id": 5, "event": None, "team_h": 2, "team_a": 4},  # postponed
        ]
        cal = ce.fixture_calendar(fx, [1, 2, 3, 4], [10, 11])
        assert cal["per_gw"][10] == {"n_fixtures": 2, "blank_teams": [4], "double_teams": [1]}
        assert cal["per_gw"][11] == {"n_fixtures": 2, "blank_teams": [], "double_teams": []}
        assert cal["unscheduled"] == [(2, 4)]
        assert cal["unscheduled_by_team"] == {2: 1, 4: 1}

    def test_moves_paired_by_position(self):
        pos = {1: 1, 2: 3, 3: 1, 4: 3, 5: 2, 6: 4}
        assert ce.pair_moves([1, 2], [4, 3], pos) == [(1, 3), (2, 4)]
        assert ce.pair_moves([5], [6], pos) == [(5, 6)]  # shape change: leftover pair

    def test_early_wc_trigger(self):
        boot = _gt("bootstrap_now.json")  # James and Joao Pedro carry 75% flags
        squad = [e["id"] for e in boot["elements"]]
        flags = ce.availability_flags(boot, squad)
        assert set(flags) == {142, 165}
        assert flags[165].startswith("doubtful 75%")
        assert ce.early_wc_trigger(flags, [142, 165, 411]) == (2, False)
        assert ce.early_wc_trigger({**flags, 411: "injured"}, [142, 165, 411]) == (3, True)


# ---------------------------------------------------------------------------
# planning: state advance, WC/FH values, end-to-end
# ---------------------------------------------------------------------------

_BASE = [
    (1, GK, 4.0), (2, GK, 1.0),
    (3, DEF, 4.0), (4, DEF, 4.0), (5, DEF, 4.0), (6, DEF, 3.5), (7, DEF, 1.0),
    (8, MID, 5.0), (9, MID, 5.0), (10, MID, 4.5), (11, MID, 4.0), (12, MID, 1.0),
    (13, FWD, 5.0), (14, FWD, 4.5), (15, FWD, 1.0),
]
# market: three clear upgrades (+3 each over the weakest starters)
_MARKET = [(101, DEF, 6.5), (102, MID, 7.0), (103, FWD, 7.0), (104, GK, 1.5)]


def _pool(h: int) -> list[HorizonCandidate]:
    return [HorizonCandidate(e, pos, 50, e, (x,) * h, (0.9,) * h)
            for e, pos, x in _BASE + _MARKET]


def _state(gw=6, ft=1, bank=0, chips=None) -> GameState:
    players = [PlayerSlot(e, pos, 50, 50) for e, pos, _ in _BASE]
    return GameState(squad=Squad(players, list(range(11)), list(range(11, 15)), 7, 8),
                     bank=bank, free_transfers=ft, chips=chips or ChipState(), current_gw=gw)


def _ctx(gws, ft=1, **cfg) -> ce.ChipContext:
    cfg.setdefault("hit_margin", 0.0)
    return ce.ChipContext(_state(gws[0], ft), _pool(len(gws)), list(gws),
                          HorizonConfig(**cfg), n_sims=2000)


class TestAdvanceState:
    def _plan(self, chip=None, ins=(), outs=(), gw=6):
        squad = [e for e, _, _ in _BASE if e not in outs] + list(ins)
        pos = {e: p for e, p, _ in _BASE + _MARKET}
        gk = [e for e in squad if pos[e] == GK]
        of = [e for e in squad if pos[e] != GK]
        lineup, bench = [gk[0]] + of[:10], [gk[1]] + of[10:]
        return GWPlan(gw=gw, chip=chip, free_transfers=2, transfers_in=list(ins),
                      transfers_out=list(outs), hits=0, squad=sorted(squad), lineup=lineup,
                      bench=bench, captain_id=lineup[1], vice_captain_id=lineup[2],
                      expected_points=50.0, bank_after=7)

    def test_transfer_week(self):
        ctx = _ctx([6, 7])
        st = _state(ft=2)
        new = ce.advance_state(st, self._plan(ins=[102], outs=[12]), {102: 57}, ctx.pos_of)
        assert new.current_gw == 7 and new.free_transfers == 2 and new.bank == 7
        slot = {p.element_id: p for p in new.squad.players}
        assert 12 not in slot and slot[102].selling_price == 57
        assert slot[1].selling_price == 50
        assert len(new.squad.lineup) == 11 and len(new.squad.bench) == 4

    def test_chip_weeks_carry_ft_and_fh_reverts(self):
        ctx = _ctx([6, 7])
        st = _state(ft=2)
        wc = ce.advance_state(st, self._plan(WC, ins=[101, 102], outs=[7, 12]),
                              {101: 50, 102: 50}, ctx.pos_of)
        assert wc.free_transfers == 2 and not wc.chips.is_available(WC, 7)
        fh = ce.advance_state(st, self._plan(FH, ins=[101], outs=[7]), {101: 50}, ctx.pos_of)
        assert fh.free_transfers == 2
        assert [p.element_id for p in fh.squad.players] == [p.element_id for p in st.squad.players]
        assert fh.bank == st.bank


class TestChipPlanValues:
    @pytest.mark.parametrize("chip", [WC, FH])
    def test_value_nonnegative_single_gw(self, chip):
        """One GW, 1 FT: the chip can replicate any no-chip move set -> >= 0,
        and with three +3 upgrades (a hit costs 4) it takes all of them."""
        ctx = _ctx([6])
        cmp = ce.compare_plans(ctx, ctx.state, [6], {6: chip})
        assert cmp.applied
        assert cmp.milp_gain >= -1e-6
        assert cmp.mc_gain > 2.0
        assert len(cmp.with_chip.plan[0].transfers_in) >= 3

    def test_wc_value_nonnegative_with_full_ft_bank(self):
        """5 FTs banked: the WC week's carried FT count equals what the
        no-chip plan can reach, so the WC plan dominates over the horizon."""
        ctx = _ctx([6, 7, 8], ft=5)
        cmp = ce.compare_plans(ctx, ctx.state, [6, 7, 8], {6: WC})
        assert cmp.applied and cmp.milp_gain >= -1e-6

    def test_unavailable_chip_is_flagged(self):
        ctx = _ctx([6])
        cs = ChipState()
        cs.use_chip(WC, 3)
        st = _state(chips=cs)
        cmp = ce.compare_plans(ctx, st, [6], {6: WC})
        assert not cmp.applied and cmp.milp_gain == pytest.approx(0.0, abs=1e-3)


class TestEvaluateEndToEnd:
    def test_small_world(self, windows):
        ctx = _ctx([6, 7, 8, 9])
        rem = ce.remaining_chips(ctx.state.chips, windows, 6)
        res = ce.evaluate_chips(ctx, rem, last_gw=7, wc_h=2, step_h=2, combos=[(6, 7)])
        assert sorted(res["trajectory"]) == [6, 7]
        assert sorted(res["tc_bb"]) == [6, 7] and sorted(res["fh"]) == [6, 7]
        assert sorted(res["wc"]) == [6, 7]
        r6 = res["tc_bb"][6]
        # the planned GW6 squad bought the upgrades the FT allows; TC = the captain's E[pts]
        assert r6["tc_gain"] == pytest.approx(r6["captain_xp"], rel=0.15)
        assert r6["tc_cal"] == pytest.approx(r6["tc_gain"] * 0.93)
        assert res["wc"][6]["applied"] and res["wc"][6]["wc_gain_milp"] >= -1e-6
        assert res["combos"]["WC6+BB7"]["applied"]
        names = {e: f"p{e}" for e, _, _ in _BASE + _MARKET}
        cal = ce.fixture_calendar([], [1, 2], list(range(6, 39)))
        txt = ce.format_report(ctx, res, rem, cal, names, {1: "AAA", 2: "BBB"}, {}, [1, 3],
                               last_gw=7)
        assert "Chip evaluation as of GW6" in txt and "WC at GW" in txt
        assert "First-half chips left: TC, BB, FH, WC" in txt

    def test_calibration_uses_the_chip_gws_half(self, windows):
        ctx = _ctx([19, 20, 21])
        rem = ce.remaining_chips(ctx.state.chips, windows, 19)
        res = ce.evaluate_chips(ctx, rem, chips={TC}, last_gw=20, step_h=2)
        r19, r20 = res["tc_bb"][19], res["tc_bb"][20]
        assert r19["tc_cal"] == pytest.approx(r19["tc_gain"] * 0.93)   # GW19: half 1, k=0
        assert r20["tc_cal"] == pytest.approx(r20["tc_gain"] * 0.68)   # GW20: half 2, k=1
        assert not res["wc"] and not res["fh"]


# ---------------------------------------------------------------------------
# DGW path: horizon blend, not a sum of single-fixture predictions
# ---------------------------------------------------------------------------


class _ConstPredictor:
    _feature_names = ["was_home"]

    def predict(self, df):
        return np.full(len(df), 3.0)


def _raw_season(tmp_path: Path, season: str) -> Path:
    raw = tmp_path / "raw" / season
    (raw / "gws").mkdir(parents=True)
    fixtures = pd.DataFrame([
        {"id": 1, "event": 1, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 3},
        {"id": 2, "event": 1, "team_h": 3, "team_a": 4, "team_h_difficulty": 3, "team_a_difficulty": 3},
        {"id": 3, "event": 2, "team_h": 1, "team_a": 3, "team_h_difficulty": 3, "team_a_difficulty": 3},
        {"id": 4, "event": 2, "team_h": 4, "team_a": 1, "team_h_difficulty": 3, "team_a_difficulty": 3},
        {"id": 5, "event": 2, "team_h": 2, "team_a": 3, "team_h_difficulty": 3, "team_a_difficulty": 3},
    ])  # GW2: team 1 plays twice (DGW), team 3 twice, team 2 and 4 once
    fixtures.to_csv(raw / "fixtures.csv", index=False)
    pd.DataFrame([{"id": t, "strength": 3, "strength_attack_home": 1100,
                   "strength_attack_away": 1100, "strength_defence_home": 1100,
                   "strength_defence_away": 1100} for t in (1, 2, 3, 4)]).to_csv(
        raw / "teams.csv", index=False)
    rows = []
    for fid, h, a in [(1, 1, 2), (2, 3, 4)]:
        for eid, team in ((11, 1), (21, 2), (31, 3), (41, 4)):
            if team in (h, a):
                rows.append({"element": eid, "GW": 1, "fixture": fid, "was_home": team == h,
                             "total_points": 2, "minutes": 90, "goals_conceded": 1})
    pd.DataFrame(rows).to_csv(raw / "gws" / "merged_gw.csv", index=False)
    return raw


def _features(gw: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "element": eid, "GW": gw, "code": 1000 + eid, "position": "MID",
        "pts_rolling_5": 3.0, "playing_prob": 1.0, "mins_rolling_3": 90.0,
        "was_home": 1.0, "fdr": 0.6, "is_dgw": 0.0, "fixture_offset": 0.0,
        "synthetic_ep": 3.0, "gw_phase": gw / 38.0,
    } for eid in (11, 21, 31, 41)])


class TestDGWBlend:
    def test_horizon_predictions_use_the_blend(self, tmp_path, monkeypatch):
        import fpl_optimizer.live.predict as live_predict
        from fpl_optimizer.live.chip_inputs import DGW_MODE, horizon_predictions

        assert DGW_MODE == "blend"
        season = "2099-00"
        _raw_season(tmp_path, season)
        feats = _features(1)  # as-of GW1 rows; GW2 (k=1) holds the double

        class _Pipe:
            def __init__(self, *a, **k):
                pass

            def build(self):
                return feats.copy()

        monkeypatch.setattr(live_predict, "FeaturePipeline", _Pipe)
        monkeypatch.setattr(live_predict, "IDResolver", lambda *a, **k: None)
        preds, rows = horizon_predictions(tmp_path, tmp_path, season, t=1, horizon=2,
                                          predictor=_ConstPredictor())
        g2 = preds[preds["GW"] == 2]
        by = dict(zip(g2["element"], g2["pred"]))
        # single fixture: 3.0; DGW: mean of one aggregated row (3.0) and the
        # per-fixture sum (6.0) = 4.5 -- NOT the prototype's 6.0 sum
        assert by[21] == pytest.approx(3.0) and by[41] == pytest.approx(3.0)
        assert by[11] == pytest.approx(4.5) and by[31] == pytest.approx(4.5)
        assert len(rows) == len(feats)

        # ... and the planner pool / chip context carry the blended value
        from fpl_optimizer.live.pool import build_live_horizon_candidates

        boot = {"elements": [{"id": e, "element_type": 3, "now_cost": 50, "team": t,
                              "status": "a", "chance_of_playing_next_round": None}
                             for e, t in ((11, 1), (21, 2), (31, 3), (41, 4))]}
        cands = build_live_horizon_candidates(boot, preds, [1, 2])
        assert {c.element_id: c.xpts[1] for c in cands}[11] == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# live inputs / script guards
# ---------------------------------------------------------------------------


class _Pred:
    def __init__(self, feats):
        self._feature_names = feats


class TestLiveInputs:
    def test_leaky_features_exact_name(self):
        from fpl_optimizer.live.chip_inputs import leaky_features

        assert leaky_features(_Pred(["fpl_xp_lag", "pts_rolling_5"])) == []
        assert leaky_features(_Pred(["fpl_xp", "fpl_xp_lag"])) == ["fpl_xp"]

    def test_play_model_decays_with_lead(self):
        from fpl_optimizer.prediction.horizon import p_play_from_playing_prob
        from fpl_optimizer.prediction.play_model import apply_play_model

        class _PM:
            def predict(self, df):
                return np.full(len(df), 0.5)

        feats = _features(6)
        preds = pd.DataFrame({"element": [11, 11, 99], "GW": [6, 8, 6], "k": [0, 2, 0],
                              "pred": [3.0, 3.0, 1.0], "p_play": [0.9, 0.8, 0.42]})
        out = apply_play_model(_PM(), feats, preds, 6)
        ratio = p_play_from_playing_prob(1.0, 2) / p_play_from_playing_prob(1.0, 0)
        assert list(out["p_play"]) == pytest.approx([0.5, 0.5 * ratio, 0.42])


def _load_script():
    spec = importlib.util.spec_from_file_location("chip_eval_script", REPO / "scripts" / "chip_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestScript:
    def test_default_model_dir_is_gameweeks(self):
        spec = importlib.util.spec_from_file_location("gw_under_test", REPO / "scripts" / "gameweek.py")
        gw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gw)
        assert _load_script().gameweek_default_model_dir() == Path(gw.DEFAULT_MODEL_DIR)

    def _args(self, tmp_path, *extra):
        fx = tmp_path / "fixtures.json"
        fx.write_text("[]", encoding="utf-8")
        return ["--entry-dir", str(GT), "--bootstrap", str(GT / "bootstrap_now.json"),
                "--fixtures", str(fx), "--model-dir", str(tmp_path),
                "--data-dir", str(tmp_path / "data"), *extra]

    def test_refuses_leaky_model(self, tmp_path, monkeypatch, capsys):
        import fpl_optimizer.live.chip_inputs as ci

        monkeypatch.setattr(ci, "load_predictor", lambda d: _Pred(["fpl_xp", "value"]))
        called = []
        monkeypatch.setattr(ci, "horizon_predictions", lambda *a, **k: called.append(1))
        assert _load_script().main(self._args(tmp_path)) == 3
        out = capsys.readouterr().out
        assert "LEAKY MODEL" in out and "REFUSING" in out and not called

    def test_never_writes_into_data(self, tmp_path):
        rc = _load_script().main(self._args(tmp_path, "--out",
                                            str(tmp_path / "data" / "x.json")))
        assert rc == 2
        assert not (tmp_path / "data" / "x.json").exists()
