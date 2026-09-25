"""R08 — scripts/gameweek.py: --chip availability checks and chip routing.

Everything external is stubbed (collector, predictions, entry API, auth,
executor); nothing touches the network or REPO/data.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

import fpl_optimizer.live.auth as auth_mod
import fpl_optimizer.live.entry as entry_mod
import fpl_optimizer.live.executor as ex
from fpl_optimizer.engine.state import ChipState, GameState
from fpl_optimizer.live.entry import LiveEntryState
from fpl_optimizer.optimizer.types import OptimizerResult

from tests.test_rules.rules_helpers import ENTRY_ID, REPO, OfflineFPL, load_gt, make_squad


def _load_gameweek():
    spec = importlib.util.spec_from_file_location(
        "gameweek_under_test", REPO / "scripts" / "gameweek.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeServer:
    """Authenticated FPL endpoints (my-team / transfers), in memory."""

    def __init__(self, squad_ids, selling, limit, bank, made=0, chips=None):
        self.chips = chips or []
        self.picks = [{"element": e, "position": i + 1, "selling_price": selling[e],
                       "purchase_price": selling[e]} for i, e in enumerate(squad_ids)]
        self.transfers = {"limit": limit, "made": made, "bank": bank, "cost": 4,
                          "status": "cost", "value": 1000}
        self.transfer_calls = []
        self.lineup_calls = []
        self.my_team_calls = 0

    def get_me(self, auth):
        return {"player": {"entry": ENTRY_ID}}

    def get_my_team(self, auth, entry_id):
        self.my_team_calls += 1
        return {"picks": [dict(p) for p in self.picks],
                "transfers": dict(self.transfers),
                "chips": [dict(c) for c in self.chips]}

    def apply_transfers(self, auth, entry_id, event, transfers, chip=None, confirm=False):
        self.transfer_calls.append({"event": event, "transfers": transfers, "chip": chip,
                                    "confirm": confirm})
        for t in transfers:
            for p in self.picks:
                if p["element"] == t["element_out"]:
                    p["element"] = t["element_in"]

    def apply_lineup(self, auth, entry_id, lineup, bench, captain_id, vice_captain_id,
                     chip=None, element_types=None):
        self.lineup_calls.append({"chip": chip, "captain": captain_id})


class FakeAuth:
    def access_token(self):
        return "tok"

    def headers(self):
        return {}


@pytest.fixture
def run_gw(monkeypatch, bootstrap, capsys, tmp_path):
    """run_gw(argv, entry_state=None, target_gw=6, server_kw=None, transfers=...)"""

    def _run(argv, entry_state=None, target_gw=6, server_kw=None,
             transfers=((221, None),)):
        gwmod = _load_gameweek()
        elements = {el["id"]: el for el in bootstrap["elements"]}
        # never write decision logs into the real data/ directory
        real_log = gwmod._write_decision_log
        monkeypatch.setattr(gwmod, "_write_decision_log",
                            lambda _dd, season, gw, rec: real_log(tmp_path, season, gw, rec))

        class FakeCollector:
            def __init__(self, data_dir=None, season=None):
                pass

            def fetch_bootstrap(self):
                return bootstrap

            def fetch_fixtures(self):
                return []

            def _target_event(self, b):
                return target_gw, None

        monkeypatch.setattr(gwmod, "LiveFPLCollector", FakeCollector)
        monkeypatch.setattr(gwmod, "predict_upcoming_gw",
                            lambda *a, **k: {e: 2.0 for e in elements})

        if entry_state is None:
            stub = OfflineFPL()
            monkeypatch.setattr(entry_mod, "_get", stub)
        else:
            monkeypatch.setattr(entry_mod, "fetch_entry_state",
                                lambda team_id, b: entry_state)

        seen = {}

        def fake_opt(gs, candidates, chip=None, max_transfers=None, **kw):
            seen.update(gs=gs, chip=chip, max_transfers=max_transfers,
                        ft=gs.free_transfers, bank=gs.bank,
                        sell={p.element_id: p.selling_price for p in gs.squad.players})
            sq = [p.element_id for p in gs.squad.players]
            outs, ins = [], []
            for out_id, in_id in transfers:
                if in_id is None:  # pick any same-type player not in the squad
                    et = elements[out_id]["element_type"]
                    in_id = next(e for e, el in elements.items()
                                 if el["element_type"] == et and e not in sq
                                 and e not in ins)
                outs.append(out_id)
                ins.append(in_id)
            new = [ins[outs.index(e)] if e in outs else e for e in sq]
            lineup = [gs.squad.players[i].element_id for i in gs.squad.lineup]
            lineup = [ins[outs.index(e)] if e in outs else e for e in lineup]
            bench = [e for e in new if e not in lineup]
            return OptimizerResult(
                squad_element_ids=new, lineup_element_ids=lineup,
                bench_element_ids=bench, captain_id=lineup[-1],
                vice_captain_id=lineup[-2], transfers_in=ins, transfers_out=outs,
                chip=chip, objective_value=50.0,
                hit_cost=4 * max(0, len(outs) - gs.free_transfers))

        monkeypatch.setattr(gwmod, "optimize_transfers", fake_opt)

        server = None
        if "--apply" in argv:
            es = entry_state
            if es is None:
                es = entry_mod.fetch_entry_state(ENTRY_ID, bootstrap)
            ids = [p.element_id for p in es.game_state.squad.players]
            kw = {"selling": {p.element_id: p.selling_price for p in es.game_state.squad.players},
                  "limit": es.game_state.free_transfers, "bank": es.game_state.bank}
            kw.update(server_kw or {})
            server = FakeServer(ids, **kw)
            monkeypatch.setattr(auth_mod, "FPLAuth", FakeAuth)
            for name in ("get_me", "get_my_team", "apply_transfers", "apply_lineup"):
                monkeypatch.setattr(ex, name, getattr(server, name))

        monkeypatch.setattr(sys, "argv", ["gameweek.py", "--team-id", str(ENTRY_ID),
                                          "--skip-build", *argv])
        gwmod.main()
        out = capsys.readouterr().out
        return out, seen, server

    return _run


def _state(gw: int, chips: ChipState, ft: int = 1) -> LiveEntryState:
    boot = load_gt("bootstrap_now.json")
    picks = sorted(load_gt("picks_gw5.json")["picks"], key=lambda p: p["position"])
    from fpl_optimizer.engine.state import PlayerSlot, Squad
    from fpl_optimizer.utils.constants import Position
    el = {e["id"]: e for e in boot["elements"]}
    players = [PlayerSlot(p["element"], Position(el[p["element"]]["element_type"]),
                          el[p["element"]]["now_cost"], el[p["element"]]["now_cost"])
               for p in picks]
    sq = Squad(players=players, lineup=list(range(11)), bench=[11, 12, 13, 14],
               captain_idx=10, vice_captain_idx=6)
    gs = GameState(squad=sq, bank=2, free_transfers=ft, chips=chips, current_gw=gw)
    return LiveEntryState(game_state=gs, team_name="t", overall_points=0,
                          overall_rank=1, picks_gw=gw - 1, upcoming_gw=gw)


class TestChipAvailabilityGate:
    def test_real_gw6_all_chips_accepted(self, run_gw):
        for chip in ("wildcard", "free_hit", "bench_boost", "triple_captain"):
            out, seen, _ = run_gw(["--chip", chip])
            assert seen["chip"] == chip and seen["ft"] == 2

    def test_used_wildcard_rejected(self, run_gw):
        cs = ChipState()
        cs.use_chip("wildcard", 11)
        out, seen, _ = run_gw(["--chip", "wildcard"], entry_state=_state(12, cs),
                              target_gw=12)
        assert "not available" in out and seen == {}

    def test_fh20_after_fh19_rejected(self, run_gw):
        cs = ChipState()
        cs.use_chip("free_hit", 19)
        cs.expire_first_half()
        out, seen, _ = run_gw(["--chip", "free_hit"], entry_state=_state(20, cs),
                              target_gw=20)
        assert "not available" in out and seen == {}

    def test_used_second_half_tc_rejected(self, run_gw):
        cs = ChipState()
        cs.use_chip("triple_captain", 25)     # second-half TC gone
        cs.expire_first_half()
        out, seen, _ = run_gw(["--chip", "triple_captain"], entry_state=_state(26, cs),
                              target_gw=26)
        assert "not available" in out and seen == {}


class TestChipRouting:
    @pytest.mark.parametrize("chip", ["wildcard", "free_hit"])
    def test_wc_fh_submitted_with_transfers_not_lineup(self, run_gw, chip):
        out, seen, srv = run_gw(["--chip", chip, "--apply", "--yes"],
                                transfers=((221, None), (306, None)))
        assert seen["chip"] == chip
        assert len(srv.transfer_calls) == 1
        assert srv.transfer_calls[0]["chip"] == chip
        assert srv.transfer_calls[0]["confirm"] is True
        assert srv.lineup_calls and srv.lineup_calls[-1]["chip"] is None

    @pytest.mark.parametrize("chip", ["bench_boost", "triple_captain"])
    def test_bb_tc_submitted_with_lineup(self, run_gw, chip):
        out, seen, srv = run_gw(["--chip", chip, "--apply", "--yes"])
        assert srv.transfer_calls[0]["chip"] is None
        assert srv.lineup_calls[-1]["chip"] == chip

    def test_transfer_pairs_same_position_and_auth_selling_price(self, run_gw, bootstrap):
        el = {e["id"]: e for e in bootstrap["elements"]}
        out, seen, srv = run_gw(["--apply", "--yes"],
                                transfers=((221, None), (586, None)),
                                server_kw={"selling": {**{e: 50 for e in range(1, 900)},
                                                       221: 44, 586: 40}})
        for t in srv.transfer_calls[0]["transfers"]:
            assert el[t["element_in"]]["element_type"] == el[t["element_out"]]["element_type"]
            assert t["purchase_price"] == el[t["element_in"]]["now_cost"]
        sells = {t["element_out"]: t["selling_price"] for t in srv.transfer_calls[0]["transfers"]}
        assert sells == {221: 44, 586: 40}


class TestAuthoritativeStateWhenAuthenticated:
    @pytest.mark.bug("R-05")
    def test_optimizer_uses_my_team_ft_bank_and_selling_prices(self, run_gw):
        """With --apply the authenticated my-team state (FT limit, bank, selling
        prices) is fetched anyway — but only AFTER the MILP ran on the public
        reconstruction. If they differ (FT top-up, price-rise rounding, a
        reconstruction bug) the plan's hits/budget are wrong."""
        out, seen, srv = run_gw(["--apply"],
                                server_kw={"limit": 1, "bank": 7,
                                           "selling": {**{e: 50 for e in range(1, 900)},
                                                       411: 150}})
        assert seen["ft"] == 1
        assert seen["bank"] == 7
        assert seen["sell"][411] == 150


PENDING_BB = [{"id": 4, "name": "bboost", "status_for_entry": "active", "is_pending": True,
               "played_by_entry": [], "number": 1, "start_event": 1, "stop_event": 19,
               "chip_type": "team"}]


class TestChipAlreadyActiveOnSite:
    """my-team POST always carries "chip"; posting null (or another chip) for a GW
    where a BB/TC is already active on the site would cancel/replace it."""

    @pytest.mark.bug("R-09")
    def test_pending_bb_not_cancelled_by_plain_apply(self, run_gw):
        out, seen, srv = run_gw(["--apply", "--yes"], server_kw={"chips": PENDING_BB})
        assert all(c["chip"] == "bench_boost" for c in srv.lineup_calls)

    @pytest.mark.bug("R-09")
    def test_second_chip_same_gw_not_submitted(self, run_gw):
        out, seen, srv = run_gw(["--chip", "triple_captain", "--apply", "--yes"],
                                server_kw={"chips": PENDING_BB})
        assert not any(c["chip"] == "triple_captain" for c in srv.lineup_calls)

    def test_matching_chip_goes_through(self, run_gw):
        out, seen, srv = run_gw(["--chip", "bench_boost", "--apply", "--yes"],
                                server_kw={"chips": PENDING_BB})
        assert srv.lineup_calls and srv.lineup_calls[-1]["chip"] == "bench_boost"
