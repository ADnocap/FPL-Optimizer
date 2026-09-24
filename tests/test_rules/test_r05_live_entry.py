"""R05 — live/entry.py: FT reconstruction, prices, chips, Free Hit revert.

Uses our real public-API ground truth (entry 8737706, GW1-5) through an offline
stub of live.entry._get, plus synthetic in-memory scenarios for GW6-38 events
(WC, FH, FH19/FH20, late joiner) that have not happened yet.
"""

from __future__ import annotations

import copy

import pytest

from fpl_optimizer.live.entry import _compute_free_transfers, fetch_entry_state
from fpl_optimizer.utils.constants import TRANSFER_HIT_COST

from tests.test_rules.rules_helpers import ENTRY_ID, load_gt

HAALAND, BRUNO, CALAFIORI, STACH, MENDY, JOAO = 411, 426, 8, 335, 586, 165


def _hist(transfers_by_gw: dict[int, int], chips: dict[int, str] | None = None,
          first: int = 1, last: int | None = None) -> dict:
    last = last or max(list(transfers_by_gw) + [first])
    return {
        "current": [{"event": g, "event_transfers": transfers_by_gw.get(g, 0),
                     "event_transfers_cost": 0} for g in range(first, last + 1)],
        "chips": [{"name": n, "event": g, "time": ""} for g, n in (chips or {}).items()],
    }


# ---------------------------------------------------------------------------
# Real ground truth
# ---------------------------------------------------------------------------
class TestRealEntryGW6:
    def test_state_entering_gw6(self, offline_fpl, bootstrap):
        offline_fpl()
        es = fetch_entry_state(ENTRY_ID, bootstrap)
        gs = es.game_state
        assert es.upcoming_gw == 6 and es.picks_gw == 5 and gs.current_gw == 6
        assert gs.free_transfers == 2          # FT rolled in GW5
        assert gs.bank == 2
        picks = load_gt("picks_gw5.json")["picks"]
        assert [p.element_id for p in gs.squad.players] == [
            p["element"] for p in sorted(picks, key=lambda p: p["position"])]
        assert [gs.squad.players[i].element_id for i in gs.squad.bench] == [301, 306, 221, 586]
        assert gs.squad.players[gs.squad.captain_idx].element_id == HAALAND
        assert gs.squad.players[gs.squad.vice_captain_idx].element_id == BRUNO
        for chip in ("wildcard", "free_hit", "bench_boost", "triple_captain"):
            for gw in (6, 7, 11, 12, 18, 19, 20, 38):
                assert gs.chips.is_available(chip, gw), (chip, gw)

    def test_purchase_and_selling_prices(self, offline_fpl, bootstrap):
        offline_fpl()
        gs = fetch_entry_state(ENTRY_ID, bootstrap).game_state
        el = {e["id"]: e for e in bootstrap["elements"]}
        by = {p.element_id: p for p in gs.squad.players}
        # bought via transfers.json
        assert by[CALAFIORI].purchase_price == 57
        assert by[STACH].purchase_price == 60
        assert by[MENDY].purchase_price == 41
        # held since GW1: season start price
        for eid, p in by.items():
            if eid in (CALAFIORI, STACH, MENDY):
                continue
            assert p.purchase_price == el[eid]["now_cost"] - el[eid]["cost_change_start"]
        for p in by.values():
            now = el[p.element_id]["now_cost"]
            exp = p.purchase_price + (now - p.purchase_price) // 2 \
                if now > p.purchase_price else now
            assert p.selling_price == exp
        assert by[HAALAND].selling_price == 155   # 155 -> 156: +0.1 not realised
        assert by[JOAO].selling_price == 76       # 75 -> 77: half of +0.2

    def test_bank_matches_transfer_ledger(self):
        """bank = 1000 - GW1 squad cost + sum(sell - buy) over transfers."""
        tr = load_gt("transfers.json")
        hist = {h["event"]: h for h in load_gt("history.json")["current"]}
        gw1_cost = hist[1]["value"] - hist[1]["bank"]
        assert gw1_cost == 1000
        bank = 1000 - gw1_cost + sum(t["element_out_cost"] - t["element_in_cost"] for t in tr)
        assert bank == hist[5]["bank"] == 2

    @pytest.mark.parametrize("upcoming,expected", [(2, 1), (3, 2), (4, 1), (5, 1), (6, 2)])
    def test_ft_trajectory(self, upcoming, expected):
        assert _compute_free_transfers(load_gt("history.json"), {}, upcoming) == expected

    def test_ft_simulation_reproduces_official_hit_charges(self):
        hist = load_gt("history.json")
        for h in hist["current"]:
            gw = h["event"]
            if gw == 1:
                continue
            ft = _compute_free_transfers(hist, {}, gw)
            assert TRANSFER_HIT_COST * max(0, h["event_transfers"] - ft) == \
                h["event_transfers_cost"], gw


# ---------------------------------------------------------------------------
# FT rules for the rest of the season
# ---------------------------------------------------------------------------
class TestFreeTransferSimulation:
    def test_cap_five(self):
        assert _compute_free_transfers(_hist({}, last=12), {}, 13) == 5

    def test_hits_do_not_go_negative(self):
        assert _compute_free_transfers(_hist({2: 4}), {}, 3) == 1

    @pytest.mark.parametrize("ft_before", [1, 2, 3, 5])
    def test_wildcard_week_carries_count_unchanged(self, ft_before):
        # reach ft_before entering GW11 by rolling, then WC with 13 transfers
        roll_until = 10
        start_rolls = ft_before - 1
        tb = {g: 0 for g in range(2, roll_until + 1)}
        # burn transfers so exactly ft_before are available in GW11
        tb[10 - start_rolls] = 5
        h = _hist({**tb, 11: 13}, last=11)
        assert _compute_free_transfers(h, {}, 11) == ft_before
        assert _compute_free_transfers(h, {11: "wildcard"}, 12) == ft_before

    def test_free_hit_week_carries_count_unchanged(self):
        h = _hist({2: 0, 3: 0, 4: 0, 5: 0, 18: 15}, last=18)
        assert _compute_free_transfers(h, {}, 18) == 5
        assert _compute_free_transfers(h, {18: "free_hit"}, 19) == 5

    def test_accrual_resumes_after_chip_week(self):
        h = _hist({10: 5, 11: 12, 12: 0}, last=12)
        ft11 = _compute_free_transfers(h, {}, 11)
        assert _compute_free_transfers(h, {11: "wildcard"}, 13) == ft11 + 1

    @pytest.mark.bug("R-04")
    def test_late_joiner_first_gw_resets_to_one(self):
        """A team whose first GW is 3 has unlimited changes before GW3's deadline
        and exactly 1 FT for GW4. The simulation assumes GW1 and gives 3."""
        h = _hist({}, first=3, last=3)
        assert _compute_free_transfers(h, {}, 4) == 1


# ---------------------------------------------------------------------------
# Chips parsing and FH revert (synthetic future weeks on top of our real data)
# ---------------------------------------------------------------------------
def _future_overrides(fh_gw: int | None = None, extra_chips=None,
                      extra_transfers=None, fh_squad_swap=None):
    """Clone GW5 as the base squad for GW6..; optionally play FH in fh_gw."""
    hist = load_gt("history.json")
    entry = load_gt("entry.json")
    picks5 = load_gt("picks_gw5.json")
    last = fh_gw or 6
    ov = {}
    for g in range(6, last + 1):
        p = copy.deepcopy(picks5)
        p["entry_history"] = dict(p["entry_history"], event=g, bank=2 + g)
        hist["current"].append(dict(hist["current"][-1], event=g,
                                    event_transfers=0, event_transfers_cost=0,
                                    bank=2 + g))
        ov[f"picks_{g}"] = p
    if fh_gw:
        fh = ov[f"picks_{fh_gw}"]
        fh["active_chip"] = "freehit"
        fh["entry_history"]["bank"] = 99
        fh["entry_history"]["event_transfers"] = len(fh_squad_swap or {})
        for pk in fh["picks"]:
            if fh_squad_swap and pk["element"] in fh_squad_swap:
                pk["element"] = fh_squad_swap[pk["element"]]
        hist["current"][-1]["event_transfers"] = len(fh_squad_swap or {})
        hist["chips"].append({"name": "freehit", "event": fh_gw, "time": ""})
    for c in extra_chips or []:
        hist["chips"].append(c)
    entry["current_event"] = last
    ov.update({"history": hist, "entry": entry})
    if extra_transfers:
        ov["transfers"] = load_gt("transfers.json") + extra_transfers
    return ov


class TestFreeHitRevertLive:
    def test_reads_pre_fh_squad_and_bank(self, offline_fpl, bootstrap):
        # FH in GW7 swapping Haaland->Watkins-ish (element 100) and Mendy->Gabriel(5)
        stub = offline_fpl(_future_overrides(
            fh_gw=7, fh_squad_swap={HAALAND: 100, MENDY: 5},
            extra_transfers=[
                {"element_in": 100, "element_in_cost": 90, "element_out": HAALAND,
                 "element_out_cost": 155, "entry": ENTRY_ID, "event": 7, "time": "t1"},
                # sell-and-rebuy Haaland inside the FH week at a different price
                {"element_in": HAALAND, "element_in_cost": 160, "element_out": 100,
                 "element_out_cost": 90, "entry": ENTRY_ID, "event": 7, "time": "t2"},
                {"element_in": 5, "element_in_cost": 60, "element_out": MENDY,
                 "element_out_cost": 41, "entry": ENTRY_ID, "event": 7, "time": "t3"},
            ]))
        es = fetch_entry_state(ENTRY_ID, bootstrap)
        gs = es.game_state
        ids = {p.element_id for p in gs.squad.players}
        assert HAALAND in ids and MENDY in ids and 100 not in ids and 5 not in ids
        assert any(u.endswith("/event/6/picks/") for u in stub.calls)
        assert gs.bank == 2 + 6                           # GW6 (pre-FH) bank
        by = {p.element_id: p for p in gs.squad.players}
        assert by[HAALAND].purchase_price == 155          # FH rebuy ignored
        assert by[MENDY].purchase_price == 41
        # FT: GW6 had 2, rolled -> 3 for GW7; FH week keeps 3 for GW8
        assert gs.free_transfers == 3
        assert not gs.chips.is_available("free_hit", 8)
        assert gs.chips.is_available("free_hit", 20)

    def test_fh19_blocks_fh20_after_reconstruction(self, offline_fpl, bootstrap):
        offline_fpl(_future_overrides(fh_gw=19))
        gs = fetch_entry_state(ENTRY_ID, bootstrap).game_state
        assert gs.current_gw == 20
        assert not gs.chips.is_available("free_hit", 20)
        assert gs.chips.is_available("free_hit", 21)
        assert gs.chips.is_available("wildcard", 20)
        # every unused first-half chip is gone
        assert gs.chips.wildcard[0] is False and gs.chips.triple_captain[0] is False

    def test_api_chip_names_parsed(self, offline_fpl, bootstrap):
        offline_fpl(_future_overrides(extra_chips=[
            {"name": "3xc", "event": 6, "time": ""},
        ]))
        gs = fetch_entry_state(ENTRY_ID, bootstrap).game_state
        assert not gs.chips.is_available("triple_captain", 7)
        assert gs.chips.is_available("triple_captain", 20)
        for chip in ("wildcard", "free_hit", "bench_boost"):
            assert gs.chips.is_available(chip, 7)

    def test_all_four_api_names_map(self):
        from fpl_optimizer.live.entry import _CHIP_NAMES
        rules = load_gt("rules_2026-27.json")
        assert set(_CHIP_NAMES) == {c["name"] for c in rules["chips"]}

    def test_unknown_chip_name_is_ignored(self, offline_fpl, bootstrap):
        offline_fpl(_future_overrides(extra_chips=[
            {"name": "manager", "event": 6, "time": ""},
        ]))
        gs = fetch_entry_state(ENTRY_ID, bootstrap).game_state
        assert gs.chips.is_available("wildcard", 7)

    def test_second_half_first_half_expired(self, offline_fpl, bootstrap):
        # simulate reaching GW20 with nothing used
        hist = load_gt("history.json")
        for g in range(6, 20):
            hist["current"].append(dict(hist["current"][-1], event=g, event_transfers=1))
        entry = load_gt("entry.json")
        entry["current_event"] = 19
        p19 = load_gt("picks_gw5.json")
        offline_fpl({"history": hist, "entry": entry, "picks_19": p19})
        gs = fetch_entry_state(ENTRY_ID, bootstrap).game_state
        assert gs.current_gw == 20
        for chip in ("wildcard", "free_hit", "bench_boost", "triple_captain"):
            assert gs.chips.is_available(chip, 20)
            assert not gs.chips.is_available(chip, 19)
