"""R01 — utils/constants.py + ChipState windows vs the official 2026-27 API rules."""

from __future__ import annotations

import itertools

import pytest

from fpl_optimizer.engine.state import ChipState
from fpl_optimizer.engine.transfers import calculate_selling_price
from fpl_optimizer.utils import constants as C
from fpl_optimizer.utils.constants import Position

# API chip name -> engine chip name
API2ENGINE = {"wildcard": "wildcard", "freehit": "free_hit",
              "bboost": "bench_boost", "3xc": "triple_captain"}


class TestSquadSettings:
    def test_squad_size(self, rules):
        assert C.SQUAD_SIZE == rules["game_settings"]["squad_squadsize"] == 15

    def test_starting_xi(self, rules):
        assert C.STARTING_XI == rules["game_settings"]["squad_squadplay"] == 11

    def test_bench_size(self, rules):
        gs = rules["game_settings"]
        assert C.BENCH_SIZE == gs["squad_squadsize"] - gs["squad_squadplay"]

    def test_club_limit(self, rules):
        assert C.MAX_PER_CLUB == rules["game_settings"]["squad_team_limit"] == 3

    def test_budget(self, rules):
        assert C.STARTING_BUDGET == rules["game_settings"]["squad_total_spend"] == 1000

    def test_position_quotas_match_element_types(self, bootstrap):
        api = {Position(t["id"]): t["squad_select"] for t in bootstrap["element_types"]}
        assert dict(C.POSITION_LIMITS) == api

    def test_formations_equal_all_legal_min_max_play(self, bootstrap):
        """Every (DEF, MID, FWD) with squad_min_play<=n<=squad_max_play, GK=1, sum 11."""
        et = {t["id"]: (t["squad_min_play"], t["squad_max_play"])
              for t in bootstrap["element_types"]}
        assert et[1] == (1, 1)  # exactly one GK
        legal = {
            (d, m, f)
            for d, m, f in itertools.product(range(6), range(6), range(4))
            if d + m + f == 10
            and et[2][0] <= d <= et[2][1]
            and et[3][0] <= m <= et[3][1]
            and et[4][0] <= f <= et[4][1]
        }
        assert set(C.VALID_FORMATIONS) == legal
        assert len(C.VALID_FORMATIONS) == 8


class TestTransferSettings:
    def test_max_banked_free_transfers(self, rules):
        # max_extra_free_transfers=4 on top of the weekly 1 => bank to 5
        assert C.MAX_FREE_TRANSFERS == 1 + rules["game_settings"]["max_extra_free_transfers"] == 5

    def test_hit_cost(self):
        assert C.TRANSFER_HIT_COST == 4

    def test_sell_on_fee_half_of_gain_floored(self, rules):
        gs = rules["game_settings"]
        assert gs["transfers_sell_on_fee"] == 0.5
        assert gs["element_sell_at_purchase_price"] is False
        for buy, now in [(50, 51), (50, 52), (50, 53), (57, 58), (155, 156), (75, 77),
                         (60, 60), (60, 58)]:
            gain = now - buy
            expected = buy + int(gain * gs["transfers_sell_on_fee"]) if gain > 0 else now
            assert calculate_selling_price(buy, now) == expected

    def test_transfers_cap_not_exceedable(self, rules):
        # transfers_cap=20 per GW; a single MILP solve can make at most 15
        assert rules["game_settings"]["transfers_cap"] == 20
        assert C.SQUAD_SIZE <= rules["game_settings"]["transfers_cap"]


class TestScoring:
    def test_scoring_table(self, rules):
        s = rules["game_config"]["scoring"]
        key = {Position.GK: "GKP", Position.DEF: "DEF", Position.MID: "MID",
               Position.FWD: "FWD"}
        for pos, k in key.items():
            assert C.POINTS_GOAL[pos] == s["goals_scored"][k]
            assert C.POINTS_CLEAN_SHEET[pos] == s["clean_sheets"][k]
            assert C.POINTS_GOALS_CONCEDED_PER_2[pos] == s["goals_conceded"][k]
        assert C.POINTS_ASSIST == s["assists"]
        assert C.POINTS_SAVES_PER_3 == s["saves"]
        assert C.POINTS_PENALTY_SAVE == s["penalties_saved"]
        assert C.POINTS_PENALTY_MISS == s["penalties_missed"]
        assert C.POINTS_YELLOW_CARD == s["yellow_cards"]
        assert C.POINTS_RED_CARD == s["red_cards"]
        assert C.POINTS_OWN_GOAL == s["own_goals"]
        assert C.POINTS_MINUTES_1_59 == s["short_play"]
        assert C.POINTS_MINUTES_60_PLUS == s["long_play"]

    def test_defcon_is_live_and_gk_excluded(self, rules):
        # Engine scores from recorded total_points (DEFCON already included);
        # this pins the official table the predictor/target assume.
        dc = rules["game_config"]["scoring"]["defensive_contribution"]
        assert dc == {"DEF": 2, "FWD": 2, "GKP": 0, "MID": 2}


class TestChipWindows:
    def test_chip_set_is_4_x_2_halves(self, rules):
        names = sorted((c["name"], c["start_event"], c["stop_event"]) for c in rules["chips"])
        assert len(names) == 8
        assert set(API2ENGINE) == {c["name"] for c in rules["chips"]}
        assert set(API2ENGINE.values()) == set(C.ALL_CHIPS)

    def test_half_boundaries(self, rules):
        stops = {c["stop_event"] for c in rules["chips"]}
        starts = {c["start_event"] for c in rules["chips"]}
        assert stops == {C.FIRST_HALF_END, C.TOTAL_GAMEWEEKS}
        assert C.SECOND_HALF_START in starts

    @pytest.mark.bug("R-01")
    def test_every_chip_every_gw_matches_official_window(self, rules):
        """ChipState().is_available(chip, gw) must equal the API window, GW1-38.

        Official: wildcard/freehit start_event=2 (NOT playable GW1);
        bboost/3xc start_event=1.
        """
        cs = ChipState()
        mismatches = []
        for gw in range(1, 39):
            for api_name, eng in API2ENGINE.items():
                official = any(
                    c["name"] == api_name and c["start_event"] <= gw <= c["stop_event"]
                    for c in rules["chips"]
                )
                if cs.is_available(eng, gw) != official:
                    mismatches.append((eng, gw, official))
        assert mismatches == []

    @pytest.mark.bug("R-01")
    @pytest.mark.parametrize("chip", ["wildcard", "free_hit"])
    def test_wc_fh_not_playable_gw1(self, chip):
        assert ChipState().is_available(chip, 1) is False

    @pytest.mark.parametrize("chip", ["bench_boost", "triple_captain"])
    def test_bb_tc_playable_gw1(self, chip):
        assert ChipState().is_available(chip, 1) is True
