"""R09 — optimizer/backtest.py (and scripts/backtest_season.py, same init) FT start.

FPL: unlimited changes before the GW1 deadline, then exactly 1 FT for GW2.
backtest.py:99 / backtest_season.py:133 start GW1 with free_transfers=1 and the
engine banks +1 -> GW2 gets 2 FTs (one free transfer too many, every season).
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_optimizer.data.loader import SeasonDataLoader
from fpl_optimizer.engine.engine import FPLGameEngine
from fpl_optimizer.engine.state import EngineAction
from fpl_optimizer.optimizer.backtest import SeasonBacktester, _optimizer_result_to_game_state
from fpl_optimizer.optimizer.squad_selection import select_squad
from fpl_optimizer.optimizer.types import build_candidate_pool

from tests.test_rules.rules_helpers import POS


def _season_loader(points_by_gw: dict[int, dict[int, int]]) -> SeasonDataLoader:
    recs = []
    for gw, pts in points_by_gw.items():
        for e in range(1, 18):
            recs.append({"element": e, "GW": gw, "fixture": gw, "total_points": pts[e],
                         "minutes": 90, "yellow_cards": 0, "red_cards": 0,
                         "value": 50})
    ld = object.__new__(SeasonDataLoader)
    ld.season = "mini"
    ld.data_dir = None
    ld._merged_gw = pd.DataFrame(recs)
    ld._gw_index = {}
    for idx, r in ld._merged_gw.iterrows():
        ld._gw_index.setdefault((int(r["element"]), int(r["GW"])), []).append(idx)
    ld._position_map = {e: POS[e] for e in range(1, 18)}
    ld._team_map = {e: e for e in range(1, 18)}
    return ld


# GW1: squad players 1-15 score 5, market 16 (DEF) / 17 (MID) score 0 -> not picked
# GW2: squad players score 2, market 16/17 score 5 (+3 each: < a -4 hit)
GW1 = {**{e: 5 for e in range(1, 16)}, 16: 0, 17: 0}
GW2 = {**{e: 2 for e in range(1, 16)}, 16: 5, 17: 5}


@pytest.mark.bug("R-02")
def test_backtest_gw1_state_gives_one_ft_in_gw2():
    ld = _season_loader({1: GW1, 2: GW2})
    res = select_squad(build_candidate_pool(ld, 1))
    st = _optimizer_result_to_game_state(res, ld, 1)
    st2, _ = FPLGameEngine(ld).step(st, EngineAction())
    assert st2.free_transfers == 1


@pytest.mark.bug("R-02")
def test_backtest_gw2_does_not_get_a_free_second_transfer():
    ld = _season_loader({1: GW1, 2: GW2})
    out = SeasonBacktester(ld).run(max_gw=2)
    gw2 = next(g for g in out.gw_results if g.gw == 2)
    # correct rules: 1 FT -> one +6 transfer; a second (+3.3) is not worth -4
    assert len(gw2.transfers_in) == 1
    assert gw2.hit_cost == 0
