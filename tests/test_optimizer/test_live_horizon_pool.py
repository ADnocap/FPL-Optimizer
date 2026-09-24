"""Tests for the live multi-GW candidate pool (availability over the horizon)."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_optimizer.live.pool import availability_multipliers, build_live_horizon_candidates


def _el(eid, status="a", chance=None, team=1, etype=3, cost=60):
    return {"id": eid, "status": status, "chance_of_playing_next_round": chance,
            "team": team, "element_type": etype, "now_cost": cost}


class TestAvailability:
    def test_fit_player_full_value(self):
        assert availability_multipliers(_el(1), 3) == [1.0, 1.0, 1.0]

    def test_flag_hits_this_gw_hard_and_recovers(self):
        m = availability_multipliers(_el(1, "d", 75), 4)
        assert m[0] == pytest.approx(0.35)  # 75% flag != 0.75x (audit)
        assert m[0] < m[1] < m[2] < m[3] <= 1.0

    def test_left_club_is_zero_everywhere(self):
        assert availability_multipliers(_el(1, "u", 0), 3) == [0.0, 0.0, 0.0]

    def test_injured_zero_now(self):
        m = availability_multipliers(_el(1, "i", 0), 3)
        assert m[0] == 0.0 and m[2] > 0.0


def test_build_live_horizon_candidates():
    boot = {"elements": [_el(1), _el(2, "i", 0), _el(3, "d", 50), _el(4, "d", 25)]}
    preds = pd.DataFrame([
        {"element": e, "GW": g, "k": g - 6, "pred": 4.0, "p_play": 0.9}
        for e in (1, 2, 3, 4) for g in (6, 7)
    ])
    cands = build_live_horizon_candidates(boot, preds, [6, 7], min_chance=50,
                                          always_include={2})
    by = {c.element_id: c for c in cands}
    assert set(by) == {1, 2, 3}  # 4 below min_chance, 2 kept (in squad)
    assert by[1].xpts == (4.0, 4.0)
    assert by[2].xpts[0] == 0.0  # injured squad player: no points this GW
    assert by[3].xpts[0] == pytest.approx(0.8) and by[3].xpts[1] > by[3].xpts[0]
    assert by[3].p_play[0] < 0.9
