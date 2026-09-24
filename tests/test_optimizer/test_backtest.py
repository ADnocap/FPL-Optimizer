"""Tests for backtest.py."""

from __future__ import annotations

import pytest

from fpl_optimizer.optimizer.backtest import (
    BacktestResult,
    GWResult,
    SeasonBacktester,
    _optimizer_result_to_game_state,
)
from fpl_optimizer.optimizer.squad_selection import select_squad
from fpl_optimizer.optimizer.types import build_candidate_pool
from fpl_optimizer.utils.constants import STARTING_BUDGET


class TestSeasonBacktester:
    """Tests for the full-season backtester (using test data with 2 GWs)."""

    def test_runs_2gw_season(self, loader):
        """Run a 2-GW hindsight backtest on test data."""
        bt = SeasonBacktester(loader)
        result = bt.run()

        assert isinstance(result, BacktestResult)
        assert len(result.gw_results) == 2
        assert result.total_points > 0

    def test_gw_results_have_correct_gws(self, loader):
        bt = SeasonBacktester(loader)
        result = bt.run()
        gws = [gw.gw for gw in result.gw_results]
        assert gws == [1, 2]

    def test_total_points_is_sum_of_net(self, loader):
        bt = SeasonBacktester(loader)
        result = bt.run()
        expected = sum(gw.net_points for gw in result.gw_results)
        assert result.total_points == expected

    def test_max_gw_limits_gameweeks(self, loader):
        bt = SeasonBacktester(loader)
        result = bt.run(max_gw=1)
        assert len(result.gw_results) == 1

    def test_gw1_has_no_transfers(self, loader):
        bt = SeasonBacktester(loader)
        result = bt.run()
        gw1 = result.gw_results[0]
        assert gw1.transfers_in == []
        assert gw1.transfers_out == []

    def test_hindsight_points_positive(self, loader):
        """Hindsight optimal should score positive points."""
        bt = SeasonBacktester(loader)
        result = bt.run()
        assert result.total_points > 0

    def test_gross_points_gte_net_points(self, loader):
        """Gross points should always be >= net points (hits can only reduce)."""
        bt = SeasonBacktester(loader)
        result = bt.run()
        for gw in result.gw_results:
            assert gw.gross_points >= gw.net_points


class TestOptimizerResultToGameState:
    """Test the helper that builds a GameState from an OptimizerResult."""

    def test_builds_valid_state(self, loader):
        candidates = build_candidate_pool(loader, gw=1)
        squad_result = select_squad(candidates)
        state = _optimizer_result_to_game_state(squad_result, loader, 1)

        assert state.current_gw == 1
        assert len(state.squad.players) == 15
        assert len(state.squad.lineup) == 11
        assert len(state.squad.bench) == 4
        assert state.bank == STARTING_BUDGET - squad_result.total_cost
        assert state.bank >= 0

    def test_state_squad_matches_result(self, loader):
        candidates = build_candidate_pool(loader, gw=1)
        squad_result = select_squad(candidates)
        state = _optimizer_result_to_game_state(squad_result, loader, 1)

        state_eids = {p.element_id for p in state.squad.players}
        assert state_eids == set(squad_result.squad_element_ids)
