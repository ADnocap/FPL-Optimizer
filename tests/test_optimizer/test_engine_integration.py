"""Integration tests: optimizer -> engine round-trip."""

from __future__ import annotations

import pytest

from fpl_optimizer.engine.constraints import (
    check_club_limits,
    is_valid_formation,
    is_valid_squad,
    is_valid_squad_composition,
)
from fpl_optimizer.engine.engine import FPLGameEngine
from fpl_optimizer.engine.state import (
    ChipState,
    GameState,
    PlayerSlot,
    Squad,
)
from fpl_optimizer.optimizer.lineup_selector import select_lineup
from fpl_optimizer.optimizer.squad_selection import select_squad
from fpl_optimizer.optimizer.transfer_optimizer import optimize_transfers
from fpl_optimizer.optimizer.types import (
    OptimizerResult,
    PlayerCandidate,
    build_candidate_pool,
    to_engine_action,
)
from fpl_optimizer.utils.constants import Position, STARTING_BUDGET


class TestBuildCandidatePool:
    """Tests for build_candidate_pool from real test data."""

    def test_builds_candidates_from_loader(self, loader):
        candidates = build_candidate_pool(loader, gw=1)
        assert len(candidates) > 0
        for c in candidates:
            assert isinstance(c, PlayerCandidate)
            assert c.price > 0
            assert c.position in list(Position)

    def test_hindsight_uses_actual_points(self, loader):
        candidates = build_candidate_pool(loader, gw=1)
        # Player 8 (MID1) has 12 points in GW1 test data
        mid1 = [c for c in candidates if c.element_id == 8]
        assert len(mid1) == 1
        assert mid1[0].predicted_points == 12.0

    def test_predicted_points_override(self, loader):
        preds = {8: 99.0, 13: 50.0}
        candidates = build_candidate_pool(loader, gw=1, predicted_points=preds)
        mid1 = [c for c in candidates if c.element_id == 8]
        assert len(mid1) == 1
        assert mid1[0].predicted_points == 99.0

    def test_empty_gw_returns_empty(self, loader):
        candidates = build_candidate_pool(loader, gw=99)
        assert candidates == []


class TestToEngineAction:
    """Tests for to_engine_action conversion."""

    def test_converts_lineup_result(self, squad_15_candidates):
        result = select_lineup(squad_15_candidates)
        action = to_engine_action(result)
        assert action.captain == result.captain_id
        assert action.vice_captain == result.vice_captain_id
        assert set(action.lineup) == set(result.lineup_element_ids)
        assert list(action.bench) == list(result.bench_element_ids)
        assert action.transfers_in == []
        assert action.transfers_out == []


class TestEngineRoundTrip:
    """Optimizer -> to_engine_action -> engine.step round trip."""

    def test_squad_selection_then_engine_step(self, loader):
        """Select initial squad with optimizer, then step through engine."""
        candidates = build_candidate_pool(loader, gw=1)
        assert len(candidates) >= 15

        result = select_squad(candidates)

        # Build GameState from result
        players = []
        for eid in result.squad_element_ids:
            pos = loader.get_player_position(eid)
            price = loader.get_player_price(eid, 1)
            players.append(PlayerSlot(
                element_id=eid, position=pos,
                purchase_price=price, selling_price=price,
            ))

        eid_to_idx = {p.element_id: i for i, p in enumerate(players)}
        lineup = [eid_to_idx[eid] for eid in result.lineup_element_ids]
        bench = [eid_to_idx[eid] for eid in result.bench_element_ids]

        squad = Squad(
            players=players, lineup=lineup, bench=bench,
            captain_idx=eid_to_idx[result.captain_id],
            vice_captain_idx=eid_to_idx[result.vice_captain_id],
        )
        state = GameState(
            squad=squad,
            bank=STARTING_BUDGET - result.total_cost,
            current_gw=1,
        )

        # Validate the squad using engine constraints
        assert is_valid_squad_composition(state.squad.players)
        assert is_valid_formation(state.squad.get_lineup_players())
        assert check_club_limits(state.squad.players, loader._team_map)

        # Step through engine
        action = to_engine_action(result)
        action.transfers_in = []
        action.transfers_out = []

        engine = FPLGameEngine(loader)
        new_state, step_result = engine.step(state, action)

        assert step_result.gw_points >= 0
        assert new_state.current_gw == 2

    def test_transfer_then_engine_step(self, loader):
        """Optimizer transfer -> engine.step round trip."""
        # First build initial squad
        gw1_candidates = build_candidate_pool(loader, gw=1)
        squad_result = select_squad(gw1_candidates)

        players = []
        for eid in squad_result.squad_element_ids:
            pos = loader.get_player_position(eid)
            price = loader.get_player_price(eid, 1)
            players.append(PlayerSlot(
                element_id=eid, position=pos,
                purchase_price=price, selling_price=price,
            ))

        eid_to_idx = {p.element_id: i for i, p in enumerate(players)}
        lineup = [eid_to_idx[eid] for eid in squad_result.lineup_element_ids]
        bench = [eid_to_idx[eid] for eid in squad_result.bench_element_ids]

        squad = Squad(
            players=players, lineup=lineup, bench=bench,
            captain_idx=eid_to_idx[squad_result.captain_id],
            vice_captain_idx=eid_to_idx[squad_result.vice_captain_id],
        )

        # Step GW1 through engine
        state = GameState(
            squad=squad,
            bank=STARTING_BUDGET - squad_result.total_cost,
            current_gw=1,
        )
        engine = FPLGameEngine(loader)
        action1 = to_engine_action(squad_result)
        action1.transfers_in = []
        action1.transfers_out = []
        state, _ = engine.step(state, action1)

        # Now optimise transfers for GW2
        gw2_candidates = build_candidate_pool(loader, gw=2)
        transfer_result = optimize_transfers(state, gw2_candidates)

        action2 = to_engine_action(transfer_result)
        new_state, step_result2 = engine.step(state, action2)

        assert step_result2.gw_points >= 0
        assert new_state.current_gw == 3

    def test_optimizer_squad_passes_engine_validation(self, loader):
        """Squad from optimizer should pass all engine constraint checks."""
        candidates = build_candidate_pool(loader, gw=1)
        result = select_squad(candidates)

        players = []
        for eid in result.squad_element_ids:
            pos = loader.get_player_position(eid)
            price = loader.get_player_price(eid, 1)
            players.append(PlayerSlot(
                element_id=eid, position=pos,
                purchase_price=price, selling_price=price,
            ))

        eid_to_idx = {p.element_id: i for i, p in enumerate(players)}
        squad = Squad(
            players=players,
            lineup=[eid_to_idx[eid] for eid in result.lineup_element_ids],
            bench=[eid_to_idx[eid] for eid in result.bench_element_ids],
            captain_idx=eid_to_idx[result.captain_id],
            vice_captain_idx=eid_to_idx[result.vice_captain_id],
        )

        assert is_valid_squad(squad, loader._team_map)
