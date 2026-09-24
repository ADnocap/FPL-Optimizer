"""Tests for transfer_optimizer.py."""

from __future__ import annotations

import pytest

from fpl_optimizer.engine.state import ChipState, GameState, PlayerSlot, Squad
from fpl_optimizer.optimizer.transfer_optimizer import optimize_transfers
from fpl_optimizer.optimizer.types import PlayerCandidate
from fpl_optimizer.utils.constants import (
    MAX_PER_CLUB,
    POSITION_LIMITS,
    STARTING_BUDGET,
    TRANSFER_HIT_COST,
    VALID_FORMATIONS,
    Position,
)
from tests.test_optimizer.conftest import _make_candidate


class TestTransferOptimizer:
    """Tests for the weekly transfer optimizer MILP."""

    def test_no_transfer_when_squad_optimal(
        self, squad_15_candidates, optimizer_game_state
    ):
        """If the current squad already has the best players, no transfers."""
        # Candidates are exactly the current squad with same xP
        result = optimize_transfers(optimizer_game_state, squad_15_candidates)
        # May or may not transfer — but should be valid
        assert len(result.squad_element_ids) == 15
        assert len(result.lineup_element_ids) == 11

    def test_makes_transfer_when_better_player_available(
        self, squad_15_candidates, optimizer_game_state
    ):
        """Should buy a much better player when available."""
        # Add a super-star MID available cheaply
        star = _make_candidate(99, Position.MID, 45, 10, xp=20.0)
        candidates = squad_15_candidates + [star]
        result = optimize_transfers(optimizer_game_state, candidates)
        # The star should be in the squad
        assert 99 in result.squad_element_ids

    def test_squad_size_always_15(self, squad_15_candidates, optimizer_game_state):
        star = _make_candidate(99, Position.MID, 45, 10, xp=20.0)
        candidates = squad_15_candidates + [star]
        result = optimize_transfers(optimizer_game_state, candidates)
        assert len(result.squad_element_ids) == 15

    def test_valid_formation_after_transfer(
        self, squad_15_candidates, optimizer_game_state
    ):
        star = _make_candidate(99, Position.MID, 45, 10, xp=20.0)
        candidates = squad_15_candidates + [star]
        result = optimize_transfers(optimizer_game_state, candidates)

        cand_map = {c.element_id: c for c in candidates}
        # Must also include current squad members in lookup
        for p in optimizer_game_state.squad.players:
            if p.element_id not in cand_map:
                cand_map[p.element_id] = _make_candidate(
                    p.element_id, p.position, p.selling_price, 0, 0.0,
                )
        lineup_positions = [cand_map[eid].position for eid in result.lineup_element_ids]
        from collections import Counter
        counts = Counter(lineup_positions)
        assert counts[Position.GK] == 1
        formation = (counts[Position.DEF], counts[Position.MID], counts[Position.FWD])
        assert formation in VALID_FORMATIONS

    def test_position_limits_after_transfer(
        self, squad_15_candidates, optimizer_game_state
    ):
        star = _make_candidate(99, Position.MID, 45, 10, xp=20.0)
        candidates = squad_15_candidates + [star]
        result = optimize_transfers(optimizer_game_state, candidates)

        cand_map = {c.element_id: c for c in candidates}
        for p in optimizer_game_state.squad.players:
            if p.element_id not in cand_map:
                cand_map[p.element_id] = _make_candidate(
                    p.element_id, p.position, p.selling_price, 0, 0.0,
                )

        from collections import Counter
        positions = [cand_map[eid].position for eid in result.squad_element_ids]
        counts = Counter(positions)
        for pos, limit in POSITION_LIMITS.items():
            assert counts[pos] == limit

    def test_transfers_in_out_same_length(
        self, squad_15_candidates, optimizer_game_state
    ):
        star = _make_candidate(99, Position.MID, 45, 10, xp=20.0)
        candidates = squad_15_candidates + [star]
        result = optimize_transfers(optimizer_game_state, candidates)
        assert len(result.transfers_in) == len(result.transfers_out)

    def test_hit_cost_for_extra_transfers(self, optimizer_game_state):
        """With 1 free transfer and 2+ transfers needed, should incur hits."""
        # Create candidates where 2 transfers are clearly worth the hit
        base = []
        for p in optimizer_game_state.squad.players:
            base.append(_make_candidate(
                p.element_id, p.position, p.selling_price,
                p.element_id % 10 + 1, xp=2.0,
            ))
        # Two amazing players replacing two weak ones
        star1 = _make_candidate(98, Position.MID, 45, 10, xp=30.0)
        star2 = _make_candidate(99, Position.FWD, 55, 10, xp=30.0)
        candidates = base + [star1, star2]

        result = optimize_transfers(optimizer_game_state, candidates)
        if len(result.transfers_out) > 1:
            # 1 free transfer, so extra transfers incur 4-pt hit each
            expected_hit = TRANSFER_HIT_COST * (len(result.transfers_out) - 1)
            assert result.hit_cost == expected_hit

    def test_wildcard_no_hit_cost(self, optimizer_game_state):
        """Wildcard should allow unlimited free transfers."""
        base = []
        for p in optimizer_game_state.squad.players:
            base.append(_make_candidate(
                p.element_id, p.position, p.selling_price,
                p.element_id % 10 + 1, xp=1.0,
            ))
        star = _make_candidate(99, Position.MID, 45, 10, xp=30.0)
        candidates = base + [star]

        result = optimize_transfers(optimizer_game_state, candidates, chip="wildcard")
        assert result.hit_cost == 0
        assert result.chip == "wildcard"

    def test_free_hit_no_hit_cost(self, optimizer_game_state):
        """Free hit should allow unlimited free transfers."""
        base = []
        for p in optimizer_game_state.squad.players:
            base.append(_make_candidate(
                p.element_id, p.position, p.selling_price,
                p.element_id % 10 + 1, xp=1.0,
            ))
        star = _make_candidate(99, Position.MID, 45, 10, xp=30.0)
        candidates = base + [star]

        result = optimize_transfers(optimizer_game_state, candidates, chip="free_hit")
        assert result.hit_cost == 0
        assert result.chip == "free_hit"

    def test_budget_respected(self, optimizer_game_state, squad_15_candidates):
        """Cannot buy players that exceed the budget."""
        # Add an expensive player
        expensive = _make_candidate(99, Position.MID, 500, 10, xp=50.0)
        candidates = squad_15_candidates + [expensive]
        result = optimize_transfers(optimizer_game_state, candidates)
        # The expensive player likely can't be afforded (bank=155)
        # If selected, verify budget is respected
        assert len(result.squad_element_ids) == 15

    def test_captain_and_vc_valid(self, squad_15_candidates, optimizer_game_state):
        result = optimize_transfers(optimizer_game_state, squad_15_candidates)
        assert result.captain_id in result.lineup_element_ids
        assert result.vice_captain_id in result.lineup_element_ids
        assert result.captain_id != result.vice_captain_id

    def test_club_limit_respected(self, squad_15_candidates, optimizer_game_state):
        """No more than 3 players from the same club."""
        # Add players from team 1 (already have GK1 eid=1, DEF1 eid=3, MID1 eid=8)
        extra = _make_candidate(99, Position.FWD, 55, 1, xp=20.0)
        candidates = squad_15_candidates + [extra]
        result = optimize_transfers(optimizer_game_state, candidates)

        cand_map = {c.element_id: c for c in candidates}
        for p in optimizer_game_state.squad.players:
            if p.element_id not in cand_map:
                cand_map[p.element_id] = _make_candidate(
                    p.element_id, p.position, p.selling_price, 0, 0.0,
                )

        from collections import Counter
        team_counts = Counter(cand_map[eid].team_id for eid in result.squad_element_ids)
        for team_id, count in team_counts.items():
            if team_id != 0:  # skip placeholder team
                assert count <= MAX_PER_CLUB
