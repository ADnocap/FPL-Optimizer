"""Tests for squad_selection.py."""

from __future__ import annotations

import pytest

from fpl_optimizer.optimizer.squad_selection import select_squad
from fpl_optimizer.optimizer.types import PlayerCandidate
from fpl_optimizer.utils.constants import (
    MAX_PER_CLUB,
    POSITION_LIMITS,
    STARTING_BUDGET,
    VALID_FORMATIONS,
    Position,
)


class TestSelectSquad:
    """Tests for the integrated squad selection MILP."""

    def test_selects_15_players(self, large_candidate_pool):
        result = select_squad(large_candidate_pool)
        assert len(result.squad_element_ids) == 15

    def test_11_starters(self, large_candidate_pool):
        result = select_squad(large_candidate_pool)
        assert len(result.lineup_element_ids) == 11

    def test_4_bench(self, large_candidate_pool):
        result = select_squad(large_candidate_pool)
        assert len(result.bench_element_ids) == 4

    def test_squad_equals_lineup_plus_bench(self, large_candidate_pool):
        result = select_squad(large_candidate_pool)
        assert set(result.squad_element_ids) == set(result.lineup_element_ids) | set(result.bench_element_ids)

    def test_position_limits(self, large_candidate_pool):
        result = select_squad(large_candidate_pool)
        cand_map = {c.element_id: c for c in large_candidate_pool}
        positions = [cand_map[eid].position for eid in result.squad_element_ids]
        from collections import Counter
        counts = Counter(positions)
        for pos, limit in POSITION_LIMITS.items():
            assert counts[pos] == limit, f"{pos}: expected {limit}, got {counts[pos]}"

    def test_valid_formation(self, large_candidate_pool):
        result = select_squad(large_candidate_pool)
        cand_map = {c.element_id: c for c in large_candidate_pool}
        lineup_positions = [cand_map[eid].position for eid in result.lineup_element_ids]
        from collections import Counter
        counts = Counter(lineup_positions)
        assert counts[Position.GK] == 1
        formation = (counts[Position.DEF], counts[Position.MID], counts[Position.FWD])
        assert formation in VALID_FORMATIONS

    def test_budget_constraint(self, large_candidate_pool):
        result = select_squad(large_candidate_pool)
        assert result.total_cost <= STARTING_BUDGET

    def test_club_limit(self, large_candidate_pool):
        result = select_squad(large_candidate_pool)
        cand_map = {c.element_id: c for c in large_candidate_pool}
        from collections import Counter
        team_counts = Counter(cand_map[eid].team_id for eid in result.squad_element_ids)
        for team_id, count in team_counts.items():
            assert count <= MAX_PER_CLUB, f"Team {team_id} has {count} players"

    def test_captain_in_lineup(self, large_candidate_pool):
        result = select_squad(large_candidate_pool)
        assert result.captain_id in result.lineup_element_ids

    def test_vice_captain_in_lineup(self, large_candidate_pool):
        result = select_squad(large_candidate_pool)
        assert result.vice_captain_id in result.lineup_element_ids

    def test_captain_distinct_from_vc(self, large_candidate_pool):
        result = select_squad(large_candidate_pool)
        assert result.captain_id != result.vice_captain_id

    def test_objective_positive(self, large_candidate_pool):
        result = select_squad(large_candidate_pool)
        assert result.objective_value > 0

    def test_tight_budget(self, large_candidate_pool):
        """With a very tight budget the solver should still find a feasible squad."""
        # Min possible ~672 without club limits, ~700 with; 750 is tight but feasible
        result = select_squad(large_candidate_pool, budget=750)
        assert len(result.squad_element_ids) == 15
        assert result.total_cost <= 750

    def test_empty_pool_raises(self):
        with pytest.raises(ValueError, match="Empty candidate pool"):
            select_squad([])

    def test_prefers_high_xp_players(self, large_candidate_pool):
        """The optimizer should prefer higher xP players in the lineup."""
        result = select_squad(large_candidate_pool)
        cand_map = {c.element_id: c for c in large_candidate_pool}
        lineup_xp = sum(cand_map[eid].predicted_points for eid in result.lineup_element_ids)
        # Should get at least 50 points from a good pool
        assert lineup_xp > 50
