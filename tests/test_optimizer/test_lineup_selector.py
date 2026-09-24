"""Tests for lineup_selector.py."""

from __future__ import annotations

import pytest

from fpl_optimizer.optimizer.lineup_selector import select_lineup
from fpl_optimizer.optimizer.types import PlayerCandidate
from fpl_optimizer.utils.constants import Position, VALID_FORMATIONS


class TestSelectLineup:
    """Tests for the lineup selection MILP."""

    def test_returns_11_starters(self, squad_15_candidates):
        result = select_lineup(squad_15_candidates)
        assert len(result.lineup_element_ids) == 11

    def test_returns_4_bench(self, squad_15_candidates):
        result = select_lineup(squad_15_candidates)
        assert len(result.bench_element_ids) == 4

    def test_all_15_accounted_for(self, squad_15_candidates):
        result = select_lineup(squad_15_candidates)
        all_ids = set(result.lineup_element_ids) | set(result.bench_element_ids)
        expected = {c.element_id for c in squad_15_candidates}
        assert all_ids == expected

    def test_valid_formation(self, squad_15_candidates):
        result = select_lineup(squad_15_candidates)
        cand_map = {c.element_id: c for c in squad_15_candidates}
        lineup_positions = [cand_map[eid].position for eid in result.lineup_element_ids]

        gk_count = sum(1 for p in lineup_positions if p == Position.GK)
        def_count = sum(1 for p in lineup_positions if p == Position.DEF)
        mid_count = sum(1 for p in lineup_positions if p == Position.MID)
        fwd_count = sum(1 for p in lineup_positions if p == Position.FWD)

        assert gk_count == 1
        assert (def_count, mid_count, fwd_count) in VALID_FORMATIONS

    def test_captain_in_lineup(self, squad_15_candidates):
        result = select_lineup(squad_15_candidates)
        assert result.captain_id in result.lineup_element_ids

    def test_vice_captain_in_lineup(self, squad_15_candidates):
        result = select_lineup(squad_15_candidates)
        assert result.vice_captain_id in result.lineup_element_ids

    def test_captain_not_vice_captain(self, squad_15_candidates):
        result = select_lineup(squad_15_candidates)
        assert result.captain_id != result.vice_captain_id

    def test_captain_is_highest_xp_starter(self, squad_15_candidates):
        """Captain should be the player with highest predicted points in lineup."""
        result = select_lineup(squad_15_candidates)
        cand_map = {c.element_id: c for c in squad_15_candidates}
        captain_xp = cand_map[result.captain_id].predicted_points
        for eid in result.lineup_element_ids:
            assert cand_map[eid].predicted_points <= captain_xp

    def test_bench_gk_first(self, squad_15_candidates):
        """Bench order should have backup GK first."""
        result = select_lineup(squad_15_candidates)
        cand_map = {c.element_id: c for c in squad_15_candidates}
        # One GK starts, one benched — benched GK should be first on bench
        bench_gks = [eid for eid in result.bench_element_ids if cand_map[eid].position == Position.GK]
        if bench_gks:
            assert result.bench_element_ids[0] in bench_gks

    def test_objective_value_positive(self, squad_15_candidates):
        result = select_lineup(squad_15_candidates)
        assert result.objective_value > 0

    def test_wrong_squad_size_raises(self):
        from tests.test_optimizer.conftest import _make_candidate
        squad = [_make_candidate(i, Position.DEF, 50, 1, 5.0) for i in range(10)]
        with pytest.raises(ValueError, match="exactly 15"):
            select_lineup(squad)

    def test_bench_outfield_sorted_by_xp(self, squad_15_candidates):
        """Non-GK bench players should be ordered by xP descending."""
        result = select_lineup(squad_15_candidates)
        cand_map = {c.element_id: c for c in squad_15_candidates}
        bench_outfield = [
            eid for eid in result.bench_element_ids
            if cand_map[eid].position != Position.GK
        ]
        xps = [cand_map[eid].predicted_points for eid in bench_outfield]
        assert xps == sorted(xps, reverse=True)
