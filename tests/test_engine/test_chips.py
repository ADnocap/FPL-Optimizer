"""Tests for chip activation, availability, Free Hit revert, and GW19 expiry."""

import pytest

from fpl_optimizer.engine.chips import (
    activate_chip,
    clear_active_chip,
    handle_gw19_expiry,
    revert_free_hit,
    validate_chip,
)
from fpl_optimizer.engine.state import ChipState, GameState


class TestChipAvailability:
    def test_all_chips_available_initially(self):
        chips = ChipState()
        for chip in ["wildcard", "free_hit", "bench_boost", "triple_captain"]:
            assert chips.is_available(chip, 1)   # First half
            assert chips.is_available(chip, 20)  # Second half

    def test_chip_used_first_half(self):
        chips = ChipState()
        chips.use_chip("wildcard", 5)
        assert not chips.is_available("wildcard", 5)
        assert not chips.is_available("wildcard", 10)
        # Second half still available
        assert chips.is_available("wildcard", 20)

    def test_chip_used_second_half(self):
        chips = ChipState()
        chips.use_chip("bench_boost", 25)
        # First half still available
        assert chips.is_available("bench_boost", 5)
        assert not chips.is_available("bench_boost", 25)

    def test_invalid_chip_name(self):
        chips = ChipState()
        with pytest.raises(ValueError):
            chips.is_available("invalid_chip", 1)


class TestGW19Expiry:
    def test_expire_unused_first_half_chips(self, sample_state):
        sample_state.current_gw = 19
        new_state = handle_gw19_expiry(sample_state)
        # All first-half chips should be expired
        assert not new_state.chips.wildcard[0]
        assert not new_state.chips.free_hit[0]
        assert not new_state.chips.bench_boost[0]
        assert not new_state.chips.triple_captain[0]
        # Second-half chips unaffected
        assert new_state.chips.wildcard[1]
        assert new_state.chips.free_hit[1]

    def test_no_expiry_before_gw19(self, sample_state):
        sample_state.current_gw = 18
        new_state = handle_gw19_expiry(sample_state)
        assert new_state.chips.wildcard[0]  # Still available

    def test_already_used_chip_stays_used(self, sample_state):
        sample_state.chips.use_chip("wildcard", 5)
        sample_state.current_gw = 19
        new_state = handle_gw19_expiry(sample_state)
        assert not new_state.chips.wildcard[0]


class TestActivateChip:
    def test_activate_wildcard(self, sample_state):
        new_state = activate_chip(sample_state, "wildcard")
        assert new_state.active_chip == "wildcard"
        assert not new_state.chips.is_available("wildcard", 1)

    def test_activate_free_hit_stashes_squad(self, sample_state):
        original_squad = sample_state.squad.copy()
        new_state = activate_chip(sample_state, "free_hit")
        assert new_state.active_chip == "free_hit"
        assert new_state.free_hit_stash is not None
        # Stashed squad should match original
        for i, p in enumerate(new_state.free_hit_stash.players):
            assert p.element_id == original_squad.players[i].element_id

    def test_cannot_use_two_chips(self, sample_state):
        state = activate_chip(sample_state, "wildcard")
        with pytest.raises(ValueError, match="already using"):
            activate_chip(state, "bench_boost")

    def test_cannot_use_unavailable_chip(self, sample_state):
        sample_state.chips.use_chip("wildcard", 1)
        with pytest.raises(ValueError, match="not available"):
            activate_chip(sample_state, "wildcard")

    def test_one_chip_per_gw(self, sample_state):
        """Only one chip allowed per gameweek."""
        state = activate_chip(sample_state, "triple_captain")
        error = validate_chip(state, "bench_boost")
        assert error is not None
        assert "already using" in error


class TestFreeHitRevert:
    def test_revert_squad(self, sample_state):
        # Activate FH, modify squad, then revert
        state = activate_chip(sample_state, "free_hit")
        # Modify a player (simulate a transfer)
        original_id = state.squad.players[6].element_id
        state.squad.players[6].element_id = 999
        # Revert
        reverted = revert_free_hit(state)
        assert reverted.free_hit_stash is None
        assert reverted.squad.players[6].element_id == original_id

    def test_no_stash_no_revert(self, sample_state):
        result = revert_free_hit(sample_state)
        assert result.squad == sample_state.squad


class TestClearActiveChip:
    def test_clear(self, sample_state):
        sample_state.active_chip = "wildcard"
        cleared = clear_active_chip(sample_state)
        assert cleared.active_chip is None
