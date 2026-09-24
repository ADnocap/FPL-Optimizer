"""Tests for transfer logic: sell prices, free transfer banking, hit costs."""

import pytest

from fpl_optimizer.engine.transfers import (
    apply_transfers,
    bank_free_transfers,
    calculate_selling_price,
    calculate_transfer_cost,
)
from fpl_optimizer.utils.constants import MAX_FREE_TRANSFERS, TRANSFER_HIT_COST


class TestSellingPrice:
    def test_no_appreciation(self):
        assert calculate_selling_price(50, 50) == 50

    def test_depreciation(self):
        # Bought at 50, now worth 45 — sell at 45
        assert calculate_selling_price(50, 45) == 45

    def test_appreciation_even(self):
        # Bought at 50, now worth 60 — sell at 50 + floor(10/2) = 55
        assert calculate_selling_price(50, 60) == 55

    def test_appreciation_odd(self):
        # Bought at 50, now worth 53 — sell at 50 + floor(3/2) = 51
        assert calculate_selling_price(50, 53) == 51

    def test_appreciation_one(self):
        # Bought at 50, now worth 51 — sell at 50 + floor(1/2) = 50
        assert calculate_selling_price(50, 51) == 50

    def test_large_appreciation(self):
        # Bought at 60, now worth 100 — sell at 60 + floor(40/2) = 80
        assert calculate_selling_price(60, 100) == 80


class TestTransferCost:
    def test_no_transfers(self):
        assert calculate_transfer_cost(0, 1) == 0

    def test_within_free_allowance(self):
        assert calculate_transfer_cost(1, 1) == 0
        assert calculate_transfer_cost(2, 3) == 0

    def test_one_extra(self):
        assert calculate_transfer_cost(2, 1) == TRANSFER_HIT_COST

    def test_multiple_extra(self):
        assert calculate_transfer_cost(5, 2) == 3 * TRANSFER_HIT_COST

    def test_zero_free_transfers(self):
        assert calculate_transfer_cost(1, 0) == TRANSFER_HIT_COST


class TestBankFreeTransfers:
    def test_basic_banking(self):
        # Start with 1, make 0 transfers -> bank to 2
        assert bank_free_transfers(1, 0, False, False) == 2

    def test_use_one_then_bank(self):
        # Start with 2, make 1 transfer -> 1 remaining + 1 = 2
        assert bank_free_transfers(2, 1, False, False) == 2

    def test_bank_up_to_max(self):
        # Start with 4, make 0 -> min(5, 5) = 5
        assert bank_free_transfers(4, 0, False, False) == 5

    def test_cap_at_max(self):
        # Start with 5, make 0 -> already at max
        assert bank_free_transfers(5, 0, False, False) == min(6, MAX_FREE_TRANSFERS)

    def test_wildcard_doesnt_reset(self):
        # 2025/26: WC doesn't reset banked transfers
        assert bank_free_transfers(3, 5, True, False) == 3

    def test_free_hit_doesnt_reset(self):
        # 2025/26: FH doesn't reset banked transfers
        assert bank_free_transfers(4, 3, False, True) == 4

    def test_use_all_then_bank_one(self):
        # Start with 2, make 2 -> 0 remaining + 1 = 1
        assert bank_free_transfers(2, 2, False, False) == 1

    def test_overspend_free_transfers(self):
        # Start with 1, make 3 -> 0 remaining + 1 = 1
        assert bank_free_transfers(1, 3, False, False) == 1


class TestApplyTransfers:
    def test_valid_transfer(self, sample_state, loader):
        # Transfer out DEF5 (element 7, team 3) for Extra_DEF1 (element 16, team 7)
        team_map = loader._team_map
        new_state, hit = apply_transfers(
            sample_state, [7], [16], loader, team_map
        )
        assert hit == 0  # 1 free transfer available
        # Check player was replaced
        assert new_state.squad.find_player_idx(16) is not None
        assert new_state.squad.find_player_idx(7) is None

    def test_transfer_hit_cost(self, sample_state, loader):
        team_map = loader._team_map
        # 2 transfers with 1 free = 1 hit
        new_state, hit = apply_transfers(
            sample_state, [7, 12], [16, 17], loader, team_map
        )
        assert hit == TRANSFER_HIT_COST

    def test_no_transfers(self, sample_state, loader):
        team_map = loader._team_map
        new_state, hit = apply_transfers(sample_state, [], [], loader, team_map)
        assert hit == 0

    def test_mismatched_lengths(self, sample_state, loader):
        team_map = loader._team_map
        with pytest.raises(ValueError, match="same length"):
            apply_transfers(sample_state, [7], [16, 17], loader, team_map)

    def test_player_not_in_squad(self, sample_state, loader):
        team_map = loader._team_map
        with pytest.raises(ValueError, match="not in squad"):
            apply_transfers(sample_state, [999], [16], loader, team_map)
