"""R02 — free-transfer banking, hits, selling price, ChipState half/FH rules.

Official sources (fetched 2026-09-24):
- premierleague.com/en/news/4059225: "four free transfers saved ahead of GW29 but
  activate the Free Hit ... still have four free transfers for GW30" => on a
  WC/FH week the FT count carries UNCHANGED (kept, no +1).
- premierleague.com/en/news/4679879 (2026/27 chips): one chip per GW; FH not in
  GW1; FH1 in GW19 blocks FH2 in GW20; first set cannot be carried over.
- API: max_extra_free_transfers=4 (bank to 5), transfers_sell_on_fee=0.5.
"""

from __future__ import annotations

import pytest

from fpl_optimizer.engine.state import ChipState
from fpl_optimizer.engine.transfers import (
    bank_free_transfers,
    calculate_selling_price,
    calculate_transfer_cost,
)


class TestFreeTransferBanking:
    @pytest.mark.parametrize("ft,made,expected", [
        (1, 0, 2), (2, 0, 3), (3, 0, 4), (4, 0, 5), (5, 0, 5),  # roll, cap 5
        (1, 1, 1), (2, 1, 2), (2, 2, 1), (5, 5, 1), (5, 2, 4),  # consume then +1
        (1, 3, 1), (2, 6, 1),                                    # hits: floor at 0, +1
    ])
    def test_normal_week(self, ft, made, expected):
        assert bank_free_transfers(ft, made, False, False) == expected

    @pytest.mark.parametrize("ft", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("made", [0, 3, 15])
    def test_wildcard_week_keeps_count_unchanged(self, ft, made):
        assert bank_free_transfers(ft, made, True, False) == ft

    @pytest.mark.parametrize("ft", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("made", [0, 3, 15])
    def test_free_hit_week_keeps_count_unchanged(self, ft, made):
        # official example: 4 saved ahead of GW29, FH GW29 -> 4 in GW30
        assert bank_free_transfers(ft, made, False, True) == ft

    def test_accrual_resumes_after_chip_week(self):
        ft = 2
        ft = bank_free_transfers(ft, 11, True, False)   # WC week
        assert ft == 2
        ft = bank_free_transfers(ft, 0, False, False)   # normal week after
        assert ft == 3


class TestHits:
    @pytest.mark.parametrize("made,ft,hit", [
        (0, 1, 0), (1, 1, 0), (2, 1, 4), (3, 1, 8), (5, 5, 0), (6, 5, 4), (3, 2, 4),
    ])
    def test_minus_4_per_extra(self, made, ft, hit):
        assert calculate_transfer_cost(made, ft) == hit


class TestSellingPrice:
    @pytest.mark.parametrize("buy,now,sell", [
        (50, 50, 50), (50, 51, 50), (50, 52, 51), (50, 53, 51), (50, 55, 52),
        (50, 49, 49), (50, 45, 45),               # drops pass through in full
        (57, 58, 57), (155, 156, 155), (75, 77, 76), (41, 41, 41),  # our squad
    ])
    def test_purchase_plus_floor_half_gain(self, buy, now, sell):
        assert calculate_selling_price(buy, now) == sell


class TestChipStateHalves:
    def test_first_half_use_does_not_touch_second(self):
        cs = ChipState()
        cs.use_chip("wildcard", 11)
        assert not cs.is_available("wildcard", 12)
        assert cs.is_available("wildcard", 20)

    def test_second_half_chip_not_usable_in_first_half(self):
        cs = ChipState()
        cs.use_chip("bench_boost", 12)
        # first-half BB used: nothing else lets you use a BB before GW20
        assert not cs.is_available("bench_boost", 19)
        assert cs.is_available("bench_boost", 20)

    def test_expire_first_half(self):
        cs = ChipState()
        cs.expire_first_half()
        for chip in ("wildcard", "free_hit", "bench_boost", "triple_captain"):
            assert not cs.is_available(chip, 19)
            assert cs.is_available(chip, 20)

    def test_gw19_is_still_first_half(self):
        cs = ChipState()
        for chip in ("wildcard", "free_hit", "bench_boost", "triple_captain"):
            assert cs.is_available(chip, 19)


class TestFreeHitGw19Gw20:
    def test_fh19_blocks_fh20(self):
        cs = ChipState()
        cs.use_chip("free_hit", 19)
        cs.expire_first_half()
        assert not cs.is_available("free_hit", 20)
        assert cs.is_available("free_hit", 21)

    def test_fh18_does_not_block_fh20(self):
        cs = ChipState()
        cs.use_chip("free_hit", 18)
        cs.expire_first_half()
        assert cs.is_available("free_hit", 20)

    def test_wc19_does_not_block_wc20(self):
        cs = ChipState()
        cs.use_chip("wildcard", 19)
        cs.expire_first_half()
        assert cs.is_available("wildcard", 20)

    def test_copy_preserves_fh_marker(self):
        cs = ChipState()
        cs.use_chip("free_hit", 19)
        assert not cs.copy().is_available("free_hit", 20)
