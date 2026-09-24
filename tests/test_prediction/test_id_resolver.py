"""Tests for IDResolver cross-source ID mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from fpl_optimizer.prediction.id_resolver import IDResolver


@pytest.fixture
def resolver(pred_data_dir: Path) -> IDResolver:
    return IDResolver(pred_data_dir)


class TestIDResolver:
    def test_code_from_element_id(self, resolver: IDResolver) -> None:
        assert resolver.code_from_element_id("2019-20", 10) == 100  # Kane
        assert resolver.code_from_element_id("2023-24", 24) == 200  # Salah
        assert resolver.code_from_element_id("2019-20", 999) is None  # not found

    def test_element_id_from_code(self, resolver: IDResolver) -> None:
        assert resolver.element_id_from_code(100, "2019-20") == 10
        assert resolver.element_id_from_code(100, "2024-25") == 15
        assert resolver.element_id_from_code(100, "2016-17") is None  # no data

    def test_understat_id(self, resolver: IDResolver) -> None:
        assert resolver.understat_id(100) == 1234
        assert resolver.understat_id(200) == 5678
        assert resolver.understat_id(99999) is None

    def test_fbref_id(self, resolver: IDResolver) -> None:
        assert resolver.fbref_id(100) == "abc123"
        assert resolver.fbref_id(300) == "ghi789"
        assert resolver.fbref_id(99999) is None

    def test_player_name(self, resolver: IDResolver) -> None:
        assert resolver.player_name(100) == "Kane"
        assert resolver.player_name(99999) == "Unknown"

    def test_all_codes_for_season(self, resolver: IDResolver) -> None:
        codes = resolver.all_codes_for_season("2023-24")
        assert set(codes) == {100, 200, 300, 400}

        codes_early = resolver.all_codes_for_season("2016-17")
        assert codes_early == []

    def test_all_codes(self, resolver: IDResolver) -> None:
        assert resolver.all_codes() == {100, 200, 300, 400}

    def test_missing_map_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            IDResolver(tmp_path)

    def test_bidirectional_consistency(self, resolver: IDResolver) -> None:
        """code -> eid -> code roundtrip."""
        for season in ["2019-20", "2023-24", "2024-25"]:
            for code in [100, 200, 300, 400]:
                eid = resolver.element_id_from_code(code, season)
                if eid is not None:
                    assert resolver.code_from_element_id(season, eid) == code
