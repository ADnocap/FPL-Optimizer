"""Tests for FotMob data collector."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fpl_optimizer.data.collectors.fotmob import (
    FotMobCollector,
    STAT_KEYS,
    _parse_stat_response,
)


def _make_fotmob_api_response(players: list[dict]) -> dict:
    """Build a FotMob API response with the given player entries."""
    stat_list = []
    for p in players:
        stat_list.append({
            "ParticipantName": p.get("name", ""),
            "StatValue": p.get("stat_value", ""),
            "SubStatValue": p.get("sub_stat_value", ""),
            "MinutesPlayed": p.get("minutes", ""),
            "MatchesPlayed": p.get("matches", ""),
        })
    return {"TopLists": [{"StatList": stat_list}]}


class TestParseStatResponse:
    """Tests for response parsing."""

    def test_parse_valid_response(self) -> None:
        raw = _make_fotmob_api_response([
            {"name": "Salah", "stat_value": "2.5", "sub_stat_value": "85%"},
            {"name": "Kane", "stat_value": "1.8", "sub_stat_value": "75%"},
        ])
        result = _parse_stat_response(raw)
        assert len(result) == 2
        assert result[0]["player_name"] == "Salah"
        assert result[0]["stat_value"] == "2.5"
        assert result[0]["sub_stat_value"] == "85%"
        assert result[1]["player_name"] == "Kane"

    def test_parse_empty_response(self) -> None:
        result = _parse_stat_response({"TopLists": []})
        assert result == []

    def test_parse_missing_top_lists(self) -> None:
        result = _parse_stat_response({})
        assert result == []


class TestFotMobCollector:
    """Tests for the collector with mocked HTTP."""

    def test_collect_season_saves_json(self, tmp_path: Path) -> None:
        """Successful collection saves a JSON file with 3 stat keys."""
        collector = FotMobCollector(data_dir=tmp_path)

        responses = []
        for _ in STAT_KEYS:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_fotmob_api_response([
                {"name": "Player A", "stat_value": "1.5", "sub_stat_value": "80%"},
            ])
            mock_resp.raise_for_status = MagicMock()
            responses.append(mock_resp)

        with patch.object(collector, "_request_with_retry", side_effect=responses):
            ok = collector.collect_season("2023-24")

        assert ok is True
        dest = tmp_path / "fotmob" / "2023-24.json"
        assert dest.exists()

        data = json.loads(dest.read_text(encoding="utf-8"))
        for key in STAT_KEYS:
            assert key in data
            assert len(data[key]) == 1
            assert data[key][0]["player_name"] == "Player A"

    def test_collect_season_uses_cache(self, tmp_path: Path) -> None:
        """If file already exists, skip fetching."""
        collector = FotMobCollector(data_dir=tmp_path)

        # Pre-create the cached file
        fotmob_dir = tmp_path / "fotmob"
        fotmob_dir.mkdir(parents=True)
        dest = fotmob_dir / "2023-24.json"
        dest.write_text('{"cached": true}', encoding="utf-8")

        with patch.object(collector, "_request_with_retry") as mock_req:
            ok = collector.collect_season("2023-24")

        assert ok is True
        mock_req.assert_not_called()

    def test_collect_season_unknown_season(self, tmp_path: Path) -> None:
        """Unknown season returns False."""
        collector = FotMobCollector(data_dir=tmp_path)
        ok = collector.collect_season("2010-11")
        assert ok is False

    def test_collect_all_iterates_seasons(self, tmp_path: Path) -> None:
        """collect_all processes each requested season."""
        collector = FotMobCollector(data_dir=tmp_path)

        with patch.object(collector, "collect_season", return_value=True) as mock:
            results = collector.collect_all(seasons=["2022-23", "2023-24"])

        assert results == {"2022-23": True, "2023-24": True}
        assert mock.call_count == 2
