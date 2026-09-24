"""Collector for FotMob player stat data via their public API."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from tqdm import tqdm

from fpl_optimizer.data.collectors.base import BaseCollector, RateLimiter, DEFAULT_DATA_DIR
from fpl_optimizer.utils.constants import AVAILABLE_SEASONS

logger = logging.getLogger(__name__)

# FotMob uses internal tournament IDs for each PL season.
# Discovered via the seasonStatLinks in the FotMob leagues API.
SEASON_TO_FOTMOB_ID: dict[str, str] = {
    "2016-17": "10418",
    "2017-18": "11522",
    "2018-19": "12776",
    "2019-20": "14022",
    "2020-21": "15382",
    "2021-22": "16390",
    "2022-23": "17664",
    "2023-24": "20720",
    "2024-25": "23685",
    "2025-26": "27110",
    "2026-27": "36781",
}

# Premier League = league 47 on FotMob
_LEAGUE_ID = "47"

# The three stat endpoints we need for prior-season features
STAT_KEYS = ("accurate_pass", "outfielder_block", "accurate_long_balls")

_BASE_URL = "https://data.fotmob.com/stats/{league}/season/{season_id}/{stat_key}.json"


class FotMobCollector(BaseCollector):
    """Collect passing / blocking / long-ball stats from FotMob.

    FotMob's public stats API requires no authentication — just a
    browser-like User-Agent header.
    """

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        # Conservative rate limit: 1 request per 3 seconds
        super().__init__(
            data_dir=data_dir,
            rate_limiter=RateLimiter(calls_per_second=1.0 / 3.0),
        )
        self.fotmob_dir = self.data_dir / "fotmob"

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    def collect_season(self, season: str) -> bool:
        """Collect FotMob stats for one season.

        Fetches 3 stat endpoints and saves a single JSON file per season.
        """
        fm_id = SEASON_TO_FOTMOB_ID.get(season)
        if fm_id is None:
            logger.error("No FotMob mapping for season %s", season)
            return False

        dest = self.fotmob_dir / f"{season}.json"
        if self._is_cached(dest):
            logger.info("FotMob %s: cached", season)
            return True

        season_data: dict[str, list[dict]] = {}
        all_ok = True

        for stat_key in STAT_KEYS:
            url = _BASE_URL.format(
                league=_LEAGUE_ID, season_id=fm_id, stat_key=stat_key,
            )
            try:
                resp = self._request_with_retry(
                    url, max_retries=3, timeout=30,
                )
                raw = resp.json()
                players = _parse_stat_response(raw)
                season_data[stat_key] = players
                logger.info(
                    "FotMob %s %s: %d players", season, stat_key, len(players),
                )
            except Exception as exc:
                logger.warning("FotMob %s %s: %s", season, stat_key, exc)
                season_data[stat_key] = []
                all_ok = False

        # Only save if at least one stat has data (avoid caching empty results)
        has_data = any(len(v) > 0 for v in season_data.values())
        if has_data:
            self.fotmob_dir.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(season_data, indent=2), encoding="utf-8")
            logger.info("FotMob %s: saved", season)
        else:
            logger.warning("FotMob %s: no data collected, skipping save", season)

        return all_ok and has_data

    def collect_all(
        self, *, max_workers: int = 1, seasons: list[str] | None = None,
    ) -> dict[str, bool]:
        """Collect FotMob data for all seasons (sequential, rate-limited)."""
        target = seasons or list(AVAILABLE_SEASONS)
        results: dict[str, bool] = {}

        for season in tqdm(target, desc="FotMob", unit="season"):
            results[season] = self.collect_season(season)

        return results

    # ------------------------------------------------------------------
    # HTTP override — add User-Agent header
    # ------------------------------------------------------------------

    def _request_with_retry(
        self,
        url: str,
        *,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """GET with a browser-like User-Agent header."""
        import requests

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            self.rate_limiter.wait()
            try:
                resp = requests.get(
                    url,
                    timeout=timeout,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                import time
                wait = 2 ** attempt
                logger.warning(
                    "Attempt %d/%d failed for %s: %s — retrying in %ds",
                    attempt + 1, max_retries, url, exc, wait,
                )
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]


def _parse_stat_response(raw: dict) -> list[dict]:
    """Extract player entries from a FotMob stat response.

    Response shape::

        {"TopLists": [{"StatList": [{...player...}, ...]}]}

    Each player entry has StatValue, SubStatValue, ParticipantName,
    MinutesPlayed, MatchesPlayed (among others).
    """
    players: list[dict] = []

    top_lists = raw.get("TopLists", [])
    for top_list in top_lists:
        for entry in top_list.get("StatList", []):
            player: dict = {
                "player_name": entry.get("ParticipantName", ""),
                "stat_value": entry.get("StatValue", ""),
                "sub_stat_value": entry.get("SubStatValue", ""),
                "minutes_played": entry.get("MinutesPlayed", ""),
                "matches_played": entry.get("MatchesPlayed", ""),
            }
            players.append(player)

    return players
