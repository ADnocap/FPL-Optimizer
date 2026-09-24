"""Collector for Understat xG/xA data via the understatapi package."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from fpl_optimizer.data.collectors.base import BaseCollector, RateLimiter, DEFAULT_DATA_DIR
from fpl_optimizer.utils.constants import AVAILABLE_SEASONS

logger = logging.getLogger(__name__)

# Understat uses the start-year as its season key: "2016-17" → "2016"
SEASON_TO_UNDERSTAT: dict[str, str] = {
    s: s.split("-")[0] for s in AVAILABLE_SEASONS
}

# Understat league name for the English Premier League
LEAGUE = "EPL"


class UnderstatCollector(BaseCollector):
    """Collect xG/xA data from Understat.

    Phase A — league-level season aggregates (fast: 9 API calls total).
    Phase B — per-match player data (slow: ~500 players × 9 seasons).
    """

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        # Understat rate limit: ~1 request per 1.5 seconds
        super().__init__(
            data_dir=data_dir,
            rate_limiter=RateLimiter(calls_per_second=1.0 / 1.5),
        )
        self.league_dir = self.data_dir / "understat" / "league"
        self.players_dir = self.data_dir / "understat" / "players"

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    def collect_season(self, season: str, *, per_match: bool = False) -> bool:
        """Collect Understat data for one season.

        Args:
            season: e.g. ``"2022-23"``
            per_match: If True, also collect per-match player data (Phase B).
        """
        ok = self._collect_league_season(season)
        if ok and per_match:
            ok = self._collect_player_matches(season)
        return ok

    def collect_all(
        self, *, max_workers: int = 1, seasons: list[str] | None = None,
    ) -> dict[str, bool]:
        """Collect league-level aggregates for seasons.

        Args:
            max_workers: Concurrent season threads (share one rate limiter).
            seasons: Subset of seasons. Defaults to all.
        """
        target = seasons or list(AVAILABLE_SEASONS)
        results: dict[str, bool] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.collect_season, s): s for s in target
            }
            with tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Understat league",
                unit="season",
            ) as pbar:
                for future in pbar:
                    season = futures[future]
                    results[season] = future.result()
                    pbar.set_postfix_str(season)
        return results

    def collect_all_per_match(
        self, *, max_workers: int = 2, seasons: list[str] | None = None,
    ) -> dict[str, bool]:
        """Collect per-match player data for seasons.

        Args:
            max_workers: Concurrent season threads (share one rate limiter).
            seasons: Subset of seasons. Defaults to all.
        """
        target = seasons or list(AVAILABLE_SEASONS)
        results: dict[str, bool] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.collect_season, s, per_match=True): s
                for s in target
            }
            with tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Understat per-match",
                unit="season",
            ) as pbar:
                for future in pbar:
                    season = futures[future]
                    results[season] = future.result()
                    pbar.set_postfix_str(season)
        return results

    # ------------------------------------------------------------------
    # Live season: forced, incremental weekly refresh
    # ------------------------------------------------------------------

    def refresh_live_season(self, season: str) -> dict[str, int]:
        """Re-fetch the CURRENT season (the normal collectors skip cached files).

        The league table is always re-downloaded; per-match files are
        re-fetched only for players whose league appearance count grew since
        the cached file was written (or who have no file yet), so a weekly run
        costs a few minutes rather than ~10. Returns counts for logging.
        """
        from understatapi import UnderstatClient

        us_season = SEASON_TO_UNDERSTAT.get(season)
        if us_season is None:
            raise ValueError(f"No Understat mapping for season {season}")

        self.rate_limiter.wait()
        with UnderstatClient() as client:
            league = client.league(league=LEAGUE).get_player_data(us_season)
        dest = self.league_dir / f"{season}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(league, indent=2), encoding="utf-8")

        season_dir = self.players_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)
        stale = []
        for player in league:
            pid = str(player.get("id", ""))
            if not pid:
                continue
            path = season_dir / f"{pid}.json"
            games = int(player.get("games") or 0)
            try:
                cached = len(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else -1
            except (json.JSONDecodeError, OSError):
                cached = -1
            if cached < games:
                stale.append(pid)

        failed = 0
        with UnderstatClient() as client:
            for pid in tqdm(stale, desc=f"Understat refresh {season}", unit="player"):
                try:
                    self.rate_limiter.wait()
                    matches = [
                        m for m in client.player(player=pid).get_match_data()
                        if str(m.get("season", "")) == us_season
                    ]
                    (season_dir / f"{pid}.json").write_text(
                        json.dumps(matches, indent=2), encoding="utf-8"
                    )
                except Exception as exc:  # one bad player must not stop the run
                    failed += 1
                    logger.warning("Understat refresh %s player %s: %s", season, pid, exc)
        logger.info(
            "Understat refresh %s: %d players, %d re-fetched, %d failed",
            season, len(league), len(stale), failed,
        )
        return {"players": len(league), "refetched": len(stale), "failed": failed}

    # ------------------------------------------------------------------
    # Phase A: league-level aggregates
    # ------------------------------------------------------------------

    def _collect_league_season(self, season: str) -> bool:
        us_season = SEASON_TO_UNDERSTAT.get(season)
        if us_season is None:
            logger.error("No Understat mapping for season %s", season)
            return False

        dest = self.league_dir / f"{season}.json"
        if self._is_cached(dest):
            logger.info("Understat league %s: cached", season)
            return True

        try:
            from understatapi import UnderstatClient

            logger.info("Understat league %s: fetching...", season)
            self.rate_limiter.wait()
            with UnderstatClient() as client:
                data = client.league(league=LEAGUE).get_player_data(us_season)

            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.info("Understat league %s: saved %d players", season, len(data))
            return True
        except ImportError:
            logger.error(
                "understatapi not installed. Install with: pip install 'fpl-optimizer[data]'"
            )
            return False
        except Exception as exc:
            logger.error("Understat league %s: %s", season, exc)
            return False

    # ------------------------------------------------------------------
    # Phase B: per-match player data
    # ------------------------------------------------------------------

    def _collect_player_matches(self, season: str) -> bool:
        """Fetch per-match xG/xA for every player in the league-level file."""
        us_season = SEASON_TO_UNDERSTAT.get(season)
        if us_season is None:
            return False

        league_file = self.league_dir / f"{season}.json"
        if not league_file.exists():
            logger.error(
                "Understat per-match %s: league file missing, run Phase A first", season
            )
            return False

        players = json.loads(league_file.read_text(encoding="utf-8"))
        season_player_dir = self.players_dir / season
        season_player_dir.mkdir(parents=True, exist_ok=True)

        all_ok = True
        try:
            from understatapi import UnderstatClient
        except ImportError:
            logger.error(
                "understatapi not installed. Install with: pip install 'fpl-optimizer[data]'"
            )
            return False

        with UnderstatClient() as client:
            for player in tqdm(players, desc=f"Understat players {season}", unit="player"):
                player_id = str(player.get("id", ""))
                if not player_id:
                    continue
                dest = season_player_dir / f"{player_id}.json"
                if self._is_cached(dest):
                    continue

                try:
                    self.rate_limiter.wait()
                    all_matches = client.player(player=player_id).get_match_data()
                    # API returns all seasons; filter to the one we want
                    match_data = [
                        m for m in all_matches
                        if str(m.get("season", "")) == us_season
                    ]
                    dest.write_text(
                        json.dumps(match_data, indent=2), encoding="utf-8"
                    )
                except Exception as exc:
                    logger.warning(
                        "Understat per-match %s player %s: %s", season, player_id, exc
                    )
                    all_ok = False

        logger.info("Understat per-match %s: done", season)
        return all_ok
