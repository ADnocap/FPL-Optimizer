"""Collector for FBref advanced stats via the soccerdata package."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from fpl_optimizer.data.collectors.base import BaseCollector, RateLimiter, DEFAULT_DATA_DIR
from fpl_optimizer.utils.constants import AVAILABLE_SEASONS

logger = logging.getLogger(__name__)

# soccerdata uses hyphenated season strings like "2023-2024"
SEASON_TO_FBREF: dict[str, str] = {
    s: f"20{s.split('-')[0][2:]}-20{s.split('-')[1]}" for s in AVAILABLE_SEASONS
}

STAT_TYPES = ["standard", "passing", "defense", "shooting"]


class FBrefCollector(BaseCollector):
    """Collect advanced stats from FBref (progressive passes, SCA, etc.).

    Uses the ``soccerdata`` package.  FBref aggressively rate-limits
    scrapers so this collector is intentionally slow (1 req / 6 s).
    """

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        # Very conservative: FBref blocks at ~10 req/min
        super().__init__(
            data_dir=data_dir,
            rate_limiter=RateLimiter(calls_per_second=1.0 / 6.0),
        )
        self.fbref_dir = self.data_dir / "fbref"

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    def collect_season(self, season: str) -> bool:
        fb_season = SEASON_TO_FBREF.get(season)
        if fb_season is None:
            logger.error("No FBref mapping for season %s", season)
            return False

        all_ok = True
        for stat_type in STAT_TYPES:
            if not self._collect_stat(season, fb_season, stat_type):
                all_ok = False
        return all_ok

    def collect_all(
        self, *, max_workers: int = 2, seasons: list[str] | None = None,
    ) -> dict[str, bool]:
        """Collect all seasons, parallelising across workers.

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
                desc="FBref seasons",
                unit="season",
            ) as pbar:
                for future in pbar:
                    season = futures[future]
                    results[season] = future.result()
                    pbar.set_postfix_str(season)
        return results

    # ------------------------------------------------------------------

    def _collect_stat(self, season: str, fb_season: str, stat_type: str) -> bool:
        dest = self.fbref_dir / f"{season}_{stat_type}.parquet"
        if self._is_cached(dest):
            logger.info("FBref %s %s: cached", season, stat_type)
            return True

        try:
            import soccerdata as sd
        except ImportError:
            logger.error(
                "soccerdata not installed. Install with: pip install 'fpl-optimizer[data]'"
            )
            return False

        try:
            logger.info("FBref %s %s: fetching...", season, stat_type)
            self.rate_limiter.wait()
            fbref = sd.FBref(leagues="ENG-Premier League", seasons=fb_season)
            df = self._read_stat(fbref, stat_type)
            if df is None or df.empty:
                logger.warning("FBref %s %s: empty result", season, stat_type)
                return False

            dest.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dest)
            logger.info("FBref %s %s: saved (%d rows)", season, stat_type, len(df))
            return True
        except Exception as exc:
            logger.error("FBref %s %s: %s", season, stat_type, exc)
            return False

    @staticmethod
    def _read_stat(fbref, stat_type: str):
        """Dispatch to the correct soccerdata method."""
        if stat_type not in STAT_TYPES:
            return None
        return fbref.read_player_season_stats(stat_type=stat_type)
