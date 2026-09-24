"""Collector for vaastav Fantasy-Premier-League GitHub CSV data."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from fpl_optimizer.data.collectors.base import BaseCollector, RateLimiter, DEFAULT_DATA_DIR
from fpl_optimizer.data.downloader import download_season
from fpl_optimizer.data.schemas import CORE_COLUMNS
from fpl_optimizer.utils.constants import AVAILABLE_SEASONS, SEASON_FILES_REQUIRED

logger = logging.getLogger(__name__)


class VaastavCollector(BaseCollector):
    """Download and validate vaastav CSV data for all seasons."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        # vaastav is fast CDN — no aggressive rate-limiting needed
        super().__init__(data_dir=data_dir, rate_limiter=RateLimiter(calls_per_second=5.0))
        self.raw_dir = self.data_dir / "raw"

    def collect_season(self, season: str) -> bool:
        """Download + validate one season. Returns True if all files valid."""
        if season not in AVAILABLE_SEASONS:
            logger.error("Unknown season: %s", season)
            return False

        # Check cache — all 5 files must exist
        season_dir = self.raw_dir / season
        if self._season_cached(season_dir):
            logger.info("vaastav %s: already cached, skipping", season)
            return True

        logger.info("vaastav %s: downloading...", season)
        ok = download_season(season, self.raw_dir)
        if not ok:
            logger.error("vaastav %s: download failed", season)
            return False

        if not self._validate_season(season_dir, season):
            logger.error("vaastav %s: validation failed", season)
            return False

        logger.info("vaastav %s: OK", season)
        return True

    def collect_all(
        self, *, max_workers: int = 4, seasons: list[str] | None = None,
    ) -> dict[str, bool]:
        """Download seasons, skipping cached ones.

        Args:
            max_workers: Number of concurrent download threads.
            seasons: Subset of seasons to collect. Defaults to all.
        """
        target_seasons = seasons or list(AVAILABLE_SEASONS)
        results: dict[str, bool] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.collect_season, s): s for s in target_seasons
            }
            with tqdm(
                as_completed(futures),
                total=len(futures),
                desc="vaastav seasons",
                unit="season",
            ) as pbar:
                for future in pbar:
                    season = futures[future]
                    results[season] = future.result()
                    pbar.set_postfix_str(season)
        return results

    # ------------------------------------------------------------------

    def _season_cached(self, season_dir: Path) -> bool:
        """True if all required files exist and are non-empty."""
        return all(
            self._is_cached(season_dir / f) for f in SEASON_FILES_REQUIRED
        )

    def _validate_season(self, season_dir: Path, season: str) -> bool:
        """Check that merged_gw.csv contains the expected core columns."""
        merged = season_dir / "gws" / "merged_gw.csv"
        if not merged.exists():
            return False
        try:
            import pandas as pd

            df = pd.read_csv(merged, nrows=5)
            missing = set(CORE_COLUMNS) - set(df.columns)
            if missing:
                logger.warning(
                    "vaastav %s: merged_gw.csv missing columns: %s", season, missing
                )
                # Missing columns are tolerable — loader fills them with defaults
            return True
        except Exception as exc:
            logger.error("vaastav %s: failed to read merged_gw.csv: %s", season, exc)
            return False
