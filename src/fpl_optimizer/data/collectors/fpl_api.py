"""Collector for the official FPL API (fantasy.premierleague.com)."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from fpl_optimizer.data.collectors.base import BaseCollector, RateLimiter, DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

FPL_API_BASE = "https://fantasy.premierleague.com/api"


class FPLAPICollector(BaseCollector):
    """Collect data from the live FPL REST API.

    Note: the API only serves **current-season** data — there is no
    historical archive.  The ``season`` parameter tags the output files
    so they can be stored alongside past snapshots.
    """

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        # FPL API rate limit: ~1 request/sec is safe
        super().__init__(
            data_dir=data_dir,
            rate_limiter=RateLimiter(calls_per_second=1.0),
        )
        self.api_dir = self.data_dir / "fpl_api"

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    def collect_season(self, season: str, *, max_workers: int = 2) -> bool:
        """Collect bootstrap, fixtures, and element summaries for *season*."""
        ok_bootstrap = self._collect_bootstrap(season)
        ok_fixtures = self._collect_fixtures(season)
        ok_elements = self._collect_element_summaries(season, max_workers=max_workers)
        return ok_bootstrap and ok_fixtures and ok_elements

    def collect_all(self) -> dict[str, bool]:
        """Collect current-season data (API only serves one season)."""
        from fpl_optimizer.utils.constants import AVAILABLE_SEASONS

        current = AVAILABLE_SEASONS[-1]
        return {current: self.collect_season(current)}

    # ------------------------------------------------------------------
    # Bootstrap-static
    # ------------------------------------------------------------------

    def _collect_bootstrap(self, season: str) -> bool:
        dest = self.api_dir / "bootstrap" / f"{season}.json"
        if self._is_cached(dest):
            logger.info("FPL API bootstrap %s: cached", season)
            return True

        url = f"{FPL_API_BASE}/bootstrap-static/"
        try:
            resp = self._request_with_retry(url)
            data = resp.json()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
            n_players = len(data.get("elements", []))
            logger.info("FPL API bootstrap %s: saved (%d players)", season, n_players)
            return True
        except Exception as exc:
            logger.error("FPL API bootstrap %s: %s", season, exc)
            return False

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    def _collect_fixtures(self, season: str) -> bool:
        dest = self.api_dir / "fixtures" / f"{season}.json"
        if self._is_cached(dest):
            logger.info("FPL API fixtures %s: cached", season)
            return True

        url = f"{FPL_API_BASE}/fixtures/"
        try:
            resp = self._request_with_retry(url)
            data = resp.json()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.info("FPL API fixtures %s: saved (%d fixtures)", season, len(data))
            return True
        except Exception as exc:
            logger.error("FPL API fixtures %s: %s", season, exc)
            return False

    # ------------------------------------------------------------------
    # Element summaries (per-player history + fixtures)
    # ------------------------------------------------------------------

    def _collect_element_summaries(
        self, season: str, *, max_workers: int = 2,
    ) -> bool:
        """Download element-summary for every player in the bootstrap file.

        Args:
            season: Season tag for output files.
            max_workers: Concurrent download threads (share one rate limiter).
        """
        bootstrap_path = self.api_dir / "bootstrap" / f"{season}.json"
        if not bootstrap_path.exists():
            logger.error(
                "FPL API element summaries %s: bootstrap missing, fetch it first",
                season,
            )
            return False

        data = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        elements = data.get("elements", [])
        if not elements:
            logger.warning("FPL API element summaries %s: no players in bootstrap", season)
            return True

        summary_dir = self.api_dir / "element_summaries" / season
        summary_dir.mkdir(parents=True, exist_ok=True)

        def _fetch_one(player: dict) -> bool:
            eid = player["id"]
            dest = summary_dir / f"{eid}.json"
            if self._is_cached(dest):
                return True
            url = f"{FPL_API_BASE}/element-summary/{eid}/"
            try:
                resp = self._request_with_retry(url)
                dest.write_text(
                    json.dumps(resp.json(), indent=2), encoding="utf-8"
                )
                return True
            except Exception as exc:
                logger.warning(
                    "FPL API element summary %s/%d: %s", season, eid, exc
                )
                return False

        all_ok = True
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_one, p): p for p in elements}
            with tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"FPL API elements {season}",
                unit="player",
            ) as pbar:
                for future in pbar:
                    if not future.result():
                        all_ok = False

        logger.info("FPL API element summaries %s: done", season)
        return all_ok
