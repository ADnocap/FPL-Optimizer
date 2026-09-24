"""Orchestrate data collection across all sources."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from fpl_optimizer.data.collectors.base import DEFAULT_DATA_DIR
from fpl_optimizer.data.collectors.id_mapping import PlayerIDMapper
from fpl_optimizer.data.collectors.vaastav import VaastavCollector
from fpl_optimizer.data.collectors.understat import UnderstatCollector
from fpl_optimizer.data.collectors.fpl_api import FPLAPICollector
from fpl_optimizer.data.collectors.fbref import FBrefCollector
from fpl_optimizer.data.collectors.fotmob import FotMobCollector
from fpl_optimizer.data.collectors.odds import OddsCollector
from fpl_optimizer.utils.constants import AVAILABLE_SEASONS

logger = logging.getLogger(__name__)

ALL_SOURCES = ("vaastav", "understat", "fpl_api", "fbref", "fotmob", "odds")


class DataOrchestrator:
    """Run collectors in priority order and report coverage.

    Priority:
        1. ID maps (prerequisite for linking sources)
        2. vaastav (foundation)
        3. Understat league-level (fills xG gap)
        4. FPL API (FDR ratings)
        5. Understat per-match (per-GW granularity)
        6. FBref (optional)
    """

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        sources: tuple[str, ...] | list[str] = ALL_SOURCES,
        seasons: list[str] | None = None,
        understat_per_match: bool = False,
        max_workers: int = 4,
    ) -> None:
        self.data_dir = data_dir
        self.sources = [s.lower().strip() for s in sources]
        self.seasons = seasons or list(AVAILABLE_SEASONS)
        self.understat_per_match = understat_per_match
        self.max_workers = max_workers

        self.id_mapper = PlayerIDMapper(data_dir=data_dir)
        self.vaastav = VaastavCollector(data_dir=data_dir)
        self.understat = UnderstatCollector(data_dir=data_dir)
        self.fpl_api = FPLAPICollector(data_dir=data_dir)
        self.fbref = FBrefCollector(data_dir=data_dir)
        self.fotmob = FotMobCollector(data_dir=data_dir)
        self.odds = OddsCollector(data_dir=data_dir)

    def _build_steps(self) -> list[tuple[str, callable]]:
        """Build the ordered list of (name, callable) collection steps."""
        steps: list[tuple[str, callable]] = []

        # 1. ID maps — always collected (needed for cross-referencing)
        steps.append(("id_maps", self._step_id_maps))

        # 2-4. Independent sources run in parallel
        steps.append(("sources", self._step_parallel_sources))

        # 5. Odds (depends on vaastav for GW dates from merged_gw.csv)
        if "odds" in self.sources:
            steps.append(("odds", self._step_odds))

        # 6. Understat per-match (depends on Understat league being done)
        if "understat" in self.sources and self.understat_per_match:
            steps.append(("understat_per_match", self._step_understat_per_match))

        return steps

    def _step_id_maps(self) -> dict[str, bool]:
        logger.info("=== ID maps ===")
        return self.id_mapper.collect_all()

    def _step_parallel_sources(self) -> dict[str, bool] | None:
        """Run vaastav, understat-league, fpl_api, fbref in parallel."""
        source_fns: dict[str, callable] = {}
        if "vaastav" in self.sources:
            source_fns["vaastav"] = self._collect_vaastav
        if "understat" in self.sources:
            source_fns["understat_league"] = self._collect_understat_league
        if "fpl_api" in self.sources:
            source_fns["fpl_api"] = self._collect_fpl_api
        if "fbref" in self.sources:
            source_fns["fbref"] = self._collect_fbref
        if "fotmob" in self.sources:
            source_fns["fotmob"] = self._collect_fotmob

        if not source_fns:
            return None

        all_results: dict[str, dict[str, bool]] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(source_fns))) as pool:
            futures = {
                pool.submit(fn): name for name, fn in source_fns.items()
            }
            with tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Sources",
                unit="source",
            ) as pbar:
                for future in pbar:
                    name = futures[future]
                    pbar.set_postfix_str(name)
                    all_results[name] = future.result()

        # Flatten into the top-level results dict (caller will merge)
        # Return a sentinel — we'll merge in run()
        self._parallel_results = all_results
        return None

    def run(self) -> dict[str, dict[str, bool]]:
        """Execute collection in priority order. Returns nested results."""
        results: dict[str, dict[str, bool]] = {}
        self._parallel_results: dict[str, dict[str, bool]] = {}

        steps = self._build_steps()
        overall = tqdm(steps, desc="Overall progress", unit="step")

        for step_name, step_fn in overall:
            overall.set_postfix_str(step_name)
            step_results = step_fn()
            if step_results is not None:
                results[step_name] = step_results

        # Merge parallel source results
        results.update(self._parallel_results)

        self.print_status(results)
        return results

    # ------------------------------------------------------------------
    # Individual source collectors (called from thread pool)
    # ------------------------------------------------------------------

    def _collect_vaastav(self) -> dict[str, bool]:
        logger.info("=== vaastav ===")
        return self.vaastav.collect_all(
            max_workers=self.max_workers, seasons=self.seasons,
        )

    def _collect_understat_league(self) -> dict[str, bool]:
        logger.info("=== Understat league-level ===")
        return self.understat.collect_all(
            max_workers=self.max_workers, seasons=self.seasons,
        )

    def _collect_fpl_api(self) -> dict[str, bool]:
        logger.info("=== FPL API ===")
        from fpl_optimizer.utils.constants import AVAILABLE_SEASONS

        current = AVAILABLE_SEASONS[-1]
        return {current: self.fpl_api.collect_season(current, max_workers=self.max_workers)}

    def _collect_fbref(self) -> dict[str, bool]:
        logger.info("=== FBref ===")
        # FBref must stay sequential — soccerdata makes its own HTTP
        # requests internally, so parallel workers bypass our rate limiter
        # and trigger 403 blocks.
        return self.fbref.collect_all(
            max_workers=1, seasons=self.seasons,
        )

    def _collect_fotmob(self) -> dict[str, bool]:
        logger.info("=== FotMob ===")
        return self.fotmob.collect_all(
            max_workers=1, seasons=self.seasons,
        )

    def _step_odds(self) -> dict[str, bool]:
        logger.info("=== Odds (historical Pinnacle) ===")
        return self.odds.collect_all(max_workers=1, seasons=self.seasons)

    def _step_understat_per_match(self) -> dict[str, bool]:
        logger.info("=== Understat per-match ===")
        return self.understat.collect_all_per_match(
            max_workers=self.max_workers, seasons=self.seasons,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_status(self, results: dict[str, dict[str, bool]]) -> None:
        """Print a human-readable summary of collection results."""
        print("\n" + "=" * 60)
        print("DATA COLLECTION STATUS")
        print("=" * 60)

        for source, season_results in results.items():
            ok = sum(1 for v in season_results.values() if v)
            total = len(season_results)
            status = "OK" if ok == total else "PARTIAL" if ok > 0 else "FAILED"
            print(f"  {source:25s}  {ok}/{total} {status}")

        # Coverage: check what's on disk
        print("\n--- File Coverage ---")
        self._print_file_coverage()
        print("=" * 60 + "\n")

    def _print_file_coverage(self) -> None:
        raw_dir = self.data_dir / "raw"
        us_league = self.data_dir / "understat" / "league"
        api_dir = self.data_dir / "fpl_api"
        fm_dir = self.data_dir / "fotmob"
        odds_dir = self.data_dir / "odds"

        for season in self.seasons:
            parts: list[str] = []
            # vaastav
            merged = raw_dir / season / "gws" / "merged_gw.csv"
            parts.append("vaastav" if merged.exists() else "-------")
            # understat league
            us_file = us_league / f"{season}.json"
            parts.append("understat" if us_file.exists() else "---------")
            # fpl_api
            bs_file = api_dir / "bootstrap" / f"{season}.json"
            parts.append("fpl_api" if bs_file.exists() else "-------")
            # fotmob
            fm_file = fm_dir / f"{season}.json"
            parts.append("fotmob" if fm_file.exists() else "------")
            # odds
            odds_file = odds_dir / f"{season}.json"
            parts.append("odds" if odds_file.exists() else "----")

            print(f"  {season}:  {' | '.join(parts)}")
