"""Collector for historical betting odds from The Odds API.

Fetches Pinnacle h2h (1X2) odds for EPL matches using the historical
endpoint.  One API call per gameweek returns all upcoming EPL events
at the snapshot closest to 2 hours before the GW's first kickoff.

Coverage: 2020-21 season onward (API data starts June 2020).
Credit cost: 10 per request (1 market × 1 region × 10x historical).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

from fpl_optimizer.data.collectors.base import BaseCollector, RateLimiter, DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

# Seasons with historical odds coverage (API starts June 2020)
ODDS_SEASONS = [
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

_HISTORICAL_URL = (
    "https://api.the-odds-api.com/v4/historical/sports/soccer_epl/odds"
)

# Mapping from Odds API team names → FPL teams.csv `name` column.
# Only entries that differ need to be listed; exact matches are automatic.
_ODDS_TO_FPL_NAME: dict[str, str] = {
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Tottenham Hotspur": "Spurs",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "Brighton and Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Sheffield United": "Sheffield Utd",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "West Bromwich Albion": "West Brom",
    "Nottingham Forest": "Nott'm Forest",
    "Luton Town": "Luton",
    "Ipswich Town": "Ipswich",
    "AFC Bournemouth": "Bournemouth",
    # football-data.co.uk variants (scripts/collect_football_data_odds.py).
    # NOTE: FPL itself renames clubs between seasons (Ipswich vs Ipswich Town);
    # features/odds.py also retries the raw name against the season's team table.
    "Man United": "Man Utd",
    "Tottenham": "Spurs",
    "Sheffield United": "Sheffield Utd",
    "Coventry": "Coventry City",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
}


def odds_team_to_fpl_name(odds_name: str) -> str:
    """Convert an Odds API team name to its FPL teams.csv equivalent."""
    return _ODDS_TO_FPL_NAME.get(odds_name, odds_name)


class OddsCollector(BaseCollector):
    """Collect historical Pinnacle h2h odds for EPL from The Odds API."""

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            # Conservative: 1 request per 2 seconds
            rate_limiter=RateLimiter(calls_per_second=0.5),
        )
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")
        self.odds_dir = self.data_dir / "odds"
        self._credits_remaining: int | None = None
        self._credits_used: int | None = None

    def collect_season(self, season: str) -> bool:
        """Collect odds for all GWs in a single EPL season."""
        if season not in ODDS_SEASONS:
            logger.info("Odds %s: not in coverage range, skipping", season)
            return True  # Not a failure — just no data for this season

        dest = self.odds_dir / f"{season}.json"
        if self._is_cached(dest):
            logger.info("Odds %s: cached", season)
            return True

        if not self.api_key:
            logger.error("ODDS_API_KEY not set — cannot collect odds")
            return False

        # Load GW dates from merged_gw.csv
        gw_dates = self._load_gw_dates(season)
        if gw_dates is None or gw_dates.empty:
            logger.warning("Odds %s: no GW dates available", season)
            return False

        season_data: dict[str, list[dict]] = {}
        n_ok = 0

        for gw in sorted(gw_dates.index):
            kickoff = gw_dates[gw]
            if pd.isna(kickoff):
                continue

            # Query 2 hours before first kickoff — odds are available but
            # no match has started, so no lookahead bias.
            query_time = kickoff - timedelta(hours=2)
            query_iso = query_time.strftime("%Y-%m-%dT%H:%M:%SZ")

            logger.info(
                "Odds %s GW%d: querying snapshot at %s", season, gw, query_iso,
            )

            events = self._fetch_snapshot(query_iso)
            if events is None:
                logger.warning("Odds %s GW%d: API error, skipping", season, gw)
                continue

            # Parse Pinnacle h2h odds from each event
            gw_matches = []
            for event in events:
                parsed = self._parse_event(event)
                if parsed is not None:
                    gw_matches.append(parsed)

            season_data[str(gw)] = gw_matches
            n_ok += 1
            logger.info(
                "Odds %s GW%d: %d matches with Pinnacle odds "
                "(credits remaining: %s)",
                season, gw, len(gw_matches), self._credits_remaining,
            )

            # Safety: stop if credits are critically low
            if self._credits_remaining is not None and self._credits_remaining < 20:
                logger.warning(
                    "Odds: credits low (%d remaining), stopping collection",
                    self._credits_remaining,
                )
                break

        if n_ok > 0:
            self.odds_dir.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(season_data, indent=2), encoding="utf-8")
            logger.info("Odds %s: saved (%d GWs)", season, n_ok)
            return True

        logger.warning("Odds %s: no GWs collected", season)
        return False

    def collect_all(
        self,
        *,
        max_workers: int = 1,
        seasons: list[str] | None = None,
    ) -> dict[str, bool]:
        """Collect odds for all applicable seasons (sequential)."""
        target = seasons or list(ODDS_SEASONS)
        # Filter to only odds-eligible seasons
        target = [s for s in target if s in ODDS_SEASONS]

        results: dict[str, bool] = {}
        for season in target:
            results[season] = self.collect_season(season)
            # Stop early if credits exhausted
            if (
                self._credits_remaining is not None
                and self._credits_remaining < 20
            ):
                logger.warning("Odds: credits exhausted, stopping")
                for remaining in target:
                    if remaining not in results:
                        results[remaining] = False
                break
        return results

    def _fetch_snapshot(self, date_iso: str) -> list[dict] | None:
        """Fetch a single historical snapshot from the API.

        Returns the list of event dicts, or None on error.
        """
        params = {
            "apiKey": self.api_key,
            "bookmakers": "pinnacle",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "date": date_iso,
        }

        self.rate_limiter.wait()
        try:
            resp = requests.get(_HISTORICAL_URL, params=params, timeout=30)

            # Track credit usage from response headers
            self._credits_remaining = _safe_int(
                resp.headers.get("x-requests-remaining")
            )
            self._credits_used = _safe_int(
                resp.headers.get("x-requests-used")
            )

            if resp.status_code == 422:
                # Timestamp out of range or no data available
                logger.debug("Odds: 422 for date %s (no data)", date_iso)
                return []
            if resp.status_code == 429:
                logger.warning("Odds: rate limited (429)")
                return None
            if resp.status_code == 401:
                logger.error("Odds: invalid API key (401)")
                return None

            resp.raise_for_status()
            body = resp.json()
            return body.get("data", [])

        except requests.RequestException as exc:
            logger.warning("Odds: request failed: %s", exc)
            return None

    @staticmethod
    def _parse_event(event: dict) -> dict | None:
        """Extract Pinnacle h2h odds from a single event dict.

        Returns a flat dict with home_team, away_team, and odds,
        or None if Pinnacle h2h data is missing.
        """
        bookmakers = event.get("bookmakers", [])
        pinnacle = None
        for bm in bookmakers:
            if bm.get("key") == "pinnacle":
                pinnacle = bm
                break

        if pinnacle is None:
            return None

        markets = pinnacle.get("markets", [])
        h2h = None
        for mkt in markets:
            if mkt.get("key") == "h2h":
                h2h = mkt
                break

        if h2h is None:
            return None

        outcomes = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        home_odds = outcomes.get(home_team)
        draw_odds = outcomes.get("Draw")
        away_odds = outcomes.get(away_team)

        if home_odds is None or draw_odds is None or away_odds is None:
            return None

        return {
            "event_id": event.get("id", ""),
            "commence_time": event.get("commence_time", ""),
            "home_team": home_team,
            "away_team": away_team,
            "home_odds": float(home_odds),
            "draw_odds": float(draw_odds),
            "away_odds": float(away_odds),
            "last_update": pinnacle.get("last_update", ""),
        }

    def _load_gw_dates(self, season: str) -> pd.Series | None:
        """Load earliest kickoff per GW from merged_gw.csv.

        Returns a Series indexed by GW number with UTC-naive datetime values.
        """
        merged_path = (
            self.data_dir / "raw" / season / "gws" / "merged_gw.csv"
        )
        if not merged_path.exists():
            return None

        try:
            df = pd.read_csv(merged_path, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(
                merged_path, encoding="latin-1", on_bad_lines="skip"
            )

        if "kickoff_time" not in df.columns or "GW" not in df.columns:
            return None

        df["GW"] = pd.to_numeric(df["GW"], errors="coerce")
        df["kickoff_dt"] = pd.to_datetime(
            df["kickoff_time"], format="mixed", errors="coerce", utc=True,
        )
        gw_dates = df.groupby("GW")["kickoff_dt"].min()
        # Strip timezone for consistent comparisons
        gw_dates = gw_dates.dt.tz_localize(None)
        return gw_dates


def _safe_int(val: str | None) -> int | None:
    """Parse a header value to int, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None
