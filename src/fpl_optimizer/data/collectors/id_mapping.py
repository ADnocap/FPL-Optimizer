"""Player ID mapping across FPL, Understat, and FBref."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from fpl_optimizer.data.collectors.base import BaseCollector, RateLimiter, DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

ID_MAP_URL = (
    "https://raw.githubusercontent.com/ChrisMusson/FPL-ID-Map/main/Master.csv"
)


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    # Decompose unicode, drop combining marks (accents)
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase + collapse whitespace
    return re.sub(r"\s+", " ", stripped.lower().strip())


class PlayerIDMapper(BaseCollector):
    """Download and query ChrisMusson's FPL-ID-Map master CSV.

    Provides fast lookups: FPL element_id -> Understat ID / FBref ID.
    """

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        super().__init__(data_dir=data_dir, rate_limiter=RateLimiter(calls_per_second=5.0))
        self.map_dir = self.data_dir / "id_maps"
        self.map_path = self.map_dir / "master_id_map.csv"
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    def collect_season(self, season: str) -> bool:
        """ID map is season-agnostic — delegates to ``collect_all``."""
        return bool(self.collect_all().get("master", False))

    def collect_all(self) -> dict[str, bool]:
        """Download the master ID map CSV."""
        if self._is_cached(self.map_path):
            logger.info("ID map: already cached at %s", self.map_path)
            return {"master": True}

        logger.info("ID map: downloading from %s", ID_MAP_URL)
        try:
            resp = self._request_with_retry(ID_MAP_URL)
            self.map_dir.mkdir(parents=True, exist_ok=True)
            self.map_path.write_text(resp.text, encoding="utf-8")
            logger.info("ID map: saved to %s", self.map_path)
            return {"master": True}
        except Exception as exc:
            logger.error("ID map: download failed: %s", exc)
            return {"master": False}

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the CSV into memory. Call after ``collect_all``."""
        if self._df is not None:
            return
        if not self.map_path.exists():
            raise FileNotFoundError(
                f"ID map not found at {self.map_path}. Run collect_all() first."
            )
        self._df = pd.read_csv(self.map_path, encoding="utf-8")
        # Normalise column names to lowercase
        self._df.columns = [c.strip().lower() for c in self._df.columns]
        logger.info("ID map: loaded %d rows", len(self._df))

    def _ensure_loaded(self) -> pd.DataFrame:
        if self._df is None:
            self.load()
        assert self._df is not None
        return self._df

    def get_understat_id(self, element_id: int, season: str | None = None) -> int | None:
        """Return Understat player ID for an FPL element_id, or None."""
        df = self._ensure_loaded()
        col_fpl = self._find_column(df, "fpl_id")
        col_us = self._find_column(df, "understat_id")
        if col_fpl is None or col_us is None:
            return None
        mask = df[col_fpl] == element_id
        if season:
            col_season = self._find_column(df, "season")
            if col_season:
                mask = mask & (df[col_season] == season)
        matches = df.loc[mask, col_us].dropna()
        if matches.empty:
            return None
        return int(matches.iloc[0])

    def get_fbref_id(self, element_id: int, season: str | None = None) -> str | None:
        """Return FBref player ID for an FPL element_id, or None."""
        df = self._ensure_loaded()
        col_fpl = self._find_column(df, "fpl_id")
        col_fb = self._find_column(df, "fbref_id")
        if col_fpl is None or col_fb is None:
            return None
        mask = df[col_fpl] == element_id
        if season:
            col_season = self._find_column(df, "season")
            if col_season:
                mask = mask & (df[col_season] == season)
        matches = df.loc[mask, col_fb].dropna()
        if matches.empty:
            return None
        return str(matches.iloc[0])

    def fuzzy_lookup(self, player_name: str) -> pd.DataFrame:
        """Return all rows whose normalized name matches *player_name*.

        Useful as a fallback when element_id is unknown.
        """
        df = self._ensure_loaded()
        target = _normalize_name(player_name)
        name_col = self._find_column(df, "player_name") or self._find_column(df, "name")
        if name_col is None:
            return pd.DataFrame()
        norm = df[name_col].astype(str).apply(_normalize_name)
        return df[norm.str.contains(target, na=False)].copy()

    # ------------------------------------------------------------------

    @staticmethod
    def _find_column(df: pd.DataFrame, fragment: str) -> str | None:
        """Find first column whose name contains *fragment* (case-insensitive)."""
        fragment_lower = fragment.lower()
        for col in df.columns:
            if fragment_lower in col.lower():
                return col
        return None
