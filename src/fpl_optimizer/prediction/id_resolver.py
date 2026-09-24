"""Cross-source ID resolution for FPL, Understat, and FBref."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SEASON_COLUMNS = [
    "16-17", "17-18", "18-19", "19-20", "20-21",
    "21-22", "22-23", "23-24", "24-25", "25-26", "26-27",
]

SEASON_TO_COL = {
    "2016-17": "16-17", "2017-18": "17-18", "2018-19": "18-19",
    "2019-20": "19-20", "2020-21": "20-21", "2021-22": "21-22",
    "2022-23": "22-23", "2023-24": "23-24", "2024-25": "24-25",
    "2025-26": "25-26", "2026-27": "26-27",
}


class IDResolver:
    """O(1) lookups between FPL element_id, stable code, understat ID, and FBref ID.

    The master ID map CSV has columns:
        code, first_name, second_name, web_name,
        16-17, 17-18, ..., 24-25,  (element_ids per season)
        fbref, understat, transfermarkt
    """

    def __init__(self, data_dir: Path) -> None:
        map_path = data_dir / "id_maps" / "master_id_map.csv"
        if not map_path.exists():
            raise FileNotFoundError(f"Master ID map not found at {map_path}")

        df = pd.read_csv(map_path, encoding="utf-8")
        df.columns = [c.strip() for c in df.columns]

        self._code_to_understat: dict[int, int] = {}
        self._code_to_fbref: dict[int, str] = {}
        self._season_eid_to_code: dict[tuple[str, int], int] = {}
        self._code_to_season_eid: dict[tuple[int, str], int] = {}
        self._code_to_name: dict[int, str] = {}
        self._code_to_full_name: dict[int, str] = {}

        for _, row in df.iterrows():
            code = int(row["code"])

            # Name mapping
            web_name = str(row.get("web_name", ""))
            second_name = str(row.get("second_name", ""))
            first_name = str(row.get("first_name", ""))
            self._code_to_name[code] = web_name or second_name

            # Full name for cross-source matching (e.g. FBref)
            if first_name and second_name and first_name != "nan" and second_name != "nan":
                self._code_to_full_name[code] = f"{first_name} {second_name}"
            elif second_name and second_name != "nan":
                self._code_to_full_name[code] = second_name

            # Understat ID
            us_id = row.get("understat")
            if pd.notna(us_id):
                self._code_to_understat[code] = int(us_id)

            # FBref ID
            fb_id = row.get("fbref")
            if pd.notna(fb_id):
                self._code_to_fbref[code] = str(fb_id)

            # Season element_id mappings
            for season, col in SEASON_TO_COL.items():
                if col in df.columns:
                    eid = row.get(col)
                    if pd.notna(eid):
                        eid_int = int(eid)
                        self._season_eid_to_code[(season, eid_int)] = code
                        self._code_to_season_eid[(code, season)] = eid_int

        # Live-season supplements: data/id_maps/live_element_code_{season}.csv
        # (columns: element_id, code, web_name) written by the fpl_live collector
        # from bootstrap-static's own `code` field. Covers players missing from
        # the ChrisMusson map (new signings, current season not yet mapped).
        for supp_path in sorted((data_dir / "id_maps").glob("live_element_code_*.csv")):
            season = supp_path.stem.replace("live_element_code_", "")
            supp = pd.read_csv(supp_path)
            n_new = 0
            for _, row in supp.iterrows():
                code = int(row["code"])
                eid = int(row["element_id"])
                if (season, eid) not in self._season_eid_to_code:
                    self._season_eid_to_code[(season, eid)] = code
                    self._code_to_season_eid[(code, season)] = eid
                    n_new += 1
                if code not in self._code_to_name:
                    self._code_to_name[code] = str(row.get("web_name", ""))
            logger.info("IDResolver: live supplement %s added %d mappings", season, n_new)

        logger.info(
            "IDResolver: %d codes, %d understat, %d fbref mappings",
            len(self._code_to_name),
            len(self._code_to_understat),
            len(self._code_to_fbref),
        )

    def code_from_element_id(self, season: str, element_id: int) -> int | None:
        """Map (season, element_id) -> stable code."""
        return self._season_eid_to_code.get((season, element_id))

    def element_id_from_code(self, code: int, season: str) -> int | None:
        """Map (code, season) -> element_id."""
        return self._code_to_season_eid.get((code, season))

    def understat_id(self, code: int) -> int | None:
        """Map code -> understat player ID."""
        return self._code_to_understat.get(code)

    def fbref_id(self, code: int) -> str | None:
        """Map code -> FBref player ID."""
        return self._code_to_fbref.get(code)

    def player_name(self, code: int) -> str:
        """Map code -> web_name (for display/debugging)."""
        return self._code_to_name.get(code, "Unknown")

    def player_full_name(self, code: int) -> str | None:
        """Map code -> 'first_name second_name' (for cross-source matching)."""
        return self._code_to_full_name.get(code)

    def all_codes_for_season(self, season: str) -> list[int]:
        """Return all stable codes that have an element_id in the given season."""
        col = SEASON_TO_COL.get(season)
        if col is None:
            return []
        return [
            code for (code, s), _ in self._code_to_season_eid.items()
            if s == season
        ]

    def all_codes(self) -> set[int]:
        """Return all known stable codes."""
        return set(self._code_to_name.keys())
