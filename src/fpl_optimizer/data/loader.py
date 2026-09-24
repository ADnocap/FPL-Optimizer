"""SeasonDataLoader: load and index historical FPL season data for fast lookups."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_optimizer.data.downloader import DEFAULT_DATA_DIR, ensure_season_data
from fpl_optimizer.data.schemas import (
    ELEMENT_TYPE_TO_POSITION,
    SEASONS_WITH_EXPECTED,
    SEASONS_WITH_POSITION,
)
from fpl_optimizer.utils.constants import Position

logger = logging.getLogger(__name__)


def _read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV, falling back to latin-1 if UTF-8 fails."""
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        logger.debug("UTF-8 failed for %s, retrying with latin-1", path)
        return pd.read_csv(path, encoding="latin-1", **kwargs)


class SeasonDataLoader:
    """Loads and indexes one season of FPL historical data for fast lookups.

    Pre-indexes merged_gw.csv by (element_id, gw) for O(1) lookups.
    Handles DGW (multiple rows per player per GW) and schema differences.
    """

    def __init__(self, season: str, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self.season = season
        self.data_dir = data_dir
        self._season_dir = ensure_season_data(season, data_dir)

        self._merged_gw: pd.DataFrame = self._load_merged_gw()
        self._player_info: pd.DataFrame = self._load_player_info()
        self._fixtures: pd.DataFrame = self._load_fixtures()
        self._teams: pd.DataFrame = self._load_teams()

        # Pre-build index: (element_id, gw) -> list of row indices (list for DGW)
        self._gw_index: dict[tuple[int, int], list[int]] = {}
        for idx, row in self._merged_gw.iterrows():
            key = (int(row["element"]), int(row["GW"]))
            self._gw_index.setdefault(key, []).append(idx)

        # Pre-build element_id -> position mapping
        self._position_map: dict[int, Position] = self._build_position_map()

        # Pre-build element_id -> team_id mapping
        self._team_map: dict[int, int] = self._build_team_map()

    def _load_merged_gw(self) -> pd.DataFrame:
        """Load merged_gw.csv with schema normalization."""
        path = self._season_dir / "gws" / "merged_gw.csv"
        df = _read_csv_safe(path, on_bad_lines="skip")

        # Exact full-row duplicates are always source-data defects (legitimate
        # DGW rows differ by fixture id) and would double-count in the
        # per-(element, gw) sums — e.g. 10 dupe rows in vaastav's 2025-26 file.
        df = df.drop_duplicates()

        # Fill missing expected stats columns with 0
        if self.season not in SEASONS_WITH_EXPECTED:
            for col in [
                "expected_goals",
                "expected_assists",
                "expected_goal_involvements",
                "expected_goals_conceded",
                "xP",
            ]:
                if col not in df.columns:
                    df[col] = 0.0

        # Ensure numeric types for key columns
        for col in ["element", "GW", "total_points", "minutes", "value"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _load_player_info(self) -> pd.DataFrame:
        """Load cleaned_players.csv for position/team info on older seasons."""
        path = self._season_dir / "cleaned_players.csv"
        if not path.exists():
            logger.debug("cleaned_players.csv not found for %s", self.season)
            return pd.DataFrame()
        return _read_csv_safe(path, on_bad_lines="skip")

    def _load_fixtures(self) -> pd.DataFrame:
        """Load fixtures.csv for DGW/BGW detection."""
        path = self._season_dir / "fixtures.csv"
        if not path.exists():
            logger.debug("fixtures.csv not found for %s", self.season)
            return pd.DataFrame()
        return _read_csv_safe(path, on_bad_lines="skip")

    def _load_teams(self) -> pd.DataFrame:
        """Load teams.csv for team name/id mappings."""
        path = self._season_dir / "teams.csv"
        if not path.exists():
            logger.debug("teams.csv not found for %s", self.season)
            return pd.DataFrame()
        return _read_csv_safe(path, on_bad_lines="skip")

    def _build_position_map(self) -> dict[int, Position]:
        """Build element_id -> Position mapping."""
        pos_map: dict[int, Position] = {}

        # Try to get position from merged_gw (2020-21+ have this column)
        if "position" in self._merged_gw.columns:
            pos_map = self._extract_positions_from_df(self._merged_gw)

        # Fall back to cleaned_players.csv
        if not pos_map and not self._player_info.empty:
            et_col = "element_type"
            id_col = "id"
            if et_col in self._player_info.columns and id_col in self._player_info.columns:
                for _, row in self._player_info[[id_col, et_col]].iterrows():
                    et = int(row[et_col])
                    if et in ELEMENT_TYPE_TO_POSITION:
                        pos_name = ELEMENT_TYPE_TO_POSITION[et]
                        pos_map[int(row[id_col])] = Position[pos_name]

        # Cross-season backfill: element IDs are stable across seasons,
        # so load a reference season that has position data
        if not pos_map:
            pos_map = self._backfill_positions()

        return pos_map

    @staticmethod
    def _extract_positions_from_df(df: pd.DataFrame) -> dict[int, Position]:
        """Extract element_id -> Position from a DataFrame with a 'position' column."""
        pos_map: dict[int, Position] = {}
        _POS_LOOKUP = {"GK": Position.GK, "GKP": Position.GK,
                       "DEF": Position.DEF, "MID": Position.MID, "FWD": Position.FWD}
        for _, row in df[["element", "position"]].drop_duplicates().iterrows():
            pos = _POS_LOOKUP.get(str(row["position"]).upper())
            if pos is not None:
                pos_map[int(row["element"])] = pos
        return pos_map

    def _backfill_positions(self) -> dict[int, Position]:
        """Backfill positions from a reference season that has position data.

        Element IDs are stable across vaastav seasons, so we can look up
        positions from any season that has the 'position' column.
        """
        from fpl_optimizer.utils.constants import AVAILABLE_SEASONS
        my_elements = set(self._merged_gw["element"].unique())
        pos_map: dict[int, Position] = {}

        for ref_season in SEASONS_WITH_POSITION:
            ref_path = self.data_dir / ref_season / "gws" / "merged_gw.csv"
            if not ref_path.exists():
                continue
            try:
                ref_df = _read_csv_safe(ref_path, on_bad_lines="skip",
                                        usecols=["element", "position"])
            except (ValueError, KeyError):
                continue
            ref_pos = self._extract_positions_from_df(ref_df)
            for eid in my_elements:
                if eid not in pos_map and eid in ref_pos:
                    pos_map[eid] = ref_pos[eid]
            if len(pos_map) >= len(my_elements):
                break

        if pos_map:
            logger.info(
                "Backfilled positions for %s: %d/%d elements",
                self.season, len(pos_map), len(my_elements),
            )
        return pos_map

    def _build_team_map(self) -> dict[int, int]:
        """Build element_id -> team_id mapping."""
        team_map: dict[int, int] = {}

        # Build name->id lookup from teams.csv if available
        team_name_to_id: dict[str, int] = {}
        if not self._teams.empty and "id" in self._teams.columns and "name" in self._teams.columns:
            for _, row in self._teams.iterrows():
                team_name_to_id[str(row["name"])] = int(row["id"])

        # Try merged_gw first (modern seasons have 'team' column)
        if "team" in self._merged_gw.columns:
            for _, row in (
                self._merged_gw[["element", "team"]].drop_duplicates("element").iterrows()
            ):
                raw_team = row["team"]
                try:
                    team_map[int(row["element"])] = int(raw_team)
                except (ValueError, TypeError):
                    # String team name (2020-21+) — resolve via teams.csv
                    team_id = team_name_to_id.get(str(raw_team))
                    if team_id is not None:
                        team_map[int(row["element"])] = team_id
        elif not self._player_info.empty and "team" in self._player_info.columns:
            id_col = "id"
            if id_col in self._player_info.columns:
                for _, row in self._player_info[[id_col, "team"]].iterrows():
                    team_map[int(row[id_col])] = int(row["team"])

        # Cross-season backfill for team mapping
        if not team_map:
            team_map = self._backfill_teams()

        return team_map

    def _backfill_teams(self) -> dict[int, int]:
        """Backfill team IDs from a reference season that has a 'team' column."""
        my_elements = set(self._merged_gw["element"].unique())
        team_map: dict[int, int] = {}

        for ref_season in SEASONS_WITH_POSITION:
            ref_path = self.data_dir / ref_season / "gws" / "merged_gw.csv"
            if not ref_path.exists():
                continue
            try:
                ref_df = _read_csv_safe(ref_path, on_bad_lines="skip",
                                        usecols=["element", "team"])
            except (ValueError, KeyError):
                continue
            if "team" not in ref_df.columns:
                continue

            # Build name->id lookup from the reference season's teams.csv
            ref_teams_path = self.data_dir / ref_season / "teams.csv"
            ref_name_to_id: dict[str, int] = {}
            if ref_teams_path.exists():
                try:
                    ref_teams = _read_csv_safe(ref_teams_path, on_bad_lines="skip")
                    if "id" in ref_teams.columns and "name" in ref_teams.columns:
                        for _, r in ref_teams.iterrows():
                            ref_name_to_id[str(r["name"])] = int(r["id"])
                except Exception:
                    pass

            for _, row in ref_df[["element", "team"]].drop_duplicates("element").iterrows():
                eid = int(row["element"])
                if eid in my_elements and eid not in team_map:
                    raw = row["team"]
                    try:
                        team_map[eid] = int(raw)
                    except (ValueError, TypeError):
                        tid = ref_name_to_id.get(str(raw))
                        if tid is not None:
                            team_map[eid] = tid
            if len(team_map) >= len(my_elements):
                break

        if team_map:
            logger.info(
                "Backfilled teams for %s: %d/%d elements",
                self.season, len(team_map), len(my_elements),
            )
        return team_map

    def get_player_gw(self, element_id: int, gw: int) -> dict | None:
        """Get a player's data for a specific gameweek.

        For DGWs, sums points and minutes across fixtures.
        Returns None if player has no data for that GW.
        """
        rows_idx = self._gw_index.get((element_id, gw))
        if not rows_idx:
            return None

        if len(rows_idx) == 1:
            return self._merged_gw.loc[rows_idx[0]].to_dict()

        # DGW: sum numeric columns, take first for non-numeric
        rows = self._merged_gw.loc[rows_idx]
        result = rows.iloc[0].to_dict()

        # Sum these columns across fixtures
        sum_cols = [
            "total_points", "minutes", "goals_scored", "assists",
            "clean_sheets", "goals_conceded", "own_goals",
            "penalties_saved", "penalties_missed", "yellow_cards",
            "red_cards", "saves", "bonus", "bps",
        ]
        for col in sum_cols:
            if col in rows.columns:
                result[col] = rows[col].sum()

        return result

    def get_gameweek_data(self, gw: int) -> pd.DataFrame:
        """Get all player data for a specific gameweek."""
        mask = self._merged_gw["GW"] == gw
        return self._merged_gw[mask].copy()

    def get_player_price(self, element_id: int, gw: int) -> int:
        """Get player price in tenths for a specific GW. Returns 0 if not found."""
        data = self.get_player_gw(element_id, gw)
        if data is None:
            return 0
        return int(data.get("value", 0))

    def get_player_position(self, element_id: int) -> Position | None:
        """Get a player's position."""
        return self._position_map.get(element_id)

    def get_player_team(self, element_id: int) -> int | None:
        """Get a player's team ID."""
        return self._team_map.get(element_id)

    def get_fixtures(self, gw: int) -> pd.DataFrame:
        """Get fixtures for a specific GW. Useful for DGW/BGW detection."""
        if self._fixtures.empty:
            return pd.DataFrame()
        # The 'event' column typically maps to GW
        event_col = "event"
        if event_col not in self._fixtures.columns:
            return pd.DataFrame()
        return self._fixtures[self._fixtures[event_col] == gw].copy()

    def get_all_element_ids(self, gw: int | None = None) -> list[int]:
        """Get all element IDs, optionally filtered to those active in a GW."""
        if gw is not None:
            return list(self._merged_gw[self._merged_gw["GW"] == gw]["element"].unique())
        return list(self._merged_gw["element"].unique())

    def get_teams_playing(self, gw: int) -> set[int]:
        """Get set of team IDs that have fixtures in the given GW."""
        fixtures = self.get_fixtures(gw)
        if fixtures.empty:
            return set()
        teams: set[int] = set()
        if "team_h" in fixtures.columns:
            teams.update(fixtures["team_h"].astype(int))
        if "team_a" in fixtures.columns:
            teams.update(fixtures["team_a"].astype(int))
        return teams

    def is_dgw(self, team_id: int, gw: int) -> bool:
        """Check if a team has a double gameweek."""
        fixtures = self.get_fixtures(gw)
        if fixtures.empty:
            return False
        team_fixtures = fixtures[
            (fixtures.get("team_h", pd.Series()) == team_id)
            | (fixtures.get("team_a", pd.Series()) == team_id)
        ]
        return len(team_fixtures) > 1

    def get_num_gameweeks(self) -> int:
        """Get the total number of gameweeks in this season."""
        return int(self._merged_gw["GW"].max())

    def get_player_form(
        self, element_id: int, gw: int, window: int = 5
    ) -> float:
        """Get rolling average points over the last `window` GWs."""
        points = []
        for prev_gw in range(max(1, gw - window), gw):
            data = self.get_player_gw(element_id, prev_gw)
            if data is not None:
                points.append(data.get("total_points", 0))
        return float(np.mean(points)) if points else 0.0

    def get_gw_average_points(self, gw: int) -> float:
        """Get average total_points across all players in a GW."""
        gw_data = self.get_gameweek_data(gw)
        if gw_data.empty:
            return 0.0
        return float(gw_data["total_points"].mean())
