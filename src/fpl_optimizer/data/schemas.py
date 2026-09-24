"""Column mappings for different season eras in the vaastav dataset.

The vaastav dataset schema has evolved over seasons:
- 2016-17 to 2019-20: ~33 columns, no position/team/xP/expected_* columns
- 2020-21 to 2022-23: ~37 columns, added some fields
- 2023-24: ~41 columns, added position, team, xP, expected_* columns
- 2024-25: ~49 columns, added manager columns
"""

# Columns that exist across all seasons in merged_gw.csv
CORE_COLUMNS = [
    "name",
    "element",       # player ID
    "GW",            # gameweek number
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "value",          # price in tenths
    "transfers_balance",
    "selected",
    "transfers_in",
    "transfers_out",
    "was_home",
    "opponent_team",
    "fixture",
    "kickoff_time",
    "round",          # same as GW in most cases
]

# Columns added in later seasons
MODERN_COLUMNS = [
    "position",       # added ~2023-24
    "team",           # added ~2023-24
    "xP",             # expected points, added ~2023-24
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
]

# Canonical column names we use internally (mapped from raw data)
CANONICAL_COLUMNS = {
    "element": "element_id",
    "GW": "gw",
    "round": "round",
    "value": "price",
}

# Seasons where position column exists in merged_gw.csv
SEASONS_WITH_POSITION = {
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27",
}

# Seasons where expected stats (xG, xA) exist
SEASONS_WITH_EXPECTED = {"2022-23", "2023-24", "2024-25", "2025-26", "2026-27"}

# Position mapping from cleaned_players.csv element_type field
ELEMENT_TYPE_TO_POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_TO_ELEMENT_TYPE = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}


def get_expected_columns(season: str) -> list[str]:
    """Return expected columns for a given season."""
    cols = list(CORE_COLUMNS)
    if season in SEASONS_WITH_EXPECTED:
        cols.extend(MODERN_COLUMNS)
    return cols
