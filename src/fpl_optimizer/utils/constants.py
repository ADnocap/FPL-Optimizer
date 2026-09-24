"""Global constants for FPL game rules and scoring."""

from enum import IntEnum


class Position(IntEnum):
    GK = 1
    DEF = 2
    MID = 3
    FWD = 4


# Squad composition requirements
SQUAD_SIZE = 15
STARTING_XI = 11
BENCH_SIZE = 4
POSITION_LIMITS = {Position.GK: 2, Position.DEF: 5, Position.MID: 5, Position.FWD: 3}
MAX_PER_CLUB = 3
STARTING_BUDGET = 1000  # £100.0m in tenths

# Valid formations (outfield only — GK is always 1)
VALID_FORMATIONS: list[tuple[int, int, int]] = [
    (3, 4, 3),  # DEF, MID, FWD
    (3, 5, 2),
    (4, 3, 3),
    (4, 4, 2),
    (4, 5, 1),
    (5, 2, 3),
    (5, 3, 2),
    (5, 4, 1),
]

# Scoring table: points per action by position
POINTS_GOAL = {Position.GK: 10, Position.DEF: 6, Position.MID: 5, Position.FWD: 4}
POINTS_ASSIST = 3
POINTS_CLEAN_SHEET = {Position.GK: 4, Position.DEF: 4, Position.MID: 1, Position.FWD: 0}
POINTS_GOALS_CONCEDED_PER_2 = {Position.GK: -1, Position.DEF: -1, Position.MID: 0, Position.FWD: 0}
POINTS_SAVES_PER_3 = 1  # GK only
POINTS_PENALTY_SAVE = 5  # GK only
POINTS_PENALTY_MISS = -2
POINTS_MINUTES_1_59 = 1
POINTS_MINUTES_60_PLUS = 2
POINTS_YELLOW_CARD = -1
POINTS_RED_CARD = -3
POINTS_OWN_GOAL = -2
# Bonus: 1-3 points for top 3 BPS in a match

# Transfer rules
TRANSFER_HIT_COST = 4  # points deducted per extra transfer
MAX_FREE_TRANSFERS = 5
INITIAL_FREE_TRANSFERS = 1
# FTs held DURING GW1 by a simulated season start (the GW1 squad is picked
# freely). engine.step then sets exactly 1 FT for GW2, as FPL does, whatever
# this value is.
GW1_FREE_TRANSFERS = INITIAL_FREE_TRANSFERS

# Chip names
CHIP_WILDCARD = "wildcard"
CHIP_FREE_HIT = "free_hit"
CHIP_BENCH_BOOST = "bench_boost"
CHIP_TRIPLE_CAPTAIN = "triple_captain"
ALL_CHIPS = [CHIP_WILDCARD, CHIP_FREE_HIT, CHIP_BENCH_BOOST, CHIP_TRIPLE_CAPTAIN]

# First GW each chip may be played (API bootstrap chips[].start_event, 2026-27):
# Wildcard/Free Hit are useless before the GW1 deadline (unlimited changes), so
# the first-half WC/FH open at GW2; BB/TC are playable from GW1.
CHIP_START_EVENT = {
    CHIP_WILDCARD: 2,
    CHIP_FREE_HIT: 2,
    CHIP_BENCH_BOOST: 1,
    CHIP_TRIPLE_CAPTAIN: 1,
}

# Season halves for chip availability
FIRST_HALF_END = 19  # GW1-19
SECOND_HALF_START = 20  # GW20-38
TOTAL_GAMEWEEKS = 38

# Seasons available in vaastav dataset
AVAILABLE_SEASONS = [
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
]

# The season currently being played (live operation). Vaastav has no per-GW
# data for it — data/raw/2026-27/ is built locally from the official FPL API
# by fpl_optimizer.data.collectors.fpl_live.
CURRENT_SEASON = "2026-27"

# Vaastav GitHub raw URL base
VAASTAV_BASE_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)

# Files to download per season
SEASON_FILES = [
    "gws/merged_gw.csv",
    "teams.csv",
    "fixtures.csv",
    "player_idlist.csv",
    "cleaned_players.csv",
    "players_raw.csv",
]

# Required files — download_season fails only if these are missing.
# teams.csv and fixtures.csv don't exist for older seasons (pre-2019-20).
SEASON_FILES_REQUIRED = [
    "gws/merged_gw.csv",
    "cleaned_players.csv",
    "player_idlist.csv",
]
