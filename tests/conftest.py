"""Shared test fixtures for FPL-Optimizer tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from fpl_optimizer.data.loader import SeasonDataLoader
from fpl_optimizer.engine.state import (
    ChipState,
    GameState,
    PlayerSlot,
    Squad,
)
from fpl_optimizer.utils.constants import Position, STARTING_BUDGET

# Path to hand-crafted test data
TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory mimicking the vaastav structure."""
    season_dir = tmp_path / "test-season"
    gws_dir = season_dir / "gws"
    gws_dir.mkdir(parents=True)

    # Copy test CSVs
    shutil.copy(TEST_DATA_DIR / "sample_merged_gw.csv", gws_dir / "merged_gw.csv")
    shutil.copy(TEST_DATA_DIR / "sample_cleaned_players.csv", season_dir / "cleaned_players.csv")
    shutil.copy(TEST_DATA_DIR / "sample_fixtures.csv", season_dir / "fixtures.csv")
    shutil.copy(TEST_DATA_DIR / "sample_teams.csv", season_dir / "teams.csv")

    return tmp_path


@pytest.fixture
def loader(test_data_dir: Path) -> SeasonDataLoader:
    """Create a SeasonDataLoader from test data."""
    # Monkey-patch ensure_season_data to skip download
    original_ensure = SeasonDataLoader.__init__

    def patched_init(self, season, data_dir):
        self.season = season
        self.data_dir = data_dir
        self._season_dir = data_dir / season
        import pandas as pd
        from fpl_optimizer.data.schemas import ELEMENT_TYPE_TO_POSITION, SEASONS_WITH_EXPECTED, SEASONS_WITH_POSITION
        self._merged_gw = self._load_merged_gw()
        self._player_info = self._load_player_info()
        self._fixtures = self._load_fixtures()
        self._teams = self._load_teams()
        self._gw_index = {}
        for idx, row in self._merged_gw.iterrows():
            key = (int(row["element"]), int(row["GW"]))
            self._gw_index.setdefault(key, []).append(idx)
        self._position_map = self._build_position_map()
        self._team_map = self._build_team_map()

    SeasonDataLoader.__init__ = patched_init
    loader = SeasonDataLoader("test-season", test_data_dir)
    SeasonDataLoader.__init__ = original_ensure
    return loader


@pytest.fixture
def sample_squad() -> Squad:
    """Create a standard 15-player test squad (4-4-2 formation)."""
    players = [
        # GK (2)
        PlayerSlot(element_id=1, position=Position.GK, purchase_price=50, selling_price=50),
        PlayerSlot(element_id=2, position=Position.GK, purchase_price=40, selling_price=40),
        # DEF (5)
        PlayerSlot(element_id=3, position=Position.DEF, purchase_price=55, selling_price=55),
        PlayerSlot(element_id=4, position=Position.DEF, purchase_price=50, selling_price=50),
        PlayerSlot(element_id=5, position=Position.DEF, purchase_price=45, selling_price=45),
        PlayerSlot(element_id=6, position=Position.DEF, purchase_price=43, selling_price=43),
        PlayerSlot(element_id=7, position=Position.DEF, purchase_price=42, selling_price=42),
        # MID (5)
        PlayerSlot(element_id=8, position=Position.MID, purchase_price=80, selling_price=80),
        PlayerSlot(element_id=9, position=Position.MID, purchase_price=65, selling_price=65),
        PlayerSlot(element_id=10, position=Position.MID, purchase_price=55, selling_price=55),
        PlayerSlot(element_id=11, position=Position.MID, purchase_price=50, selling_price=50),
        PlayerSlot(element_id=12, position=Position.MID, purchase_price=45, selling_price=45),
        # FWD (3)
        PlayerSlot(element_id=13, position=Position.FWD, purchase_price=100, selling_price=100),
        PlayerSlot(element_id=14, position=Position.FWD, purchase_price=70, selling_price=70),
        PlayerSlot(element_id=15, position=Position.FWD, purchase_price=55, selling_price=55),
    ]

    # 4-4-2 formation: GK + 4 DEF + 4 MID + 2 FWD
    lineup = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]  # 11 starters
    bench = [1, 6, 11, 14]  # 4 bench players

    return Squad(
        players=players,
        lineup=lineup,
        bench=bench,
        captain_idx=7,   # MID1 (element 8) as captain
        vice_captain_idx=12,  # FWD1 (element 13) as vice
    )


@pytest.fixture
def sample_state(sample_squad: Squad) -> GameState:
    """Create a standard game state at GW1."""
    budget_used = sum(p.purchase_price for p in sample_squad.players)
    return GameState(
        squad=sample_squad,
        bank=STARTING_BUDGET - budget_used,
        free_transfers=1,
        chips=ChipState(),
        current_gw=1,
        total_points=0,
        active_chip=None,
        free_hit_stash=None,
    )
