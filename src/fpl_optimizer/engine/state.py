"""Game state dataclasses for FPL simulation."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from fpl_optimizer.utils.constants import (
    ALL_CHIPS,
    INITIAL_FREE_TRANSFERS,
    STARTING_BUDGET,
    Position,
)


@dataclass
class PlayerSlot:
    """A player in the squad with purchase price tracking."""

    element_id: int
    position: Position
    purchase_price: int  # in tenths (e.g., 100 = £10.0m)
    selling_price: int  # calculated with 50% appreciation rule

    def copy(self) -> PlayerSlot:
        return PlayerSlot(
            element_id=self.element_id,
            position=self.position,
            purchase_price=self.purchase_price,
            selling_price=self.selling_price,
        )


@dataclass
class Squad:
    """The 15-player squad with lineup/bench assignments."""

    players: list[PlayerSlot]  # 15 players
    lineup: list[int]  # 11 indices into players (starting XI)
    bench: list[int]  # 4 indices into players, ordered by sub priority
    captain_idx: int  # index into players
    vice_captain_idx: int  # index into players

    def get_player(self, idx: int) -> PlayerSlot:
        return self.players[idx]

    def get_lineup_players(self) -> list[PlayerSlot]:
        return [self.players[i] for i in self.lineup]

    def get_bench_players(self) -> list[PlayerSlot]:
        return [self.players[i] for i in self.bench]

    def find_player_idx(self, element_id: int) -> int | None:
        """Find the index of a player by element_id. Returns None if not found."""
        for i, p in enumerate(self.players):
            if p.element_id == element_id:
                return i
        return None

    def copy(self) -> Squad:
        return Squad(
            players=[p.copy() for p in self.players],
            lineup=list(self.lineup),
            bench=list(self.bench),
            captain_idx=self.captain_idx,
            vice_captain_idx=self.vice_captain_idx,
        )


@dataclass
class ChipState:
    """Tracks chip availability per half-season."""

    # Index 0 = first half (GW1-19), index 1 = second half (GW20-38)
    wildcard: list[bool] = field(default_factory=lambda: [True, True])
    free_hit: list[bool] = field(default_factory=lambda: [True, True])
    bench_boost: list[bool] = field(default_factory=lambda: [True, True])
    triple_captain: list[bool] = field(default_factory=lambda: [True, True])
    # 2025-26 rule: Free Hit cannot be used in both GW19 and GW20
    free_hit_last_used_gw: int | None = None

    def is_available(self, chip: str, gw: int) -> bool:
        """Check if a chip is available for the given GW."""
        half = 0 if gw <= 19 else 1
        available = self._get_chip_list(chip)[half]
        if not available:
            return False
        # 2025-26: Free Hit cannot be used in both GW19 and GW20
        if chip == "free_hit" and self.free_hit_last_used_gw is not None:
            if (self.free_hit_last_used_gw == 19 and gw == 20) or (
                self.free_hit_last_used_gw == 20 and gw == 19
            ):
                return False
        return True

    def use_chip(self, chip: str, gw: int) -> None:
        """Mark a chip as used."""
        half = 0 if gw <= 19 else 1
        self._get_chip_list(chip)[half] = False
        if chip == "free_hit":
            self.free_hit_last_used_gw = gw

    def expire_first_half(self) -> None:
        """Expire unused first-half chips (called after GW19)."""
        self.wildcard[0] = False
        self.free_hit[0] = False
        self.bench_boost[0] = False
        self.triple_captain[0] = False

    def _get_chip_list(self, chip: str) -> list[bool]:
        chip_map = {
            "wildcard": self.wildcard,
            "free_hit": self.free_hit,
            "bench_boost": self.bench_boost,
            "triple_captain": self.triple_captain,
        }
        if chip not in chip_map:
            raise ValueError(f"Unknown chip: {chip}. Valid: {ALL_CHIPS}")
        return chip_map[chip]

    def copy(self) -> ChipState:
        return ChipState(
            wildcard=list(self.wildcard),
            free_hit=list(self.free_hit),
            bench_boost=list(self.bench_boost),
            triple_captain=list(self.triple_captain),
            free_hit_last_used_gw=self.free_hit_last_used_gw,
        )


@dataclass
class GameState:
    """Complete game state at any point during a season."""

    squad: Squad
    bank: int = STARTING_BUDGET  # remaining budget in tenths
    free_transfers: int = INITIAL_FREE_TRANSFERS
    chips: ChipState = field(default_factory=ChipState)
    current_gw: int = 1
    total_points: int = 0
    active_chip: str | None = None  # chip active this GW
    free_hit_stash: Squad | None = None  # saved squad during Free Hit

    def copy(self) -> GameState:
        return GameState(
            squad=self.squad.copy(),
            bank=self.bank,
            free_transfers=self.free_transfers,
            chips=self.chips.copy(),
            current_gw=self.current_gw,
            total_points=self.total_points,
            active_chip=self.active_chip,
            free_hit_stash=self.free_hit_stash.copy() if self.free_hit_stash else None,
        )


@dataclass
class EngineAction:
    """Action input to the game engine for one GW."""

    transfers_out: list[int] = field(default_factory=list)  # element_ids to sell
    transfers_in: list[int] = field(default_factory=list)  # element_ids to buy
    captain: int | None = None  # element_id (None = keep current)
    vice_captain: int | None = None  # element_id (None = keep current)
    chip: str | None = None  # chip to activate (None = no chip)
    lineup: list[int] | None = None  # element_ids for starting XI (None = keep current)
    bench: list[int] | None = None  # element_ids for bench order (None = keep current)


@dataclass
class StepResult:
    """Result of processing one gameweek."""

    gw_points: int  # points scored this GW (before hits)
    hit_cost: int  # points deducted for extra transfers
    net_points: int  # gw_points - hit_cost
    captain_points: int  # points from captain (after multiplier)
    bench_points: int  # points from bench (only counts with Bench Boost)
    auto_subs: list[tuple[int, int]]  # list of (out_element_id, in_element_id)
    captain_failover: bool  # True if vice-captain was used
