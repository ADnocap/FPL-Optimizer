"""Shared helpers for the 2026-27 rules regression suite.

Written by the 2026-09-24 rules audit (every FPL rule the system encodes,
checked against the official 2026-27 API config and our real GW1-5 history).
Tests marked ``bug`` pin a rules bug that audit found and fixed.

Ground truth (never mutated), tests/test_data/rules_2026_27/:
  rules_2026-27.json        official chips / game_settings / scoring
  history.json, transfers.json, entry.json, picks_gw{1..5}.json,
  live_gw{1..5}.json, bootstrap_now.json   (entry 8737706 GW1-5; the
  bootstrap/live files are trimmed to that entry's 18 players)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fpl_optimizer.engine.state import ChipState, GameState, PlayerSlot, Squad
from fpl_optimizer.utils.constants import Position

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GT = REPO / "tests" / "test_data" / "rules_2026_27"

ENTRY_ID = 8737706


def load_gt(name: str):
    with open(GT / name, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def rules():
    return load_gt("rules_2026-27.json")


@pytest.fixture(scope="session")
def bootstrap():
    return load_gt("bootstrap_now.json")


# --------------------------------------------------------------------------
# Synthetic 15-man squad: eids 1..15, list index = eid - 1
#   GK 1,2 | DEF 3-7 | MID 8-12 | FWD 13-15      default 4-4-2
# Market players: 16 DEF, 17 MID, 18 FWD, 19 GK, 20 DEF, 21 MID
# --------------------------------------------------------------------------
POS = {1: Position.GK, 2: Position.GK}
POS.update({e: Position.DEF for e in (3, 4, 5, 6, 7, 16, 20)})
POS.update({e: Position.MID for e in (8, 9, 10, 11, 12, 17, 21)})
POS.update({e: Position.FWD for e in (13, 14, 15, 18)})
POS[19] = Position.GK

DEFAULT_LINEUP = (1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14)
DEFAULT_BENCH = (2, 12, 7, 15)  # GK first, then outfield priority


def make_squad(
    lineup=DEFAULT_LINEUP,
    bench=DEFAULT_BENCH,
    captain=13,
    vice=8,
    price: int = 50,
) -> Squad:
    players = [
        PlayerSlot(element_id=e, position=POS[e], purchase_price=price,
                   selling_price=price)
        for e in range(1, 16)
    ]
    return Squad(
        players=players,
        lineup=[e - 1 for e in lineup],
        bench=[e - 1 for e in bench],
        captain_idx=captain - 1,
        vice_captain_idx=vice - 1,
    )


def make_state(gw: int = 6, ft: int = 1, bank: int = 0, **squad_kw) -> GameState:
    return GameState(
        squad=make_squad(**squad_kw),
        bank=bank,
        free_transfers=ft,
        chips=ChipState(),
        current_gw=gw,
    )


def row(pts: int = 2, minutes: int = 90, yc: int = 0, rc: int = 0,
        value: int = 50) -> dict:
    return {"total_points": pts, "minutes": minutes, "yellow_cards": yc,
            "red_cards": rc, "value": value}


class FakeLoader:
    """Duck-typed SeasonDataLoader: rows[(eid, gw)] -> dict (already DGW-summed)."""

    def __init__(self, rows: dict | None = None, positions: dict | None = None,
                 teams: dict | None = None):
        self.rows = rows or {}
        self._position_map = dict(positions or POS)
        self._team_map = dict(teams or {e: e for e in self._position_map})

    # API used by the engine
    def get_player_gw(self, eid, gw):
        r = self.rows.get((eid, gw))
        return dict(r) if r is not None else None

    def get_player_price(self, eid, gw):
        r = self.rows.get((eid, gw))
        return int(r.get("value", 0)) if r else 0

    def get_player_position(self, eid):
        return self._position_map.get(eid)

    def get_player_team(self, eid):
        return self._team_map.get(eid)


def full_rows(gw: int, pts=None, minutes=None, value=50, eids=range(1, 22)):
    """Every player plays 90' and scores pts[eid] (default 2)."""
    pts = pts or {}
    minutes = minutes or {}
    return {
        (e, gw): row(pts.get(e, 2), minutes.get(e, 90), value=value)
        for e in eids
    }


# --------------------------------------------------------------------------
# Offline FPL public API, backed by GT files (optionally overridden in-memory)
# --------------------------------------------------------------------------
class OfflineFPL:
    def __init__(self, overrides: dict | None = None):
        self.data = {
            "entry": load_gt("entry.json"),
            "history": load_gt("history.json"),
            "transfers": load_gt("transfers.json"),
        }
        for gw in range(1, 6):
            self.data[f"picks_{gw}"] = load_gt(f"picks_gw{gw}.json")
        if overrides:
            self.data.update(copy.deepcopy(overrides))
        self.calls: list[str] = []

    def __call__(self, url: str):
        self.calls.append(url)
        tail = url.split("/api/", 1)[1].strip("/").split("/")
        # entry/{id} | entry/{id}/history | entry/{id}/transfers |
        # entry/{id}/event/{gw}/picks
        if len(tail) == 2:
            return copy.deepcopy(self.data["entry"])
        if tail[2] == "history":
            return copy.deepcopy(self.data["history"])
        if tail[2] == "transfers":
            return copy.deepcopy(self.data["transfers"])
        if tail[2] == "event":
            return copy.deepcopy(self.data[f"picks_{int(tail[3])}"])
        raise KeyError(url)


@pytest.fixture
def offline_fpl(monkeypatch):
    """Factory: offline_fpl(overrides) patches live.entry._get and returns the stub."""
    import fpl_optimizer.live.entry as entry_mod

    def _install(overrides: dict | None = None) -> OfflineFPL:
        stub = OfflineFPL(overrides)
        monkeypatch.setattr(entry_mod, "_get", stub)
        return stub

    return _install
