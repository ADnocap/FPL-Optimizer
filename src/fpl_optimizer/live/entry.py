"""Fetch a real FPL team (entry) from the public API and build a GameState.

Uses only public endpoints (no login):

- ``entry/{id}/``                 -> current event
- ``entry/{id}/history/``         -> chips played, per-GW bank/transfers
- ``entry/{id}/event/{gw}/picks/``-> squad, lineup, captain, bank (post-deadline)
- ``entry/{id}/transfers/``       -> full transfer history with prices

Purchase prices are reconstructed from the transfer history; players held
since GW1 use ``now_cost - cost_change_start`` (their season-start price).
Selling prices apply the 50%-of-appreciation rule.

Limitation: transfers already made for the UPCOMING gameweek are only visible
on the authenticated ``my-team`` endpoint, so run recommendations before
making transfers on the site.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from fpl_optimizer.engine.state import ChipState, GameState, PlayerSlot, Squad
from fpl_optimizer.utils.constants import MAX_FREE_TRANSFERS, Position

logger = logging.getLogger(__name__)

FPL_API_BASE = "https://fantasy.premierleague.com/api"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# API chip names -> engine chip names
_CHIP_NAMES = {
    "wildcard": "wildcard",
    "freehit": "free_hit",
    "bboost": "bench_boost",
    "3xc": "triple_captain",
}


@dataclass
class LiveEntryState:
    """A real FPL team's state, ready for the transfer optimizer."""

    game_state: GameState
    team_name: str
    overall_points: int
    overall_rank: int | None
    picks_gw: int  # GW the squad snapshot comes from
    upcoming_gw: int  # GW being planned


def _get(url: str) -> dict | list:
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def offline_fetcher(entry_dir):
    """A ``fetcher`` for :func:`fetch_entry_state` that reads saved JSONs.

    ``entry_dir`` holds the public-API responses: ``entry.json``,
    ``history.json``, ``transfers.json`` and ``picks_gw{n}.json`` — no network.
    """
    import json
    from pathlib import Path

    d = Path(entry_dir)

    def _read(url: str) -> dict | list:
        tail = url.split("/entry/", 1)[1].strip("/").split("/")
        if len(tail) == 1:
            name = "entry.json"
        elif tail[1] in ("history", "transfers"):
            name = f"{tail[1]}.json"
        elif tail[1] == "event" and len(tail) >= 4 and tail[3] == "picks":
            name = f"picks_gw{tail[2]}.json"
        else:
            raise ValueError(f"offline_fetcher: unsupported endpoint {url}")
        path = d / name
        if not path.exists():
            raise FileNotFoundError(f"offline_fetcher: {path} missing (for {url})")
        return json.loads(path.read_text(encoding="utf-8"))

    return _read


def _compute_free_transfers(
    history: dict, chips_by_gw: dict[int, str], upcoming_gw: int
) -> int:
    """Simulate the free-transfer bank from the entry's first GW to the upcoming GW.

    Changes before a team's FIRST deadline are unlimited (GW1 for us, later for
    a late joiner); the following GW starts with exactly 1 FT. Then each GW:
    FT' = min(5, max(FT - made, 0) + 1), except WC/FH weeks where the count is
    carried unchanged (official: "four saved ahead of GW29 ... activate the
    Free Hit ... still have four free transfers for GW30").
    """
    transfers_by_gw = {
        row["event"]: row.get("event_transfers", 0)
        for row in history.get("current", [])
    }
    first_gw = min(transfers_by_gw, default=1)
    ft = 1  # after the first entered GW everyone has 1 FT
    for gw in range(first_gw + 1, upcoming_gw):
        made = transfers_by_gw.get(gw, 0)
        chip = chips_by_gw.get(gw)
        if chip in ("wildcard", "free_hit"):
            # WC/FH: banked FTs carry through UNCHANGED (no consumption, no +1)
            ft = min(MAX_FREE_TRANSFERS, ft)
        else:
            ft = min(MAX_FREE_TRANSFERS, max(ft - made, 0) + 1)
    return ft


def fetch_entry_state(
    team_id: int,
    bootstrap: dict,
    fetcher=None,
) -> LiveEntryState:
    """Fetch a team's live state from the public FPL API.

    Parameters
    ----------
    team_id : int
        The FPL entry ID (visible in the URL on the Points page).
    bootstrap : dict
        Current bootstrap-static JSON (for prices and positions).
    fetcher : callable(url) -> JSON, optional
        Replaces the HTTP GET (e.g. :func:`offline_fetcher` for saved JSONs).
    """
    _get_json = fetcher or _get
    entry = _get_json(f"{FPL_API_BASE}/entry/{team_id}/")
    history = _get_json(f"{FPL_API_BASE}/entry/{team_id}/history/")
    transfers = _get_json(f"{FPL_API_BASE}/entry/{team_id}/transfers/")

    current_event = entry.get("current_event")
    if not current_event:
        raise ValueError(
            f"Entry {team_id} has no completed gameweek yet — "
            "use initial-squad mode instead."
        )
    upcoming_gw = current_event + 1

    picks_data = _get_json(f"{FPL_API_BASE}/entry/{team_id}/event/{current_event}/picks/")
    # Free Hit squads revert after the GW — the picks endpoint permanently
    # records the temporary FH squad, so read the REAL squad from the GW before.
    if picks_data.get("active_chip") == "freehit" and current_event > 1:
        picks_data = _get_json(
            f"{FPL_API_BASE}/entry/{team_id}/event/{current_event - 1}/picks/"
        )
    picks = picks_data["picks"]
    entry_history = picks_data.get("entry_history", {})

    elements = {el["id"]: el for el in bootstrap["elements"]}

    # Chips played (engine names), keyed by GW
    chips_by_gw: dict[int, str] = {}
    for chip in history.get("chips", []):
        engine_name = _CHIP_NAMES.get(chip["name"])
        if engine_name:
            chips_by_gw[chip["event"]] = engine_name

    # Free Hit squads revert — exclude that GW's transfers from price tracking
    fh_gws = {gw for gw, c in chips_by_gw.items() if c == "free_hit"}
    latest_buy_price: dict[int, int] = {}
    for tr in sorted(transfers, key=lambda t: (t["event"], t.get("time") or "")):
        if tr["event"] in fh_gws:
            continue
        latest_buy_price[tr["element_in"]] = tr["element_in_cost"]

    # Build the squad from picks (position 1-11 = lineup, 12-15 = bench order)
    players: list[PlayerSlot] = []
    lineup: list[int] = []
    bench: list[int] = []
    captain_idx = 0
    vice_captain_idx = 0
    for i, pick in enumerate(sorted(picks, key=lambda p: p["position"])):
        eid = pick["element"]
        el = elements.get(eid)
        if el is None:
            raise ValueError(f"Element {eid} not in bootstrap — season mismatch?")
        now_cost = el["now_cost"]
        purchase = latest_buy_price.get(
            eid, now_cost - el.get("cost_change_start", 0)
        )
        if now_cost > purchase:
            selling = purchase + (now_cost - purchase) // 2
        else:
            selling = now_cost
        players.append(
            PlayerSlot(
                element_id=eid,
                position=Position(el["element_type"]),
                purchase_price=purchase,
                selling_price=selling,
            )
        )
        if pick["position"] <= 11:
            lineup.append(i)
        else:
            bench.append(i)
        if pick.get("is_captain"):
            captain_idx = i
        if pick.get("is_vice_captain"):
            vice_captain_idx = i

    squad = Squad(
        players=players,
        lineup=lineup,
        bench=bench,
        captain_idx=captain_idx,
        vice_captain_idx=vice_captain_idx,
    )

    # Chip availability
    chip_state = ChipState()
    for gw, chip in chips_by_gw.items():
        chip_state.use_chip(chip, gw)
    if upcoming_gw > 19:
        chip_state.expire_first_half()

    game_state = GameState(
        squad=squad,
        bank=entry_history.get("bank", 0),
        free_transfers=_compute_free_transfers(history, chips_by_gw, upcoming_gw),
        chips=chip_state,
        current_gw=upcoming_gw,
        total_points=entry.get("summary_overall_points", 0),
    )

    return LiveEntryState(
        game_state=game_state,
        team_name=entry.get("name", ""),
        overall_points=entry.get("summary_overall_points", 0),
        overall_rank=entry.get("summary_overall_rank"),
        picks_gw=current_event,
        upcoming_gw=upcoming_gw,
    )
