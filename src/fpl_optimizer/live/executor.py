"""Submit lineups, transfers, and chips to the real FPL API.

Endpoint contracts (verified against AIrsenal / fpl-picker, Aug 2026):

- Lineup / captain / bench order / bboost / 3xc:
    POST /api/my-team/{entry_id}/
    {"picks": [{"element", "position" 1-15, "is_captain", "is_vice_captain"}],
     "chip": null | "bboost" | "3xc"}

- Transfers / wildcard / freehit (two-step: validate then commit):
    POST /api/transfers/
    {"confirmed": false|true, "entry": id, "event": gw,
     "transfers": [{"element_in", "element_out", "purchase_price",
                    "selling_price"}],
     "chip": null | "wildcard" | "freehit"}
    Success on commit = HTTP 200 with EMPTY body (do not JSON-decode).

Selling prices MUST come from the authenticated GET /api/my-team/{id}/
(public now_cost is wrong after price rises).
"""

from __future__ import annotations

import logging

import requests

from fpl_optimizer.live.auth import FPLAuth

logger = logging.getLogger(__name__)

FPL_API_BASE = "https://fantasy.premierleague.com/api"

# engine chip name -> API chip name per endpoint
_MYTEAM_CHIPS = {"bench_boost": "bboost", "triple_captain": "3xc"}
_TRANSFER_CHIPS = {"wildcard": "wildcard", "free_hit": "freehit"}


class FPLWriteError(RuntimeError):
    pass


def get_me(auth: FPLAuth) -> dict:
    """Verify the token and return the authenticated manager (entry id)."""
    resp = requests.get(f"{FPL_API_BASE}/me/", headers=auth.headers(), timeout=30)
    if resp.status_code != 200:
        raise FPLWriteError(f"/me/ failed ({resp.status_code}): {resp.text[:200]}")
    return resp.json()


def get_my_team(auth: FPLAuth, entry_id: int) -> dict:
    """Authenticated team state: picks with selling/purchase prices, bank,
    free transfers, chip availability, pending transfers."""
    resp = requests.get(
        f"{FPL_API_BASE}/my-team/{entry_id}/", headers=auth.headers(), timeout=30
    )
    if resp.status_code != 200:
        raise FPLWriteError(
            f"my-team GET failed ({resp.status_code}): {resp.text[:200]}"
        )
    return resp.json()


def apply_lineup(
    auth: FPLAuth,
    entry_id: int,
    lineup_element_ids: list[int],
    bench_element_ids: list[int],
    captain_id: int,
    vice_captain_id: int,
    chip: str | None = None,
    element_types: dict[int, int] | None = None,
) -> None:
    """POST the starting XI, bench order, captaincy, and optional BB/TC chip.

    The server requires positions 1-11 ordered GK, DEF, MID, FWD and the
    substitute GK pinned at position 12 (bench outfielders keep their given
    sub-priority order at 13-15).  *element_types* maps element_id ->
    element_type (1-4); if omitted it is fetched from bootstrap-static.
    """
    if chip is not None and chip not in _MYTEAM_CHIPS:
        raise ValueError(
            f"my-team POST only accepts bench_boost/triple_captain, got {chip}"
        )
    if element_types is None:
        resp = requests.get(
            f"{FPL_API_BASE}/bootstrap-static/", headers=auth.headers(), timeout=30
        )
        resp.raise_for_status()
        element_types = {
            el["id"]: el["element_type"] for el in resp.json()["elements"]
        }

    xi = sorted(lineup_element_ids, key=lambda eid: element_types[eid])
    bench_gk = [e for e in bench_element_ids if element_types[e] == 1]
    bench_out = [e for e in bench_element_ids if element_types[e] != 1]
    ordered = xi + bench_gk + bench_out
    if len(ordered) != 15 or len(bench_gk) != 1:
        raise ValueError(
            f"Invalid squad shape: {len(xi)} starters, {len(bench_gk)} bench GK"
        )
    picks = []
    for pos, eid in enumerate(ordered, start=1):
        picks.append(
            {
                "element": eid,
                "position": pos,
                "is_captain": eid == captain_id,
                "is_vice_captain": eid == vice_captain_id,
            }
        )
    payload = {"picks": picks, "chip": _MYTEAM_CHIPS.get(chip) if chip else None}
    resp = requests.post(
        f"{FPL_API_BASE}/my-team/{entry_id}/",
        json=payload,
        headers=auth.headers(),
        timeout=30,
    )
    if resp.status_code != 200:
        raise FPLWriteError(
            f"lineup POST failed ({resp.status_code}): {resp.text[:400]}"
        )
    logger.info("Lineup applied (%d picks, chip=%s)", len(picks), chip)


def apply_transfers(
    auth: FPLAuth,
    entry_id: int,
    event: int,
    transfers: list[dict],
    chip: str | None = None,
    confirm: bool = False,
) -> dict | None:
    """Validate (confirm=False) or commit (confirm=True) transfers.

    Each transfer dict: {"element_in", "element_out", "purchase_price",
    "selling_price"} — prices in tenths.  purchase_price = buy price now
    (bootstrap now_cost of element_in); selling_price = the authenticated
    my-team selling_price of element_out.

    Returns the validation response dict on dry-run, None on committed success.
    """
    if chip is not None and chip not in _TRANSFER_CHIPS:
        raise ValueError(
            f"transfers POST only accepts wildcard/free_hit, got {chip}"
        )
    api_chip = _TRANSFER_CHIPS.get(chip) if chip else None
    payload = {
        "confirmed": confirm,
        "entry": entry_id,
        "event": event,
        "transfers": transfers,
        "chip": api_chip,
        # older/parallel client shape — keep consistent for compatibility
        "wildcard": api_chip == "wildcard",
        "freehit": api_chip == "freehit",
    }
    resp = requests.post(
        f"{FPL_API_BASE}/transfers/",
        json=payload,
        headers=auth.headers(),
        timeout=30,
    )
    if resp.status_code != 200:
        raise FPLWriteError(
            f"transfers POST ({'commit' if confirm else 'dry-run'}) failed "
            f"({resp.status_code}): {resp.text[:400]}"
        )
    if not confirm:
        try:
            return resp.json()
        except ValueError:
            return {}
    # committed success is an empty 200
    logger.info("Transfers committed (%d moves, chip=%s)", len(transfers), chip)
    return None
