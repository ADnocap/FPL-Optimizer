"""Weekly gameweek driver for the live 2026-27 season.

Chains: pre-deadline snapshot -> data refresh -> model predictions ->
MILP optimization -> printed recommendations.

Modes
-----
Initial squad (GW1 / wildcard from scratch):
    python scripts/gameweek.py --fresh-squad

Weekly transfers for a real team:
    python scripts/gameweek.py --team-id 1234567

Useful flags:
    --skip-refresh      don't re-download element summaries (~10 min)
    --model-dir PATH    predictor to use (default models/prod_2026-27 when it
                        exists, else models/full_pregame)
    --max-transfers N   cap transfers considered (default: optimizer decides)
    --chip NAME         evaluate with a chip (wildcard/free_hit/bench_boost/
                        triple_captain)
    --ep                use FPL's own EP instead of the model (sanity check)

Find your team ID on fantasy.premierleague.com -> Points tab -> the number
in the URL: /entry/<TEAM_ID>/event/1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fpl_optimizer.data.collectors.fpl_live import LiveFPLCollector
from fpl_optimizer.live.pool import build_live_candidates
from fpl_optimizer.live.predict import ep_reference, predict_upcoming_gw
from fpl_optimizer.optimizer.squad_selection import select_squad
from fpl_optimizer.optimizer.transfer_optimizer import optimize_transfers
from fpl_optimizer.utils.constants import CURRENT_SEASON

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gameweek")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
# Prefer the model retrained with 2025-26 (DEFCON-era) data when available
_PROD = REPO_ROOT / "models" / "prod_2026-27"
DEFAULT_MODEL_DIR = _PROD if _PROD.exists() else REPO_ROOT / "models" / "full_pregame"


def _fmt_player(eid: int, elements: dict, teams: dict, pts: dict) -> str:
    el = elements[eid]
    pos = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}[el["element_type"]]
    return (
        f"{el['web_name']:<20} {pos:<4} {teams[el['team']]:<4} "
        f"{el['now_cost'] / 10:>5.1f}m  xPts={pts.get(eid, 0.0):.2f}"
    )


def _resolve_element(name_or_id: str, elements: dict[int, dict]) -> int:
    """web_name (case-insensitive, unique) or numeric element id -> element id."""
    if name_or_id.isdigit():
        return int(name_or_id)
    hits = [el for el in elements.values()
            if el["web_name"].lower() == name_or_id.lower()]
    if len(hits) != 1:
        raise SystemExit(
            f"--captain/--vice '{name_or_id}' matches {len(hits)} players"
            + (": " + ", ".join(f"{h['web_name']} (id {h['id']})" for h in hits)
               if hits else "") + " — pass the element id instead."
        )
    return hits[0]["id"]


def _override_captaincy(result, captain: str | None, vice: str | None,
                        elements: dict[int, dict]):
    """Replace the MILP's captain/vice with user picks from the same XI."""
    import dataclasses

    cap_id = _resolve_element(captain, elements) if captain else result.captain_id
    vice_id = _resolve_element(vice, elements) if vice else result.vice_captain_id
    if vice_id == cap_id:  # keep them distinct: fall back to the MILP's other pick
        vice_id = (result.captain_id if result.captain_id != cap_id
                   else result.vice_captain_id)
    for label, eid in (("captain", cap_id), ("vice", vice_id)):
        if eid not in result.lineup_element_ids:
            raise SystemExit(f"--{label} {eid} is not in the recommended XI")
    return dataclasses.replace(result, captain_id=cap_id, vice_captain_id=vice_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-id", type=int, default=None)
    parser.add_argument("--fresh-squad", action="store_true",
                        help="pick a full squad from scratch (GW1 or wildcard)")
    parser.add_argument("--budget", type=int, default=1000,
                        help="budget in tenths for --fresh-squad")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--season", default=CURRENT_SEASON)
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--skip-build", action="store_true",
                        help="reuse existing season files (no API calls for data)")
    parser.add_argument("--max-transfers", type=int, default=None)
    parser.add_argument("--chip", default=None,
                        choices=[None, "wildcard", "free_hit", "bench_boost",
                                 "triple_captain"])
    parser.add_argument("--ep", action="store_true",
                        help="use FPL's own EP instead of the trained model")
    parser.add_argument("--min-chance", type=int, default=75)
    parser.add_argument("--captain", default=None, metavar="NAME_OR_ID",
                        help="override the MILP captain (web_name or element id; "
                             "must be in the recommended XI)")
    parser.add_argument("--vice", default=None, metavar="NAME_OR_ID",
                        help="override the MILP vice-captain (same rules)")
    parser.add_argument("--apply", action="store_true",
                        help="submit to the FPL API (dry-run validation unless --yes; "
                             "needs FPL_REFRESH_TOKEN in .env — see live/auth.py)")
    parser.add_argument("--yes", action="store_true",
                        help="with --apply: actually commit transfers and lineup")
    args = parser.parse_args()

    # Default --team-id from .env (FPL_TEAM_ID)
    if args.team_id is None and not args.fresh_squad:
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
        if os.environ.get("FPL_TEAM_ID"):
            args.team_id = int(os.environ["FPL_TEAM_ID"])
            print(f"Using FPL_TEAM_ID={args.team_id} from .env")

    # Fail fast on auth BEFORE any expensive work: refresh tokens can be
    # revoked by a browser/app login on the same FPL session.
    if args.apply:
        from fpl_optimizer.live.auth import FPLAuth, FPLAuthError

        try:
            FPLAuth().access_token()
        except FPLAuthError as exc:
            print(f"AUTH CHECK FAILED — nothing will be submitted.\n{exc}")
            return
        print("Auth OK (token refreshed)")

    collector = LiveFPLCollector(data_dir=args.data_dir, season=args.season)
    bootstrap = collector.fetch_bootstrap()
    fixtures = collector.fetch_fixtures()

    gw, _ = collector._target_event(bootstrap)
    if gw is None:
        print("No upcoming gameweek found — season over?")
        return

    # Deadline info
    event = next(e for e in bootstrap["events"] if e["id"] == gw)
    print(f"\n=== Planning GW{gw} — deadline {event['deadline_time']} ===\n")

    # 0. Fetch live team state FIRST (fail fast, before expensive data work)
    entry_state = None
    if args.team_id is not None and not args.fresh_squad:
        from fpl_optimizer.live.entry import fetch_entry_state

        try:
            entry_state = fetch_entry_state(args.team_id, bootstrap)
        except ValueError as exc:
            print(f"Cannot load team {args.team_id}: {exc}")
            print("Hint: before GW1 completes, use --fresh-squad instead.")
            return
        gs_chips = entry_state.game_state.chips
        if args.chip and not gs_chips.is_available(args.chip, gw):
            print(f"ERROR: chip '{args.chip}' is not available for GW{gw} "
                  "(already used, expired, or blocked this GW).")
            return

    # 1. Data refresh
    if not args.skip_build:
        collector.snapshot_predeadline()
        if not args.skip_refresh:
            collector.refresh_element_summaries(bootstrap)
        collector.build_season_files(
            bootstrap=bootstrap, fixtures=fixtures, include_upcoming=True
        )
        # Live player-prop odds (feeds props_* features; optional — needs
        # ODDS_API_KEY with prop-market access; ~40 credits per GW)
        try:
            from collect_props import collect_live_props

            collect_live_props(args.season, gw, args.data_dir)
        except Exception as exc:
            print(f"(props snapshot skipped: {exc})")

    # 2. Predictions
    # FPL's EP already embeds chance_of_playing (EP_FORMULA.md), so the pool
    # must not scale by availability a second time in EP mode.
    using_ep = args.ep
    if args.ep:
        predictions = ep_reference(bootstrap)
        print("Predictions: FPL EP (ep_next)")
    else:
        predictions = predict_upcoming_gw(
            args.data_dir, args.model_dir, args.season, gw
        )
        if not predictions:
            print("WARNING: model produced no predictions — falling back to EP.")
            predictions = ep_reference(bootstrap)
            using_ep = True
        else:
            print(f"Predictions: {args.model_dir.name} model "
                  f"({len(predictions)} players)")

    elements = {el["id"]: el for el in bootstrap["elements"]}
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}

    # 3. Optimize
    if args.fresh_squad or args.team_id is None:
        if args.apply:
            print("ERROR: --apply is not supported in fresh-squad mode — the "
                  "initial squad must be entered on the site; for a wildcard "
                  "rebuild use --team-id with --chip wildcard (rides on the "
                  "transfers endpoint).")
            return
        if not args.fresh_squad:
            print("No --team-id given: running fresh-squad mode.\n")
        candidates = build_live_candidates(
            bootstrap, predictions,
            min_chance=args.min_chance,
            availability_scaling=not using_ep,
        )
        result = select_squad(candidates, budget=args.budget)
        header = f"Fresh squad (cost {result.total_cost / 10:.1f}m)"
    else:
        gs = entry_state.game_state
        print(f"Team: {entry_state.team_name}  |  "
              f"{entry_state.overall_points} pts, rank "
              f"{entry_state.overall_rank:,}" if entry_state.overall_rank
              else f"Team: {entry_state.team_name}")
        print(f"Bank: {gs.bank / 10:.1f}m  |  free transfers: "
              f"{gs.free_transfers}  |  squad from GW{entry_state.picks_gw}\n")

        squad_ids = {p.element_id for p in gs.squad.players}
        candidates = build_live_candidates(
            bootstrap, predictions,
            min_chance=args.min_chance, always_include=squad_ids,
            availability_scaling=not using_ep,
        )
        result = optimize_transfers(
            gs, candidates, chip=args.chip, max_transfers=args.max_transfers
        )
        header = "Recommended plan"

        if result.transfers_out:
            print("Transfers:")
            for out_id, in_id in zip(result.transfers_out, result.transfers_in):
                print(f"  OUT: {_fmt_player(out_id, elements, teams, predictions)}")
                print(f"  IN:  {_fmt_player(in_id, elements, teams, predictions)}")
            if result.hit_cost:
                print(f"  Hit cost: -{result.hit_cost} pts")
        else:
            print("Transfers: none (roll the free transfer)")
        print()

    # 3b. Captain override (bookmaker/EP signal has beaten the model's
    # form-driven captain pick early season — see live-season notes).
    if args.captain or args.vice:
        result = _override_captaincy(result, args.captain, args.vice, elements)

    # 4. Report
    print(f"=== {header} — expected XI points: {result.objective_value:.1f} ===\n")
    print("Starting XI:")
    for eid in result.lineup_element_ids:
        tag = ""
        if eid == result.captain_id:
            tag = "  (C)"
        elif eid == result.vice_captain_id:
            tag = "  (V)"
        print(f"  {_fmt_player(eid, elements, teams, predictions)}{tag}")
    print("\nBench (sub priority order):")
    for eid in result.bench_element_ids:
        print(f"  {_fmt_player(eid, elements, teams, predictions)}")
    if args.chip:
        print(f"\nChip evaluated: {args.chip}")

    # 5. Optional API submission
    applied = False
    if args.apply and args.team_id:
        from fpl_optimizer.live.auth import FPLAuth
        from fpl_optimizer.live.executor import (
            apply_lineup, apply_transfers, get_me, get_my_team,
        )

        auth = FPLAuth()
        me = get_me(auth)
        me_entry = (me.get("player") or {}).get("entry")
        if me_entry and me_entry != args.team_id:
            print(f"\nWARNING: token belongs to entry {me_entry}, "
                  f"not {args.team_id} — aborting apply.")
            return
        my_team = get_my_team(auth, args.team_id)
        selling = {p["element"]: p["selling_price"] for p in my_team["picks"]}
        buy_price = {el["id"]: el["now_cost"] for el in bootstrap["elements"]}

        if result.transfers_out:
            # FPL requires every transfer pair to be the SAME position type.
            # The MILP only guarantees the resulting squad is valid, so its
            # out/in lists may cross positions (e.g. MID out, DEF in) even
            # when the squad shape is unchanged — re-pair them by type here.
            el_type = {el["id"]: el["element_type"] for el in bootstrap["elements"]}
            outs_by_type: dict[int, list[int]] = {}
            for out_id in result.transfers_out:
                outs_by_type.setdefault(el_type[out_id], []).append(out_id)
            payload = []
            for in_id in result.transfers_in:
                pos = el_type[in_id]
                if not outs_by_type.get(pos):
                    print(
                        f"ERROR: cannot pair incoming element {in_id} (type "
                        f"{pos}) with an outgoing player of the same type; "
                        "not submitting."
                    )
                    return
                out_id = outs_by_type[pos].pop()
                payload.append(
                    {
                        "element_in": in_id,
                        "element_out": out_id,
                        "purchase_price": buy_price[in_id],
                        "selling_price": selling.get(
                            out_id, buy_price.get(out_id, 0)
                        ),
                    }
                )
            chip = args.chip if args.chip in ("wildcard", "free_hit") else None
            # NOTE: the current FPL API applies transfers even when the
            # payload says confirmed=false (observed GW3 2026-27: the
            # follow-up confirmed=true POST failed with "Element in is
            # already picked").  So there is no safe dry-run — submit once
            # with confirmed=true and verify the resulting squad instead.
            print("\nTransfer payload:")
            for t in payload:
                print(f"  out {t['element_out']} (sell {t['selling_price']}) -> "
                      f"in {t['element_in']} (buy {t['purchase_price']})")
            if args.yes:
                apply_transfers(
                    auth, args.team_id, gw, payload, chip=chip, confirm=True
                )
                after = get_my_team(auth, args.team_id)
                have = {p["element"] for p in after["picks"]}
                missing = [t["element_in"] for t in payload
                           if t["element_in"] not in have]
                lingering = [t["element_out"] for t in payload
                             if t["element_out"] in have]
                if missing or lingering:
                    print(f"ERROR: squad after transfers does not match — "
                          f"missing ins {missing}, still present outs "
                          f"{lingering}; not touching the lineup.")
                    return
                print(f"Transfers COMMITTED and verified "
                      f"(bank now {after['transfers']['bank'] / 10:.1f}m).")
            else:
                print("Not submitted — re-run with --yes to commit.")
        if args.yes:
            chip = args.chip if args.chip in ("bench_boost", "triple_captain") else None
            apply_lineup(
                auth, args.team_id,
                result.lineup_element_ids, result.bench_element_ids,
                result.captain_id, result.vice_captain_id, chip=chip,
                element_types={
                    el["id"]: el["element_type"] for el in bootstrap["elements"]
                },
            )
            print("Lineup/captain APPLIED.")
            applied = True

    print(f"\nDeadline: {event['deadline_time']}"
          + ("" if applied else
             " — apply on fantasy.premierleague.com or re-run with --apply --yes")
          + "\n")


if __name__ == "__main__":
    main()
