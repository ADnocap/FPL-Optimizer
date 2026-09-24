"""Weekly gameweek driver for the live 2026-27 season.

Chains: pre-deadline snapshot -> data refresh -> model predictions ->
MILP optimization -> printed recommendations.

Modes
-----
Initial squad (GW1 only — budget is a flat --budget, default 100.0m):
    python scripts/gameweek.py --fresh-squad

Wildcard / Free Hit (budget = bank + real selling prices, not 100.0m):
    python scripts/gameweek.py --team-id 1234567 --chip wildcard

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

Multi-GW planning (receding horizon; default off = single-GW optimizer):
    --horizon N         plan transfers jointly over the next N GWs (FT
                        banking, hits, fixture swings, blanks/doubles);
                        only this GW's moves are executed. 3 is the
                        backtested choice (scripts/backtest_horizon.py).
    --chip-plan SPEC    chip schedule the planner must respect, e.g.
                        "tc:7,wc:11,bb:12" (tc/wc/bb/fh). The plan is a human
                        decision (SEASON_GUIDE.md); chips outside the horizon
                        or already used are ignored. --chip overrides this GW.
    --discount D        per-GW discount of future points (default 0.85)
    --hit-margin M      extra penalty per -4 hit (default 4: a hit must
                        promise > 8 planned points — guards against
                        over-predicted buys; 0 overtrades badly in backtests)
    --max-hits K        at most K hits per GW (0 = free transfers only)
    --ft-value V        value of each FT banked after the horizon (default 0)

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


def _sync_with_my_team(gs, my_team: dict) -> list[str]:
    """Overwrite the public-API reconstruction with authenticated my-team values.

    The public endpoints only let us *simulate* the FT bank and selling prices;
    my-team is the source of truth (FT top-ups, price-rise rounding, anything
    FPL changes mid-season). Returns a note per corrected value; raises
    ValueError when the squad itself differs (moves already made on the site).
    """
    notes: list[str] = []
    tr = my_team.get("transfers") or {}
    if tr.get("made"):
        raise ValueError(
            f"{tr['made']} transfer(s) already made for this GW on the site — the "
            "public squad is stale; undo them or plan this GW manually."
        )
    picks = {p["element"]: p for p in my_team.get("picks", [])}
    ours = {p.element_id for p in gs.squad.players}
    if picks and set(picks) != ours:
        raise ValueError(
            f"my-team squad differs from the public picks "
            f"(+{sorted(set(picks) - ours)} / -{sorted(ours - set(picks))})"
        )
    limit = tr.get("limit")
    if isinstance(limit, int) and limit != gs.free_transfers:
        notes.append(f"free transfers {gs.free_transfers} -> {limit}")
        gs.free_transfers = limit
    bank = tr.get("bank")
    if isinstance(bank, int) and bank != gs.bank:
        notes.append(f"bank {gs.bank} -> {bank}")
        gs.bank = bank
    for p in gs.squad.players:
        mp = picks.get(p.element_id, {})
        sp = mp.get("selling_price")
        if isinstance(sp, int) and sp != p.selling_price:
            notes.append(f"selling price {p.element_id}: {p.selling_price} -> {sp}")
            p.selling_price = sp
        if isinstance(mp.get("purchase_price"), int):
            p.purchase_price = mp["purchase_price"]
    return notes


def _refresh_side_sources(data_dir: Path, season: str) -> None:
    """Understat per-match data and h2h odds for the live season.

    The model was trained with both populated; serving them empty (as in
    GW1-5 2026-27) cost most of the top-of-ranking accuracy. Failures are
    non-fatal but loud — the data-health block will flag the gap.
    """
    try:
        from fpl_optimizer.data.collectors.understat import UnderstatCollector
        from fpl_optimizer.data.collectors.understat_ids import (
            build_understat_id_supplement,
        )

        UnderstatCollector(data_dir=data_dir).refresh_live_season(season)
        build_understat_id_supplement(data_dir, season)
    except Exception as exc:
        print(f"WARNING: understat refresh failed ({exc}) — understat features "
              "will be stale or empty this GW.")
    try:
        from collect_football_data_odds import build_season_odds

        build_season_odds(season, data_dir, include_upcoming=True)
    except Exception as exc:
        print(f"WARNING: h2h odds refresh failed ({exc}) — odds features may be "
              "empty for the upcoming GW.")


def _write_decision_log(data_dir: Path, season: str, gw: int, record: dict) -> Path:
    """Persist what the model saw and recommended (post-mortems need it)."""
    import json
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = data_dir / "live" / season / "decisions" / f"gw{gw}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-id", type=int, default=None)
    parser.add_argument("--fresh-squad", action="store_true",
                        help="pick a full squad from scratch with --budget (GW1 "
                             "only; for a wildcard use --team-id --chip wildcard)")
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
    parser.add_argument("--horizon", type=int, default=None, metavar="N",
                        help="multi-GW planner over N GWs (team mode only)")
    parser.add_argument("--chip-plan", default=None, metavar="SPEC",
                        help='chip schedule for the planner, e.g. "tc:7,wc:11,bb:12"')
    parser.add_argument("--discount", type=float, default=None,
                        help="per-GW discount (default: HorizonConfig)")
    parser.add_argument("--hit-margin", type=float, default=None,
                        help="extra penalty per -4 hit (default: HorizonConfig)")
    parser.add_argument("--ft-value", type=float, default=None,
                        help="value per FT banked after the horizon (default: HorizonConfig)")
    parser.add_argument("--max-hits", type=int, default=None,
                        help="max -4 hits per GW for the planner (0 = FT-only; "
                             "default: HorizonConfig)")
    parser.add_argument("--apply", action="store_true",
                        help="submit to the FPL API (dry-run validation unless --yes; "
                             "needs FPL_REFRESH_TOKEN in .env — see live/auth.py)")
    parser.add_argument("--yes", action="store_true",
                        help="with --apply: actually commit transfers and lineup")
    args = parser.parse_args()
    if args.horizon is not None:
        if args.horizon < 1:
            parser.error("--horizon must be >= 1")
        if args.ep:
            parser.error("--horizon needs model predictions (FPL EP covers one GW only)")
        if args.fresh_squad:
            parser.error("--horizon plans transfers for an existing team (use --team-id)")
    if args.chip_plan and args.horizon is None:
        parser.error("--chip-plan is only used by the multi-GW planner (add --horizon N)")
    chip_plan: dict[int, str] = {}
    if args.chip_plan:
        from fpl_optimizer.optimizer.horizon_optimizer import parse_chip_plan

        try:
            chip_plan = parse_chip_plan(args.chip_plan)
        except ValueError as exc:
            parser.error(str(exc))

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
        if args.apply:
            # We are authenticated anyway: plan on the real FT count, bank and
            # selling prices instead of the public-API reconstruction.
            from fpl_optimizer.live.auth import FPLAuth
            from fpl_optimizer.live.executor import get_my_team

            try:
                notes = _sync_with_my_team(
                    entry_state.game_state, get_my_team(FPLAuth(), args.team_id)
                )
            except ValueError as exc:
                print(f"ERROR: {exc}")
                return
            for note in notes:
                print(f"my-team sync: {note}")

    # 1. Data refresh
    if not args.skip_build:
        collector.snapshot_predeadline()
        if not args.skip_refresh:
            collector.refresh_element_summaries(bootstrap)
        collector.build_season_files(
            bootstrap=bootstrap, fixtures=fixtures, include_upcoming=True
        )
        _refresh_side_sources(args.data_dir, args.season)
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
    health: list[str] = []
    horizon_preds = None
    horizon_gws: list[int] = []
    if args.horizon is not None and args.team_id is not None:
        from fpl_optimizer.live.predict import predict_horizon_live

        horizon_gws = list(range(gw, min(gw + args.horizon, 39)))
        horizon_preds = predict_horizon_live(
            args.data_dir, args.model_dir, args.season, gw, len(horizon_gws),
            health=health,
        )
        if horizon_preds.empty:
            print("ERROR: no horizon predictions — cannot run the multi-GW planner.")
            return
        k0 = horizon_preds[horizon_preds["k"] == 0]
        predictions = dict(zip(k0["element"].astype(int), k0["pred"].astype(float)))
        print(f"Predictions: {args.model_dir.name} model, horizon GW{horizon_gws[0]}"
              f"-{horizon_gws[-1]} ({len(predictions)} players this GW)")
    elif args.ep:
        predictions = ep_reference(bootstrap)
        print("Predictions: FPL EP (ep_next)")
    else:
        predictions = predict_upcoming_gw(
            args.data_dir, args.model_dir, args.season, gw, health=health
        )
        if not predictions:
            print("WARNING: model produced no predictions — falling back to EP.")
            predictions = ep_reference(bootstrap)
            using_ep = True
        else:
            print(f"Predictions: {args.model_dir.name} model "
                  f"({len(predictions)} players)")

    if health:
        print("\n!!! DATA HEALTH — the model will see these features empty/constant:")
        for w in health:
            print(f"    - {w}")
        print("    Fix the data (or read the recommendation with caution).\n")
    else:
        print("Data health: all model features populated like training.")

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
        if horizon_preds is not None:
            from fpl_optimizer.live.pool import build_live_horizon_candidates
            from fpl_optimizer.optimizer.horizon_optimizer import (
                HorizonConfig,
                optimize_horizon,
            )

            if args.chip:
                chip_plan[gw] = args.chip
            h_cands = build_live_horizon_candidates(
                bootstrap, horizon_preds, horizon_gws,
                min_chance=args.min_chance, always_include=squad_ids,
            )
            cfg = HorizonConfig(max_transfers_per_gw=args.max_transfers)
            if args.discount is not None:
                cfg.discount = args.discount
            if args.hit_margin is not None:
                cfg.hit_margin = args.hit_margin
            if args.max_hits is not None:
                cfg.max_hits_per_gw = args.max_hits
            if args.ft_value is not None:
                cfg.ft_value = args.ft_value
            hres = optimize_horizon(gs, h_cands, horizon_gws, chip_plan, cfg)
            import dataclasses

            # report THIS GW's expected points (the objective spans the horizon)
            result = dataclasses.replace(
                hres.first, objective_value=hres.plan[0].expected_points)
            xp_by = {c.element_id: c.xpts for c in h_cands}
            print(f"Multi-GW plan (GW{horizon_gws[0]}-{horizon_gws[-1]}, discount "
                  f"{cfg.discount}, hit margin {cfg.hit_margin}, max hits/GW "
                  f"{cfg.max_hits_per_gw}, FT value {cfg.ft_value}, "
                  f"solve {hres.solve_seconds:.0f}s, {hres.status}); "
                  "only the first GW is executed — re-run next week:")
            for k, p in enumerate(hres.plan):
                names_in = ", ".join(elements[e]["web_name"] for e in p.transfers_in)
                names_out = ", ".join(elements[e]["web_name"] for e in p.transfers_out)
                chip_txt = f" [{p.chip}]" if p.chip else ""
                moves = (f"OUT {names_out} -> IN {names_in}" if p.transfers_in
                         else "no transfers")
                print(f"  GW{p.gw}{chip_txt}: FT {p.free_transfers}, {moves}"
                      + (f", hits -{4 * p.hits}" if p.hits else "")
                      + f" | C {elements[p.captain_id]['web_name']}"
                      f" ({xp_by.get(p.captain_id, (0,) * (k + 1))[k]:.1f})"
                      f" | xPts {p.expected_points:.1f} | bank {p.bank_after / 10:.1f}m")
            print()
        else:
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
    # The chip this GW: --chip, or the planner's chip-plan entry for this GW
    chip_used = result.chip if horizon_preds is not None else args.chip
    if chip_used:
        print(f"\nChip {'planned' if horizon_preds is not None else 'evaluated'}"
              f" this GW: {chip_used}")

    log_path = _write_decision_log(args.data_dir, args.season, gw, {
        "gw": gw,
        "deadline": event["deadline_time"],
        "model_dir": str(args.model_dir),
        "mode": ("ep" if using_ep else "model")
                + (f"+horizon{args.horizon}" if horizon_preds is not None else ""),
        "args": {k: v for k, v in vars(args).items() if k not in ("yes",)},
        "data_health": health,
        "team_state": None if entry_state is None else {
            "free_transfers": entry_state.game_state.free_transfers,
            "bank": entry_state.game_state.bank,
            "squad": [p.element_id for p in entry_state.game_state.squad.players],
        },
        "predictions": {int(k): round(float(v), 3) for k, v in predictions.items()},
        "transfers_out": list(getattr(result, "transfers_out", []) or []),
        "transfers_in": list(getattr(result, "transfers_in", []) or []),
        "hit_cost": getattr(result, "hit_cost", 0),
        "lineup": list(result.lineup_element_ids),
        "bench": list(result.bench_element_ids),
        "captain": result.captain_id,
        "vice": result.vice_captain_id,
        "captain_override": args.captain,
        "vice_override": args.vice,
        "chip": chip_used,
        "expected_xi_points": result.objective_value,
    })
    print(f"\nDecision log: {log_path}")

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
        # One chip per GW, and the my-team POST always carries "chip": posting
        # null/another chip would cancel or replace one already active on the
        # site (e.g. BB clicked in the app). Require the flag to match it.
        api2engine = {"wildcard": "wildcard", "freehit": "free_hit",
                      "bboost": "bench_boost", "3xc": "triple_captain"}
        pending = [
            api2engine.get(c.get("name"))
            for c in my_team.get("chips", [])
            if c.get("status_for_entry") == "active" or c.get("is_pending")
        ]
        pending = [c for c in pending if c]
        if pending and args.chip not in pending:
            print(f"ERROR: chip '{pending[0]}' is already active for GW{gw} on "
                  f"the site. Re-run with --chip {pending[0]} (plans for it and "
                  "keeps it) or cancel it on the site. Nothing submitted.")
            return
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
            chip = chip_used if chip_used in ("wildcard", "free_hit") else None
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
            chip = chip_used if chip_used in ("bench_boost", "triple_captain") else None
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
