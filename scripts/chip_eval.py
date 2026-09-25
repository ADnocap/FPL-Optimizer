"""Chip decision support: expected gain of every remaining chip in every GW.

Reports numbers for the chip decision points; it never changes the approved
chip plan (SEASON_GUIDE.md), never submits anything, never snapshots and
never writes into data/.  Method, definitions and the calibration table:
``src/fpl_optimizer/optimizer/chip_eval.py``.

For each remaining chip and candidate GW g (t = the upcoming GW):
  TC   E[effective captain points] (captain, or the vice on failover) on the
       planned GW-g squad (no-chip receding-horizon plan, gameweek.py
       --horizon 3 replayed on the as-of-t predictions)
  BB   E[bench points under BB] - E[auto-sub points] on that squad
  FH   plan with FH at g vs the no-chip plan (optimize_horizon, same state /
       config / horizon / pool), and vs holding the GW-g squad
  WC   plan with WC at g vs without over g..g+5 (optimize_horizon)
plus the BGW/DGW/unscheduled-fixture map, the squad's availability and the
plan's early-WC trigger (>= 3 starters flagged/out).  Gains are Monte-Carlo
expectations (who plays, FPL auto-subs, captain failover); 'cal' columns
scale them by the 2025-26 replay's realised/predicted ratios for the chip
GW's season half.

Run after the weekly data build (the live season files must contain the
upcoming GW's synthetic rows, e.g. after ``scripts/gameweek.py``):

    python scripts/chip_eval.py --team-id 8737706
    python scripts/chip_eval.py --team-id 8737706 --chips tc,bb --last-gw 19
    python scripts/chip_eval.py --team-id 8737706 --wc-gws 8,11 --wc-bb 11:12,11:13

Offline (no network):
    python scripts/chip_eval.py --entry-dir <dir with entry/history/transfers/
        picks_gw{n}.json> --bootstrap bootstrap.json --fixtures fixtures.json

The model defaults to the weekly run's (scripts/gameweek.py DEFAULT_MODEL_DIR).
A model that uses the same-GW ``fpl_xp`` (a lookahead leak; the leak-fixed
pipeline no longer even produces it) is refused unless --allow-leaky-model.

P(play): the horizon predictor's calibrated lookup by default; --pplay-dir
<dir with pplay.lgb> uses the sharper LightGBM classifier
(``train-pplay --features <features.parquet> --pplay-dir <out>`` trains one).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_optimizer.utils.constants import CURRENT_SEASON  # noqa: E402

logger = logging.getLogger("chip_eval")
FPL_API = "https://fantasy.premierleague.com/api"


def gameweek_default_model_dir() -> Path:
    """scripts/gameweek.py's DEFAULT_MODEL_DIR (the model the weekly run uses)."""
    spec = importlib.util.spec_from_file_location(
        "_gameweek_defaults", REPO_ROOT / "scripts" / "gameweek.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return Path(mod.DEFAULT_MODEL_DIR)


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _api_json(endpoint: str):
    import requests

    r = requests.get(f"{FPL_API}/{endpoint}", headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    return r.json()


def _team_id_from_env() -> int | None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    v = os.environ.get("FPL_TEAM_ID")
    return int(v) if v else None


def _leak_banner(model_dir: Path, leaks: list[str], allowed: bool) -> str:
    bar = "!" * 78
    verdict = ("--allow-leaky-model given: continuing, numbers are NOT trustworthy"
               if allowed else "REFUSING to evaluate (pass --allow-leaky-model to override)")
    return (f"{bar}\n!!! LEAKY MODEL: {model_dir}\n!!! features {leaks} = the same-GW FPL xP "
            "(post-deadline ep_this, a lookahead leak).\n!!! The leak-fixed pipeline serves "
            "fpl_xp_lag instead, so this model would also see the feature as NaN.\n"
            f"!!! Retrain (scripts/train_predictor.py) or pass --model-dir <leak-fixed model>.\n"
            f"!!! {verdict}\n{bar}")


def cmd_train_pplay(args) -> int:
    import pandas as pd

    from fpl_optimizer.prediction.play_model import PlayModel, load_played_labels

    if args.features is None or args.pplay_dir is None:
        print("train-pplay needs --features <parquet> and --pplay-dir <output dir>")
        return 2
    if _is_inside(args.pplay_dir, args.data_dir):
        print(f"Refusing to write the P(play) model into {args.data_dir} (data/ is read-only here)")
        return 2
    df = pd.read_parquet(args.features)
    lab = load_played_labels(args.data_dir, sorted(df["season"].unique()))
    df = df.merge(lab, on=["season", "element", "GW"], how="inner")
    PlayModel.train(df, df["played"].to_numpy()).save(args.pplay_dir)
    print(f"P(play) model saved to {args.pplay_dir} ({len(df)} rows)")
    return 0


def cmd_eval(args) -> int:
    from fpl_optimizer.live.chip_inputs import (
        horizon_predictions,
        leaky_features,
        load_predictor,
        offline_entry_state,
        understat_gap,
    )
    from fpl_optimizer.live.pool import build_live_horizon_candidates
    from fpl_optimizer.optimizer import chip_eval as ce
    from fpl_optimizer.optimizer.horizon_optimizer import HorizonConfig

    if args.out is not None and _is_inside(args.out, args.data_dir):
        print(f"Refusing to write {args.out}: nothing is written into {args.data_dir}")
        return 2

    # --- inputs: bootstrap / fixtures / entry --------------------------------
    bootstrap = _load_json(args.bootstrap) if args.bootstrap else _api_json("bootstrap-static/")
    fixtures = _load_json(args.fixtures) if args.fixtures else _api_json("fixtures/")
    print(f"bootstrap: {args.bootstrap or 'FPL API (live)'}\n"
          f"fixtures:  {args.fixtures or 'FPL API (live)'}")
    if args.entry_dir:
        es = offline_entry_state(args.entry_dir, bootstrap, args.team_id)
        print(f"entry:     offline JSONs in {args.entry_dir}")
    else:
        from fpl_optimizer.live.entry import fetch_entry_state

        team_id = args.team_id or _team_id_from_env()
        if team_id is None:
            print("Need --team-id (or FPL_TEAM_ID in .env) or --entry-dir")
            return 2
        es = fetch_entry_state(team_id, bootstrap)
        print(f"entry:     {team_id} (public API)")
    state = es.game_state
    t = es.upcoming_gw
    squad_ids = [p.element_id for p in state.squad.players]

    # --- model / leak guard / P(play) -----------------------------------------
    predictor = load_predictor(args.model_dir)
    leaks = leaky_features(predictor)
    print(f"model:     {args.model_dir} ({len(predictor._feature_names)} features)")
    if leaks:
        print(_leak_banner(args.model_dir, leaks, args.allow_leaky_model))
        if not args.allow_leaky_model:
            return 3
    play_model = None
    if args.pplay_dir is not None:
        from fpl_optimizer.prediction.play_model import PlayModel

        play_model = PlayModel.load(args.pplay_dir)
        if play_model is None:
            print(f"WARNING: no pplay.lgb in --pplay-dir {args.pplay_dir} -> FALLING BACK to "
                  "the horizon predictor's calibrated P(play) lookup")
    print("P(play):   " + (f"LightGBM classifier {args.pplay_dir}" if play_model is not None
                           else "calibrated lookup on playing_prob x lead "
                                "(prediction/horizon.py; no --pplay-dir model)"))

    # --- which GWs are needed ----------------------------------------------------
    windows = ce.official_chip_windows(bootstrap["chips"])
    remaining = ce.remaining_chips(state.chips, windows, t)
    chips = {ce.SHORT_TO_CHIP[c.strip()] for c in args.chips.split(",") if c.strip()}
    if args.last_gw is None:  # the current half's chips
        args.last_gw = 19 if t <= 19 else 38
    last_gw = min(args.last_gw, 38)
    wc_gws = [int(g) for g in args.wc_gws.split(",")] if args.wc_gws else None
    combos = [tuple(int(x) for x in c.split(":")) for c in args.wc_bb.split(",") if c.strip()]
    cand_gws = [g for rc in remaining if rc.chip in chips for g in rc.gws if g <= last_gw]
    if ce.CHIP_WILDCARD in chips and wc_gws:
        cand_gws += [g for g in wc_gws]
    cand_gws += [w for w, _ in combos]
    if not cand_gws:
        print("No selected chip is playable in the window.")
        return 0
    reach = max(args.step_h, args.wc_h if (ce.CHIP_WILDCARD in chips or combos) else 1)
    end = min(38, max(cand_gws) + reach - 1)
    gws = list(range(t, end + 1))

    # --- predictions / pool ---------------------------------------------------------
    health: list[str] = []
    preds, feats = horizon_predictions(args.data_dir, args.model_dir, args.season, t,
                                       len(gws), predictor=predictor,
                                       play_model=play_model, health=health)
    if preds.empty:
        print(f"No GW{t} feature rows in {args.data_dir}/raw/{args.season}: build the live "
              "season files with the upcoming GW first (scripts/gameweek.py).")
        return 2
    gap = understat_gap(predictor._feature_names, feats, t)
    if gap:
        health.append(f"understat features all NaN at GW{t} but used by the model: {gap} "
                      "(elite attackers under-rated -> captain/TC numbers unreliable)")
    if health:
        print("\n!!! DATA HEALTH — the model will see these features empty/constant:")
        for w in health:
            print(f"    - {w}")
    cands = build_live_horizon_candidates(bootstrap, preds, gws, min_chance=args.min_chance,
                                          always_include=set(squad_ids))
    cfg = HorizonConfig(time_limit=args.time_limit)
    if args.discount is not None:
        cfg.discount = args.discount
    if args.hit_margin is not None:
        cfg.hit_margin = args.hit_margin
    if args.max_hits is not None:
        cfg.max_hits_per_gw = args.max_hits
    ctx = ce.ChipContext(state, cands, gws, cfg, n_sims=args.n_sims)
    print(f"horizon:   GW{gws[0]}-{gws[-1]} ({len(cands)} candidates); planner discount "
          f"{cfg.discount}, hit margin {cfg.hit_margin}, max hits {cfg.max_hits_per_gw}\n")

    res = ce.evaluate_chips(ctx, remaining, chips=chips, last_gw=last_gw, wc_gws=wc_gws,
                            wc_h=args.wc_h, step_h=args.step_h, combos=combos,
                            progress=lambda m: logger.info(m))

    # --- report ----------------------------------------------------------------------
    names = {e["id"]: e["web_name"] for e in bootstrap["elements"]}
    teams = {tm["id"]: tm["short_name"] for tm in bootstrap["teams"]}
    flags = ce.availability_flags(bootstrap, squad_ids)
    xp_t, _ = ctx.maps(t)
    model_xi, _, _, _ = ce.best_lineup(squad_ids, ctx.pos_of, xp_t)
    starters = set(model_xi) | {state.squad.players[i].element_id for i in state.squad.lineup}
    calendar = ce.fixture_calendar(fixtures, sorted(teams), list(range(t, 39)))
    report = ce.format_report(ctx, res, remaining, calendar, names, teams, flags, starters,
                              last_gw)
    if args.out:
        out = {k: v for k, v in res.items() if not k.startswith("_")}
        out.update({
            "model_dir": str(args.model_dir), "pplay": str(args.pplay_dir) if play_model else None,
            "squad": squad_ids, "data_health": health,
            "remaining": [{"chip": rc.chip, "half": rc.window.half, "gws": list(rc.gws)}
                          for rc in remaining],
            "calibration": ce.CALIB, "calendar": calendar,
            "flags": flags, "starters": sorted(starters),
            "horizon": preds.to_dict(orient="records"),
        })
        Path(args.out).write_text(json.dumps(out, indent=1, default=lambda o: o.item()
                                             if hasattr(o, "item") else str(o)),
                                  encoding="utf-8")
    print(report)
    if args.out:
        print(f"saved {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="eval", choices=["eval", "train-pplay"])
    ap.add_argument("--team-id", type=int, default=None)
    ap.add_argument("--entry-dir", type=Path, default=None,
                    help="offline entry JSONs (entry/history/transfers/picks_gw{n}.json)")
    ap.add_argument("--bootstrap", type=Path, default=None,
                    help="bootstrap-static JSON file (default: FPL API, read-only)")
    ap.add_argument("--fixtures", type=Path, default=None,
                    help="fixtures JSON file (default: FPL API, read-only)")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--season", default=CURRENT_SEASON)
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="point model (default: scripts/gameweek.py DEFAULT_MODEL_DIR)")
    ap.add_argument("--allow-leaky-model", action="store_true",
                    help="evaluate even if the model uses the same-GW fpl_xp (leak)")
    ap.add_argument("--pplay-dir", type=Path, default=None,
                    help="LightGBM P(play) model dir (default: calibrated lookup)")
    ap.add_argument("--chips", default="tc,bb,fh,wc")
    ap.add_argument("--last-gw", type=int, default=None,
                    help="last candidate GW (default: 19 up to GW19, else 38)")
    ap.add_argument("--wc-gws", default=None, help="WC candidate GWs, e.g. 8,11 "
                    "(default: every playable GW up to --last-gw)")
    ap.add_argument("--wc-h", type=int, default=6, help="WC evaluation horizon (GWs)")
    ap.add_argument("--step-h", type=int, default=3,
                    help="base planner horizon (= gameweek.py --horizon) and FH horizon")
    ap.add_argument("--wc-bb", default="", help="WC:BB combos, e.g. 11:12,11:13")
    ap.add_argument("--min-chance", type=int, default=75)
    ap.add_argument("--discount", type=float, default=None)
    ap.add_argument("--hit-margin", type=float, default=None)
    ap.add_argument("--max-hits", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=60.0, help="seconds per MILP solve")
    ap.add_argument("--n-sims", type=int, default=4000)
    ap.add_argument("--features", type=Path, default=None, help="train-pplay: features parquet")
    ap.add_argument("--out", type=Path, default=None, help="save results as JSON")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):  # player names on a cp1252 console/pipe
        sys.stdout.reconfigure(errors="replace")
    if args.cmd == "train-pplay":
        return cmd_train_pplay(args)
    if args.model_dir is None:
        args.model_dir = gameweek_default_model_dir()
    return cmd_eval(args)


if __name__ == "__main__":
    sys.exit(main())
