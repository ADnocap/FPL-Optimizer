#!/usr/bin/env python3
"""Season replay of the multi-GW (receding-horizon) planner vs the single-GW MILP.

Two steps (predictions are cached so the replay variants reuse them):

1. ``predict`` — train an evaluation model strictly on seasons before the
   target season (leak-fixed feature set: vaastav's same-GW ``fpl_xp`` is
   replaced by its previous-GW value ``fpl_xp_lag``), then for EVERY GW t of
   the target season build horizon predictions for GWs t..t+H-1 as of the
   GW t deadline (``fpl_optimizer.prediction.horizon``) and save them::

     python scripts/backtest_horizon.py predict --season 2024-25 \
         --features features_all.parquet --out preds_2024-25.parquet

   ``--features`` is a FeaturePipeline output parquet (all seasons); build
   one with ``FeaturePipeline(...).build().to_parquet(...)`` if needed.

2. ``replay`` — replay the season through the rules engine (auto-subs,
   captain failover, FT banking, hits, chips) for each variant::

     python scripts/backtest_horizon.py replay --season 2024-25 \
         --preds preds_2024-25.parquet --variants single1 single h4 h3:m1

   Variant syntax: ``single`` (current optimize_transfers, unconstrained;
   ``single:m2`` = hit margin 2, the live default), ``single1`` (max 1
   transfer/GW), ``hN[:dD][:mM][:fF][:xK][:bfixed][:cSPEC]``
   (horizon N, discount D, hit margin M, FT terminal value F, at most K hits
   per GW, legacy fixed bench weights, ``wC`` = cost C per Wildcard-week move,
   chip plan SPEC like ``tc7-wc11-bb12``;
   ``m100`` and ``x0`` both mean "never take a hit").

GW1 squad: ``select_squad`` on the GW1 predictions for every variant (so the
comparison isolates the transfer policy), 0 FTs during GW1 -> 1 FT for GW2
(FPL rule).  ``--start-picks picks.json --start-gw N`` instead replays from a
real FPL squad (public picks JSON of GW N-1).

Caveats: prices assumed flat over the horizon; the final fixtures.csv is used
for future fixtures (later rescheduling is known to the backtest); no
availability flags historically.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logger = logging.getLogger("backtest_horizon")

ALL_SEASONS = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
               "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
PARAMS_FAST = {
    "objective": "regression", "metric": "mae", "num_leaves": 127,
    "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 5, "min_child_samples": 10, "n_estimators": 800,
    "verbose": -1, "lambda_l1": 1.0, "lambda_l2": 1.0, "seed": 42, "num_threads": 4,
}
PARAMS_FULL = {**PARAMS_FAST, "learning_rate": 0.01, "n_estimators": 2000}


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------


def fix_xp_leak(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Replace the leaky same-GW ``fpl_xp`` with ``fpl_xp_lag``.

    vaastav's xP is FPL's ep_this scraped AFTER the GW (recomputed with the
    GW's own points in form).  The previous GW's value is point-in-time safe;
    live, the pre-deadline snapshot's ep_this is exactly that quantity.
    No-op when the frame already has ``fpl_xp_lag`` or lacks ``fpl_xp``.
    """
    if "fpl_xp" not in df.columns or "fpl_xp_lag" in df.columns:
        return df
    df = df.copy()
    xp = df["fpl_xp"].copy()
    gw_abs = xp.fillna(0).abs().groupby([df["season"], df["GW"]]).transform("sum")
    xp[gw_abs == 0] = np.nan  # whole GW missing -> NaN, not a real 0
    df["_xp"] = xp
    order = df.sort_values(["season", "code", "GW"]).index
    lag = df.loc[order].groupby(["season", "code"])["_xp"].shift(1)
    df["fpl_xp_lag"] = lag.reindex(df.index)
    snap = data_dir / "live" / "2026-27" / "snapshots"
    live: dict[tuple[int, int], float] = {}
    for p in snap.glob("gw*_bootstrap.json"):
        n = int(p.name[2:].split("_")[0])
        for e in json.loads(p.read_text(encoding="utf-8"))["elements"]:
            v = e.get("ep_this")
            live[(e["id"], n)] = float(v) if v is not None else np.nan
    m = df["season"] == "2026-27"
    if m.any():
        df.loc[m, "fpl_xp_lag"] = [
            live.get((int(el), int(g)), np.nan) if pd.notna(el) else np.nan
            for el, g in zip(df.loc[m, "element"], df.loc[m, "GW"])
        ]
    return df.drop(columns=["_xp", "fpl_xp"])


def cmd_predict(args: argparse.Namespace) -> None:
    from fpl_optimizer.prediction.horizon import FixtureContext, predict_horizon
    from fpl_optimizer.prediction.minutes import load_predictor
    from fpl_optimizer.prediction.model import _NON_FEATURE_COLS, PointPredictor

    target = args.season
    t_idx = ALL_SEASONS.index(target)
    train_seasons = ALL_SEASONS[:t_idx]
    val_season = train_seasons[-1]
    t0 = time.time()
    import gc

    df = pd.read_parquet(args.features, filters=[("season", "in", train_seasons + [target])])
    gmax = df.groupby(["season", "GW"])["target"].transform("max")
    df = df[(gmax > 0) & df["position"].notna()].reset_index(drop=True)
    if not args.keep_leaky_xp:
        df = fix_xp_leak(df, args.data_dir)
    # float32 halves memory (LightGBM bins to float32 internally anyway)
    f64 = [c for c in df.columns if df[c].dtype == "float64"]
    df[f64] = df[f64].astype("float32")
    gc.collect()
    feats = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    print(f"features: {len(df)} rows, {len(feats)} feature cols ({time.time() - t0:.0f}s)",
          flush=True)

    if args.model:  # reuse an evaluation model saved by an earlier run (any kind)
        model = load_predictor(args.model)
        print(f"loaded {model.kind} model {args.model}", flush=True)
    elif args.recipe:  # the production recipe (scripts/train_predictor.py)
        import train_predictor as tp
        from fpl_optimizer.prediction.minutes import OOF_KEY, expanding_minutes_oof

        recipe = tp.resolve_recipe(None if args.recipe == "default" else args.recipe)
        df, _ = tp.apply_exclusions(df, recipe)
        tr_df, va_df = tp._split_val(df[df["season"].isin(train_seasons)], train_seasons)
        oof = None
        if recipe["kind"] == "minutes_blend":
            hist = df[df["season"].isin(train_seasons)]
            oof = pd.concat([hist[OOF_KEY], expanding_minutes_oof(hist, recipe["mm_rounds"])],
                            axis=1)
        model = tp.fit_model(recipe, tr_df, va_df, oof)
        del tr_df, va_df, oof
        gc.collect()
        print(f"trained {recipe['kind']} recipe on {train_seasons[0]}..{val_season} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if args.save_model:
            model.save(args.save_model)
    else:
        tr = df["season"].isin(train_seasons)
        last = int(df.loc[df["season"] == val_season, "GW"].max())
        va = tr & (df["season"] == val_season) & (df["GW"] > last - 8)
        cols = feats + ["position", "target"]
        model = PointPredictor(params=PARAMS_FULL if args.full else PARAMS_FAST,
                               early_stopping_rounds=50)
        model.train(df.loc[tr & ~va, cols], df.loc[va, cols])
        print(f"trained on {train_seasons[0]}..{val_season} ({time.time() - t0:.0f}s)",
              flush=True)
        if args.save_model:
            model.save(args.save_model)

    season_df = df[df["season"] == target].copy()
    del df
    gc.collect()
    ctx = FixtureContext.from_raw_dir(args.data_dir / "raw" / target)
    max_gw = args.max_gw or int(season_df["GW"].max())
    out = []
    for t in range(1, max_gw + 1):
        if not (season_df["GW"] == t).any():
            continue
        hp = predict_horizon(model, season_df, ctx, t, args.horizon, args.dgw_mode,
                             args.market)
        hp.insert(0, "t", t)
        out.append(hp)
    preds = pd.concat(out, ignore_index=True)
    # attach targets (actual points) where the GW has been played
    tgt = season_df.groupby(["element", "GW"], as_index=False)["target"].sum()
    preds = preds.merge(tgt, on=["element", "GW"], how="left")
    preds.to_parquet(args.out)
    print(f"saved {len(preds)} rows -> {args.out} ({time.time() - t0:.0f}s)", flush=True)
    _report_horizon_quality(preds)


def _report_horizon_quality(preds: pd.DataFrame) -> None:
    """Per-offset ranking quality vs actual points (+ persistence baseline)."""
    rows = []
    base = preds[preds["k"] == 0][["t", "element", "pred"]].rename(columns={"pred": "p0"})
    for k in sorted(preds["k"].unique()):
        d = preds[(preds["k"] == k) & preds["target"].notna()]
        d = d.merge(base, on=["t", "element"], how="left")
        sp, sp_p, bias = [], [], []
        for _, g in d.groupby("GW"):
            if len(g) < 50:
                continue
            sp.append(g["pred"].corr(g["target"], method="spearman"))
            sp_p.append(g["p0"].fillna(0).corr(g["target"], method="spearman"))
            bias.append(float((g["pred"] - g["target"]).mean()))
        rows.append({"k": int(k), "n_gw": len(sp), "spearman": np.mean(sp),
                     "spearman_persist": np.mean(sp_p), "bias": np.mean(bias)})
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def parse_variant(spec: str) -> dict:
    parts = spec.split(":")
    head = parts[0]
    v = {"name": spec, "kind": "single" if head.startswith("single") else "horizon"}
    if head == "single":
        v["max_transfers"] = None
    elif head.startswith("single"):
        v["max_transfers"] = int(head[len("single"):])
    elif head.startswith("h"):
        v["horizon"] = int(head[1:])
    else:
        raise ValueError(spec)
    v.update(discount=0.85, hit_margin=0.0, ft_value=0.0, bench_mode="dnp", chips="",
             max_hits=None, wc_move_cost=0.0)
    for p in parts[1:]:
        if p.startswith("d"):
            v["discount"] = float(p[1:])
        elif p.startswith("m"):
            v["hit_margin"] = float(p[1:])
        elif p.startswith("w"):
            v["wc_move_cost"] = float(p[1:])
        elif p.startswith("f"):
            v["ft_value"] = float(p[1:])
        elif p == "bfixed":
            v["bench_mode"] = "fixed"
        elif p.startswith("x"):
            v["max_hits"] = int(p[1:])
        elif p.startswith("c"):
            v["chips"] = p[1:].replace("-", ",")
        else:
            raise ValueError(f"bad variant part {p!r} in {spec!r}")
    return v


class Replayer:
    def __init__(self, season: str, preds: pd.DataFrame, data_dir: Path,
                 init_budget: int = 1000) -> None:
        self.init_budget = init_budget  # GW1 squad spend (rest stays in the bank)
        from fpl_optimizer.data.loader import SeasonDataLoader
        from fpl_optimizer.engine.engine import FPLGameEngine

        self.season = season
        self.loader = SeasonDataLoader(season, data_dir / "raw")
        self.engine = FPLGameEngine(self.loader)
        self.preds = preds
        self.by_t = {t: g for t, g in preds.groupby("t")}
        self.team_map = self.loader._team_map
        self.gws = sorted(int(t) for t in self.by_t)

    def price(self, eid: int, gw: int) -> int:
        p = self.loader.get_player_price(eid, gw)
        if p <= 0:
            p = self.loader.get_player_price(eid, max(1, gw - 1))
        return p

    def single_candidates(self, t: int):
        from fpl_optimizer.optimizer.types import build_candidate_pool

        g = self.by_t[t]
        k0 = g[g["k"] == 0]
        return build_candidate_pool(self.loader, t, dict(zip(k0["element"], k0["pred"])))

    def horizon_candidates(self, t: int, horizon: int, squad_ids: set[int]):
        from fpl_optimizer.optimizer.horizon_optimizer import HorizonCandidate

        g = self.by_t[t]
        g = g[g["k"] < horizon]
        xp: dict[int, list[float]] = {}
        pp: dict[int, list[float]] = {}
        for eid, k, pred, p_play in g[["element", "k", "pred", "p_play"]].itertuples(index=False):
            xp.setdefault(int(eid), [0.0] * horizon)[int(k)] = float(pred)
            pp.setdefault(int(eid), [0.0] * horizon)[int(k)] = float(p_play)
        cands = []
        for eid in set(xp) | squad_ids:
            pos = self.loader.get_player_position(eid)
            team = self.team_map.get(eid)
            price = self.price(eid, t)
            if pos is None or team is None or price <= 0:
                continue
            cands.append(HorizonCandidate(
                eid, pos, price, team,
                tuple(xp.get(eid, [0.0] * horizon)),
                tuple(pp.get(eid, [0.0] * horizon)),
            ))
        return cands

    def initial_state(self):
        from fpl_optimizer.engine.state import GameState, PlayerSlot, Squad
        from fpl_optimizer.optimizer.squad_selection import select_squad
        from fpl_optimizer.utils.constants import GW1_FREE_TRANSFERS, STARTING_BUDGET

        first = self.gws[0]
        init = select_squad(self.single_candidates(first), budget=self.init_budget)
        players = [PlayerSlot(e, self.loader.get_player_position(e),
                              self.price(e, first), self.price(e, first))
                   for e in init.squad_element_ids]
        idx = {p.element_id: i for i, p in enumerate(players)}
        squad = Squad(players, [idx[e] for e in init.lineup_element_ids],
                      [idx[e] for e in init.bench_element_ids],
                      idx[init.captain_id], idx[init.vice_captain_id])
        state = GameState(squad=squad, bank=STARTING_BUDGET - init.total_cost,
                          free_transfers=GW1_FREE_TRANSFERS, current_gw=first)
        return state, init

    def run(self, v: dict, start_state=None, start_gw: int | None = None) -> dict:
        from fpl_optimizer.engine.state import EngineAction
        from fpl_optimizer.optimizer.horizon_optimizer import (
            HorizonConfig,
            optimize_horizon,
            parse_chip_plan,
        )
        from fpl_optimizer.optimizer.transfer_optimizer import optimize_transfers
        from fpl_optimizer.optimizer.types import to_engine_action
        from fpl_optimizer.utils.constants import TOTAL_GAMEWEEKS

        chip_plan = parse_chip_plan(v.get("chips"))
        rows = []
        if start_state is None:
            state, init = self.initial_state()
            gws = self.gws
            first_action = to_engine_action(init)
            first_action.transfers_in, first_action.transfers_out = [], []
        else:
            state = start_state
            gws = [g for g in self.gws if g >= start_gw]
            first_action = None
        n_fail = 0
        for gw in gws:
            t0 = time.time()
            info = {}
            if first_action is not None and gw == self.gws[0]:
                action = first_action
            elif v["kind"] == "single":
                chip = chip_plan.get(gw)
                if chip and not state.chips.is_available(chip, gw):
                    chip = None
                res = optimize_transfers(state, self.single_candidates(gw), chip=chip,
                                         max_transfers=v["max_transfers"],
                                         hit_margin=v["hit_margin"],
                                         squad_teams={
                                             sp.element_id: self.loader.get_player_team(sp.element_id)
                                             for sp in state.squad.players
                                             if self.loader.get_player_team(sp.element_id) is not None
                                         })
                action = to_engine_action(res)
            else:
                h = min(v["horizon"], TOTAL_GAMEWEEKS - gw + 1)
                squad_ids = {p.element_id for p in state.squad.players}
                cfg = HorizonConfig(discount=v["discount"], hit_margin=v["hit_margin"],
                                    ft_value=v["ft_value"], bench_mode=v["bench_mode"],
                                    max_hits_per_gw=v["max_hits"],
                                    wc_move_cost=v["wc_move_cost"],
                                    time_limit=v.get("time_limit", 60))
                res = optimize_horizon(state, self.horizon_candidates(gw, h, squad_ids),
                                       list(range(gw, gw + h)), chip_plan, cfg)
                action = to_engine_action(res.first)
                info = {"status": res.status, "n_cand": res.n_candidates,
                        "bench_w1": res.bench_weights[0][1][0]}
            solve_s = time.time() - t0
            ft_before = state.free_transfers
            try:
                state, step = self.engine.step(state, action)
            except ValueError as exc:
                n_fail += 1
                logger.warning("%s GW%d engine rejected action (%s) -> no-op", v["name"], gw, exc)
                state, step = self.engine.step(state, EngineAction())
                action = EngineAction()
            rows.append({
                "variant": v["name"], "gw": gw, "net": step.net_points,
                "gross": step.gw_points, "hit": step.hit_cost,
                "n_transfers": len(action.transfers_out), "ft_before": ft_before,
                "chip": action.chip, "auto_subs": len(step.auto_subs),
                "bench_pts": step.bench_points, "captain_pts": step.captain_points,
                "failover": step.captain_failover, "solve_s": round(solve_s, 2), **info,
            })
        df = pd.DataFrame(rows)
        return {
            "variant": v["name"], "net": int(df["net"].sum()),
            "gross": int(df["gross"].sum()), "hits": int(df["hit"].sum()),
            "transfers": int(df["n_transfers"].sum()),
            "auto_subs": int(df["auto_subs"].sum()),
            "sub_gws": int((df["auto_subs"] > 0).sum()),
            "engine_rejects": n_fail, "max_solve_s": float(df["solve_s"].max()),
            "mean_solve_s": float(df["solve_s"].mean()),
            "rows": df,
        }


def _state_from_picks(picks_path: Path, replayer: Replayer, start_gw: int):
    from fpl_optimizer.engine.state import GameState, PlayerSlot, Squad

    data = json.loads(Path(picks_path).read_text(encoding="utf-8"))
    picks = sorted(data["picks"], key=lambda p: p["position"])
    prev_gw = start_gw - 1
    players, lineup, bench, cap, vice = [], [], [], 0, 1
    for i, p in enumerate(picks):
        eid = p["element"]
        price = replayer.price(eid, prev_gw)
        players.append(PlayerSlot(eid, replayer.loader.get_player_position(eid), price, price))
        (lineup if p["position"] <= 11 else bench).append(i)
        cap = i if p.get("is_captain") else cap
        vice = i if p.get("is_vice_captain") else vice
    bank = int(data.get("entry_history", {}).get("bank", 0))
    return GameState(squad=Squad(players, lineup, bench, cap, vice), bank=bank,
                     free_transfers=1, current_gw=start_gw)


def cmd_replay(args: argparse.Namespace) -> None:
    preds = pd.read_parquet(args.preds)
    rep = Replayer(args.season, preds, args.data_dir, init_budget=args.init_budget)
    summaries = []
    all_rows = []
    for spec in args.variants:
        v = parse_variant(spec)
        v["time_limit"] = args.time_limit
        start_state = None
        if args.start_picks:
            start_state = _state_from_picks(args.start_picks, rep, args.start_gw)
        t0 = time.time()
        s = rep.run(v, start_state, args.start_gw)
        all_rows.append(s.pop("rows"))
        s["secs"] = round(time.time() - t0)
        summaries.append(s)
        print(json.dumps(s), flush=True)
    out = pd.DataFrame(summaries)
    print(out.to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)
        pd.concat(all_rows).to_csv(str(args.out).replace(".csv", "_gw.csv"), index=False)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--season", required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--dgw-mode", default="blend", choices=["row", "sum", "blend"])
    p.add_argument("--market", default="keep", choices=["keep", "drop"],
                   help="drop = no odds/props at any horizon GW (consistent footing)")
    p.add_argument("--model", type=Path, default=None,
                   help="load this model dir (either kind) instead of training")
    p.add_argument("--recipe", default=None,
                   help="train with scripts/train_predictor.py's RECIPE: 'default' or a "
                        "JSON override (else the legacy L2 PARAMS_FAST/--full)")
    p.add_argument("--max-gw", type=int, default=None)
    p.add_argument("--full", action="store_true", help="full (slow) LightGBM params")
    p.add_argument("--keep-leaky-xp", action="store_true")
    p.add_argument("--save-model", type=Path, default=None,
                   help="also save the evaluation model (loadable with load_predictor)")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    r = sub.add_parser("replay")
    r.add_argument("--season", required=True)
    r.add_argument("--preds", type=Path, required=True)
    r.add_argument("--variants", nargs="+", default=["single1", "single", "h4"])
    r.add_argument("--time-limit", type=float, default=60.0)
    r.add_argument("--init-budget", type=int, default=1000,
                   help="GW1 squad spend in tenths (vary to get other starting squads)")
    r.add_argument("--start-picks", type=Path, default=None)
    r.add_argument("--start-gw", type=int, default=None)
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = ap.parse_args()
    {"predict": cmd_predict, "replay": cmd_replay}[args.cmd](args)


if __name__ == "__main__":
    main()
