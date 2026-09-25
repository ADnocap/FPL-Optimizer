#!/usr/bin/env python3
"""Train the production point predictor for the live season.

The committed, reproducible recipe for the model of record. Every component
is switchable through RECIPE (override with ``--recipe JSON|path.json``;
``params`` is merged key by key, a ``null`` value deletes a key):

- ``kind``: ``"minutes_blend"`` (default) = minutes model + 0.5 factorized
  P60*E[pts|60+] + P(1-59)*E[pts|1-59] + 0.5 stacked points model
  (fpl_optimizer.prediction.minutes; +0.015 per-GW Spearman on 2024-25 and
  2025-26, 38/38 GWs each) | ``"point"`` = one PointPredictor.
- ``params``: LightGBM params of the point models. Default huber (alpha 0.9)
  + min_child_samples 50 (+0.012 per-GW Spearman, 38/38 GWs on both holdouts).
- ``calibrate``: monotone quadratic raw->points map fitted on the
  early-stopping val split, applied to the FINAL (blended) output. Huber
  predicts near the conditional median; the MILP needs the points scale.
- ``exclude_features``: features never served at a live deadline, plus the
  neutral-but-fragile h2h odds (fpl_optimizer.prediction.feature_sets.
  EXCLUDED_FEATURES) — train what
  you serve.
- ``include_current_season``: PROD also trains on the current season's
  completed GWs (never used as the val season: early stopping stays on the
  last 8 GWs of the last COMPLETE season).
- ``refit``: after early stopping, retrain on train+val at the best
  iteration (neutral in tests; off by default).

Phases (features built once, or loaded from ``--features-cache``):
1. EVAL  — train <= 2024-25 (val = last 8 GWs of 2024-25), holdout 2025-26;
           ``--second-fold`` also trains <= 2023-24 -> holdout 2024-25.
           Reports MAE, per-GW Pearson/Spearman, top-10 mean actual, captain
           points (argmax among value >= 5.0m), stratified bins, and the
           holdout split by GWs with / without real xP (fpl_xp_lag).
2. LIVE  — trained on complete seasons only, evaluated on the current
           season's completed GWs (report only; few GWs = noisy).
3. PROD  — the recipe above; saved atomically (staging dir, ``.prev``
           rollback) with metadata (kind, recipe, train seasons, current-
           season GWs, calibration) and serving_reference.json.

``--features-cache PATH`` loads the FeaturePipeline parquet if it exists and
else builds and saves it (~10 min saved per run). REBUILD IT
(``--rebuild-cache``) after any pipeline change (the cache records
FEATURE_PIPELINE_VERSION and is refused on mismatch) and after new GWs are
played (it is a snapshot of the season files).

Usage:
    python scripts/train_predictor.py [--out models/prod_2026-27]
        [--features-cache features.parquet] [--recipe '{"refit": true}']
        [--no-eval] [--no-live] [--second-fold]
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fpl_optimizer.prediction.feature_sets import EXCLUDED_FEATURES  # noqa: E402
from fpl_optimizer.prediction.minutes import MM_ROUNDS  # noqa: E402
from fpl_optimizer.utils.constants import CURRENT_SEASON  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

COMPLETE_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]
EVAL_TRAIN_SEASONS = COMPLETE_SEASONS[:-1]          # <= 2024-25
EVAL_HOLDOUT = COMPLETE_SEASONS[-1]                 # 2025-26
FOLD2_TRAIN_SEASONS = COMPLETE_SEASONS[:-2]         # <= 2023-24
FOLD2_HOLDOUT = COMPLETE_SEASONS[-2]                # 2024-25
ALL_SEASONS = COMPLETE_SEASONS + [CURRENT_SEASON]
# Backward-compatible name: complete seasons only (the in-progress season is
# added through RECIPE["include_current_season"], never here — _split_val
# would make every one of its rows validation).
PROD_SEASONS = COMPLETE_SEASONS

PARAMS = {
    "objective": "huber",
    "alpha": 0.9,
    "metric": "mae",
    "num_leaves": 127,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "n_estimators": 2000,
    "verbose": -1,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "seed": 42,
    "num_threads": 4,
}

RECIPE = {
    "kind": "minutes_blend",           # "point" | "minutes_blend"
    "params": PARAMS,                  # LightGBM params of the point model(s)
    "early_stopping_rounds": 50,
    "calibrate": True,                 # quadratic raw->points map on the val split
    "exclude_features": EXCLUDED_FEATURES,
    "include_current_season": True,    # PROD trains on the current season's completed GWs
    "refit": False,                    # retrain on train+val at the best iteration
    "w_fact": 0.5,                     # minutes_blend: weight of the factorized part
    "mm_rounds": MM_ROUNDS,            # minutes_blend: minutes-model boosting rounds
}


def resolve_recipe(override: str | None) -> dict:
    """RECIPE with a JSON override (inline or a file path) merged in."""
    recipe = copy.deepcopy(RECIPE)
    if not override:
        return recipe
    p = Path(override)
    text = p.read_text(encoding="utf-8") if p.suffix == ".json" and p.exists() else override
    over = json.loads(text)
    unknown = set(over) - set(recipe)
    if unknown:
        raise SystemExit(f"unknown recipe keys: {sorted(unknown)}")
    for key, val in over.items():
        if key == "params":
            for pk, pv in val.items():
                if pv is None:
                    recipe["params"].pop(pk, None)
                else:
                    recipe["params"][pk] = pv
        else:
            recipe[key] = val
    if recipe["kind"] not in ("point", "minutes_blend"):
        raise SystemExit(f"recipe kind must be 'point' or 'minutes_blend', got {recipe['kind']!r}")
    return recipe


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _split_val(df: pd.DataFrame, seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/val split: val = last 8 GWs of the last season in *seasons*."""
    last = seasons[-1]
    if last == CURRENT_SEASON:
        raise ValueError(
            f"{CURRENT_SEASON} is in progress: it must never be the val season "
            "(all its rows would become validation) — use include_current_season"
        )
    subset = df[df["season"].isin(seasons)]
    last_rows = subset[subset["season"] == last]
    max_gw = int(last_rows["GW"].max())
    val_mask = (subset["season"] == last) & (subset["GW"] > max_gw - 8)
    return subset[~val_mask].copy(), subset[val_mask].copy()


def apply_exclusions(df: pd.DataFrame, recipe: dict) -> tuple[pd.DataFrame, list[str]]:
    """Drop the recipe's excluded features (models train on every other
    non-meta column, so an absent column is an unused feature)."""
    excl = [c for c in recipe["exclude_features"] if c in df.columns]
    return df.drop(columns=excl), excl


def prod_split(df: pd.DataFrame, include_current: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PROD rows: complete seasons (val = last 8 GWs of the last one) plus,
    when *include_current*, every completed current-season GW in TRAIN."""
    train_df, val_df = _split_val(df, COMPLETE_SEASONS)
    if include_current:
        cur = df[df["season"] == CURRENT_SEASON]
        train_df = pd.concat([train_df, cur])
    return train_df, val_df


def load_features(args) -> pd.DataFrame:
    from fpl_optimizer.prediction.feature_pipeline import (
        FeaturePipeline,
        StaleFeatureCache,
        load_or_build_feature_cache,
    )
    from fpl_optimizer.prediction.id_resolver import IDResolver

    t0 = time.time()
    if args.features_cache:
        try:
            df = load_or_build_feature_cache(
                args.features_cache, args.data_dir, ALL_SEASONS, rebuild=args.rebuild_cache,
            )
        except StaleFeatureCache as exc:
            raise SystemExit(f"Stale feature cache: {exc}") from None
        print(f"Features {'rebuilt' if args.rebuild_cache else 'from cache'} "
              f"{args.features_cache} (built {df.attrs.get('built_utc')})")
    else:
        print(f"Building features for {len(ALL_SEASONS)} seasons...")
        df = FeaturePipeline(args.data_dir, IDResolver(args.data_dir), ALL_SEASONS).build()
    print(f"Features: {len(df)} rows x {len(df.columns)} cols in {time.time() - t0:.0f}s")

    # Drop synthetic/unplayed gameweeks: fpl_live's live rebuild writes
    # upcoming-GW rows with all stats zeroed. A real GW always has at least
    # one player with points (appearance minimum), so an all-zero-target
    # (season, GW) group can only be synthetic.
    gw_max_target = df.groupby(["season", "GW"])["target"].transform("max")
    synthetic = ~(gw_max_target > 0)  # catches all-zero AND all-NaN groups
    if synthetic.any():
        dropped = df.loc[synthetic, ["season", "GW"]].drop_duplicates()
        print(f"Dropping {int(synthetic.sum())} synthetic/unplayed rows: "
              + ", ".join(f"{s} GW{int(g)}" for s, g in dropped.itertuples(index=False)))
        df = df[~synthetic]
    no_pos = df["position"].isna()
    if no_pos.any():  # never trained on nor predicted (no position model)
        print(f"Dropping {int(no_pos.sum())} rows without a position")
        df = df[~no_pos]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def fit_model(recipe: dict, train_df: pd.DataFrame, val_df: pd.DataFrame,
              minutes_oof: pd.DataFrame | None = None):
    """Train one model of the recipe's kind (+ calibration / refit)."""
    from fpl_optimizer.prediction.minutes import MinutesBlendPredictor
    from fpl_optimizer.prediction.model import PointPredictor

    t0 = time.time()
    if recipe["kind"] == "minutes_blend":
        model = MinutesBlendPredictor(
            params=recipe["params"], w_fact=recipe["w_fact"],
            early_stopping_rounds=recipe["early_stopping_rounds"],
            mm_rounds=recipe["mm_rounds"],
        )
        model.train(train_df, val_df, minutes_oof=minutes_oof)
    else:
        model = PointPredictor(params=recipe["params"],
                               early_stopping_rounds=recipe["early_stopping_rounds"])
        model.train(train_df, val_df)
    best = model.best_iterations()
    print(f"  trained {recipe['kind']} in {time.time() - t0:.0f}s; best iterations {best}")
    if recipe["calibrate"]:
        cal = model.fit_calibration(val_df)
        print(f"  calibration (on {cal['n']} val rows): y = {cal['coef'][2]:+.3f} "
              f"{cal['coef'][1]:+.3f}*x {cal['coef'][0]:+.4f}*x^2")
    if recipe["refit"]:
        t1 = time.time()
        if recipe["kind"] == "minutes_blend":
            model.refit(train_df, val_df, minutes_oof=minutes_oof)
        else:
            model.refit(train_df, val_df)
        print(f"  refit on train+val at the best iterations in {time.time() - t1:.0f}s")
    return model


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

_MIN_GW_ROWS = 20


def per_gw_metrics(holdout: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    """One row per (season, GW): Spearman, Pearson, MAE, top-10, captain, xP coverage."""
    res = holdout[["season", "GW", "target", "value"]].copy()
    res["pred"] = preds
    res["xp_cov"] = (holdout["fpl_xp_lag"].notna().to_numpy()
                     if "fpl_xp_lag" in holdout.columns else False)
    rows = []
    for (season, gw), g in res.groupby(["season", "GW"]):
        g = g[g["target"].notna()]
        if len(g) < _MIN_GW_ROWS:
            continue
        y, p = g["target"].to_numpy(float), g["pred"].to_numpy(float)
        order = np.argsort(-p)
        elig = (g["value"] >= 50).to_numpy()  # captain candidates: >= 5.0m
        cap = float(y[elig][np.argmax(p[elig])]) if elig.any() else float("nan")
        rows.append({
            "season": season, "GW": int(gw), "n": len(g),
            "spearman": float(pd.Series(p).corr(pd.Series(y), method="spearman")),
            "pearson": float(np.corrcoef(p, y)[0, 1]),
            "mae": float(np.mean(np.abs(p - y))),
            "bias": float(np.mean(p - y)),
            "top10": float(np.mean(y[order[:10]])),
            "captain": cap,
            "xp_cov": float(g["xp_cov"].mean()),
        })
    return pd.DataFrame(rows)


def _gw_ranges(gws) -> str:
    """[1, 2, 3, 7, 9, 10] -> '1-3,7,9-10'."""
    out, run = [], []
    for g in sorted(int(x) for x in gws):
        if run and g == run[-1] + 1:
            run.append(g)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [g]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


def _summ(pg: pd.DataFrame) -> dict:
    out = {"n_gw": int(len(pg))}
    for m in ["spearman", "pearson", "mae", "bias", "top10", "captain"]:
        out[m] = float(pg[m].mean()) if len(pg) else float("nan")
    return out


def evaluate(model, holdout: pd.DataFrame, label: str) -> dict:
    """Print and return the holdout report (see module docstring)."""
    preds = model.predict(holdout)
    actual = holdout["target"].to_numpy(float)
    valid = ~np.isnan(actual)
    mae = float(np.mean(np.abs(preds[valid] - actual[valid])))
    rmse = float(np.sqrt(np.mean((preds[valid] - actual[valid]) ** 2)))
    corr = float(np.corrcoef(preds[valid], actual[valid])[0, 1])
    pg = per_gw_metrics(holdout, preds)
    s = _summ(pg)
    print(f"\n=== {label} ===")
    print(f"  rows {int(valid.sum())} | MAE {mae:.4f}  RMSE {rmse:.4f}  "
          f"overall Pearson {corr:.4f} | mean pred {preds[valid].mean():.3f} "
          f"vs actual {actual[valid].mean():.3f}")
    print(f"  per-GW ({s['n_gw']} GWs): Spearman {s['spearman']:.4f}  "
          f"Pearson {s['pearson']:.4f}  top-10 {s['top10']:.3f}  captain {s['captain']:.2f}")
    with_xp = pg[pg["xp_cov"] >= 0.5]
    without_xp = pg[pg["xp_cov"] < 0.5]
    split = {"with_xp": _summ(with_xp), "without_xp": _summ(without_xp)}
    for name, key, part in (("with real xP", "with_xp", with_xp),
                            ("without xP", "without_xp", without_xp)):
        split[key]["gws"] = part["GW"].tolist()
        if len(part):
            ps = split[key]
            print(f"    {name:<12} ({ps['n_gw']:>2} GWs: {_gw_ranges(part['GW'])}): "
                  f"Spearman {ps['spearman']:.4f}  Pearson {ps['pearson']:.4f}  "
                  f"MAE {ps['mae']:.4f}  top-10 {ps['top10']:.3f}  captain {ps['captain']:.2f}")
    by_pos = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        m = (holdout["position"] == pos).to_numpy() & valid
        if m.sum():
            by_pos[pos] = float(np.mean(np.abs(preds[m] - actual[m])))
    print("  MAE by position: " + "  ".join(f"{k} {v:.4f}" for k, v in by_pos.items()))
    out = {"mae": mae, "rmse": rmse, "corr": corr, "gw_corr": s["pearson"],
           "gw_spearman": s["spearman"], "top10": s["top10"], "captain": s["captain"],
           "n_gw": s["n_gw"], "xp_split": split, "mae_by_position": by_pos,
           "per_gw": pg.to_dict(orient="records")}
    try:
        from fpl_optimizer.prediction.stratified_metrics import (
            format_report, stratified_metrics,
        )

        minutes = holdout["min_total"].to_numpy(float)[valid] if "min_total" in holdout else None
        strat = stratified_metrics(actual[valid], preds[valid], minutes)
        print(format_report(strat, label=f"stratified: {label}"))
        out["stratified"] = strat
    except Exception as exc:  # metrics module optional — never break training
        print(f"  (stratified metrics unavailable: {exc})")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def main() -> None:
    from fpl_optimizer.prediction.feature_pipeline import FEATURE_PIPELINE_VERSION
    from fpl_optimizer.prediction.minutes import OOF_KEY, expanding_minutes_oof

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "prod_2026-27")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--features-cache", type=Path, default=None,
                        help="FeaturePipeline parquet: loaded if present, else built and saved")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="rebuild --features-cache even if it exists")
    parser.add_argument("--recipe", default=None,
                        help="JSON (inline or a .json file) merged over RECIPE")
    parser.add_argument("--no-eval", action="store_true",
                        help="skip the 2025-26 holdout evaluation phase")
    parser.add_argument("--second-fold", action="store_true",
                        help="also evaluate train <= 2023-24 -> holdout 2024-25")
    parser.add_argument("--no-live", action="store_true",
                        help="skip the current-season LIVE check")
    parser.add_argument("--no-prod", action="store_true",
                        help="evaluate only; do not train/save the production model")
    args = parser.parse_args()

    recipe = resolve_recipe(args.recipe)
    print("Recipe: " + json.dumps({k: v for k, v in recipe.items() if k != "exclude_features"},
                                  default=_json_default))
    print(f"  exclude_features: {len(recipe['exclude_features'])} names")
    t_start = time.time()
    timings: dict[str, float] = {}

    df = load_features(args)
    df, excl = apply_exclusions(df, recipe)
    cur = df[df["season"] == CURRENT_SEASON]
    cur_gws = sorted(int(g) for g in cur["GW"].unique())
    print(f"Excluded {len(excl)} features; {CURRENT_SEASON} completed GWs: {cur_gws}")
    timings["features"] = time.time() - t_start

    minutes_oof = None
    if recipe["kind"] == "minutes_blend":
        t0 = time.time()
        print("\nNested minutes predictions (season S from a model fit on seasons < S)...")
        oof = expanding_minutes_oof(df, recipe["mm_rounds"])
        minutes_oof = pd.concat([df[OOF_KEY], oof], axis=1)
        timings["minutes_oof"] = time.time() - t0
        print(f"  done in {timings['minutes_oof']:.0f}s")

    metrics: dict = {"recipe": recipe, "feature_pipeline_version": FEATURE_PIPELINE_VERSION}

    phases = []
    if not args.no_eval:
        phases.append(("eval", EVAL_TRAIN_SEASONS, EVAL_HOLDOUT))
        if args.second_fold:
            phases.append(("eval_fold2", FOLD2_TRAIN_SEASONS, FOLD2_HOLDOUT))
    for name, train_seasons, holdout_season in phases:
        t0 = time.time()
        train_df, val_df = _split_val(df, train_seasons)
        holdout = df[df["season"] == holdout_season]
        print(f"\n{name.upper()} phase: train <= {train_seasons[-1]} ({len(train_df)} rows), "
              f"val {len(val_df)}, holdout {holdout_season} ({len(holdout)})")
        model = fit_model(recipe, train_df, val_df, minutes_oof)
        metrics[f"{name}_{holdout_season}"] = evaluate(
            model, holdout, f"HOLDOUT {holdout_season} ({recipe['kind']})")
        timings[name] = time.time() - t0
        del model, train_df, val_df

    live_model = None
    if not args.no_live and len(cur):
        t0 = time.time()
        train_df, val_df = _split_val(df, COMPLETE_SEASONS)
        print(f"\nLIVE check: train <= {COMPLETE_SEASONS[-1]} ({len(train_df)} rows), "
              f"val {len(val_df)}, test {CURRENT_SEASON} GW{cur_gws[0]}-{cur_gws[-1]} ({len(cur)})")
        live_model = fit_model(recipe, train_df, val_df, minutes_oof)
        metrics[f"live_{CURRENT_SEASON}"] = evaluate(
            live_model, cur, f"LIVE {CURRENT_SEASON} GW{cur_gws[0]}-{cur_gws[-1]} "
                             f"(complete seasons only; {len(cur_gws)} GWs = noisy)")
        timings["live"] = time.time() - t0

    if args.no_prod:
        print(f"\nDone (no PROD) in {time.time() - t_start:.0f}s; timings {timings}")
        return

    t0 = time.time()
    include_cur = bool(recipe["include_current_season"]) and len(cur) > 0
    train_df, val_df = prod_split(df, include_cur)
    print(f"\nPROD phase: train {len(train_df)} rows (complete seasons"
          + (f" + {CURRENT_SEASON} GW{cur_gws[0]}-{cur_gws[-1]}" if include_cur else "")
          + f"), val {len(val_df)} ({COMPLETE_SEASONS[-1]} last 8 GWs)")
    if live_model is not None and not include_cur:
        prod = live_model  # identical rows: the LIVE model IS the PROD model
        print("  (reusing the LIVE model: no current-season rows to add)")
    else:
        prod = fit_model(recipe, train_df, val_df, minutes_oof)
    timings["prod"] = time.time() - t0

    # Atomic promote: train into a staging dir, keep the previous model as
    # a .prev rollback, then rename — readers (gameweek.py) never see a torn
    # model dir, and the old model+report survive for comparison.
    staging = args.out.with_name(args.out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    prod.save(staging)
    # Serving reference for the live data-health check (live/predict.py):
    # non-null rate / std per feature on recent rows of playing players.
    from fpl_optimizer.live.predict import SERVING_REFERENCE_FILE, serving_reference

    recent = df[df["season"].isin(COMPLETE_SEASONS[-3:]) & (df["GW"] >= 2)]
    (staging / SERVING_REFERENCE_FILE).write_text(
        json.dumps(serving_reference(recent, prod.feature_names), indent=1),
        encoding="utf-8",
    )
    trained = {
        "kind": recipe["kind"],
        "train_seasons": COMPLETE_SEASONS,
        "current_season": CURRENT_SEASON,
        "current_season_gws": cur_gws if include_cur else [],
        "val": f"{COMPLETE_SEASONS[-1]} last 8 GWs",
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "best_iterations": prod.best_iterations(),
        "calibration": prod.calibration,
        "n_features": len(prod.feature_names),
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    metrics["prod"] = trained
    metrics["timings_s"] = {k: round(v) for k, v in timings.items()}
    (staging / "training_report.json").write_text(
        json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8"
    )
    meta_path = staging / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update({k: v for k, v in trained.items() if k != "calibration"})
    meta["recipe"] = recipe
    meta_path.write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")

    prev = args.out.with_name(args.out.name + ".prev")
    if args.out.exists():
        if prev.exists():
            shutil.rmtree(prev)
        args.out.rename(prev)
    staging.rename(args.out)
    print(f"\nProduction model ({recipe['kind']}, {len(prod.feature_names)} input features) "
          f"saved to {args.out} (previous model kept at {prev.name})")

    fi = prod.feature_importance()
    if not fi.empty:
        print("\nTop 15 features:")
        for i, row in fi.head(15).reset_index(drop=True).iterrows():
            print(f"  #{i + 1:>2} {row['feature']:<35} {row['importance']:.4g}")
    print(f"\nTotal {time.time() - t_start:.0f}s; timings (s): "
          + ", ".join(f"{k} {v:.0f}" for k, v in timings.items()))


if __name__ == "__main__":
    main()
