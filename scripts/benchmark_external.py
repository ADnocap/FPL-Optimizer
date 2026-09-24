#!/usr/bin/env python3
"""Benchmark our point predictor against external public projections.

Outcome-stratified evaluation (Zeros/Blanks/Tickers/Haulers conditional
MAE/RMSE, see fpl_optimizer.prediction.stratified_metrics) of:

1. OUR model — expanding-window eval folds (train from 2016-17, val = last
   8 GWs of the last training season, LightGBM params identical to
   scripts/train_predictor.py) tested on 2021-22 .. 2025-26. The 2025-26
   fold reproduces the honest holdout of the model of record.
2. theFPLkiwi — archived weekly projection CSVs from
   github.com/theFPLkiwi/theFPLkiwi. The public archive STOPS at 2023-24
   (and 23-24 has only GWs 1/3/4/18); there are NO 2024-25 or 2025-26
   projections, so the comparison runs on 2021-22, 2022-23 and 2023-24,
   scored on the exact same (player, GW) rows as our fold models.
3. OpenFPL (github.com/daniegr/OpenFPL, arXiv 2508.09992) — ships
   pretrained XGBoost ensembles but NO feature-engineering code (only
   models/, a 6-row samples.csv and play.ipynb), so scoring it on our
   holdout would mean reimplementing their full FPL+Understat rolling
   feature pipeline from the paper. Infeasible in a day — we instead quote
   their PUBLISHED per-category RMSE (prospective test on 2024-25) next to
   our own 2024-25 fold scored with the same bins.

Usage:
    python scripts/benchmark_external.py ours    # long: features + 5 fold trainings
    python scripts/benchmark_external.py kiwi    # download + score theFPLkiwi
    python scripts/benchmark_external.py report  # final comparison table
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fpl_optimizer.prediction.stratified_metrics import (  # noqa: E402
    OUTCOME_BINS,
    format_report,
    per_gw_spearman,
    stratified_metrics,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"

ALL_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]
# Expanding-window fold test seasons (train = everything before each)
FOLD_TEST_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

# Same params as scripts/train_predictor.py (kept in sync by hand — scripts/
# is not an importable package).
PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 127,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
    "n_estimators": 2000,
    "verbose": -1,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
}

OURS_JSON = RUNS_DIR / "benchmark_ours_stratified.json"
KIWI_JSON = RUNS_DIR / "benchmark_kiwi_stratified.json"
KIWI_CACHE = RUNS_DIR / "benchmark_kiwi"

# theFPLkiwi archive layout: season -> repo folder (files named FPL_GW{n}.csv)
_KIWI_RAW = "https://raw.githubusercontent.com/theFPLkiwi/theFPLkiwi/main"
KIWI_FOLDERS = {
    "2021-22": "Old_Seasons/FPL_projections_21_22",
    "2022-23": "Old_Seasons/FPL_projections_22_23",
    "2023-24": "FPL_projections_23_24",
}
# Unconditional expected-points section marker (naming drifts per file:
# "Overall xPts by GW" 21-22 GW1, "xPts" 21-22/22-23, "GW pts"/"Pts" 23-24)
_KIWI_PTS_SECTIONS = {"xPts", "GW pts", "Pts", "Overall xPts by GW"}

# OpenFPL paper (arXiv 2508.09992): per-category RMSE at the one-GW
# horizon, prospective test on GWs 32-38 of 2024-25 only. Their categories:
# Zeros (did not play), Blanks (played, <=2 pts), Tickers (3-4),
# Haulers (>=5) — ours differ only in that played-but-<=0-pt rows land in
# Zeros instead of Blanks (rare). FPL Review = the commercial benchmark.
OPENFPL_PUBLISHED = {
    "test_season": "2024-25 GW32-38",
    "OpenFPL": {"Zeros": 0.818, "Blanks": 1.291, "Tickers": 1.517, "Haulers": 5.142},
    "FPL Review": {"Zeros": 0.689, "Blanks": 1.189, "Tickers": 1.594, "Haulers": 5.172},
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_actuals(season: str) -> pd.DataFrame:
    """Per-(element, GW) actual points and minutes, DGW-aggregated."""
    path = REPO_ROOT / "data" / "raw" / season / "gws" / "merged_gw.csv"
    try:
        gw = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        gw = pd.read_csv(path, encoding="latin-1")
    out = (
        gw.groupby(["element", "GW"], as_index=False)
        .agg(target=("total_points", "sum"), minutes=("minutes", "sum"))
    )
    if "position" in gw.columns:
        pos = gw[["element", "position"]].dropna().drop_duplicates("element")
        out = out.merge(pos, on="element", how="left")
    return out


def _preds_path(season: str) -> Path:
    return RUNS_DIR / f"benchmark_ours_preds_{season}.parquet"


def _score(df: pd.DataFrame, pred_col: str, label: str) -> dict:
    """Stratified metrics + per-GW Spearman for one prediction column."""
    m = stratified_metrics(
        df["target"].values, df[pred_col].values, df["minutes"].values
    )
    rho = per_gw_spearman(df["target"].values, df[pred_col].values, df["GW"].values)
    m["per_gw_spearman"] = rho["mean"]
    print(format_report(m, label=label))
    print(f"Per-GW Spearman: {rho['mean']:.4f} over {len(rho['per_gw'])} GWs\n")
    return m


# ---------------------------------------------------------------------------
# Subcommand: ours
# ---------------------------------------------------------------------------

def run_ours(data_dir: Path) -> None:
    from fpl_optimizer.prediction.feature_pipeline import FeaturePipeline
    from fpl_optimizer.prediction.id_resolver import IDResolver
    from fpl_optimizer.prediction.model import PointPredictor

    print(f"Building features for {len(ALL_SEASONS)} seasons...")
    t0 = time.time()
    resolver = IDResolver(data_dir)
    df = FeaturePipeline(data_dir, resolver, ALL_SEASONS).build()
    print(f"Features: {len(df)} rows x {len(df.columns)} cols "
          f"in {time.time() - t0:.0f}s", flush=True)

    # Drop synthetic/unplayed GWs (same guard as scripts/train_predictor.py)
    gw_max_target = df.groupby(["season", "GW"])["target"].transform("max")
    synthetic = ~(gw_max_target > 0)
    if synthetic.any():
        dropped = df.loc[synthetic, ["season", "GW"]].drop_duplicates()
        print(f"Dropping {int(synthetic.sum())} synthetic/unplayed rows: "
              + ", ".join(f"{s} GW{int(g)}" for s, g in dropped.itertuples(index=False)))
        df = df[~synthetic]

    results: dict = {"params": PARAMS, "folds": {}}
    for test_season in FOLD_TEST_SEASONS:
        train_seasons = ALL_SEASONS[: ALL_SEASONS.index(test_season)]
        subset = df[df["season"].isin(train_seasons)]
        last = train_seasons[-1]
        max_gw = int(subset.loc[subset["season"] == last, "GW"].max())
        val_mask = (subset["season"] == last) & (subset["GW"] > max_gw - 8)
        train_df, val_df = subset[~val_mask], subset[val_mask]
        test_df = df[df["season"] == test_season].copy()

        print(f"\n--- Fold {test_season}: train={len(train_df)} "
              f"val={len(val_df)} test={len(test_df)} ---", flush=True)
        t0 = time.time()
        model = PointPredictor(params=PARAMS, early_stopping_rounds=50)
        model.train(train_df, val_df)
        test_df["pred"] = model.predict(test_df)
        print(f"Trained + predicted in {time.time() - t0:.0f}s")

        # Actual minutes (not a feature — joined only for DNP binning)
        actuals = _load_actuals(test_season)
        test_df = test_df.merge(
            actuals[["element", "GW", "minutes"]], on=["element", "GW"], how="left"
        )
        keep = ["element", "GW", "season", "position", "target", "minutes", "pred"]
        test_df = test_df[keep].dropna(subset=["target"])
        test_df.to_parquet(_preds_path(test_season), index=False)

        results["folds"][test_season] = _score(
            test_df, "pred", f"OURS — {test_season} "
            f"(train {train_seasons[0]}..{train_seasons[-1]})"
        )

    OURS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved metrics to {OURS_JSON}")


# ---------------------------------------------------------------------------
# Subcommand: kiwi
# ---------------------------------------------------------------------------

def _fetch_kiwi_file(season: str, gw: int) -> str | None:
    """Download (with local cache) one kiwi weekly CSV; None if absent."""
    cache = KIWI_CACHE / season / f"FPL_GW{gw}.csv"
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    url = f"{_KIWI_RAW}/{KIWI_FOLDERS[season]}/FPL_GW{gw}.csv"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def _parse_kiwi_projection(text: str, gw: int) -> pd.DataFrame:
    """Extract the unconditional xPts for *gw* from one kiwi weekly CSV.

    The header is one row of sections: ``ID,Name,Pos,Price,Team,<summary
    cols>,<section>,<gw>,<gw+1>,...,<section>,...``. Section names drift
    across seasons (xPts / "GW pts" / Pts for the unconditional expected
    points block), so we locate the marker by exact token match, then the
    column labeled str(gw) before the next non-numeric token.
    """
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    sec_idx = next(
        (i for i, tok in enumerate(header) if tok.strip() in _KIWI_PTS_SECTIONS),
        None,
    )
    if sec_idx is None:
        raise ValueError(f"no xPts section in header: {header[:12]}...")
    col_idx = None
    for i in range(sec_idx + 1, len(header)):
        tok = header[i].strip()
        if not tok.isdigit():
            break
        if int(tok) == gw:
            col_idx = i
            break
    if col_idx is None:
        raise ValueError(f"GW{gw} column not found after section {header[sec_idx]!r}")

    out = []
    for row in rows[1:]:
        if len(row) <= col_idx or not row[0].strip().isdigit():
            continue
        try:
            pred = float(row[col_idx])
        except ValueError:
            continue
        out.append({
            "element": int(row[0]),
            "kiwi_name": row[1].strip(),
            "kiwi_pos": row[2].strip().upper(),
            "GW": gw,
            "kiwi_pred": pred,
        })
    return pd.DataFrame(out)


def run_kiwi() -> None:
    results: dict = {"seasons": {}}
    for season in KIWI_FOLDERS:
        frames = []
        for gw in range(1, 39):
            text = _fetch_kiwi_file(season, gw)
            if text is None:
                continue
            frames.append(_parse_kiwi_projection(text, gw))
        if not frames:
            print(f"{season}: no kiwi files found, skipping")
            continue
        kiwi = pd.concat(frames, ignore_index=True)
        gws = sorted(kiwi["GW"].unique())
        print(f"\n{'=' * 70}\n{season}: {len(gws)} kiwi GW files "
              f"({', '.join(map(str, gws))}), {len(kiwi)} projections")

        actuals = _load_actuals(season)
        joined = kiwi.merge(actuals, on=["element", "GW"], how="inner")
        join_rate = len(joined) / len(kiwi)
        pos_ok = float("nan")
        if "position" in joined.columns:
            has_pos = joined.dropna(subset=["position"])
            pos_ok = (
                has_pos["kiwi_pos"].str[:2] == has_pos["position"].str[:2]
            ).mean()
        print(f"Join rate on (element, GW): {join_rate:.1%}; "
              f"position agreement (ID-mapping check): {pos_ok:.1%}")
        if pos_ok < 0.95:
            print("WARNING: kiwi ID column does not look like the FPL "
                  "element id for this season — skipping")
            continue

        season_res: dict = {
            "gws": [int(g) for g in gws],
            "n_rows": len(joined),
            "join_rate": join_rate,
        }
        season_res["kiwi"] = _score(joined, "kiwi_pred", f"theFPLkiwi — {season}")

        # Our fold model on the exact same rows
        if _preds_path(season).exists():
            ours = pd.read_parquet(_preds_path(season))
            both = joined.merge(
                ours[["element", "GW", "pred"]], on=["element", "GW"], how="inner"
            )
            print(f"Common rows with our fold preds: {len(both)}")
            season_res["kiwi_common"] = _score(
                both, "kiwi_pred", f"theFPLkiwi — {season} (common rows)"
            )
            season_res["ours_common"] = _score(
                both, "pred", f"OURS — {season} (common rows)"
            )
        else:
            print(f"({_preds_path(season).name} missing — run 'ours' first "
                  f"for the head-to-head)")
        results["seasons"][season] = season_res

    KIWI_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to {KIWI_JSON}")


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------

def _fmt_bins(m: dict, metric: str) -> str:
    return "  ".join(
        f"{name[0]}={m['bins'][name][metric]:.3f}" for name in OUTCOME_BINS
    )


def run_report() -> None:
    print("Outcome bins: Zeros (0 pts/DNP), Blanks (1-2), Tickers (3-4), "
          "Haulers (5+)\n")

    if OURS_JSON.exists():
        ours = json.loads(OURS_JSON.read_text(encoding="utf-8"))
        print("--- OURS (expanding-window folds, conditional RMSE) ---")
        for season, m in ours["folds"].items():
            o = m["overall"]
            print(f"{season}: MAE={o['mae']:.4f} Spearman={o['spearman']:.4f} "
                  f"GW-Spearman={m['per_gw_spearman']:.4f} | {_fmt_bins(m, 'rmse')}")
        our_2425 = ours["folds"].get("2024-25")
    else:
        print(f"({OURS_JSON.name} missing — run 'ours')")
        our_2425 = None

    print("\n--- vs OpenFPL published (arXiv 2508.09992, RMSE, "
          f"test {OPENFPL_PUBLISHED['test_season']}) ---")
    for name in ("OpenFPL", "FPL Review"):
        pub = OPENFPL_PUBLISHED[name]
        print(f"{name:>22}: " + "  ".join(
            f"{b[0]}={pub[b]:.3f}" for b in OUTCOME_BINS))
    if our_2425:
        print(f"{'OURS 2024-25 full':>22}: {_fmt_bins(our_2425, 'rmse')}")
    p2425 = _preds_path("2024-25")
    if p2425.exists():
        sub = pd.read_parquet(p2425)
        sub = sub[sub["GW"] >= 32]
        m = stratified_metrics(
            sub["target"].values, sub["pred"].values, sub["minutes"].values
        )
        print(f"{'OURS 2024-25 GW32-38':>22}: {_fmt_bins(m, 'rmse')}  "
              f"(same GWs as published numbers)")

    if KIWI_JSON.exists():
        kiwi = json.loads(KIWI_JSON.read_text(encoding="utf-8"))
        print("\n--- vs theFPLkiwi (same rows, conditional RMSE) ---")
        for season, res in kiwi["seasons"].items():
            if "kiwi_common" not in res:
                continue
            for label, key in (("kiwi", "kiwi_common"), ("ours", "ours_common")):
                m = res[key]
                o = m["overall"]
                print(f"{season} {label:>4}: MAE={o['mae']:.4f} "
                      f"GW-Spearman={m['per_gw_spearman']:.4f} | "
                      f"{_fmt_bins(m, 'rmse')}  (n={o['n']})")
    else:
        print(f"\n({KIWI_JSON.name} missing — run 'kiwi')")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["ours", "kiwi", "report"])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()
    RUNS_DIR.mkdir(exist_ok=True)
    if args.mode == "ours":
        run_ours(args.data_dir)
    elif args.mode == "kiwi":
        run_kiwi()
    else:
        run_report()


if __name__ == "__main__":
    main()
