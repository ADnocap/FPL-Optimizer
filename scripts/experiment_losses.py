#!/usr/bin/env python3
"""Loss/weighting A/B experiments for the point predictor (ROADMAP #7).

Builds features ONCE (2016-17..2025-26, cached to parquet under runs/),
then trains four configs on 2016-17..2024-25 (val = last 8 GWs of 2024-25
for early stopping) and evaluates every config on the 2025-26 season:

1. baseline         — production params (scripts/train_predictor.py PARAMS),
                      objective=regression, metric=mae. Reproduces the honest
                      2025-26 holdout run of models/prod_2026-27.
2. tweedie          — objective=tweedie, tweedie_variance_power=1.3.
                      Tweedie requires non-negative targets but FPL points
                      can be negative (red card + own goals etc., min ~-4),
                      so the TRAIN/VAL target is shifted by +TARGET_SHIFT
                      before fitting and predictions are un-shifted before
                      scoring. The shift is bumped automatically if the
                      observed minimum is below -TARGET_SHIFT.
3. weights          — baseline objective + entropy-binned balanced sample
                      weights (OpenFPL's trick): KBinsDiscretizer(kmeans,
                      4 bins) on the raw train target, weight = inverse bin
                      frequency, clipped at the 95th percentile, normalized
                      to mean 1. Validation stays unweighted so early
                      stopping selects on unweighted MAE.
4. tweedie+weights  — configs 2 and 3 combined (weights are binned on the
                      raw, un-shifted target; kmeans binning is
                      shift-invariant so this is equivalent either way).

Metrics per config (on 2025-26): MAE, RMSE, mean per-GW Spearman, and
Zeros/Blanks/Tickers/Haulers conditional RMSE (uses
fpl_optimizer.prediction.stratified_metrics when available, else the inline
fallback below with strata zeros<=0 / blanks 1-3 / tickers 4-9 /
haulers>=10 points).

Everything printed is tee'd to runs/experiment_losses.log; the full
comparison table is also written to runs/experiment_losses_results.json.

Usage:
    python -X utf8 scripts/experiment_losses.py [--rebuild-features] [--quick]

--quick trains tiny models on 3 seasons to smoke-test the plumbing only —
its numbers are meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
LOG_PATH = RUNS_DIR / "experiment_losses.log"
RESULTS_PATH = RUNS_DIR / "experiment_losses_results.json"
FEATURES_CACHE = RUNS_DIR / "experiment_losses_features.parquet"

TRAIN_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]
EVAL_SEASON = "2025-26"
ALL_SEASONS = TRAIN_SEASONS + [EVAL_SEASON]

# Mirrors scripts/train_predictor.py PARAMS (the models/prod_2026-27 recipe).
BASELINE_PARAMS = {
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

# metric stays "mae": the shift is a constant, so early stopping on shifted
# MAE selects the same iteration as un-shifted MAE — keeps model selection
# comparable with the baseline.
TWEEDIE_PARAMS = {
    **BASELINE_PARAMS,
    "objective": "tweedie",
    "tweedie_variance_power": 1.3,
}

TARGET_SHIFT = 4.0  # min possible FPL score is ~-4; bumped if data disagrees

# Conditional-RMSE strata on ACTUAL points (inline fallback — Task B's
# stratified_metrics.py is preferred at runtime when present).
STRATA = [
    ("zeros", -np.inf, 0.0),  # <= 0 pts (includes negatives)
    ("blanks", 1.0, 3.0),
    ("tickers", 4.0, 9.0),
    ("haulers", 10.0, np.inf),
]
STRATA_NAMES = [name for name, _, _ in STRATA]


class _Tee:
    """Duplicate a text stream to the console and the log file."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
        self.flush()
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


def _stratified_rmse_inline(actual: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Conditional RMSE per actual-points stratum (fallback implementation)."""
    out: dict[str, float] = {}
    for name, lo, hi in STRATA:
        m = (actual >= lo) & (actual <= hi)
        if m.any():
            out[name] = float(np.sqrt(np.mean((preds[m] - actual[m]) ** 2)))
        else:
            out[name] = float("nan")
    return out


def _resolve_stratified_rmse():
    """Prefer fpl_optimizer.prediction.stratified_metrics (Task B) when importable
    and call-compatible; otherwise use the inline fallback."""
    try:
        from fpl_optimizer.prediction import stratified_metrics as sm
    except ImportError:
        return _stratified_rmse_inline, "inline"

    for fn_name in ("stratified_rmse", "conditional_rmse"):
        fn = getattr(sm, fn_name, None)
        if not callable(fn):
            continue
        try:  # probe call-compatibility: (actual, preds) -> mapping of floats
            probe = fn(np.array([0.0, 2.0, 5.0, 12.0]), np.array([1.0, 2.0, 4.0, 8.0]))
            if isinstance(probe, dict) and probe:
                return fn, f"stratified_metrics.{fn_name}"
        except Exception:
            continue
    return _stratified_rmse_inline, "inline"


def balanced_sample_weights(
    y: np.ndarray,
    n_bins: int = 4,
    clip_pct: float = 95.0,
) -> np.ndarray:
    """Entropy-binned balanced weights (OpenFPL's trick).

    KBinsDiscretizer(kmeans) bins the target, each row is weighted by the
    inverse frequency of its bin (balanced: n / (n_bins * bin_count)),
    weights are clipped at the *clip_pct* percentile and normalized to
    mean 1. ``subsample=None`` keeps the kmeans binning deterministic.
    """
    from sklearn.preprocessing import KBinsDiscretizer

    disc = KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="kmeans", subsample=None
    )
    bins = disc.fit_transform(y.reshape(-1, 1)).ravel().astype(int)
    counts = np.bincount(bins, minlength=n_bins)
    w = len(y) / (n_bins * counts[bins].astype(np.float64))
    w = np.minimum(w, np.percentile(w, clip_pct))
    w = w / w.mean()

    print("  Weight bins (kmeans on train target):")
    edges = disc.bin_edges_[0]
    for b in range(n_bins):
        bw = len(y) / (n_bins * counts[b]) if counts[b] else float("nan")
        print(
            f"    bin {b}: [{edges[b]:6.2f}, {edges[b + 1]:6.2f}] "
            f"n={counts[b]:>7d}  raw_w={bw:.3f}"
        )
    print(
        f"  Clipped at p{clip_pct:.0f}={np.percentile(w, clip_pct):.3f}, "
        f"normalized: mean={w.mean():.3f} min={w.min():.3f} max={w.max():.3f}"
    )
    return w


def build_or_load_features(
    data_dir: Path, seasons: list[str], cache: Path, rebuild: bool
) -> pd.DataFrame:
    """Build the feature DataFrame once (or load the cached parquet)."""
    if cache.exists() and not rebuild:
        df = pd.read_parquet(cache)
        cached = sorted(df["season"].unique())
        if cached == sorted(seasons):
            print(f"Loaded cached features from {cache}: "
                  f"{len(df)} rows x {len(df.columns)} cols")
            return df
        print(f"Cache season mismatch ({cached}), rebuilding...")

    from fpl_optimizer.prediction.feature_pipeline import FeaturePipeline
    from fpl_optimizer.prediction.id_resolver import IDResolver

    print(f"Building features for {len(seasons)} seasons...")
    t0 = time.time()
    resolver = IDResolver(data_dir)
    df = FeaturePipeline(data_dir, resolver, seasons).build()
    print(f"Features: {len(df)} rows x {len(df.columns)} cols "
          f"in {time.time() - t0:.0f}s")

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    print(f"Cached features to {cache}")
    return df


def prepare_splits(
    df: pd.DataFrame, train_seasons: list[str], eval_season: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic-GW drop + train/val/eval split (train_predictor.py recipe).

    val = last 8 GWs of the last training season (2024-25).
    """
    # Drop synthetic/unplayed gameweeks (fpl_live writes upcoming-GW rows
    # with all stats zeroed; a real GW always has at least one scorer).
    gw_max_target = df.groupby(["season", "GW"])["target"].transform("max")
    synthetic = ~(gw_max_target > 0)
    if synthetic.any():
        dropped = df.loc[synthetic, ["season", "GW"]].drop_duplicates()
        print(f"Dropping {int(synthetic.sum())} synthetic/unplayed rows: "
              + ", ".join(f"{s} GW{int(g)}"
                          for s, g in dropped.itertuples(index=False)))
        df = df[~synthetic]

    # Uniform NaN-target drop (weights + tweedie both need finite labels;
    # applied to every config equally so the A/B stays fair).
    n_nan = int(df["target"].isna().sum())
    if n_nan:
        print(f"Dropping {n_nan} NaN-target rows")
        df = df[df["target"].notna()]

    subset = df[df["season"].isin(train_seasons)].copy()
    last = train_seasons[-1]
    max_gw = int(subset.loc[subset["season"] == last, "GW"].max())
    val_mask = (subset["season"] == last) & (subset["GW"] > max_gw - 8)
    train_df = subset[~val_mask].copy()
    val_df = subset[val_mask].copy()
    eval_df = df[df["season"] == eval_season].copy()

    # Rows with unknown position get predict()'s constant fallback (2.0),
    # which the tweedie shift would distort — drop them from eval uniformly.
    known_pos = eval_df["position"].isin(["GK", "DEF", "MID", "FWD"])
    if (~known_pos).any():
        print(f"Dropping {int((~known_pos).sum())} unknown-position eval rows")
        eval_df = eval_df[known_pos]

    print(f"Split: train={len(train_df)} val={len(val_df)} "
          f"(last 8 GWs of {last}) eval({eval_season})={len(eval_df)}")
    return train_df, val_df, eval_df


def evaluate_config(
    preds: np.ndarray, eval_df: pd.DataFrame, stratified_fn
) -> dict:
    """MAE, RMSE, mean per-GW Spearman, per-position MAE, conditional RMSE."""
    from scipy.stats import spearmanr

    actual = eval_df["target"].to_numpy(dtype=np.float64)
    mae = float(np.mean(np.abs(preds - actual)))
    rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))

    per_pos_mae = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        m = (eval_df["position"] == pos).to_numpy()
        if m.any():
            per_pos_mae[pos] = float(np.mean(np.abs(preds[m] - actual[m])))

    ed = eval_df.copy()
    ed["pred"] = preds
    gw_rhos = [
        float(spearmanr(g["pred"], g["target"]).statistic)
        for _, g in ed.groupby("GW")
        if len(g) > 20
    ]
    gw_spearman = float(np.nanmean(gw_rhos))

    strat = {k: float(v) for k, v in stratified_fn(actual, preds).items()}
    return {
        "mae": mae,
        "rmse": rmse,
        "gw_spearman": gw_spearman,
        "per_position_mae": per_pos_mae,
        "stratified_rmse": strat,
    }


def run_config(
    name: str,
    params: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    stratified_fn,
    sample_weight: np.ndarray | None = None,
    shift: float = 0.0,
) -> dict:
    """Train one config and score it on the eval season."""
    from fpl_optimizer.prediction.model import PointPredictor

    print(f"\n--- Config: {name} ---")
    print(f"  objective={params['objective']}"
          + (f" (tweedie_variance_power={params['tweedie_variance_power']})"
             if "tweedie_variance_power" in params else "")
          + f" shift={shift:+.1f} weights={'yes' if sample_weight is not None else 'no'}")

    tdf, vdf = train_df, val_df
    if shift:
        tdf = train_df.copy()
        tdf["target"] = tdf["target"] + shift
        vdf = val_df.copy()
        vdf["target"] = vdf["target"] + shift

    t0 = time.time()
    predictor = PointPredictor(params=params, early_stopping_rounds=50)
    predictor.train(tdf, vdf, sample_weight=sample_weight)
    best_iters = {
        pos: getattr(m, "best_iteration", None)
        for pos, m in predictor._models.items()
    }
    print(f"  Trained in {time.time() - t0:.0f}s, best_iteration={best_iters}")

    preds = predictor.predict(eval_df) - shift
    metrics = evaluate_config(preds, eval_df, stratified_fn)
    metrics["name"] = name
    metrics["params"] = params
    metrics["shift"] = shift
    metrics["weighted"] = sample_weight is not None
    metrics["best_iteration"] = best_iters
    metrics["train_seconds"] = round(time.time() - t0, 1)

    strat = metrics["stratified_rmse"]
    print(f"  MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} "
          f"GW-Spearman={metrics['gw_spearman']:.4f}")
    print("  Conditional RMSE: "
          + "  ".join(f"{k}={strat.get(k, float('nan')):.4f}"
                      for k in STRATA_NAMES))
    print("  Per-position MAE: "
          + "  ".join(f"{k}={v:.4f}"
                      for k, v in metrics["per_position_mae"].items()))
    return metrics


def print_table(results: list[dict]) -> None:
    cols = ["MAE", "RMSE", "GW-Sp"] + [n.capitalize() for n in STRATA_NAMES]
    header = f"{'Config':<22}" + "".join(f"{c:>9}" for c in cols)
    print("\n" + "=" * len(header))
    print("COMPARISON TABLE (eval season 2025-26; GW-Sp = mean per-GW Spearman;")
    print("Zeros/Blanks/Tickers/Haulers = conditional RMSE on actual <=0 / 1-3 / 4-9 / >=10 pts)")
    print("=" * len(header))
    print(header)
    for r in results:
        strat = r["stratified_rmse"]
        vals = [r["mae"], r["rmse"], r["gw_spearman"]] + [
            strat.get(n, float("nan")) for n in STRATA_NAMES
        ]
        print(f"{r['name']:<22}" + "".join(f"{v:>9.4f}" for v in vals))
    print("=" * len(header))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--rebuild-features", action="store_true",
                        help="ignore the cached feature parquet")
    parser.add_argument("--quick", action="store_true",
                        help="tiny smoke-test run (3 seasons, small trees) — "
                             "numbers are meaningless")
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_PATH, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    print(f"\n{'#' * 70}\n# experiment_losses run "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')} quick={args.quick}\n{'#' * 70}")

    if args.quick:
        train_seasons = ["2023-24", "2024-25"]
        seasons = train_seasons + [EVAL_SEASON]
        cache = RUNS_DIR / "experiment_losses_features_quick.parquet"
        overrides = {"n_estimators": 50, "num_leaves": 31}
    else:
        train_seasons = TRAIN_SEASONS
        seasons = ALL_SEASONS
        cache = FEATURES_CACHE
        overrides = {}

    stratified_fn, strat_source = _resolve_stratified_rmse()
    print(f"Stratified RMSE implementation: {strat_source}")

    df = build_or_load_features(args.data_dir, seasons, cache, args.rebuild_features)
    train_df, val_df, eval_df = prepare_splits(df, train_seasons, EVAL_SEASON)

    # Tweedie target shift: constant +4 unless the data has a lower minimum.
    min_target = float(min(train_df["target"].min(), val_df["target"].min()))
    shift = max(TARGET_SHIFT, -min_target)
    if shift != TARGET_SHIFT:
        print(f"NOTE: min train/val target {min_target} < -{TARGET_SHIFT}; "
              f"bumping shift to +{shift}")
    print(f"Tweedie target shift: +{shift} (min train/val target={min_target})")

    # Balanced weights, binned once on the raw (un-shifted) train target.
    print("\nComputing entropy-binned balanced sample weights...")
    weights = balanced_sample_weights(train_df["target"].to_numpy(dtype=np.float64))

    baseline_params = {**BASELINE_PARAMS, **overrides}
    tweedie_params = {**TWEEDIE_PARAMS, **overrides}

    results = [
        run_config("baseline", baseline_params,
                   train_df, val_df, eval_df, stratified_fn),
        run_config("tweedie(1.3,+4)", tweedie_params,
                   train_df, val_df, eval_df, stratified_fn, shift=shift),
        run_config("balanced-weights", baseline_params,
                   train_df, val_df, eval_df, stratified_fn,
                   sample_weight=weights),
        run_config("tweedie+weights", tweedie_params,
                   train_df, val_df, eval_df, stratified_fn,
                   sample_weight=weights, shift=shift),
    ]

    print_table(results)

    # Recommendation: we consume rankings (transfers/captaincy), so mean
    # per-GW Spearman is the primary criterion; MAE breaks ties.
    best = max(results, key=lambda r: (r["gw_spearman"], -r["mae"]))
    base = results[0]
    print(f"\nBest by mean per-GW Spearman: {best['name']} "
          f"({best['gw_spearman']:.4f} vs baseline {base['gw_spearman']:.4f}; "
          f"MAE {best['mae']:.4f} vs {base['mae']:.4f})")

    RESULTS_PATH.write_text(
        json.dumps({
            "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "quick": args.quick,
            "train_seasons": train_seasons,
            "eval_season": EVAL_SEASON,
            "tweedie_shift": shift,
            "stratified_impl": strat_source,
            "results": results,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
