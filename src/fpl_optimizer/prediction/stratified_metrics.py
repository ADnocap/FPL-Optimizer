"""Outcome-stratified evaluation metrics for point prediction.

Global MAE hides *where* a model is weak: most player-GWs are zeros, so a
model can look good by nailing DNPs while missing every haul. This module
bins actual outcomes into four categories and reports conditional errors
per bin (the OpenFPL-style diagnostic, arXiv 2508.09992):

- **Zeros**:   0 points or fewer, or did not play (minutes == 0)
- **Blanks**:  1-2 points (played, no return)
- **Tickers**: 3-4 points (appearance + CS/bonus scraps)
- **Haulers**: 5+ points (attacking return or better)

Pure-numpy/pandas helpers — no model dependency — so external projections
(theFPLkiwi, OpenFPL, ...) can be scored identically to our own model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOME_BINS = ["Zeros", "Blanks", "Tickers", "Haulers"]

# Minimum players in a GW for its Spearman to be trustworthy
_MIN_GW_SIZE = 20


def assign_outcome_bins(
    y_true: np.ndarray,
    minutes: np.ndarray | None = None,
) -> np.ndarray:
    """Label each observation with its outcome bin.

    Parameters
    ----------
    y_true : np.ndarray
        Actual FPL points per player-GW.
    minutes : np.ndarray | None
        Actual minutes played. When given, ``minutes == 0`` forces the
        ``Zeros`` bin (DNP) regardless of points.

    Returns
    -------
    np.ndarray
        Array of bin labels (str), same shape as ``y_true``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    labels = np.empty(len(y_true), dtype=object)
    labels[y_true <= 0] = "Zeros"  # includes negative-point outings
    labels[(y_true >= 1) & (y_true <= 2)] = "Blanks"
    labels[(y_true >= 3) & (y_true <= 4)] = "Tickers"
    labels[y_true >= 5] = "Haulers"
    if minutes is not None:
        labels[np.asarray(minutes, dtype=np.float64) == 0] = "Zeros"
    return labels.astype(str)


def _mae_rmse(err: np.ndarray) -> tuple[float, float]:
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    minutes: np.ndarray | None = None,
) -> dict:
    """Overall + per-outcome-bin error metrics.

    NaN targets (and their predictions) are dropped before scoring.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Actual and predicted points, aligned.
    minutes : np.ndarray | None
        Actual minutes (optional, refines the Zeros/DNP bin).

    Returns
    -------
    dict
        ``{"overall": {mae, rmse, spearman, n}, "bins": {name: {mae, rmse,
        n, share, mean_true, mean_pred}}}``. Bins with no members report
        ``n=0`` and NaN errors.
    """
    from scipy.stats import spearmanr

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[valid], y_pred[valid]
    if minutes is not None:
        minutes = np.asarray(minutes, dtype=np.float64)[valid]

    err = y_pred - y_true
    mae, rmse = _mae_rmse(err)
    rho = float(spearmanr(y_true, y_pred).statistic) if len(y_true) > 2 else float("nan")

    labels = assign_outcome_bins(y_true, minutes)
    bins: dict[str, dict] = {}
    for name in OUTCOME_BINS:
        mask = labels == name
        n = int(mask.sum())
        if n:
            b_mae, b_rmse = _mae_rmse(err[mask])
            bins[name] = {
                "mae": b_mae,
                "rmse": b_rmse,
                "n": n,
                "share": n / len(y_true),
                "mean_true": float(y_true[mask].mean()),
                "mean_pred": float(y_pred[mask].mean()),
            }
        else:
            bins[name] = {
                "mae": float("nan"), "rmse": float("nan"),
                "n": 0, "share": 0.0,
                "mean_true": float("nan"), "mean_pred": float("nan"),
            }

    return {
        "overall": {"mae": mae, "rmse": rmse, "spearman": rho, "n": int(len(y_true))},
        "bins": bins,
    }


def per_gw_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gws: np.ndarray,
    min_gw_size: int = _MIN_GW_SIZE,
) -> dict:
    """Mean Spearman rank correlation computed within each gameweek.

    This is the metric that matters for squad selection: we consume
    *rankings* within a GW, not absolute point values.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Actual and predicted points, aligned.
    gws : np.ndarray
        Gameweek id per row (groups rows into ranking pools).
    min_gw_size : int
        GWs with fewer rows than this are skipped.

    Returns
    -------
    dict
        ``{"mean": float, "per_gw": {gw: rho}}``.
    """
    from scipy.stats import spearmanr

    df = pd.DataFrame({
        "gw": np.asarray(gws),
        "y": np.asarray(y_true, dtype=np.float64),
        "p": np.asarray(y_pred, dtype=np.float64),
    }).dropna()
    per_gw: dict = {}
    for gw, grp in df.groupby("gw"):
        if len(grp) >= min_gw_size:
            per_gw[gw] = float(spearmanr(grp["y"], grp["p"]).statistic)
    mean = float(np.nanmean(list(per_gw.values()))) if per_gw else float("nan")
    return {"mean": mean, "per_gw": per_gw}


def format_report(metrics: dict, label: str = "") -> str:
    """Render a :func:`stratified_metrics` dict as an aligned text table."""
    o = metrics["overall"]
    lines = []
    if label:
        lines.append(f"=== {label} ===")
    lines.append(
        f"Overall: MAE={o['mae']:.4f}  RMSE={o['rmse']:.4f}  "
        f"Spearman={o['spearman']:.4f}  n={o['n']}"
    )
    lines.append(f"  {'bin':<8} {'n':>7} {'share':>6} {'MAE':>7} {'RMSE':>7} "
                 f"{'mean_true':>9} {'mean_pred':>9}")
    for name in OUTCOME_BINS:
        b = metrics["bins"][name]
        lines.append(
            f"  {name:<8} {b['n']:>7} {b['share']:>6.1%} {b['mae']:>7.3f} "
            f"{b['rmse']:>7.3f} {b['mean_true']:>9.2f} {b['mean_pred']:>9.2f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # Worked example: 12 player-GWs with a deliberately hauler-blind model.
    rng = np.random.default_rng(0)
    y_true = np.array([0, 0, 0, 0, 1, 2, 2, 3, 4, 6, 9, 13], dtype=float)
    minutes = np.array([0, 0, 0, 12, 60, 90, 90, 90, 90, 90, 90, 90], dtype=float)
    y_pred = np.clip(y_true * 0.4 + rng.normal(1.0, 0.5, len(y_true)), 0, None)

    m = stratified_metrics(y_true, y_pred, minutes)
    print(format_report(m, label="toy example"))

    gws = np.array([1] * 6 + [2] * 6)
    rho = per_gw_spearman(y_true, y_pred, gws, min_gw_size=3)
    print(f"\nPer-GW Spearman: mean={rho['mean']:.4f}  per_gw={ {k: round(v, 3) for k, v in rho['per_gw'].items()} }")

    # Sanity checks
    labels = assign_outcome_bins(y_true, minutes)
    assert list(labels[:4]) == ["Zeros"] * 4          # DNP + 0-pt cameo
    assert list(labels[4:7]) == ["Blanks"] * 3        # 1-2 pts
    assert list(labels[7:9]) == ["Tickers"] * 2       # 3-4 pts
    assert list(labels[9:]) == ["Haulers"] * 3        # 5+ pts
    assert m["bins"]["Haulers"]["mae"] > m["bins"]["Zeros"]["mae"]  # hauler-blind
    assert abs(sum(b["share"] for b in m["bins"].values()) - 1.0) < 1e-9
    print("\nAll sanity checks passed.")
