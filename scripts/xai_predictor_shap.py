#!/usr/bin/env python3
"""SHAP-based XAI analysis of the LightGBM point predictor.

Generates:
  feature_importance_gain.png   -- LightGBM built-in gain importance (4-panel)
  shap_bar_global.png           -- Mean |SHAP| averaged across positions
  shap_bar_per_position.png     -- Top-15 per GK/DEF/MID/FWD (4-panel)
  shap_dependence_fpl_xp.png    -- fpl_xp value vs SHAP (coloured by pts_rolling_5)
  shap_dependence_pts_rolling.png -- pts_rolling_5 dependence plot
  shap_waterfall_example.png    -- Waterfall for one MID player instance
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

POSITIONS = ["GK", "DEF", "MID", "FWD"]
PALETTE = {"GK": "#4C72B0", "DEF": "#55A868", "MID": "#C44E52", "FWD": "#8172B2"}
PURPLE = "#37006E"


# ── helpers ────────────────────────────────────────────────────────────────────

def _feature_group(name: str) -> str:
    """Assign a feature to a human-readable category."""
    if name == "fpl_xp":
        return "FPL xP"
    if name.startswith("odds_"):
        return "Odds"
    if name.startswith("opp_") or name in ("fdr", "was_home", "is_dgw"):
        return "Opponent / Fixture"
    if name.startswith("prev_"):
        return "Prior Season"
    if name in ("xg_rolling_3", "xg_rolling_5", "xg_rolling_10",
                "xa_rolling_3", "xa_rolling_5", "xa_rolling_10",
                "npxg_rolling_5", "npxg_rolling_10",
                "shots_rolling_5", "key_passes_rolling_5",
                "xgchain_rolling_5", "xgchain_rolling_10",
                "xgbuildup_rolling_5"):
        return "Understat"
    if name in ("value", "selected_norm", "season_avg_pts",
                "season_total_mins", "games_played",
                "gw_phase", "pts_per_min_5", "pts_form_delta",
                "goals_vs_xg_5", "assists_vs_xa_5"):
        return "Global / Derived"
    return "FPL Rolling"


def _group_color(group: str) -> str:
    mapping = {
        "FPL xP":           "#E64B35",
        "FPL Rolling":      "#4DBBD5",
        "Understat":        "#00A087",
        "Opponent / Fixture":"#3C5488",
        "Prior Season":     "#F39B7F",
        "Odds":             "#8491B4",
        "Global / Derived": "#91D1C2",
    }
    return mapping.get(group, "#AAAAAA")


def load_predictor(model_dir: Path):
    from fpl_optimizer.prediction.model import PointPredictor
    log.info("Loading predictor from %s", model_dir)
    return PointPredictor.load(model_dir)


def build_features(data_dir: Path, season: str) -> pd.DataFrame:
    """Run FeaturePipeline for one season. Returns raw feature DataFrame."""
    from fpl_optimizer.prediction.id_resolver import IDResolver
    from fpl_optimizer.prediction.feature_pipeline import FeaturePipeline

    log.info("Building features for %s…", season)
    resolver = IDResolver(data_dir)
    pipeline = FeaturePipeline(data_dir, resolver, seasons=[season])
    df = pipeline.build()
    log.info("  → %d rows, %d cols", len(df), len(df.columns))
    return df


def prepare_X(df: pd.DataFrame, feature_names: list[str], position: str | None = None,
               n_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Slice, fill NaN with 0, optionally subsample."""
    if position:
        df = df[df["position"] == position]
    X = df[feature_names].copy()
    # LightGBM was trained with NaN→0 fill for missing external features
    X = X.fillna(0.0)
    if n_samples and len(X) > n_samples:
        X = X.sample(n_samples, random_state=seed)
    return X


# ── Figure 1: LightGBM built-in gain importance ────────────────────────────────

def plot_gain_importance(predictor, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LightGBM Feature Importance (Gain)\nper Position", fontsize=14, fontweight="bold")

    for ax, pos in zip(axes.flat, POSITIONS):
        if pos not in predictor._models:
            ax.set_visible(False)
            continue
        booster = predictor._models[pos]
        imps = booster.feature_importance(importance_type="gain")
        names = predictor._feature_names
        df_imp = pd.DataFrame({"feature": names, "gain": imps})
        df_imp = df_imp.nlargest(15, "gain")
        groups = [_feature_group(f) for f in df_imp["feature"]]
        colors = [_group_color(g) for g in groups]

        ax.barh(range(len(df_imp)), df_imp["gain"].values, color=colors)
        ax.set_yticks(range(len(df_imp)))
        ax.set_yticklabels(df_imp["feature"].values, fontsize=8)
        ax.set_title(pos, fontsize=11, fontweight="bold", color=PALETTE[pos])
        ax.invert_yaxis()
        ax.set_xlabel("Gain")

    # legend
    from matplotlib.patches import Patch
    groups_all = ["FPL xP", "FPL Rolling", "Understat", "Opponent / Fixture",
                  "Prior Season", "Odds", "Global / Derived"]
    legend_handles = [Patch(facecolor=_group_color(g), label=g) for g in groups_all]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = out_dir / "feature_importance_gain.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


# ── SHAP computation ────────────────────────────────────────────────────────────

def compute_shap(predictor, df: pd.DataFrame, n_samples: int) -> dict[str, np.ndarray]:
    """Returns {position: shap_values (n, n_features)} and X_dict."""
    import shap
    shap_dict: dict[str, np.ndarray] = {}
    X_dict: dict[str, pd.DataFrame] = {}

    for pos in POSITIONS:
        if pos not in predictor._models:
            continue
        X = prepare_X(df, predictor._feature_names, position=pos, n_samples=n_samples)
        if len(X) == 0:
            log.warning("No rows for position %s", pos)
            continue
        booster = predictor._models[pos]
        log.info("SHAP TreeExplainer for %s (%d rows)…", pos, len(X))
        explainer = shap.TreeExplainer(booster)
        sv = explainer.shap_values(X)
        shap_dict[pos] = sv
        X_dict[pos] = X
        log.info("  → SHAP shape %s", sv.shape)

    return shap_dict, X_dict


# ── Figure 2: Global SHAP bar chart ────────────────────────────────────────────

def plot_shap_bar_global(shap_dict: dict, feature_names: list[str], out_dir: Path) -> None:
    # Stack all positions, weight by sample count
    all_sv = []
    for sv in shap_dict.values():
        all_sv.append(np.abs(sv))
    combined = np.concatenate(all_sv, axis=0)
    mean_abs = combined.mean(axis=0)  # (n_features,)

    df_imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df_imp = df_imp.nlargest(20, "mean_abs_shap")
    groups = [_feature_group(f) for f in df_imp["feature"]]
    colors = [_group_color(g) for g in groups]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(df_imp)), df_imp["mean_abs_shap"].values, color=colors)
    ax.set_yticks(range(len(df_imp)))
    ax.set_yticklabels(df_imp["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Global Feature Importance (SHAP)\nAll Positions Combined — Top 20", fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    groups_all = ["FPL xP", "FPL Rolling", "Understat", "Opponent / Fixture",
                  "Prior Season", "Odds", "Global / Derived"]
    handles = [Patch(facecolor=_group_color(g), label=g) for g in groups_all
               if _group_color(g) in colors]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    out = out_dir / "shap_bar_global.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


# ── Figure 3: Per-position SHAP bars ───────────────────────────────────────────

def plot_shap_bar_per_position(shap_dict: dict, feature_names: list[str], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SHAP Feature Importance — Top 15 per Position", fontsize=14, fontweight="bold")

    for ax, pos in zip(axes.flat, POSITIONS):
        if pos not in shap_dict:
            ax.set_visible(False)
            continue
        sv = shap_dict[pos]
        mean_abs = np.abs(sv).mean(axis=0)
        df_imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        df_imp = df_imp.nlargest(15, "mean_abs_shap")
        groups = [_feature_group(f) for f in df_imp["feature"]]
        colors = [_group_color(g) for g in groups]

        ax.barh(range(len(df_imp)), df_imp["mean_abs_shap"].values, color=colors)
        ax.set_yticks(range(len(df_imp)))
        ax.set_yticklabels(df_imp["feature"].values, fontsize=8)
        ax.set_title(pos, fontsize=11, fontweight="bold", color=PALETTE[pos])
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP|")

    from matplotlib.patches import Patch
    groups_all = ["FPL xP", "FPL Rolling", "Understat", "Opponent / Fixture",
                  "Prior Season", "Odds", "Global / Derived"]
    handles = [Patch(facecolor=_group_color(g), label=g) for g in groups_all]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = out_dir / "shap_bar_per_position.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


# ── Figure 4: SHAP dependence — fpl_xp ─────────────────────────────────────────

def plot_shap_dependence(shap_dict: dict, X_dict: dict, feature_names: list[str],
                          feat: str, color_feat: str | None, out_dir: Path,
                          out_name: str) -> None:
    if feat not in feature_names:
        log.warning("Feature %s not in feature_names, skipping dependence plot", feat)
        return

    feat_idx = feature_names.index(feat)
    color_idx = feature_names.index(color_feat) if color_feat and color_feat in feature_names else None

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"SHAP Dependence: {feat}", fontsize=13, fontweight="bold")

    for ax, pos in zip(axes.flat, POSITIONS):
        if pos not in shap_dict:
            ax.set_visible(False)
            continue
        sv = shap_dict[pos]
        X = X_dict[pos]
        x_vals = X[feat].values
        y_vals = sv[:, feat_idx]

        if color_idx is not None and color_feat in X.columns:
            c_vals = X[color_feat].values
            sc = ax.scatter(x_vals, y_vals, c=c_vals, cmap="viridis", alpha=0.4, s=8)
            plt.colorbar(sc, ax=ax, label=color_feat)
        else:
            ax.scatter(x_vals, y_vals, alpha=0.4, s=8, color=PALETTE[pos])

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel(f"SHAP({feat})", fontsize=9)
        ax.set_title(pos, fontsize=10, fontweight="bold", color=PALETTE[pos])

    plt.tight_layout()
    out = out_dir / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


# ── Figure 5: Waterfall for one player ─────────────────────────────────────────

def plot_waterfall(predictor, df: pd.DataFrame, feature_names: list[str],
                   out_dir: Path, position: str = "MID") -> None:
    import shap

    X = prepare_X(df, feature_names, position=position, n_samples=None)
    if len(X) == 0:
        log.warning("No rows for waterfall plot (%s)", position)
        return

    booster = predictor._models[position]
    explainer = shap.TreeExplainer(booster)

    # Pick the player with highest predicted points (most interesting example)
    preds = booster.predict(X)
    best_idx = int(np.argmax(preds))
    x_instance = X.iloc[[best_idx]]
    pred_val = float(preds[best_idx])

    sv_instance = explainer.shap_values(x_instance)[0]  # (n_features,)
    expected = float(explainer.expected_value)

    # Top-12 by |SHAP|
    order = np.argsort(np.abs(sv_instance))[::-1][:12]
    top_names = [feature_names[i] for i in order]
    top_sv = sv_instance[order]

    fig, ax = plt.subplots(figsize=(9, 6))

    bar_colors = ["#E64B35" if v > 0 else "#4DBBD5" for v in top_sv]
    bars = ax.barh(range(len(top_sv)), top_sv, color=bar_colors)
    ax.set_yticks(range(len(top_sv)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value", fontsize=10)
    ax.set_title(
        f"Instance Explanation — {position} player\n"
        f"Predicted: {pred_val:.2f} pts  |  Base value (mean prediction): {expected:.2f} pts",
        fontsize=11, fontweight="bold"
    )

    # annotate values
    for bar, val in zip(bars, top_sv):
        ax.text(val + 0.01 * np.sign(val), bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", fontsize=7)

    plt.tight_layout()
    out = out_dir / "shap_waterfall_example.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SHAP XAI for FPL point predictor")
    parser.add_argument("--model", default="models/point_predictor",
                        help="Path to saved PointPredictor directory")
    parser.add_argument("--season", default="2023-24",
                        help="Season to use for SHAP data (e.g. 2023-24)")
    parser.add_argument("--out-dir", default="reports/figures/predictor",
                        help="Output directory for figures")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Max samples per position for SHAP (for speed)")
    parser.add_argument("--data-dir", default=None,
                        help="Root data directory (auto-detected if not set)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    model_dir = repo_root / args.model
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        from fpl_optimizer.data.downloader import DEFAULT_DATA_DIR
        data_dir = DEFAULT_DATA_DIR.parent if DEFAULT_DATA_DIR.name == "raw" else DEFAULT_DATA_DIR

    # ── Load model ────────────────────────────────────────────────────────────
    predictor = load_predictor(model_dir)
    feature_names = predictor._feature_names
    log.info("Model loaded: %d features, positions: %s",
             len(feature_names), list(predictor._models.keys()))

    # ── Figure 1: Gain importance (no data needed) ────────────────────────────
    log.info("=== Figure 1: Gain importance ===")
    plot_gain_importance(predictor, out_dir)

    # ── Build feature data ────────────────────────────────────────────────────
    log.info("=== Building feature data for %s ===", args.season)
    try:
        df = build_features(data_dir, args.season)
    except Exception as exc:
        log.error("FeaturePipeline failed: %s", exc)
        log.error("Cannot compute SHAP values without feature data. Exiting.")
        sys.exit(1)

    if df.empty:
        log.error("Feature pipeline returned empty DataFrame. Exiting.")
        sys.exit(1)

    # Filter to rows that have a target (played rows)
    if "total_points" in df.columns:
        df = df[df["total_points"].notna()]
    log.info("Using %d rows for SHAP analysis", len(df))

    # ── Compute SHAP ──────────────────────────────────────────────────────────
    log.info("=== Computing SHAP values (n_samples=%d per position) ===", args.n_samples)
    shap_dict, X_dict = compute_shap(predictor, df, n_samples=args.n_samples)

    if not shap_dict:
        log.error("SHAP computation produced no results. Exiting.")
        sys.exit(1)

    # ── Figure 2: Global SHAP bar ─────────────────────────────────────────────
    log.info("=== Figure 2: Global SHAP bar ===")
    plot_shap_bar_global(shap_dict, feature_names, out_dir)

    # ── Figure 3: Per-position SHAP bars ─────────────────────────────────────
    log.info("=== Figure 3: Per-position SHAP bars ===")
    plot_shap_bar_per_position(shap_dict, feature_names, out_dir)

    # ── Figure 4a: SHAP dependence — fpl_xp ──────────────────────────────────
    log.info("=== Figure 4a: fpl_xp dependence ===")
    plot_shap_dependence(shap_dict, X_dict, feature_names,
                          feat="fpl_xp", color_feat="pts_rolling_5",
                          out_dir=out_dir, out_name="shap_dependence_fpl_xp.png")

    # ── Figure 4b: SHAP dependence — pts_rolling_5 ───────────────────────────
    log.info("=== Figure 4b: pts_rolling_5 dependence ===")
    plot_shap_dependence(shap_dict, X_dict, feature_names,
                          feat="pts_rolling_5", color_feat="fpl_xp",
                          out_dir=out_dir, out_name="shap_dependence_pts_rolling.png")

    # ── Figure 5: Waterfall ───────────────────────────────────────────────────
    log.info("=== Figure 5: Waterfall example (MID) ===")
    plot_waterfall(predictor, df, feature_names, out_dir, position="MID")

    log.info("=== All figures saved to %s ===", out_dir)


if __name__ == "__main__":
    main()
