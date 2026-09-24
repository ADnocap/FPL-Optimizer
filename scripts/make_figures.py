#!/usr/bin/env python3
"""Generate the repo's comparison figures into figures/.

All numbers are from recorded experiment logs (runs/) and the 2026-08-21/22
benchmark + A/B sweep. Sources noted per figure. Regenerate any time with:

    python -X utf8 scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "figures"

# Validated categorical palette (dataviz reference, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, GRAY = "#e34948", "#9a9a94"
SURFACE, INK, MUTED = "#fcfcfb", "#1f1f1e", "#6e6e68"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": "#e8e8e4",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
})


def _bar_labels(ax, bars, fmt="{:.2f}", dy=0.01, fontsize=9):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()),
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize, color=INK)


def fig1_sota() -> None:
    """Ours vs OpenFPL vs FPL Review — OpenFPL's published protocol.

    Source: scripts/benchmark_external.py (train <=2023-24, eval 2024-25
    GW32-38, conditional RMSE) vs numbers published in arXiv 2508.09992.
    """
    bins = ["Zeros\n(didn't play / 0 pts)", "Blanks\n(1–2 pts)",
            "Tickers\n(3–4 pts)", "Haulers\n(5+ pts)"]
    ours = [0.591, 1.066, 1.689, 4.164]
    openfpl = [0.818, 1.291, 1.517, 5.142]
    review = [0.689, 1.189, 1.594, 5.172]

    x = np.arange(len(bins))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    b1 = ax.bar(x - w, ours, w, label="FPL-Optimizer (ours)", color=BLUE)
    b2 = ax.bar(x, openfpl, w, label="OpenFPL (best public)", color=ORANGE)
    b3 = ax.bar(x + w, review, w, label="FPL Review (commercial)", color=AQUA)
    for bars in (b1, b2, b3):
        _bar_labels(ax, bars)
    ax.set_xticks(x, bins)
    ax.set_ylabel("RMSE (points)  —  lower is better")
    ax.set_title(
        "Prediction error vs the published field, by outcome type\n"
        "OpenFPL's exact protocol: train ≤2023-24, evaluate 2024-25 GW32–38",
        loc="left",
    )
    ax.legend(frameon=False, loc="upper left")
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_sota_comparison.png", dpi=180)
    plt.close(fig)


def fig2_kiwi() -> None:
    """Head-to-head vs theFPLkiwi on identical (player, GW) rows.

    Source: scripts/benchmark_external.py kiwi mode (expanding-window folds).
    """
    seasons = ["2021-22\n(38 GWs)", "2022-23\n(27 GWs)", "2023-24\n(4 GWs)"]
    ours_sp = [0.756, 0.798, 0.740]
    kiwi_sp = [0.694, 0.701, 0.617]
    ours_mae = [1.022, 0.884, 1.010]
    kiwi_mae = [1.304, 1.291, 1.518]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    x = np.arange(len(seasons))
    w = 0.34
    b1 = ax1.bar(x - w / 2, ours_sp, w, label="FPL-Optimizer (ours)", color=BLUE)
    b2 = ax1.bar(x + w / 2, kiwi_sp, w, label="theFPLkiwi", color=ORANGE)
    _bar_labels(ax1, b1, "{:.3f}")
    _bar_labels(ax1, b2, "{:.3f}")
    ax1.set_xticks(x, seasons)
    ax1.set_ylabel("Per-GW rank correlation — higher is better")
    ax1.set_title("Ranking quality")
    ax1.set_ylim(0, 0.95)
    ax1.legend(frameon=False, loc="lower left")

    b3 = ax2.bar(x - w / 2, ours_mae, w, color=BLUE)
    b4 = ax2.bar(x + w / 2, kiwi_mae, w, color=ORANGE)
    _bar_labels(ax2, b3)
    _bar_labels(ax2, b4)
    ax2.set_xticks(x, seasons)
    ax2.set_ylabel("MAE (points) — lower is better")
    ax2.set_title("Prediction error")
    ax2.margins(y=0.15)

    fig.suptitle("Head-to-head vs theFPLkiwi — identical player-gameweek rows")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_kiwi_head2head.png", dpi=180)
    plt.close(fig)


def fig3_experiments() -> None:
    """Everything we tried, by its effect on the metric we consume.

    Sources: runs/experiment_losses.log (full 2025-26 eval) and
    runs/experiment_defcon.log + the props A/B (2025-26 GW31-38 eval).
    """
    rows = [
        # (label, delta per-GW Spearman vs its baseline, verdict, panel)
        ("Prop-odds features", 0.0076, "SHIPPED", 1),
        ("DEFCON features", -0.0005, "kept — neutral", 1),
        ("Tweedie loss", 0.0016, "rejected: worse MAE/haulers", 0),
        ("Balanced weights", -0.0854, "rejected", 0),
        ("Tweedie + weights", -0.0629, "rejected", 0),
    ]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for panel, ax, xlim, title in (
        (0, ax0, (-0.115, 0.115),
         "Loss-function experiments\n(eval: full 2025-26 season)"),
        (1, ax1, (-0.0115, 0.0115),
         "Feature experiments\n(eval: 2025-26 GW31–38, in-era training)"),
    ):
        sub = [r for r in rows if r[3] == panel]
        vals = [r[1] for r in sub]
        colors = [BLUE if v > 0.003 else RED if v < -0.003 else GRAY
                  for v in vals]
        y = np.arange(len(sub))
        bars = ax.barh(y, vals, 0.5, color=colors)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_yticks(y, [r[0] for r in sub], fontsize=10)
        ax.set_xlim(*xlim)
        ax.invert_yaxis()
        # Annotations in a fixed column on the side opposite the bar, so
        # they never collide with y-labels or leave the axes.
        for b, (label, v, verdict, _) in zip(bars, sub):
            on_right = v < 0  # negative bars annotate in the empty right half
            ax.annotate(
                f"{v:+.4f}\n{verdict}",
                (xlim[1] * 0.98 if on_right else xlim[0] * 0.98,
                 b.get_y() + b.get_height() / 2),
                ha="right" if on_right else "left", va="center",
                fontsize=8.8, color=INK, linespacing=1.4,
            )
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Δ per-GW rank correlation vs baseline")
        ax.grid(axis="y", visible=False)
    fig.suptitle("The experiment ledger — measured, not assumed")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_experiment_ledger.png", dpi=180)
    plt.close(fig)


def fig4_props() -> None:
    """Props features win the ranking metric in all 8 held-out GWs.

    Source: scripts/experiment_props.py A/B (2025-26 GW31-38).
    """
    gws = list(range(31, 39))
    with_p = [0.738, 0.721, 0.768, 0.726, 0.703, 0.713, 0.734, 0.759]
    without = [0.725, 0.711, 0.759, 0.721, 0.698, 0.707, 0.733, 0.748]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(gws, without, "-o", color=ORANGE, linewidth=2, markersize=7,
            label="Without props")
    ax.plot(gws, with_p, "-o", color=BLUE, linewidth=2, markersize=7,
            label="With props  (wins 8/8)")
    for g, a, b in zip(gws, with_p, without):
        ax.vlines(g, b, a, color=MUTED, linewidth=0.8, alpha=0.5)
    ax.set_xticks(gws, [f"GW{g}" for g in gws])
    ax.set_ylabel("Per-GW rank correlation — higher is better")
    ax.set_title(
        "Betting-market features: better ranking in every held-out gameweek\n"
        "2025-26 GW31–38 · identical training & params · sign test p ≈ 0.004",
        loc="left",
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_props_all8.png", dpi=180)
    plt.close(fig)


def fig5_hauler_gap() -> None:
    """The DEFCON regime shift and how it closes with in-era data.

    Sources: benchmark fold evals (pre-DEFCON seasons), the standard eval
    (cross-era), experiment_defcon.py (in-era GW31-38).
    """
    labels = [
        "Pre-DEFCON seasons\n(2021-22 … 2024-25,\nhonest folds)",
        "2025-26, model trained\nonly on pre-DEFCON data\n(cross-era eval)",
        "2025-26 GW31–38, model\ntrained incl. GW1–30\n(in-era eval)",
    ]
    vals = [3.62, 5.84, 5.23]  # pre-DEFCON = mean of fold hauler RMSEs
    colors = [BLUE, RED, ORANGE]

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    bars = ax.bar(np.arange(3), vals, 0.5, color=colors)
    _bar_labels(ax, bars)
    ax.set_xticks(np.arange(3), labels, fontsize=9.5)
    ax.set_ylabel("Hauler RMSE (5+ point scores) — lower is better")
    ax.set_title("The DEFCON regime shift: big scores got harder to predict —\n"
                 "and in-era training data wins most of it back")
    ax.annotate("regime\nshift", xy=(0.5, 4.7), ha="center", fontsize=9,
                color=MUTED)
    ax.annotate("in-era data\nrecovers ~53%", xy=(1.5, 5.6), ha="center",
                fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_hauler_gap.png", dpi=180)
    plt.close(fig)


def fig6_backtest() -> None:
    """Season-replay net points vs human references.

    Sources: scripts/backtest_season.py (2025-26), README-era backtests
    (2024-25), public season records.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.3), sharey=True)
    for ax, season, bars_, refs in (
        (ax1, "2024-25 (Salah season)",
         [("MILP, 1 transfer/GW", 2918), ("MILP, unconstrained", 3171)],
         [("Best human (~2,810)", 2810), ("Oracle ceiling (3,713)", 3713)]),
        (ax2, "2025-26 (DEFCON season)",
         [("MILP, 1 transfer/GW", 2261), ("MILP, unconstrained", 2566)],
         []),
    ):
        x = np.arange(len(bars_))
        bb = ax.bar(x, [v for _, v in bars_], 0.5, color=[BLUE, AQUA])
        _bar_labels(ax, bb, "{:.0f}")
        ax.set_xticks(x, [l for l, _ in bars_], fontsize=9.5)
        for label, v in refs:
            ax.axhline(v, color=MUTED, linewidth=1.2, linestyle="--")
            ax.annotate(label, (0.99, v), xycoords=("axes fraction", "data"),
                        xytext=(0, 4), textcoords="offset points",
                        ha="right", fontsize=8.5, color=MUTED)
        ax.set_title(season, fontsize=11)
    ax1.set_ylabel("Net season points (no chips played)")
    fig.suptitle("Honest season replays — model trained only on prior seasons")
    fig.text(0.5, 0.005,
             "Chips excluded (+50–120 typical) · no availability filtering "
             "possible historically · direct AQUA-bar labels satisfy contrast relief",
             ha="center", fontsize=8, color=MUTED, alpha=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_season_backtests.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig1_sota()
    fig2_kiwi()
    fig3_experiments()
    fig4_props()
    fig5_hauler_gap()
    fig6_backtest()
    print(f"Wrote 6 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
