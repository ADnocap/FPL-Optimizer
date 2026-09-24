"""Pre-game features from players_raw.csv and merged_gw.csv.

Features extracted:
1. **fpl_xp** — FPL's official pre-match expected points (ep_this from
   bootstrap-static API). Available 2020-21+. Used unshifted — this is
   pre-deadline information.
2. **synthetic_ep** — Reconstructed EP from its known components:
   ``(form_proxy + fixture_offset) * playing_prob * dgw_mult``.
   Available for ALL seasons since it's built from rolling features.
   Fills the gap for 2016-2020 where xP doesn't exist.
3. **Set-piece flags** — Binary indicators for penalty/corner/FK takers.
   From players_raw.csv (end-of-season snapshot, stable within season).
   Available 2020-21+.
4. **playing_prob** — Probability of playing derived from recent minutes
   pattern. A key component of EP that the model can use directly.

The synthetic_ep formula mirrors FPL's own:
    ep_this = (form + fixture_offset) * (chance_of_playing / 100)
We approximate:
    form ≈ pts_rolling_5 (shifted, from vaastav)
    fixture_offset ≈ quantized team-vs-opponent strength differential
    chance_of_playing ≈ derived from mins_rolling_3 pattern
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "fpl_xp",
    "synthetic_ep",
    "playing_prob",
    "fixture_offset",
    "is_penalty_taker",
    "is_corner_taker",
    "is_freekick_taker",
    "set_piece_order_sum",
]


def compute_players_raw_features(
    data_dir: Path,
    season: str,
    merged_gw: pd.DataFrame,
) -> pd.DataFrame:
    """Extract pre-game features from merged_gw and players_raw.csv.

    Parameters
    ----------
    data_dir : Path
        Root data directory.
    season : str
        Season string, e.g. ``"2023-24"``.
    merged_gw : pd.DataFrame
        Per-player-per-fixture data with columns ``element``, ``GW``,
        and optionally ``xP``.

    Returns
    -------
    pd.DataFrame
        One row per (element, GW) with feature columns.
    """
    empty = pd.DataFrame(columns=["element", "GW"] + FEATURE_COLS)
    if merged_gw.empty:
        return empty

    # Get unique (element, GW) pairs
    base = merged_gw[["element", "GW"]].drop_duplicates()

    # ------------------------------------------------------------------
    # 1. fpl_xp from merged_gw (xP column = ep_this, available 2020-21+)
    # ------------------------------------------------------------------
    xp_df = merged_gw[["element", "GW"]].copy()
    if "xP" in merged_gw.columns:
        xp_df["fpl_xp"] = pd.to_numeric(merged_gw["xP"], errors="coerce")
    else:
        xp_df["fpl_xp"] = float("nan")
    # Average across DGW fixtures
    xp_agg = xp_df.groupby(["element", "GW"], as_index=False).agg(
        {"fpl_xp": "mean"}
    )
    base = base.merge(xp_agg, on=["element", "GW"], how="left")

    # ------------------------------------------------------------------
    # 2. Synthetic EP components (available for ALL seasons)
    # ------------------------------------------------------------------
    # Build per-player rolling form and minutes from merged_gw (shifted)
    synth = _build_synthetic_ep(merged_gw)
    if not synth.empty:
        base = base.merge(synth, on=["element", "GW"], how="left")
    else:
        base["synthetic_ep"] = float("nan")
        base["playing_prob"] = float("nan")
        base["fixture_offset"] = float("nan")

    # ------------------------------------------------------------------
    # 3. Set-piece features from players_raw.csv
    # ------------------------------------------------------------------
    sp_df = _load_set_piece_features(data_dir, season)
    if not sp_df.empty:
        base = base.merge(sp_df, on="element", how="left")
    else:
        for col in ["is_penalty_taker", "is_corner_taker",
                     "is_freekick_taker", "set_piece_order_sum"]:
            base[col] = float("nan")

    n_xp = base["fpl_xp"].notna().sum()
    n_synth = base["synthetic_ep"].notna().sum()
    logger.info(
        "players_raw %s: fpl_xp=%d/%d (%.0f%%), synthetic_ep=%d/%d (%.0f%%)",
        season, n_xp, len(base), 100.0 * n_xp / max(len(base), 1),
        n_synth, len(base), 100.0 * n_synth / max(len(base), 1),
    )

    return base[["element", "GW"] + FEATURE_COLS]


def _build_synthetic_ep(merged_gw: pd.DataFrame) -> pd.DataFrame:
    """Build synthetic expected points from EP components.

    Mimics FPL's formula:
        ep_this = (form + fixture_offset) * chance_of_playing / 100

    We approximate each component from available data:
    - form ≈ pts_rolling_5 (shifted rolling avg of total_points)
    - fixture_offset ≈ quantized strength differential (own - opponent)
    - playing_prob ≈ sigmoid of recent minutes pattern
    """
    df = merged_gw.copy()

    # Ensure numeric
    for col in ["total_points", "minutes", "element", "GW"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregate DGW rows first
    sum_cols = ["total_points", "minutes"]
    sum_cols = [c for c in sum_cols if c in df.columns]
    agg_dict = {c: "sum" for c in sum_cols}

    # Track number of fixtures per (element, GW) for DGW multiplier
    df["_n_fixtures"] = 1
    agg_dict["_n_fixtures"] = "sum"

    # Keep opponent strength info (take mean for DGW)
    if "opponent_team" in df.columns:
        agg_dict["opponent_team"] = "first"
    if "team" in df.columns:
        agg_dict["team"] = "first"

    df = (
        df.sort_values(["element", "GW"])
        .groupby(["element", "GW"], as_index=False)
        .agg(agg_dict)
    )

    # ---- Form proxy: shifted rolling 5 avg of total_points ----
    df = df.sort_values(["element", "GW"]).reset_index(drop=True)
    grouped = df.groupby("element")
    shifted_pts = grouped["total_points"].shift(1)
    df["_form"] = shifted_pts.groupby(df["element"]).transform(
        lambda s: s.rolling(window=5, min_periods=1).mean()
    )

    # ---- Playing probability from minutes pattern ----
    shifted_mins = grouped["minutes"].shift(1)
    # Average recent minutes (last 3 GWs)
    avg_mins_3 = shifted_mins.groupby(df["element"]).transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    # Convert to probability: 0 mins → ~0, 45 mins → ~0.75, 90 mins → ~1.0
    # Using a simple sigmoid-like mapping
    df["playing_prob"] = np.clip(avg_mins_3 / 90.0, 0.0, 1.0)

    # ---- Fixture offset: own team strength - opponent strength ----
    # We approximate from the opponent rolling stats since we don't have
    # teams.csv strength ratings in the per-GW data. Use opponent goals
    # conceded as a proxy: more goals conceded = easier fixture = positive offset.
    # For simplicity, use a form-based offset:
    # If opponent concedes more points, fixture is easier.
    if "opponent_team" in df.columns and "team" in df.columns:
        # Build team-level rolling form (team's avg points per player per GW)
        team_form = (
            df.groupby(["team", "GW"], as_index=False)["total_points"]
            .mean()
            .rename(columns={"total_points": "_team_pts_avg"})
        )
        team_form = team_form.sort_values(["team", "GW"])
        shifted_team = team_form.groupby("team")["_team_pts_avg"].shift(1)
        team_form["_team_form"] = shifted_team.groupby(team_form["team"]).transform(
            lambda s: s.rolling(window=5, min_periods=1).mean()
        )

        # Get own team form
        df = df.merge(
            team_form[["team", "GW", "_team_form"]],
            on=["team", "GW"], how="left",
        )
        # Get opponent team form
        df = df.merge(
            team_form[["team", "GW", "_team_form"]].rename(
                columns={"team": "opponent_team", "_team_form": "_opp_form"}
            ),
            on=["opponent_team", "GW"], how="left",
        )
        # Fixture offset: positive when our team is stronger
        strength_diff = df["_team_form"].fillna(0) - df["_opp_form"].fillna(0)
        # Quantize to {-1.0, -0.5, 0.0, +0.5, +1.0} like FPL does
        df["fixture_offset"] = (strength_diff * 2).round() / 2
        df["fixture_offset"] = df["fixture_offset"].clip(-1.0, 1.0)
    else:
        df["fixture_offset"] = 0.0

    # ---- Synthetic EP: (form + fixture_offset) * playing_prob * dgw_mult ----
    form_adjusted = (df["_form"].fillna(0) + df["fixture_offset"])
    dgw_mult = df["_n_fixtures"].clip(lower=1)
    df["synthetic_ep"] = form_adjusted * df["playing_prob"] * dgw_mult

    return df[["element", "GW", "synthetic_ep", "playing_prob", "fixture_offset"]]


def _load_set_piece_features(data_dir: Path, season: str) -> pd.DataFrame:
    """Load set-piece order flags from players_raw.csv."""
    raw_path = data_dir / "raw" / season / "players_raw.csv"
    if not raw_path.exists():
        return pd.DataFrame(columns=["element", "is_penalty_taker",
                                      "is_corner_taker", "is_freekick_taker",
                                      "set_piece_order_sum"])

    try:
        players_raw = pd.read_csv(raw_path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        players_raw = pd.read_csv(raw_path, encoding="latin-1", on_bad_lines="skip")

    sp_cols = {
        "penalties_order": "is_penalty_taker",
        "corners_and_indirect_freekicks_order": "is_corner_taker",
        "direct_freekicks_order": "is_freekick_taker",
    }

    id_col = "id" if "id" in players_raw.columns else "element"
    if id_col not in players_raw.columns:
        return pd.DataFrame()

    sp_df = players_raw[[id_col]].copy().rename(columns={id_col: "element"})

    has_any = False
    for raw_col, feat_col in sp_cols.items():
        if raw_col in players_raw.columns:
            order = pd.to_numeric(players_raw[raw_col], errors="coerce")
            sp_df[feat_col] = (order.notna() & (order <= 2)).astype(float)
            has_any = True
        else:
            sp_df[feat_col] = float("nan")

    if has_any:
        sp_flags = [c for c in sp_cols.values() if c in sp_df.columns]
        sp_df["set_piece_order_sum"] = sp_df[sp_flags].sum(axis=1)
    else:
        sp_df["set_piece_order_sum"] = float("nan")

    return sp_df.drop_duplicates("element")
