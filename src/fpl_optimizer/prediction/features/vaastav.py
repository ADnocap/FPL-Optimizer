"""Rolling features computed from vaastav merged_gw.csv data.

All rolling windows use `.shift(1)` before `.rolling()` to prevent lookahead:
the feature for GW=k uses only data from GW < k.

DGW rows (multiple fixtures in one gameweek) are pre-aggregated by summing
numeric columns per (element, GW) before any rolling computation.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Rolling feature specifications
# ---------------------------------------------------------------------------
# Each entry: (output_col, source_col, window, agg)
# agg is one of: "mean", "sum", "std"

_ROLLING_SPECS: list[tuple[str, str, int, str]] = [
    # Points
    ("pts_rolling_3", "total_points", 3, "mean"),
    ("pts_rolling_5", "total_points", 5, "mean"),
    ("pts_rolling_10", "total_points", 10, "mean"),
    # Minutes
    ("mins_rolling_3", "minutes", 3, "mean"),
    ("mins_rolling_5", "minutes", 5, "mean"),
    ("mins_std_5", "minutes", 5, "std"),
    # Goals
    ("goals_rolling_3", "goals_scored", 3, "sum"),
    ("goals_rolling_5", "goals_scored", 5, "sum"),
    # Assists
    ("assists_rolling_3", "assists", 3, "sum"),
    ("assists_rolling_5", "assists", 5, "sum"),
    # Clean sheets
    ("cs_rolling_5", "clean_sheets", 5, "mean"),
    ("cs_rolling_10", "clean_sheets", 10, "mean"),
    # Bonus
    ("bonus_rolling_5", "bonus", 5, "mean"),
    ("bonus_rolling_10", "bonus", 10, "mean"),
    # BPS
    ("bps_rolling_5", "bps", 5, "mean"),
    ("bps_rolling_10", "bps", 10, "mean"),
    # ICT
    ("ict_rolling_3", "ict_index", 3, "mean"),
    ("ict_rolling_5", "ict_index", 5, "mean"),
    ("ict_rolling_10", "ict_index", 10, "mean"),
    # ICT sub-components
    ("influence_rolling_5", "influence", 5, "mean"),
    ("creativity_rolling_5", "creativity", 5, "mean"),
    ("threat_rolling_5", "threat", 5, "mean"),
    # Saves (GK points source)
    ("saves_rolling_5", "saves", 5, "mean"),
    # Goals conceded (DEF/GK clean sheet proxy)
    ("goals_conceded_rolling_5", "goals_conceded", 5, "mean"),
    # Transfers balance (crowd wisdom)
    ("transfers_balance_rolling_3", "transfers_balance", 3, "mean"),
    # Discipline
    ("yellows_rolling_5", "yellow_cards", 5, "sum"),
    ("reds_rolling_10", "red_cards", 10, "sum"),
    # Starts (rotation signal, 2022-23+ only)
    ("starts_rolling_5", "starts", 5, "mean"),
    # FPL expected stats (2022-23+ only) — post-game, shifted safe
    ("fpl_xg_rolling_3", "expected_goals", 3, "mean"),
    ("fpl_xg_rolling_5", "expected_goals", 5, "mean"),
    ("fpl_xa_rolling_3", "expected_assists", 3, "mean"),
    ("fpl_xa_rolling_5", "expected_assists", 5, "mean"),
    ("fpl_xgi_rolling_3", "expected_goal_involvements", 3, "mean"),
    ("fpl_xgi_rolling_5", "expected_goal_involvements", 5, "mean"),
    ("fpl_xgc_rolling_5", "expected_goals_conceded", 5, "mean"),
    # Transfers in/out separately (crowd wisdom — pre-game)
    ("transfers_in_rolling_3", "transfers_in", 3, "mean"),
    ("transfers_in_rolling_5", "transfers_in", 5, "mean"),
    ("transfers_out_rolling_3", "transfers_out", 3, "mean"),
    ("transfers_out_rolling_5", "transfers_out", 5, "mean"),
    # Detailed stats from old seasons (2016-17 to 2018-19 only).
    # tackles/recoveries ALSO exist 2025-26+ (DEFCON era), so those two
    # rolling features light up again for recent seasons.
    ("fpl_key_passes_rolling_5", "key_passes", 5, "mean"),
    ("tackles_rolling_5", "tackles", 5, "mean"),
    ("completed_passes_rolling_5", "completed_passes", 5, "mean"),
    ("big_chances_created_rolling_5", "big_chances_created", 5, "mean"),
    ("recoveries_rolling_5", "recoveries", 5, "mean"),
    ("dribbles_rolling_5", "dribbles", 5, "mean"),
    # DEFCON-era stats (2025-26+): defensive_contribution is the awarded
    # DEFCON stat per match; cbit/cbirt are derived threshold trackers
    # (DEF scores at 10 CBIT, MID/FWD at 12 CBIRT) built pre-loop.
    ("defcon_rolling_3", "defensive_contribution", 3, "mean"),
    ("defcon_rolling_5", "defensive_contribution", 5, "mean"),
    ("cbi_rolling_5", "clearances_blocks_interceptions", 5, "mean"),
    ("cbit_rolling_5", "cbit", 5, "mean"),
    ("cbirt_rolling_5", "cbirt", 5, "mean"),
]

# Columns that get summed during DGW aggregation
_NUMERIC_SUM_COLS = [
    "total_points", "minutes", "goals_scored", "assists",
    "clean_sheets", "bonus", "bps",
    "saves", "goals_conceded", "yellow_cards", "red_cards",
    "starts", "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
    "transfers_balance", "transfers_in", "transfers_out",
    # Old-season detailed stats (2016-19 only, NaN for newer seasons)
    "key_passes", "tackles", "completed_passes",
    "big_chances_created", "recoveries", "dribbles",
    # DEFCON-era stats (2025-26+, NaN for older seasons)
    "defensive_contribution", "clearances_blocks_interceptions",
]

# Columns that are floats in the raw data and get summed during DGW agg
_FLOAT_SUM_COLS = ["influence", "creativity", "threat", "ict_index"]

# Columns taken from the last row in a DGW group (known before deadline)
# (team is carried through for the team-share feature, not emitted as output)
_LAST_COLS = ["value", "selected", "team"]

# All output feature columns (rolling specs + expanding + non-rolling)
FEATURE_COLUMNS: list[str] = (
    [spec[0] for spec in _ROLLING_SPECS]
    + ["season_avg_pts", "season_total_mins", "games_played",
       "value", "selected_norm", "value_momentum", "selected_momentum",
       "pts_rolling_5_home", "pts_rolling_5_away", "minutes_share_5",
       "team_xgi_share_5"]
)

# was_home normalisation map (raw CSVs mix bools, strings and ints)
_WAS_HOME_MAP = {
    True: True, False: False, "True": True, "False": False, 1: True, 0: False,
}


def compute_vaastav_features(merged_gw: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling features from merged_gw data.

    Parameters
    ----------
    merged_gw : pd.DataFrame
        Raw merged_gw data. Must contain at minimum the columns:
        element, GW, total_points, minutes, goals_scored, assists,
        clean_sheets, bonus, bps, influence, creativity, threat,
        ict_index, value, selected.
        May contain multiple rows per (element, GW) for double gameweeks.
        Optional columns: ``was_home`` (venue-split form) and ``team``
        (team xGI share) — the dependent features are NaN when absent.

    Returns
    -------
    pd.DataFrame
        One row per (element, GW) with columns ``element``, ``GW``,
        and the feature columns listed in :data:`FEATURE_COLUMNS`.
    """
    df = merged_gw.copy()

    # Ensure float types for ICT columns (they come as strings in some seasons)
    for col in _FLOAT_SUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Keep the per-fixture rows: the venue split needs was_home, which the
    # DGW aggregation below discards.
    raw_df = df

    # ------------------------------------------------------------------
    # 1. Aggregate DGW rows: sum numeric columns, take last for non-summed
    # ------------------------------------------------------------------
    sum_cols = [c for c in _NUMERIC_SUM_COLS + _FLOAT_SUM_COLS if c in df.columns]
    last_cols = [c for c in _LAST_COLS if c in df.columns]

    agg_dict: dict[str, str] = {}
    for c in sum_cols:
        agg_dict[c] = "sum"
    for c in last_cols:
        agg_dict[c] = "last"

    df = (
        df.sort_values(["element", "GW"])
        .groupby(["element", "GW"], as_index=False)
        .agg(agg_dict)
    )

    # ------------------------------------------------------------------
    # 2. Sort and compute rolling features per player
    # ------------------------------------------------------------------
    df = df.sort_values(["element", "GW"]).reset_index(drop=True)

    # Derived DEFCON threshold trackers (2025-26+ columns; NaN elsewhere):
    # cbit = CBI + tackles (DEF DEFCON threshold is 10 CBIT);
    # cbirt = cbit + recoveries (MID/FWD threshold is 12 CBIRT).
    if {"clearances_blocks_interceptions", "tackles"} <= set(df.columns):
        df["cbit"] = df["clearances_blocks_interceptions"] + df["tackles"]
        if "recoveries" in df.columns:
            df["cbirt"] = df["cbit"] + df["recoveries"]

    grouped = df.groupby("element")

    # Shifted series cache: {source_col: shifted Series}
    _shifted: dict[str, pd.Series] = {}

    def _get_shifted(col: str) -> pd.Series:
        if col not in _shifted:
            _shifted[col] = grouped[col].shift(1)
        return _shifted[col]

    # Compute each rolling feature (per-group to avoid cross-player contamination)
    for out_col, src_col, window, agg in _ROLLING_SPECS:
        # Guard: if source column doesn't exist (e.g. newer-season-only columns),
        # set output to NaN and skip
        if src_col not in df.columns:
            df[out_col] = float("nan")
            continue

        shifted = _get_shifted(src_col)
        if agg == "mean":
            df[out_col] = shifted.groupby(df["element"]).transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            )
        elif agg == "sum":
            df[out_col] = shifted.groupby(df["element"]).transform(
                lambda s: s.rolling(window=window, min_periods=1).sum()
            )
        elif agg == "std":
            df[out_col] = shifted.groupby(df["element"]).transform(
                lambda s: s.rolling(window=window, min_periods=1).std()
            )
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

    # ------------------------------------------------------------------
    # 3. Expanding (season-level) features — also shifted to avoid lookahead
    # ------------------------------------------------------------------
    shifted_pts = _get_shifted("total_points")
    shifted_mins = _get_shifted("minutes")

    df["season_avg_pts"] = shifted_pts.groupby(df["element"]).transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    df["season_total_mins"] = shifted_mins.groupby(df["element"]).transform(
        lambda s: s.expanding(min_periods=1).sum()
    )

    # games_played: count of prior GWs where the player had minutes > 0
    # shift(1) already done, so we count non-NaN values where minutes > 0
    played_flag = (shifted_mins > 0).astype(float)
    # Replace NaN (from the shift) with 0 for the flag
    played_flag = played_flag.fillna(0.0)
    df["games_played"] = played_flag.groupby(df["element"]).transform(
        lambda s: s.expanding(min_periods=1).sum()
    )

    # ------------------------------------------------------------------
    # 4. Non-rolling features (known at prediction time, from current row)
    # ------------------------------------------------------------------
    # value is already in the df from the aggregation step
    df["selected_norm"] = df["selected"] / 1e7

    # Value and selected momentum (change from prior GW — pre-game signals)
    df["value_momentum"] = df["value"] - grouped["value"].shift(1)
    df["selected_momentum"] = df["selected_norm"] - grouped["selected"].shift(1) / 1e7

    # ------------------------------------------------------------------
    # 5. Minutes share: fraction of available minutes over the last 5 GWs
    # ------------------------------------------------------------------
    # rolling-5 SUM of prior minutes / (5 * 90). Unlike mins_rolling_5 (a
    # mean over observed GWs), this always divides by the full 450, so
    # early-season values reflect true availability, not per-game averages.
    df["minutes_share_5"] = shifted_mins.groupby(df["element"]).transform(
        lambda s: s.rolling(window=5, min_periods=1).sum()
    ) / 450.0

    # ------------------------------------------------------------------
    # 6. Venue-split rolling form (last 5 home / last 5 away matches)
    # ------------------------------------------------------------------
    df = _add_venue_split_features(df, raw_df)

    # ------------------------------------------------------------------
    # 7. Team attacking share: player rolling-5 xGI / team rolling-5 xGI
    # ------------------------------------------------------------------
    df = _add_team_xgi_share(df)

    # ------------------------------------------------------------------
    # 8. Select output columns
    # ------------------------------------------------------------------
    output_cols = ["element", "GW"] + FEATURE_COLUMNS
    # Only keep columns that exist (value is guaranteed)
    output_cols = [c for c in output_cols if c in df.columns]

    return df[output_cols].copy()


def _add_venue_split_features(
    df: pd.DataFrame, raw_df: pd.DataFrame
) -> pd.DataFrame:
    """Add pts_rolling_5_home / pts_rolling_5_away to ``df``.

    Each is the mean of ``total_points`` over the player's last 5 gameweeks
    at that venue *strictly before* the current GW (a DGW with two fixtures
    at the same venue counts as one observation with summed points, matching
    the module's per-GW aggregation). The value is defined at every GW row —
    e.g. at an away GW, pts_rolling_5_home still reflects the last 5 home
    matches played so far. NaN until the player has a prior match at that
    venue.

    ``df`` must be sorted by (element, GW); ``raw_df`` holds the per-fixture
    rows (before DGW aggregation) carrying ``was_home``.
    """
    for venue_flag, out_col in (
        (True, "pts_rolling_5_home"),
        (False, "pts_rolling_5_away"),
    ):
        if "was_home" not in raw_df.columns:
            df[out_col] = float("nan")
            continue

        was_home = raw_df["was_home"].map(_WAS_HOME_MAP)
        venue_obs = (
            raw_df.loc[was_home == venue_flag, ["element", "GW", "total_points"]]
            .groupby(["element", "GW"], as_index=False)["total_points"]
            .sum()
            .sort_values(["element", "GW"])
        )

        if venue_obs.empty:
            df[out_col] = float("nan")
            continue

        # Rolling mean over the last 5 venue observations INCLUDING the
        # current one — the strictly-prior value is recovered below.
        tmp_col = f"_{out_col}_incl"
        venue_obs[tmp_col] = venue_obs.groupby("element")["total_points"].transform(
            lambda s: s.rolling(window=5, min_periods=1).mean()
        )

        df = df.merge(
            venue_obs[["element", "GW", tmp_col]], on=["element", "GW"], how="left"
        )
        # ffill carries the latest venue rolling value forward across
        # non-venue GWs; shift(1) then makes it strictly prior to each GW.
        df[out_col] = df.groupby("element")[tmp_col].transform(
            lambda s: s.ffill().shift(1)
        )
        df = df.drop(columns=[tmp_col])

    return df


def _add_team_xgi_share(df: pd.DataFrame) -> pd.DataFrame:
    """Add team_xgi_share_5: player rolling-5 xGI / team rolling-5 xGI.

    Both numerator and denominator are shifted rolling-5 SUMS of FPL
    ``expected_goal_involvements`` (player rows vs the per-GW total across
    all of the team's players), so the share is the fraction of the team's
    recent attacking output flowing through this player. NaN when the
    source column is missing (pre-2022-23 seasons), the team is unknown,
    or there is no prior data.
    """
    if "team" not in df.columns or "expected_goal_involvements" not in df.columns:
        df["team_xgi_share_5"] = float("nan")
        return df

    # Team per-GW xGI totals, then shifted rolling-5 sum per team
    team_gw = (
        df.groupby(["team", "GW"], as_index=False)["expected_goal_involvements"]
        .sum()
        .rename(columns={"expected_goal_involvements": "_team_xgi"})
        .sort_values(["team", "GW"])
    )
    team_gw["_team_xgi_roll5"] = team_gw.groupby("team")["_team_xgi"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).sum()
    )

    df = df.merge(
        team_gw[["team", "GW", "_team_xgi_roll5"]], on=["team", "GW"], how="left"
    )

    player_roll5 = (
        df.groupby("element")["expected_goal_involvements"]
        .shift(1)
        .groupby(df["element"])
        .transform(lambda s: s.rolling(window=5, min_periods=1).sum())
    )

    # Guard against zero/NaN denominators (early season, xGI all zero)
    denom = df["_team_xgi_roll5"].where(df["_team_xgi_roll5"] > 1e-6)
    df["team_xgi_share_5"] = player_roll5 / denom
    df = df.drop(columns=["_team_xgi_roll5"])

    return df
