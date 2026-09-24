"""Opponent / fixture context features for point prediction.

Computes 8 features per (element, GW) that describe the difficulty and
recent defensive record of the opponent team, plus fixture metadata
(home/away, DGW flag, FDR).

Features
--------
was_home               : 1.0 if home, 0.0 if away (averaged for DGW)
opp_goals_conceded_r5  : Rolling 5-GW avg of goals conceded by opponent team
opp_pts_conceded_r5    : Rolling 5-GW avg of total_points given up by opponent
fdr                    : Fixture Difficulty Rating / 5 (from fixtures.csv)
opp_strength           : Opponent overall strength / 1500 (from teams.csv)
opp_attack_strength    : Opponent attack strength / 1500 (home/away aware)
opp_defence_strength   : Opponent defence strength / 1500 (home/away aware)
is_dgw                 : 1.0 if player's team has >1 fixture in the GW
"""

from __future__ import annotations

import pandas as pd

ROLLING_WINDOW = 5

FEATURE_COLS = [
    "was_home",
    "opp_goals_conceded_r5",
    "opp_pts_conceded_r5",
    "fdr",
    "opp_strength",
    "opp_attack_strength",
    "opp_defence_strength",
    "is_dgw",
]


def _team_goals_conceded_per_gw(merged_gw: pd.DataFrame) -> pd.DataFrame:
    """Compute total goals conceded by each team in each GW.

    A team's goals-conceded in a GW equals the sum of ``goals_conceded``
    across all players on that team in that GW, *divided by the number of
    players* who have a ``goals_conceded`` value (since every outfield player
    on the pitch shares the same value).  In practice we just take the max
    per (team, GW, fixture) to get the match-level goals conceded, then sum
    across fixtures in that GW (for DGW teams).

    Returns a DataFrame with columns (team, GW, team_goals_conceded).
    """
    # Take max goals_conceded per (team, GW, fixture) to get match-level value
    match_gc = (
        merged_gw.groupby(["team", "GW", "fixture"], as_index=False)["goals_conceded"]
        .max()
    )
    # Sum across fixtures in the same GW (handles DGWs)
    team_gw_gc = (
        match_gc.groupby(["team", "GW"], as_index=False)["goals_conceded"]
        .sum()
        .rename(columns={"goals_conceded": "team_goals_conceded"})
    )
    return team_gw_gc


def _team_pts_conceded_per_gw(merged_gw: pd.DataFrame) -> pd.DataFrame:
    """Compute total FPL points conceded TO opponents by each team per GW.

    For each team T in GW g, this is the sum of ``total_points`` earned by
    all players on the *opponent* teams that T faced in GW g.

    Returns a DataFrame with columns (team, GW, team_pts_conceded).
    """
    # For each row in merged_gw we know: the player's team, and opponent_team.
    # Points conceded BY team T = sum of total_points for all players whose
    # team == opponent of T *in the same fixture*.
    #
    # Strategy: for each (fixture, GW), compute total points for each side,
    # then attribute those as "conceded" by the other side.

    # Sum total_points per (team, GW, fixture)
    pts_by_team_fixture = (
        merged_gw.groupby(["team", "GW", "fixture"], as_index=False)["total_points"]
        .sum()
    )

    # We need the opponent for each (team, fixture).  From merged_gw, each
    # player row has (team, opponent_team, fixture).  Take the first row per
    # (team, fixture) to get the mapping.
    team_opp_map = (
        merged_gw[["team", "opponent_team", "GW", "fixture"]]
        .drop_duplicates(subset=["team", "GW", "fixture"])
    )

    # Merge to get opponent's points in the same fixture
    conceded = team_opp_map.merge(
        pts_by_team_fixture.rename(
            columns={"team": "opponent_team", "total_points": "opp_pts"}
        ),
        on=["opponent_team", "GW", "fixture"],
        how="left",
    )
    conceded["opp_pts"] = conceded["opp_pts"].fillna(0)

    # Sum across fixtures in the same GW (DGWs)
    team_gw_pts = (
        conceded.groupby(["team", "GW"], as_index=False)["opp_pts"]
        .sum()
        .rename(columns={"opp_pts": "team_pts_conceded"})
    )
    return team_gw_pts


def _rolling_team_stats(
    team_gc: pd.DataFrame, team_pts: pd.DataFrame
) -> pd.DataFrame:
    """Apply rolling-5-GW average with shift(1) to team-level stats.

    Returns DataFrame with columns
    (team, GW, opp_goals_conceded_r5, opp_pts_conceded_r5).
    The values represent the *opponent's* recent defensive weakness, so when
    we later merge via ``opponent_team`` the name already matches the feature.
    """
    merged = team_gc.merge(team_pts, on=["team", "GW"], how="outer")
    merged = merged.sort_values(["team", "GW"]).reset_index(drop=True)

    # Shift and rolling per team using vectorised groupby transforms
    shifted_gc = merged.groupby("team")["team_goals_conceded"].shift(1)
    shifted_pts = merged.groupby("team")["team_pts_conceded"].shift(1)

    merged["opp_goals_conceded_r5"] = (
        shifted_gc
        .groupby(merged["team"])
        .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    merged["opp_pts_conceded_r5"] = (
        shifted_pts
        .groupby(merged["team"])
        .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean())
    )

    return merged[["team", "GW", "opp_goals_conceded_r5", "opp_pts_conceded_r5"]]


def _fixture_difficulty(
    merged_gw: pd.DataFrame, fixtures_df: pd.DataFrame
) -> pd.DataFrame:
    """Look up FDR from fixtures.csv for each player-fixture row.

    Returns DataFrame with columns (element, GW, fixture, fdr) or an empty
    DataFrame if fixtures_df is empty.
    """
    if fixtures_df.empty:
        return pd.DataFrame(columns=["element", "GW", "fixture", "fdr"])

    # Build a lookup: for each fixture id, get (team_h_difficulty, team_a_difficulty)
    fix_lookup = fixtures_df[["id", "event", "team_h", "team_a",
                               "team_h_difficulty", "team_a_difficulty"]].copy()
    fix_lookup = fix_lookup.rename(columns={"id": "fixture_id", "event": "GW"})

    # Deduplicate player rows to (element, GW, fixture, team, was_home)
    player_rows = merged_gw[["element", "GW", "fixture", "team", "was_home"]].copy()
    player_rows = player_rows.drop_duplicates()

    # Merge on fixture id
    player_rows = player_rows.merge(
        fix_lookup.rename(columns={"fixture_id": "fixture"}),
        on=["fixture", "GW"],
        how="left",
    )

    # FDR: if the player is home, use team_h_difficulty; if away, team_a_difficulty
    player_rows["fdr"] = player_rows.apply(
        lambda r: (
            r["team_h_difficulty"] / 5.0
            if r["was_home"]
            else r["team_a_difficulty"] / 5.0
        )
        if pd.notna(r.get("team_h_difficulty"))
        else float("nan"),
        axis=1,
    )
    return player_rows[["element", "GW", "fixture", "fdr"]]


def _opponent_team_strengths(
    merged_gw: pd.DataFrame, teams_df: pd.DataFrame
) -> pd.DataFrame:
    """Look up opponent team strength ratings from teams.csv.

    Returns DataFrame with columns
    (element, GW, fixture, opp_strength, opp_attack_strength, opp_defence_strength)
    or an empty DataFrame if teams_df is empty.
    """
    if teams_df.empty:
        return pd.DataFrame(
            columns=[
                "element", "GW", "fixture",
                "opp_strength", "opp_attack_strength", "opp_defence_strength",
            ]
        )

    # Prepare team strength lookup keyed by team id
    teams = teams_df[
        ["id", "strength", "strength_attack_home", "strength_attack_away",
         "strength_defence_home", "strength_defence_away"]
    ].copy()

    # Deduplicate player rows
    player_rows = merged_gw[
        ["element", "GW", "fixture", "opponent_team", "was_home"]
    ].drop_duplicates()

    # Merge opponent team info
    player_rows = player_rows.merge(
        teams.rename(columns={"id": "opponent_team"}),
        on="opponent_team",
        how="left",
    )

    player_rows["opp_strength"] = player_rows["strength"] / 1500.0

    # Opponent attack/defence: if the opponent is home (player is away),
    # use _home variants; if opponent is away (player is home), use _away.
    opp_is_home = ~player_rows["was_home"]
    player_rows["opp_attack_strength"] = player_rows["strength_attack_away"] / 1500.0
    player_rows.loc[opp_is_home, "opp_attack_strength"] = (
        player_rows.loc[opp_is_home, "strength_attack_home"] / 1500.0
    )
    player_rows["opp_defence_strength"] = player_rows["strength_defence_away"] / 1500.0
    player_rows.loc[opp_is_home, "opp_defence_strength"] = (
        player_rows.loc[opp_is_home, "strength_defence_home"] / 1500.0
    )

    return player_rows[
        ["element", "GW", "fixture",
         "opp_strength", "opp_attack_strength", "opp_defence_strength"]
    ]


def compute_opponent_features(
    merged_gw: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute 8 opponent/fixture features per (element, GW).

    Parameters
    ----------
    merged_gw : pd.DataFrame
        Per-player-per-fixture data with columns including
        element, GW, was_home, opponent_team, team, total_points,
        goals_conceded, fixture.  May have multiple rows per (element, GW)
        for double gameweeks.
    fixtures_df : pd.DataFrame
        From fixtures.csv with columns event, team_h, team_a,
        team_h_difficulty, team_a_difficulty.  May be empty for old seasons.
    teams_df : pd.DataFrame
        From teams.csv with columns id, strength, strength_attack_home,
        strength_attack_away, strength_defence_home, strength_defence_away.
        May be empty for old seasons.

    Returns
    -------
    pd.DataFrame
        One row per (element, GW) with columns:
        element, GW, was_home, opp_goals_conceded_r5, opp_pts_conceded_r5,
        fdr, opp_strength, opp_attack_strength, opp_defence_strength, is_dgw.
    """
    if merged_gw.empty:
        return pd.DataFrame(columns=["element", "GW"] + FEATURE_COLS)

    # ------------------------------------------------------------------
    # 1. was_home (per fixture row, will be averaged for DGW later)
    # ------------------------------------------------------------------
    rows = merged_gw[
        ["element", "GW", "fixture", "was_home", "opponent_team", "team"]
    ].drop_duplicates()
    rows["was_home_f"] = rows["was_home"].astype(float)

    # ------------------------------------------------------------------
    # 2-3. Rolling team-level stats (goals/pts conceded)
    # ------------------------------------------------------------------
    team_gc = _team_goals_conceded_per_gw(merged_gw)
    team_pts = _team_pts_conceded_per_gw(merged_gw)
    rolling = _rolling_team_stats(team_gc, team_pts)

    # Merge rolling stats via opponent_team
    rows = rows.merge(
        rolling.rename(columns={"team": "opponent_team"}),
        on=["opponent_team", "GW"],
        how="left",
    )

    # ------------------------------------------------------------------
    # 4. FDR from fixtures.csv
    # ------------------------------------------------------------------
    fdr_df = _fixture_difficulty(merged_gw, fixtures_df)
    if not fdr_df.empty:
        rows = rows.merge(fdr_df, on=["element", "GW", "fixture"], how="left")
    else:
        rows["fdr"] = float("nan")

    # ------------------------------------------------------------------
    # 5-7. Opponent team strengths from teams.csv
    # ------------------------------------------------------------------
    strength_df = _opponent_team_strengths(merged_gw, teams_df)
    if not strength_df.empty:
        rows = rows.merge(
            strength_df, on=["element", "GW", "fixture"], how="left"
        )
    else:
        rows["opp_strength"] = float("nan")
        rows["opp_attack_strength"] = float("nan")
        rows["opp_defence_strength"] = float("nan")

    # ------------------------------------------------------------------
    # 8. is_dgw flag
    # ------------------------------------------------------------------
    fixture_counts = (
        merged_gw[["team", "GW", "fixture"]]
        .drop_duplicates()
        .groupby(["team", "GW"], as_index=False)["fixture"]
        .count()
        .rename(columns={"fixture": "n_fixtures"})
    )
    rows = rows.merge(fixture_counts, on=["team", "GW"], how="left")
    rows["is_dgw"] = (rows["n_fixtures"] > 1).astype(float)

    # ------------------------------------------------------------------
    # Aggregate DGW rows: average per-fixture features, one row per
    # (element, GW).
    # ------------------------------------------------------------------
    agg_cols = {
        "was_home_f": "mean",
        "opp_goals_conceded_r5": "mean",
        "opp_pts_conceded_r5": "mean",
        "fdr": "mean",
        "opp_strength": "mean",
        "opp_attack_strength": "mean",
        "opp_defence_strength": "mean",
        "is_dgw": "max",
    }
    result = rows.groupby(["element", "GW"], as_index=False).agg(agg_cols)
    result = result.rename(columns={"was_home_f": "was_home"})

    return result[["element", "GW"] + FEATURE_COLS]
