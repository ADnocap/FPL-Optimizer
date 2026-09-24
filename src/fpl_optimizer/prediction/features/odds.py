"""Betting odds features for point prediction.

Computes match-level implied probability features from historical
Pinnacle h2h odds, then maps them to each player via their team.

Anti-lookahead guarantee
------------------------
The odds collector stores a snapshot taken ~2 hours BEFORE the GW's
first kickoff.  At that point, all odds are pre-match and publicly
available — no information leaks from match outcomes.

Features (per element per GW)
-----------------------------
odds_team_win_prob    : Normalised implied probability that the player's
                        team wins their match (averaged for DGW).
odds_team_draw_prob   : Normalised implied draw probability.
odds_team_loss_prob   : Normalised implied probability that the player's
                        team loses.
odds_team_strength    : win_prob - loss_prob  (signed favouritism measure).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from fpl_optimizer.data.collectors.odds import odds_team_to_fpl_name

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "odds_team_win_prob",
    "odds_team_draw_prob",
    "odds_team_loss_prob",
    "odds_team_strength",
]


def _load_season_odds(data_dir: Path, season: str) -> dict[int, list[dict]]:
    """Load odds JSON for a season.

    Returns {gw_int: [match_dict, ...]} or empty dict if unavailable.
    """
    odds_path = data_dir / "odds" / f"{season}.json"
    if not odds_path.exists():
        return {}

    try:
        with open(odds_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Odds %s: failed to load JSON: %s", season, exc)
        return {}

    # Keys are GW numbers as strings
    return {int(gw): matches for gw, matches in raw.items()}


def _build_team_name_to_id(
    merged_gw: pd.DataFrame,
    teams_df: pd.DataFrame,
    data_dir: Path,
    season: str,
) -> dict[str, int]:
    """Build a mapping from team name → numeric team ID.

    Tries three sources in order:
    1. teams.csv (available 2019-20+)
    2. master_team_list.csv (covers all seasons from vaastav)
    3. Empty fallback
    """
    mapping: dict[str, int] = {}

    # Source 1: teams.csv
    if not teams_df.empty and "name" in teams_df.columns and "id" in teams_df.columns:
        for _, row in teams_df.iterrows():
            name = str(row["name"])
            tid = int(row["id"])
            mapping[name] = tid

    # Source 2: master_team_list.csv (fallback for old seasons)
    if not mapping:
        mtl_path = data_dir / "id_maps" / "master_team_list.csv"
        if mtl_path.exists():
            try:
                mtl = pd.read_csv(mtl_path, encoding="utf-8")
                season_rows = mtl[mtl["season"] == season]
                for _, row in season_rows.iterrows():
                    name = str(row["team_name"])
                    tid = int(row["team"])
                    mapping[name] = tid
            except Exception as exc:
                logger.debug("master_team_list.csv load failed: %s", exc)

    return mapping


def _match_odds_to_teams(
    gw_matches: list[dict],
    name_to_id: dict[str, int],
) -> list[dict]:
    """Convert odds matches to rows with numeric team IDs and probabilities.

    Returns a list of dicts, each with:
        team (int), gw (not set here), win_prob, draw_prob, loss_prob
    Two rows per match (one per team).
    """
    def _lookup(raw_name: str) -> int | None:
        # Mapped name first, then the raw source name — FPL renames clubs
        # between seasons (e.g. "Ipswich" 2024-25 vs "Ipswich Town" 2026-27),
        # so a static alias table can't be right for every season.
        tid = name_to_id.get(odds_team_to_fpl_name(raw_name))
        if tid is None:
            tid = name_to_id.get(raw_name)
        return tid

    rows = []
    for match in gw_matches:
        home_id = _lookup(match["home_team"])
        away_id = _lookup(match["away_team"])

        if home_id is None or away_id is None:
            logger.warning(
                "Odds: dropping match, unmapped team name(s): %r (id=%s), %r (id=%s)",
                match["home_team"], home_id, match["away_team"], away_id,
            )
            continue

        home_odds = match["home_odds"]
        draw_odds = match["draw_odds"]
        away_odds = match["away_odds"]

        # Normalise implied probabilities (remove overround)
        raw_home = 1.0 / home_odds
        raw_draw = 1.0 / draw_odds
        raw_away = 1.0 / away_odds
        total = raw_home + raw_draw + raw_away

        if total <= 0:
            continue

        home_win_p = raw_home / total
        draw_p = raw_draw / total
        away_win_p = raw_away / total

        # Home team row
        rows.append({
            "team": home_id,
            "win_prob": home_win_p,
            "draw_prob": draw_p,
            "loss_prob": away_win_p,
        })
        # Away team row
        rows.append({
            "team": away_id,
            "win_prob": away_win_p,
            "draw_prob": draw_p,
            "loss_prob": home_win_p,
        })

    return rows


def compute_odds_features(
    data_dir: Path,
    season: str,
    merged_gw: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute odds-based features per (element, GW).

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``odds/{season}.json``.
    season : str
        Season string, e.g. ``"2023-24"``.
    merged_gw : pd.DataFrame
        Per-player-per-fixture data with columns ``element``, ``GW``,
        ``team`` (numeric ID).
    teams_df : pd.DataFrame
        From teams.csv with columns ``id``, ``name``.

    Returns
    -------
    pd.DataFrame
        One row per (element, GW) with columns:
        element, GW, odds_team_win_prob, odds_team_draw_prob,
        odds_team_loss_prob, odds_team_strength.
    """
    empty = pd.DataFrame(columns=["element", "GW"] + FEATURE_COLS)

    if merged_gw.empty:
        return empty

    season_odds = _load_season_odds(data_dir, season)
    if not season_odds:
        logger.info("Odds %s: no data available, returning NaN features", season)
        # Return DataFrame with NaN for all odds features
        base = merged_gw[["element", "GW"]].drop_duplicates()
        for col in FEATURE_COLS:
            base[col] = float("nan")
        return base

    name_to_id = _build_team_name_to_id(merged_gw, teams_df, data_dir, season)
    if not name_to_id:
        logger.warning("Odds %s: no team name→id mapping available", season)
        base = merged_gw[["element", "GW"]].drop_duplicates()
        for col in FEATURE_COLS:
            base[col] = float("nan")
        return base

    # Build team-level odds DataFrame: (team, GW) → probabilities
    all_team_rows: list[dict] = []
    for gw, matches in season_odds.items():
        for row in _match_odds_to_teams(matches, name_to_id):
            row["GW"] = gw
            all_team_rows.append(row)

    if not all_team_rows:
        logger.warning("Odds %s: no matches mapped to teams", season)
        base = merged_gw[["element", "GW"]].drop_duplicates()
        for col in FEATURE_COLS:
            base[col] = float("nan")
        return base

    team_odds = pd.DataFrame(all_team_rows)

    # For DGWs a team may have multiple matches in one GW — average the probs
    team_odds = (
        team_odds.groupby(["team", "GW"], as_index=False)
        .agg({"win_prob": "mean", "draw_prob": "mean", "loss_prob": "mean"})
    )

    # Rename to final feature names
    team_odds = team_odds.rename(columns={
        "win_prob": "odds_team_win_prob",
        "draw_prob": "odds_team_draw_prob",
        "loss_prob": "odds_team_loss_prob",
    })
    team_odds["odds_team_strength"] = (
        team_odds["odds_team_win_prob"] - team_odds["odds_team_loss_prob"]
    )

    # Map to players via their team
    # Get unique (element, GW, team) from merged_gw
    player_teams = (
        merged_gw[["element", "GW", "team"]]
        .drop_duplicates(subset=["element", "GW"])
    )
    player_teams["team"] = pd.to_numeric(player_teams["team"], errors="coerce")

    result = player_teams.merge(team_odds, on=["team", "GW"], how="left")

    n_with_odds = result["odds_team_win_prob"].notna().sum()
    n_total = len(result)
    logger.info(
        "Odds %s: %d/%d player-GW rows have odds data (%.1f%%)",
        season, n_with_odds, n_total,
        100.0 * n_with_odds / n_total if n_total > 0 else 0,
    )

    return result[["element", "GW"] + FEATURE_COLS]
