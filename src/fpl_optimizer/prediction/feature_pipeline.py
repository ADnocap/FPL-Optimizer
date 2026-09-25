"""Orchestrates all feature modules into a single training-ready DataFrame.

Builds one row per (code, season, gw) with all features + position + target.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fpl_optimizer.prediction.id_resolver import IDResolver, SEASON_TO_COL
from fpl_optimizer.prediction.features.vaastav import compute_vaastav_features
from fpl_optimizer.prediction.features.understat import compute_understat_features
from fpl_optimizer.prediction.features.prior_season import compute_prior_season_features
from fpl_optimizer.prediction.features.opponent import compute_opponent_features
from fpl_optimizer.prediction.features.odds import compute_odds_features
from fpl_optimizer.prediction.features.props import compute_props_features
from fpl_optimizer.prediction.features.players_raw import compute_players_raw_features
from fpl_optimizer.prediction.features.minutes_history import (
    MH_FEATURES,
    MINUTES_LABELS,
    add_depth_and_availability,
    compute_minutes_history_features,
)

logger = logging.getLogger(__name__)

# Bump whenever the pipeline's output changes (new/changed feature columns).
# Feature caches (scripts/train_predictor.py --features-cache) store it and are
# refused when it differs — a stale cache silently trains on the old features.
#   1: legacy (<= 2026-09-24)
#   2: fpl_xp -> fpl_xp_lag, understat same-day fix (season-hardening)
#   3: minutes history (mh_/xs_/ps_/dp_/av_ features + min_total/min_max/mcls labels)
FEATURE_PIPELINE_VERSION = 3

# Position mapping (element_type int -> string label)
_ELEMENT_TYPE_TO_POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


class FeaturePipeline:
    """Orchestrates feature engineering across all seasons.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``raw/``, ``understat/``, ``fbref/``,
        and ``id_maps/`` sub-directories.
    id_resolver : IDResolver
        Cross-source ID resolver.
    seasons : list[str]
        Seasons to process, e.g. ``["2016-17", "2017-18", ...]``.
    """

    def __init__(
        self,
        data_dir: Path,
        id_resolver: IDResolver,
        seasons: list[str],
    ) -> None:
        self.data_dir = data_dir
        self.id_resolver = id_resolver
        self.seasons = seasons

    def build(self) -> pd.DataFrame:
        """Build the full feature DataFrame across all seasons.

        Returns
        -------
        pd.DataFrame
            One row per (code, season, gw) with columns:
            - ``code``, ``season``, ``GW``
            - All feature columns from vaastav, understat, prior_season, opponent
            - ``position`` (str: GK/DEF/MID/FWD)
            - ``total_points`` (target variable)
        """
        all_season_dfs: list[pd.DataFrame] = []

        try:
            from tqdm.auto import tqdm
            season_iter = tqdm(self.seasons, desc="Seasons", unit="season")
        except ImportError:
            season_iter = self.seasons

        for season in season_iter:
            logger.info("Processing season %s...", season)
            try:
                season_df = self._build_season(season)
                if not season_df.empty:
                    all_season_dfs.append(season_df)
                    logger.info(
                        "  %s: %d rows, %d columns",
                        season, len(season_df), len(season_df.columns),
                    )
            except Exception as exc:
                logger.error("Failed to process season %s: %s", season, exc)
                continue

        if not all_season_dfs:
            logger.warning("No data produced for any season")
            return pd.DataFrame()

        result = pd.concat(all_season_dfs, ignore_index=True)

        # Back-fill position for old seasons using cross-season code mapping.
        # A player's FPL position (GK/DEF/MID/FWD) is stable across seasons,
        # so we can copy it from any season that has it.
        if "position" in result.columns and "code" in result.columns:
            missing_pos = result["position"].isna()
            if missing_pos.any():
                # Build code -> position from rows that have position
                code_pos = (
                    result.loc[~missing_pos, ["code", "position"]]
                    .drop_duplicates("code")
                    .set_index("code")["position"]
                )
                filled = result.loc[missing_pos, "code"].map(code_pos)
                result.loc[missing_pos, "position"] = filled.values
                n_filled = filled.notna().sum()
                n_still_missing = filled.isna().sum()
                logger.info(
                    "Position back-fill: %d filled from other seasons, "
                    "%d still missing",
                    n_filled, n_still_missing,
                )

        # Minutes depth-chart ranks (season, GW, team, position) + availability
        # proxies — after the back-fill so 2016-19 rows have a position.
        result = add_depth_and_availability(result)

        logger.info(
            "Feature pipeline complete: %d rows x %d columns",
            len(result), len(result.columns),
        )
        return result

    def _build_season(self, season: str) -> pd.DataFrame:
        """Build features for a single season."""
        raw_dir = self.data_dir / "raw" / season

        # 1. Load merged_gw
        merged_path = raw_dir / "gws" / "merged_gw.csv"
        if not merged_path.exists():
            logger.warning("No merged_gw.csv for %s", season)
            return pd.DataFrame()

        try:
            merged_gw = pd.read_csv(merged_path, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            merged_gw = pd.read_csv(merged_path, encoding="latin-1", on_bad_lines="skip")

        # Ensure numeric types
        for col in ["element", "GW", "total_points", "minutes", "value",
                     "goals_scored", "assists", "clean_sheets", "bonus", "bps",
                     "goals_conceded", "selected",
                     "saves", "yellow_cards", "red_cards", "starts",
                     "expected_goals", "expected_assists",
                     "expected_goal_involvements", "expected_goals_conceded",
                     "transfers_balance"]:
            if col in merged_gw.columns:
                merged_gw[col] = pd.to_numeric(merged_gw[col], errors="coerce")

        for col in ["influence", "creativity", "threat", "ict_index"]:
            if col in merged_gw.columns:
                merged_gw[col] = pd.to_numeric(merged_gw[col], errors="coerce").fillna(0.0)

        # Ensure was_home is boolean
        if "was_home" in merged_gw.columns:
            merged_gw["was_home"] = merged_gw["was_home"].map(
                {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
            ).fillna(False)

        # Ensure 'team' column exists and is consistent with opponent_team
        merged_gw = self._ensure_team_column(merged_gw, raw_dir)

        # 2. Map element_id -> code
        merged_gw["code"] = merged_gw["element"].apply(
            lambda eid: self.id_resolver.code_from_element_id(season, int(eid))
            if pd.notna(eid) else None
        )
        # Drop rows without a code mapping
        n_before = len(merged_gw)
        merged_gw = merged_gw.dropna(subset=["code"])
        merged_gw["code"] = merged_gw["code"].astype(int)
        n_dropped = n_before - len(merged_gw)
        if n_dropped > 0:
            logger.debug("  %s: dropped %d rows without code mapping", season, n_dropped)

        if merged_gw.empty:
            return pd.DataFrame()

        # 2b. Minutes history (as-of features + the minutes labels) — needs the
        # code and numeric team columns set above. A failure must not drop the
        # season: log it and serve NaN (serving_health flags it live).
        try:
            minutes_df = compute_minutes_history_features(
                merged_gw, self.data_dir, season, self.id_resolver,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Minutes history failed for %s: %s", season, exc)
            minutes_df = pd.DataFrame()

        # 3. Compute vaastav rolling features (uses element column)
        vaastav_df = compute_vaastav_features(merged_gw)

        # 4. Compute understat features (needs GW dates)
        gw_dates = self._extract_gw_dates(merged_gw)
        understat_df = compute_understat_features(
            self.data_dir, season, self.id_resolver, gw_dates,
        )

        # 5. Compute prior-season features
        prior_df = compute_prior_season_features(
            self.data_dir, season, self.id_resolver,
        )

        # 6. Compute opponent features
        fixtures_df = self._load_csv_safe(raw_dir / "fixtures.csv")
        teams_df = self._load_csv_safe(raw_dir / "teams.csv")
        opponent_df = compute_opponent_features(merged_gw, fixtures_df, teams_df)

        # 7. Compute odds features (Pinnacle implied probabilities)
        odds_df = compute_odds_features(
            self.data_dir, season, merged_gw, teams_df,
        )

        # 8. Compute player-prop odds features (anytime scorer, assists, SOT)
        props_df = compute_props_features(
            merged_gw, self.data_dir, season, self.id_resolver, teams_df,
        )

        # 9. Compute players_raw features (set pieces, xP)
        players_raw_df = compute_players_raw_features(
            self.data_dir, season, merged_gw,
        )

        # 9. Get position mapping from cleaned_players.csv
        position_df = self._load_positions(raw_dir, merged_gw)

        # 8. Get target (total_points) — aggregate DGW rows
        target_df = (
            merged_gw.groupby(["element", "GW"], as_index=False)
            .agg({"total_points": "sum", "code": "first"})
        )

        # 9. Merge everything together
        # Start with vaastav (has element, GW)
        result = vaastav_df.copy()

        # Add code from merged_gw mapping (element -> code)
        eid_to_code = merged_gw[["element", "code"]].drop_duplicates("element")
        result = result.merge(eid_to_code, on="element", how="left")

        # Add understat features (on code, GW)
        if not understat_df.empty:
            result = result.merge(understat_df, on=["code", "GW"], how="left")

        # Add prior-season features (on code, broadcast to all GWs)
        if not prior_df.empty:
            result = result.merge(prior_df, on="code", how="left")

        # Add opponent features (on element, GW)
        if not opponent_df.empty:
            result = result.merge(opponent_df, on=["element", "GW"], how="left")

        # Add odds features (on element, GW)
        if not odds_df.empty:
            result = result.merge(odds_df, on=["element", "GW"], how="left")

        # Add player-prop features (on element, GW)
        if not props_df.empty:
            result = result.merge(props_df, on=["element", "GW"], how="left")

        # Add players_raw features (on element, GW)
        if not players_raw_df.empty:
            result = result.merge(players_raw_df, on=["element", "GW"], how="left")

        # Add position
        if not position_df.empty:
            result = result.merge(position_df, on="element", how="left")
        else:
            result["position"] = None

        # Add target
        result = result.merge(
            target_df[["element", "GW", "total_points"]].rename(
                columns={"total_points": "target"}
            ),
            on=["element", "GW"],
            how="left",
        )

        # Add minutes history + labels (on element, GW). The depth-chart /
        # availability features are added in build(), after the position
        # back-fill (they rank within team x position).
        if not minutes_df.empty:
            result = result.merge(minutes_df, on=["element", "GW"], how="left")
        else:
            for col in [c for c in MH_FEATURES if not c.startswith(("dp_", "av_"))]:
                result[col] = float("nan")
            for col in MINUTES_LABELS:
                result[col] = float("nan")

        # Add derived interaction features
        result = self._add_derived_features(result)

        # Add season column
        result["season"] = season

        return result

    @staticmethod
    def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add derived interaction features from existing rolling features."""
        eps = 1e-6
        # Points per minute efficiency
        if "pts_rolling_5" in df.columns and "mins_rolling_5" in df.columns:
            df["pts_per_min_5"] = df["pts_rolling_5"] / (df["mins_rolling_5"] + eps)
        # Form momentum (short vs long rolling delta)
        if "pts_rolling_3" in df.columns and "pts_rolling_10" in df.columns:
            df["pts_form_delta"] = df["pts_rolling_3"] - df["pts_rolling_10"]
        # xG overperformance (goals per match vs xG per match)
        if "goals_rolling_5" in df.columns and "xg_rolling_5" in df.columns:
            df["goals_vs_xg_5"] = (df["goals_rolling_5"] / 5.0) - df["xg_rolling_5"]
        # xA overperformance (assists per match vs xA per match)
        if "assists_rolling_5" in df.columns and "xa_rolling_5" in df.columns:
            df["assists_vs_xa_5"] = (df["assists_rolling_5"] / 5.0) - df["xa_rolling_5"]
        # Seasonal position (captures early-season rotation, end-of-season patterns)
        if "GW" in df.columns:
            df["gw_phase"] = df["GW"] / 38.0
        # Transfer demand ratio (crowd buys vs sells — pre-game wisdom)
        if "transfers_in_rolling_3" in df.columns and "transfers_out_rolling_3" in df.columns:
            df["transfer_ratio_3"] = df["transfers_in_rolling_3"] / (
                df["transfers_out_rolling_3"] + eps
            )
        # ICT momentum (short vs long)
        if "ict_rolling_3" in df.columns and "ict_rolling_10" in df.columns:
            df["ict_form_delta"] = df["ict_rolling_3"] - df["ict_rolling_10"]
        # BPS momentum (short vs long)
        if "bps_rolling_5" in df.columns and "bps_rolling_10" in df.columns:
            df["bps_form_delta"] = df["bps_rolling_5"] - df["bps_rolling_10"]
        # Nailedness proxy (fraction of recent games started)
        if "starts_rolling_5" in df.columns and "mins_rolling_5" in df.columns:
            df["nailedness"] = df["starts_rolling_5"]  # already 0-1 scale as mean
        # Bonus rate (bonus per point — identifies consistent bonus earners)
        if "bonus_rolling_5" in df.columns and "pts_rolling_5" in df.columns:
            df["bonus_rate_5"] = df["bonus_rolling_5"] / (df["pts_rolling_5"] + eps)
        return df

    def _extract_gw_dates(self, merged_gw: pd.DataFrame) -> pd.Series:
        """Extract earliest kickoff_time per GW as a Series indexed by GW."""
        if "kickoff_time" not in merged_gw.columns:
            # Fallback: create synthetic dates (1 week apart)
            gws = sorted(merged_gw["GW"].unique())
            base = pd.Timestamp("2020-01-01")
            return pd.Series(
                {gw: base + pd.Timedelta(weeks=int(gw) - 1) for gw in gws}
            )

        merged_gw = merged_gw.copy()
        merged_gw["kickoff_dt"] = pd.to_datetime(
            merged_gw["kickoff_time"], format="mixed", errors="coerce", utc=True,
        )
        gw_dates = merged_gw.groupby("GW")["kickoff_dt"].min()
        # Strip timezone to avoid comparison issues with naive timestamps
        gw_dates = gw_dates.dt.tz_localize(None)
        return gw_dates

    def _load_positions(
        self, raw_dir: Path, merged_gw: pd.DataFrame
    ) -> pd.DataFrame:
        """Load position mapping from cleaned_players.csv or merged_gw."""
        # Try cleaned_players.csv first
        cp_path = raw_dir / "cleaned_players.csv"
        if cp_path.exists():
            try:
                cp = pd.read_csv(cp_path, encoding="utf-8", on_bad_lines="skip")
            except UnicodeDecodeError:
                cp = pd.read_csv(cp_path, encoding="latin-1", on_bad_lines="skip")
            if "element_type" in cp.columns and "id" in cp.columns:
                cp["position"] = cp["element_type"].map(_ELEMENT_TYPE_TO_POS)
                return cp[["id", "position"]].rename(columns={"id": "element"})

        # Fallback: position column in merged_gw
        if "position" in merged_gw.columns:
            pos_df = merged_gw[["element", "position"]].drop_duplicates("element")
            # Normalise non-standard position labels
            pos_df["position"] = pos_df["position"].replace(
                {"GKP": "GK", "AM": "MID"}
            )
            return pos_df

        return pd.DataFrame(columns=["element", "position"])

    @staticmethod
    def _ensure_team_column(merged_gw: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
        """Ensure ``team`` and ``opponent_team`` use the same numeric ID space.

        The opponent feature code groups by ``team`` and merges via
        ``opponent_team``, so they must use the same identifier type.
        Other code (teams.csv, fixtures.csv merges) also expects numeric IDs.

        Handles three cases:
        1. team already numeric → keep as-is
        2. team is string names, opponent_team is numeric → convert team
           to numeric IDs by deriving a name→id mapping from fixture pairs
        3. No team column → derive from opponent_team via fixture structure
        """
        merged_gw = merged_gw.copy()

        has_opp = "opponent_team" in merged_gw.columns
        if has_opp:
            merged_gw["opponent_team"] = pd.to_numeric(
                merged_gw["opponent_team"], errors="coerce"
            )

        has_team = "team" in merged_gw.columns and merged_gw["team"].notna().mean() > 0.5

        if has_team:
            # Check if team is already numeric
            numeric_team = pd.to_numeric(merged_gw["team"], errors="coerce")
            if numeric_team.notna().mean() > 0.5:
                merged_gw["team"] = numeric_team
                return merged_gw

            # team is string — convert to numeric using opponent_team mapping.
            # For fixture F with teams A vs B: players on A have opponent_team
            # equal to B's numeric ID, and vice versa. So A's own numeric ID
            # is what B's players see as opponent_team.
            if has_opp and "fixture" in merged_gw.columns:
                opp_map = (
                    merged_gw[["fixture", "team", "opponent_team"]]
                    .drop_duplicates(subset=["fixture", "team"])
                )
                name_to_id: dict[str, int] = {}
                for _, group in opp_map.groupby("fixture"):
                    pairs = group[["team", "opponent_team"]].values.tolist()
                    if len(pairs) == 2:
                        name_a, opp_id_a = pairs[0]
                        name_b, opp_id_b = pairs[1]
                        try:
                            # A's opponent_team is B's ID, B's opponent_team is A's ID
                            name_to_id[str(name_a)] = int(opp_id_b)
                            name_to_id[str(name_b)] = int(opp_id_a)
                        except (ValueError, TypeError):
                            pass

                if name_to_id:
                    merged_gw["team"] = merged_gw["team"].map(name_to_id)
                    return merged_gw

            # Cannot convert — drop and try to derive below
            merged_gw = merged_gw.drop(columns=["team"])

        # No usable team column — derive from fixture structure
        if has_opp and "fixture" in merged_gw.columns:
            fixture_teams = (
                merged_gw[["fixture", "opponent_team"]]
                .drop_duplicates()
            )
            fixture_groups = fixture_teams.groupby("fixture")["opponent_team"].apply(set)
            team_lookup: dict[tuple, object] = {}
            for fix_id, teams in fixture_groups.items():
                if len(teams) == 2:
                    t_list = list(teams)
                    team_lookup[(fix_id, t_list[0])] = t_list[1]
                    team_lookup[(fix_id, t_list[1])] = t_list[0]

            if team_lookup:
                merged_gw["team"] = merged_gw.apply(
                    lambda r: team_lookup.get((r["fixture"], r["opponent_team"])),
                    axis=1,
                )
                if merged_gw["team"].notna().mean() > 0.5:
                    return merged_gw

        # Last resort
        if "team" not in merged_gw.columns:
            merged_gw["team"] = pd.NA
        return merged_gw

    @staticmethod
    def _load_csv_safe(path: Path) -> pd.DataFrame:
        """Load a CSV, returning empty DataFrame if missing.

        Drops exact full-row duplicates — always source-data defects (real DGW
        rows differ by fixture id) that would double-count in per-GW sums.
        """
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(
                path, encoding="utf-8", on_bad_lines="skip"
            ).drop_duplicates()
        except UnicodeDecodeError:
            try:
                return pd.read_csv(
                    path, encoding="latin-1", on_bad_lines="skip"
                ).drop_duplicates()
            except Exception:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()


class StaleFeatureCache(RuntimeError):
    """A feature cache was built by a different pipeline version / season set."""


def load_or_build_feature_cache(
    path: Path,
    data_dir: Path,
    seasons: list[str],
    rebuild: bool = False,
) -> pd.DataFrame:
    """FeaturePipeline output for *seasons*, cached as a parquet at *path*.

    The multi-season build takes ~10 min; training re-runs reuse the cache.
    The cache records :data:`FEATURE_PIPELINE_VERSION`, the season list and
    the build time in the parquet metadata (``DataFrame.attrs``). A cache from
    another pipeline version or season list raises :class:`StaleFeatureCache`
    — rebuild it (``rebuild=True``) whenever the pipeline changes (bump
    FEATURE_PIPELINE_VERSION with it) AND whenever new GWs have been played
    (the cache is a snapshot of the season files at build time).
    """
    from datetime import datetime, timezone

    path = Path(path)
    if path.exists() and not rebuild:
        df = pd.read_parquet(path)
        meta = dict(df.attrs)
        version = meta.get("feature_pipeline_version")
        if version != FEATURE_PIPELINE_VERSION:
            raise StaleFeatureCache(
                f"{path} was built by feature pipeline v{version}, the code is "
                f"v{FEATURE_PIPELINE_VERSION}: rebuild it (--rebuild-cache)"
            )
        if list(meta.get("seasons", [])) != list(seasons):
            raise StaleFeatureCache(
                f"{path} holds seasons {meta.get('seasons')}, need {list(seasons)}: "
                "rebuild it (--rebuild-cache)"
            )
        logger.info("Loaded feature cache %s (built %s)", path, meta.get("built_utc"))
        return df

    df = FeaturePipeline(data_dir, IDResolver(data_dir), list(seasons)).build()
    df.attrs = {
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "seasons": list(seasons),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)
    logger.info("Saved feature cache %s (%d rows x %d cols)", path, len(df), len(df.columns))
    return df
