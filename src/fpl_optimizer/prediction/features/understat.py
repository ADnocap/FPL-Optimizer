"""Rolling features computed from understat per-match JSON data.

Each player has a JSON file at data/understat/players/{season}/{understat_id}.json
containing a list of match dicts with fields like xG, xA, npxG, shots, key_passes,
date, etc.

Temporal alignment: for each GW, only matches with date < the GW kickoff date are
included. This prevents lookahead bias since we only use information available
before the GW deadline.

All rolling windows use ``min_periods=1`` so that features are produced even when
a player has fewer matches than the window size.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from fpl_optimizer.prediction.id_resolver import IDResolver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rolling feature specifications
# ---------------------------------------------------------------------------
# Each entry: (output_col, source_col, window)
# All aggregations are "mean".

_ROLLING_SPECS: list[tuple[str, str, int]] = [
    # xG
    ("xg_rolling_3", "xG", 3),
    ("xg_rolling_5", "xG", 5),
    ("xg_rolling_10", "xG", 10),
    # xA
    ("xa_rolling_3", "xA", 3),
    ("xa_rolling_5", "xA", 5),
    ("xa_rolling_10", "xA", 10),
    # npxG
    ("npxg_rolling_5", "npxG", 5),
    ("npxg_rolling_10", "npxG", 10),
    # Shots
    ("shots_rolling_5", "shots", 5),
    # Key passes
    ("key_passes_rolling_5", "key_passes", 5),
    # xGChain (value created through pass sequences, broader than xA)
    ("xgchain_rolling_5", "xGChain", 5),
    ("xgchain_rolling_10", "xGChain", 10),
    # xGBuildup (progressive play excluding terminal actions)
    ("xgbuildup_rolling_5", "xGBuildup", 5),
]

# All output feature columns
FEATURE_COLUMNS: list[str] = [spec[0] for spec in _ROLLING_SPECS]


def _load_player_matches(json_path: Path) -> pd.DataFrame:
    """Load and parse a single player's understat JSON file.

    Returns a DataFrame with columns: date, xG, xA, npxG, shots, key_passes,
    xGChain, xGBuildup, sorted by date ascending. Numeric fields are cast
    from strings to float.
    """
    with open(json_path, encoding="utf-8") as f:
        matches = json.load(f)

    _COLS = ["date", "xG", "xA", "npxG", "shots", "key_passes", "xGChain", "xGBuildup"]

    if not matches:
        return pd.DataFrame(columns=_COLS)

    df = pd.DataFrame(matches)

    # Parse date — understat dates can be "2023-08-12" or "2023-08-12 17:30:00"
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)

    # Cast numeric columns from string to float
    for col in ("xG", "xA", "npxG", "shots", "key_passes", "xGChain", "xGBuildup"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    df = df[_COLS].copy()
    df = df.sort_values("date").reset_index(drop=True)

    return df


def compute_understat_features(
    data_dir: Path,
    season: str,
    id_resolver: IDResolver,
    gw_dates: pd.Series,
) -> pd.DataFrame:
    """Compute rolling features from understat per-match data.

    Parameters
    ----------
    data_dir : Path
        Root data directory (contains ``understat/players/{season}/``).
    season : str
        Season string like ``"2023-24"``.
    id_resolver : IDResolver
        Maps stable code -> understat_id.
    gw_dates : pd.Series
        Series indexed by GW number (int), values are datetime (earliest
        kickoff for that GW). Used for temporal alignment: only understat
        matches with date < gw_date are included.

    Returns
    -------
    pd.DataFrame
        Columns: ``code``, ``GW``, plus the feature columns from
        :data:`FEATURE_COLUMNS`. One row per (code, GW) for each player
        that has an understat mapping, even if all features are NaN.
    """
    codes = id_resolver.all_codes_for_season(season)
    gw_numbers = sorted(gw_dates.index)

    if not codes or not gw_numbers:
        return pd.DataFrame(columns=["code", "GW"] + FEATURE_COLUMNS)

    all_rows: list[dict] = []

    try:
        from tqdm.auto import tqdm
        code_iter = tqdm(codes, desc="Understat players", unit="player", leave=False)
    except ImportError:
        code_iter = codes

    for code in code_iter:
        us_id = id_resolver.understat_id(code)

        if us_id is None:
            # No understat mapping — emit NaN rows for all GWs
            for gw in gw_numbers:
                row: dict = {"code": code, "GW": gw}
                for col in FEATURE_COLUMNS:
                    row[col] = float("nan")
                all_rows.append(row)
            continue

        json_path = data_dir / "understat" / "players" / season / f"{us_id}.json"

        if not json_path.exists():
            # JSON file missing — emit NaN rows for all GWs
            for gw in gw_numbers:
                row = {"code": code, "GW": gw}
                for col in FEATURE_COLUMNS:
                    row[col] = float("nan")
                all_rows.append(row)
            logger.debug(
                "No understat JSON for code=%d, understat_id=%d at %s",
                code, us_id, json_path,
            )
            continue

        # Load matches once for this player
        matches_df = _load_player_matches(json_path)

        if matches_df.empty:
            for gw in gw_numbers:
                row = {"code": code, "GW": gw}
                for col in FEATURE_COLUMNS:
                    row[col] = float("nan")
                all_rows.append(row)
            continue

        # For each GW, filter matches before GW date and compute rolling
        for gw in gw_numbers:
            gw_date = pd.Timestamp(gw_dates[gw])
            eligible = matches_df[matches_df["date"] < gw_date]

            row = {"code": code, "GW": gw}

            if eligible.empty:
                for col in FEATURE_COLUMNS:
                    row[col] = float("nan")
            else:
                # Compute rolling features on the eligible matches
                for out_col, src_col, window in _ROLLING_SPECS:
                    series = eligible[src_col]
                    rolling_val = series.rolling(
                        window=window, min_periods=1
                    ).mean().iloc[-1]
                    row[out_col] = rolling_val

            all_rows.append(row)

    result = pd.DataFrame(all_rows, columns=["code", "GW"] + FEATURE_COLUMNS)
    result["code"] = result["code"].astype(int)
    result["GW"] = result["GW"].astype(int)

    logger.info(
        "Understat features: %d rows for %d players across %d GWs",
        len(result), len(codes), len(gw_numbers),
    )

    return result
