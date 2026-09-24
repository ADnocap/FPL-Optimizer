"""Prior-season features from Understat league and FBref full-season aggregates.

These features are constant per player per season (no per-GW variation).
They use **full-season** aggregates from season S-1, which means they are
safe from lookahead bias when used as features for season S.

Produces 12 features per player:
  Understat (6): prev_xg_per90, prev_xa_per90, prev_npxg_per90,
                 prev_shots_per90, prev_key_passes_per90, prev_minutes
  FBref    (6): prev_sot_per90, prev_pass_cmp_pct, prev_prog_dist_per90,
                prev_tkl_int_per90, prev_blocks_per90, prev_gls_per90
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_optimizer.data.collectors.id_mapping import _normalize_name
from fpl_optimizer.prediction.id_resolver import IDResolver

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Season mapping
# -------------------------------------------------------------------------

PREV_SEASON: dict[str, str] = {
    "2017-18": "2016-17",
    "2018-19": "2017-18",
    "2019-20": "2018-19",
    "2020-21": "2019-20",
    "2021-22": "2020-21",
    "2022-23": "2021-22",
    "2023-24": "2022-23",
    "2024-25": "2023-24",
    "2025-26": "2024-25",
    "2026-27": "2025-26",
}

# All 12 output feature column names
PRIOR_FEATURE_COLUMNS: list[str] = [
    "prev_xg_per90",
    "prev_xa_per90",
    "prev_npxg_per90",
    "prev_shots_per90",
    "prev_key_passes_per90",
    "prev_minutes",
    "prev_sot_per90",
    "prev_pass_cmp_pct",
    "prev_prog_dist_per90",
    "prev_tkl_int_per90",
    "prev_blocks_per90",
    "prev_gls_per90",
]

# Minimum minutes threshold for per-90 calculations
_MIN_MINUTES_THRESHOLD = 90


# -------------------------------------------------------------------------
# Column-finding helper for FBref parquets
# -------------------------------------------------------------------------


def _find_col(columns: list[str], *fragments: str) -> str | None:
    """Find first column whose name contains ALL given fragments (case-insensitive).

    Parameters
    ----------
    columns : list[str]
        Available column names.
    *fragments : str
        Substrings that must all appear in the column name.

    Returns
    -------
    str | None
        The matching column name, or None if not found.
    """
    for col in columns:
        col_lower = col.lower()
        if all(f.lower() in col_lower for f in fragments):
            return col
    return None


# -------------------------------------------------------------------------
# Understat league features
# -------------------------------------------------------------------------


def _load_understat_features(
    data_dir: Path,
    prev_season: str,
    id_resolver: IDResolver,
    codes: list[int],
) -> pd.DataFrame:
    """Load understat league JSON for *prev_season* and compute 6 features.

    Returns a DataFrame with columns (code, prev_xg_per90, prev_xa_per90,
    prev_npxg_per90, prev_shots_per90, prev_key_passes_per90, prev_minutes).
    """
    understat_cols = [
        "prev_xg_per90",
        "prev_xa_per90",
        "prev_npxg_per90",
        "prev_shots_per90",
        "prev_key_passes_per90",
        "prev_minutes",
    ]

    json_path = data_dir / "understat" / "league" / f"{prev_season}.json"

    if not json_path.exists():
        logger.warning("Understat league file not found: %s", json_path)
        return _empty_df(codes, understat_cols)

    try:
        with open(json_path, encoding="utf-8") as f:
            players = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read understat league data: %s", exc)
        return _empty_df(codes, understat_cols)

    # Build understat_id -> player dict
    us_by_id: dict[int, dict] = {}
    for p in players:
        try:
            us_by_id[int(p["id"])] = p
        except (ValueError, KeyError):
            continue

    # Map each code to its understat data
    rows: list[dict] = []
    for code in codes:
        us_id = id_resolver.understat_id(code)
        if us_id is None or us_id not in us_by_id:
            rows.append({"code": code})
            continue

        p = us_by_id[us_id]
        try:
            minutes = float(p.get("time", 0))
            xg = float(p.get("xG", 0))
            xa = float(p.get("xA", 0))
            npxg = float(p.get("npxG", 0))
            shots = float(p.get("shots", 0))
            key_passes = float(p.get("key_passes", 0))
        except (ValueError, TypeError):
            rows.append({"code": code})
            continue

        nineties = minutes / 90.0

        row: dict = {"code": code, "prev_minutes": minutes}

        if minutes >= _MIN_MINUTES_THRESHOLD:
            row["prev_xg_per90"] = xg / nineties
            row["prev_xa_per90"] = xa / nineties
            row["prev_npxg_per90"] = npxg / nineties
            row["prev_shots_per90"] = shots / nineties
            row["prev_key_passes_per90"] = key_passes / nineties
        # else: per-90 values stay NaN (not set)

        rows.append(row)

    df = pd.DataFrame(rows)
    # Ensure all columns exist
    for col in understat_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[["code"] + understat_cols]


# -------------------------------------------------------------------------
# FBref features
# -------------------------------------------------------------------------


def _load_fbref_features(
    data_dir: Path,
    prev_season: str,
    id_resolver: IDResolver,
    codes: list[int],
) -> pd.DataFrame:
    """Load FBref parquets for *prev_season* and compute 6 features.

    Returns a DataFrame with columns (code, prev_sot_per90, prev_pass_cmp_pct,
    prev_prog_dist_per90, prev_tkl_int_per90, prev_blocks_per90, prev_gls_per90).
    """
    fbref_cols = [
        "prev_sot_per90",
        "prev_pass_cmp_pct",
        "prev_prog_dist_per90",
        "prev_tkl_int_per90",
        "prev_blocks_per90",
        "prev_gls_per90",
    ]

    fbref_dir = data_dir / "fbref"

    # Build reverse mappings: fbref_id -> code and normalized name -> code
    fbref_id_to_code: dict[str, int] = {}
    name_to_code: dict[str, int] = {}
    for code in codes:
        fb_id = id_resolver.fbref_id(code)
        if fb_id is not None:
            fbref_id_to_code[fb_id] = code

        # Use full name (first + second) for FBref matching, since FBref
        # parquets contain full player names like "Mohamed Salah"
        full_name = id_resolver.player_full_name(code)
        if full_name:
            name_to_code[_normalize_name(full_name)] = code
        # Also add web_name as fallback for single-name players (e.g. "Jorginho")
        pname = id_resolver.player_name(code)
        if pname and pname != "Unknown":
            norm = _normalize_name(pname)
            if norm not in name_to_code:
                name_to_code[norm] = code

    # Initialize result dict: code -> feature dict
    result: dict[int, dict] = {code: {"code": code} for code in codes}

    # --- Shooting: prev_sot_per90 ---
    _extract_shooting(fbref_dir, prev_season, fbref_id_to_code, name_to_code, result)

    # --- Passing: prev_pass_cmp_pct, prev_prog_dist_per90 ---
    _extract_passing(fbref_dir, prev_season, fbref_id_to_code, name_to_code, result)

    # --- Defense: prev_tkl_int_per90, prev_blocks_per90 ---
    _extract_defense(fbref_dir, prev_season, fbref_id_to_code, name_to_code, result)

    # --- Standard: prev_gls_per90 ---
    _extract_standard(fbref_dir, prev_season, fbref_id_to_code, name_to_code, result)

    df = pd.DataFrame(list(result.values()))
    for col in fbref_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[["code"] + fbref_cols]


def _match_fbref_row_to_code(
    row: pd.Series,
    player_col: str,
    fbref_id_to_code: dict[str, int],
    name_to_code: dict[str, int],
) -> int | None:
    """Try to match a FBref row to a stable code.

    Strategy: name matching via _normalize_name.
    """
    # Match by player name
    raw_name = row.get(player_col)
    if pd.notna(raw_name) and isinstance(raw_name, str):
        norm = _normalize_name(raw_name)
        code = name_to_code.get(norm)
        if code is not None:
            return code

    return None


def _read_parquet_safe(path: Path) -> pd.DataFrame | None:
    """Read a parquet file, returning None on any error."""
    if not path.exists():
        logger.warning("FBref parquet not found: %s", path)
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Failed to read FBref parquet %s: %s", path, exc)
        return None


def _get_player_col(df: pd.DataFrame) -> str | None:
    """Find the player name column in a FBref parquet."""
    return _find_col(list(df.columns), "player")


def _get_90s_col(df: pd.DataFrame) -> str | None:
    """Find the 90s column in a FBref parquet."""
    return _find_col(list(df.columns), "90s")


def _extract_shooting(
    fbref_dir: Path,
    prev_season: str,
    fbref_id_to_code: dict[str, int],
    name_to_code: dict[str, int],
    result: dict[int, dict],
) -> None:
    """Extract prev_sot_per90 from shooting parquet."""
    df = _read_parquet_safe(fbref_dir / f"{prev_season}_shooting.parquet")
    if df is None:
        return

    cols = list(df.columns)
    player_col = _get_player_col(df)
    # SoT/90 is provided directly in the data
    sot_per90_col = _find_col(cols, "sot", "/90") or _find_col(cols, "sot", "90")
    nineties_col = _get_90s_col(df)
    sot_col = _find_col(cols, "sot") if sot_per90_col is None else None

    if player_col is None:
        return

    for _, row in df.iterrows():
        code = _match_fbref_row_to_code(row, player_col, fbref_id_to_code, name_to_code)
        if code is None or code not in result:
            continue

        if sot_per90_col is not None:
            val = pd.to_numeric(row.get(sot_per90_col), errors="coerce")
            if pd.notna(val):
                result[code]["prev_sot_per90"] = float(val)
        elif sot_col is not None and nineties_col is not None:
            sot_val = pd.to_numeric(row.get(sot_col), errors="coerce")
            nineties_val = pd.to_numeric(row.get(nineties_col), errors="coerce")
            if pd.notna(sot_val) and pd.notna(nineties_val) and nineties_val >= 1.0:
                result[code]["prev_sot_per90"] = float(sot_val / nineties_val)


def _extract_passing(
    fbref_dir: Path,
    prev_season: str,
    fbref_id_to_code: dict[str, int],
    name_to_code: dict[str, int],
    result: dict[int, dict],
) -> None:
    """Extract prev_pass_cmp_pct and prev_prog_dist_per90 from passing parquet."""
    df = _read_parquet_safe(fbref_dir / f"{prev_season}_passing.parquet")
    if df is None:
        return

    cols = list(df.columns)
    player_col = _get_player_col(df)
    nineties_col = _get_90s_col(df)
    cmp_pct_col = _find_col(cols, "total", "cmp%") or _find_col(cols, "cmp%")
    prog_dist_col = _find_col(cols, "prgdist") or _find_col(cols, "prog", "dist")

    if player_col is None:
        return

    for _, row in df.iterrows():
        code = _match_fbref_row_to_code(row, player_col, fbref_id_to_code, name_to_code)
        if code is None or code not in result:
            continue

        # Pass completion %
        if cmp_pct_col is not None:
            val = pd.to_numeric(row.get(cmp_pct_col), errors="coerce")
            if pd.notna(val):
                result[code]["prev_pass_cmp_pct"] = float(val)

        # Progressive passing distance per 90
        if prog_dist_col is not None and nineties_col is not None:
            dist_val = pd.to_numeric(row.get(prog_dist_col), errors="coerce")
            nineties_val = pd.to_numeric(row.get(nineties_col), errors="coerce")
            if pd.notna(dist_val) and pd.notna(nineties_val) and nineties_val >= 1.0:
                result[code]["prev_prog_dist_per90"] = float(dist_val / nineties_val)


def _extract_defense(
    fbref_dir: Path,
    prev_season: str,
    fbref_id_to_code: dict[str, int],
    name_to_code: dict[str, int],
    result: dict[int, dict],
) -> None:
    """Extract prev_tkl_int_per90 and prev_blocks_per90 from defense parquet."""
    df = _read_parquet_safe(fbref_dir / f"{prev_season}_defense.parquet")
    if df is None:
        return

    cols = list(df.columns)
    player_col = _get_player_col(df)
    nineties_col = _get_90s_col(df)

    # Tkl+Int is often a single column — but verify it has numeric data
    tkl_int_col = _find_col(cols, "tkl+int")
    if tkl_int_col is not None:
        if pd.to_numeric(df[tkl_int_col], errors="coerce").notna().mean() < 0.1:
            tkl_int_col = None  # Column exists but has no numeric data

    # Fallback: separate Tackles_Tkl and Int columns
    tkl_col = _find_col(cols, "tackles", "tkl") if tkl_int_col is None else None
    if tkl_col is not None:
        if pd.to_numeric(df[tkl_col], errors="coerce").notna().mean() < 0.1:
            tkl_col = None  # Column exists but has no numeric data

    int_col = _find_col(cols, "int") if tkl_int_col is None else None
    # Be careful: "int" might also match "Challenges_Tkl%" etc. — use the standalone Int col
    if int_col is not None and "challenge" in int_col.lower():
        int_col = None
    # Try to find the standalone Int column more precisely
    if tkl_int_col is None and int_col is None:
        for c in cols:
            c_lower = c.lower()
            if c_lower.endswith("_int") or c_lower.endswith("int"):
                if "tkl" not in c_lower and "challenge" not in c_lower:
                    int_col = c
                    break
    # Additional fallback: TklW (tackles won) as proxy for total tackles
    # FBref's automated-browser responses have TklW populated but not Tkl
    tklw_col = _find_col(cols, "tklw") if tkl_int_col is None else None

    blocks_col = _find_col(cols, "blocks", "blocks") or _find_col(cols, "blocks")
    # Avoid the sub-columns Blocks_Sh, Blocks_Pass — prefer Blocks_Blocks
    if blocks_col is not None and ("_sh" in blocks_col.lower() or "_pass" in blocks_col.lower()):
        # Try to find the main blocks column
        for c in cols:
            if c.lower().endswith("blocks_blocks") or c.lower() == "blocks_blocks":
                blocks_col = c
                break
    # Verify blocks column has numeric data
    if blocks_col is not None:
        if pd.to_numeric(df[blocks_col], errors="coerce").notna().mean() < 0.1:
            blocks_col = None

    if player_col is None:
        return

    for _, row in df.iterrows():
        code = _match_fbref_row_to_code(row, player_col, fbref_id_to_code, name_to_code)
        if code is None or code not in result:
            continue

        nineties_val = pd.to_numeric(row.get(nineties_col), errors="coerce") if nineties_col else np.nan

        # Tackles + Interceptions per 90
        if tkl_int_col is not None and nineties_col is not None:
            val = pd.to_numeric(row.get(tkl_int_col), errors="coerce")
            if pd.notna(val) and pd.notna(nineties_val) and nineties_val >= 1.0:
                result[code]["prev_tkl_int_per90"] = float(val / nineties_val)
        elif tkl_col is not None and int_col is not None and nineties_col is not None:
            tkl_val = pd.to_numeric(row.get(tkl_col), errors="coerce")
            int_val = pd.to_numeric(row.get(int_col), errors="coerce")
            if (
                pd.notna(tkl_val)
                and pd.notna(int_val)
                and pd.notna(nineties_val)
                and nineties_val >= 1.0
            ):
                result[code]["prev_tkl_int_per90"] = float(
                    (tkl_val + int_val) / nineties_val
                )
        elif int_col is not None and nineties_col is not None:
            # Fallback: use TklW (tackles won) + Int, or just Int if TklW unavailable
            int_val = pd.to_numeric(row.get(int_col), errors="coerce")
            tklw_val = pd.to_numeric(row.get(tklw_col), errors="coerce") if tklw_col else np.nan
            if pd.notna(int_val) and pd.notna(nineties_val) and nineties_val >= 1.0:
                combined = int_val + (tklw_val if pd.notna(tklw_val) else 0.0)
                result[code]["prev_tkl_int_per90"] = float(combined / nineties_val)

        # Blocks per 90
        if blocks_col is not None and nineties_col is not None:
            val = pd.to_numeric(row.get(blocks_col), errors="coerce")
            if pd.notna(val) and pd.notna(nineties_val) and nineties_val >= 1.0:
                result[code]["prev_blocks_per90"] = float(val / nineties_val)


def _extract_standard(
    fbref_dir: Path,
    prev_season: str,
    fbref_id_to_code: dict[str, int],
    name_to_code: dict[str, int],
    result: dict[int, dict],
) -> None:
    """Extract prev_gls_per90 from standard parquet."""
    df = _read_parquet_safe(fbref_dir / f"{prev_season}_standard.parquet")
    if df is None:
        return

    cols = list(df.columns)
    player_col = _get_player_col(df)
    # Per 90 Minutes_Gls is the goals per 90 directly
    gls_per90_col = _find_col(cols, "per 90", "gls") or _find_col(cols, "90", "gls")
    nineties_col = _get_90s_col(df)
    # Fallback: Performance_Gls / 90s
    gls_col = _find_col(cols, "performance", "gls") if gls_per90_col is None else None

    if player_col is None:
        return

    for _, row in df.iterrows():
        code = _match_fbref_row_to_code(row, player_col, fbref_id_to_code, name_to_code)
        if code is None or code not in result:
            continue

        if gls_per90_col is not None:
            val = pd.to_numeric(row.get(gls_per90_col), errors="coerce")
            if pd.notna(val):
                result[code]["prev_gls_per90"] = float(val)
        elif gls_col is not None and nineties_col is not None:
            gls_val = pd.to_numeric(row.get(gls_col), errors="coerce")
            nineties_val = pd.to_numeric(row.get(nineties_col), errors="coerce")
            if pd.notna(gls_val) and pd.notna(nineties_val) and nineties_val >= 1.0:
                result[code]["prev_gls_per90"] = float(gls_val / nineties_val)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _empty_df(codes: list[int], feature_cols: list[str]) -> pd.DataFrame:
    """Return a DataFrame of NaNs for *codes* with the given feature columns."""
    df = pd.DataFrame({"code": codes})
    for col in feature_cols:
        df[col] = np.nan
    return df


# -------------------------------------------------------------------------
# FotMob features (fallback for missing FBref passing/defense data)
# -------------------------------------------------------------------------

# FotMob feature columns that can fill FBref gaps
_FOTMOB_FALLBACK_COLS = ["prev_pass_cmp_pct", "prev_blocks_per90", "prev_prog_dist_per90"]


def _build_fotmob_name_index(
    data_dir: Path,
    id_resolver: IDResolver,
    codes: list[int],
) -> dict[str, int]:
    """Build a rich normalized-name -> code mapping for FotMob matching.

    FotMob uses display names like "Ruben Dias", "Ben White", "Rodri"
    while our ID map has legal names ("Ruben dos Santos Gato Alves Dias",
    "Benjamin White") and web_names ("Rúben", "White").  We index by
    multiple name variants to maximise match rate.
    """
    name_to_code: dict[str, int] = {}

    # Read master ID map directly for first_name / second_name / web_name
    map_path = data_dir / "id_maps" / "master_id_map.csv"
    code_set = set(codes)
    try:
        id_df = pd.read_csv(map_path, encoding="utf-8")
        id_df.columns = [c.strip() for c in id_df.columns]
    except Exception:
        id_df = pd.DataFrame()

    for _, row in id_df.iterrows():
        code = int(row.get("code", -1))
        if code not in code_set:
            continue

        first = str(row.get("first_name", "")).strip()
        second = str(row.get("second_name", "")).strip()
        web = str(row.get("web_name", "")).strip()
        if first == "nan":
            first = ""
        if second == "nan":
            second = ""
        if web == "nan":
            web = ""

        # Variant 1: full legal name ("Ruben dos Santos Gato Alves Dias")
        if first and second:
            name_to_code.setdefault(_normalize_name(f"{first} {second}"), code)

        # Variant 2: web_name ("Rúben", "Salah", "Rodri")
        if web:
            name_to_code.setdefault(_normalize_name(web), code)

        # Variant 3: first_name + web_name ("Ben White" → first="Benjamin" web="White")
        if first and web and web != first:
            name_to_code.setdefault(_normalize_name(f"{first} {web}"), code)

        # Variant 4: first_name alone (catches single-name players like "Rodri")
        if first:
            name_to_code.setdefault(_normalize_name(first), code)

        # Variant 5: second_name alone
        if second and second != first:
            name_to_code.setdefault(_normalize_name(second), code)

        # Variant 6: first_name + last token of second_name
        #   "Bruno" + "Borges Fernandes" → "bruno fernandes"
        #   "Thiago" + "Emiliano da Silva" → "thiago silva"
        if first and second and " " in second:
            last_token = second.split()[-1]
            name_to_code.setdefault(
                _normalize_name(f"{first} {last_token}"), code,
            )

        # Variant 7: extract nickname from first_name if present
        #   "Rodrigo 'Rodri'" → "rodri"
        if first and "'" in first:
            import re
            nick = re.search(r"'([^']+)'", first)
            if nick:
                name_to_code.setdefault(_normalize_name(nick.group(1)), code)

    # Also add from id_resolver as fallback
    for code in codes:
        full_name = id_resolver.player_full_name(code)
        if full_name:
            name_to_code.setdefault(_normalize_name(full_name), code)
        pname = id_resolver.player_name(code)
        if pname and pname != "Unknown":
            name_to_code.setdefault(_normalize_name(pname), code)

    return name_to_code


def _load_fotmob_features(
    data_dir: Path,
    prev_season: str,
    id_resolver: IDResolver,
    codes: list[int],
) -> pd.DataFrame:
    """Load FotMob JSON for *prev_season* and extract 3 features.

    Returns a DataFrame with columns (code, prev_pass_cmp_pct,
    prev_blocks_per90, prev_prog_dist_per90).
    """
    json_path = data_dir / "fotmob" / f"{prev_season}.json"

    if not json_path.exists():
        logger.warning("FotMob file not found: %s", json_path)
        return _empty_df(codes, _FOTMOB_FALLBACK_COLS)

    try:
        with open(json_path, encoding="utf-8") as f:
            season_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read FotMob data: %s", exc)
        return _empty_df(codes, _FOTMOB_FALLBACK_COLS)

    name_to_code = _build_fotmob_name_index(data_dir, id_resolver, codes)
    prefix_idx = _build_prefix_index(name_to_code)

    result: dict[int, dict] = {code: {"code": code} for code in codes}

    # --- accurate_pass: SubStatValue = pass completion % ---
    for entry in season_data.get("accurate_pass", []):
        code = _match_fotmob_name(entry.get("player_name", ""), name_to_code, prefix_idx)
        if code is not None and code in result:
            try:
                # SubStatValue contains the completion percentage (e.g. "82%")
                raw = str(entry.get("sub_stat_value", ""))
                val = float(raw.replace("%", "").strip())
                result[code]["prev_pass_cmp_pct"] = val
            except (ValueError, TypeError):
                pass

    # --- outfielder_block: StatValue = blocks per 90 ---
    for entry in season_data.get("outfielder_block", []):
        code = _match_fotmob_name(entry.get("player_name", ""), name_to_code, prefix_idx)
        if code is not None and code in result:
            try:
                val = float(entry.get("stat_value", ""))
                result[code]["prev_blocks_per90"] = val
            except (ValueError, TypeError):
                pass

    # --- accurate_long_balls: StatValue = long balls per 90 (proxy) ---
    for entry in season_data.get("accurate_long_balls", []):
        code = _match_fotmob_name(entry.get("player_name", ""), name_to_code, prefix_idx)
        if code is not None and code in result:
            try:
                val = float(entry.get("stat_value", ""))
                result[code]["prev_prog_dist_per90"] = val
            except (ValueError, TypeError):
                pass

    df = pd.DataFrame(list(result.values()))
    for col in _FOTMOB_FALLBACK_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[["code"] + _FOTMOB_FALLBACK_COLS]


def _match_fotmob_name(
    name: str,
    name_to_code: dict[str, int],
    _prefix_index: dict[str, int] | None = None,
) -> int | None:
    """Match a FotMob player name to a code via normalized name lookup.

    Falls back to prefix matching for shortened first names
    (e.g. "Ben White" → "Benjamin White").
    """
    if not name:
        return None

    norm = _normalize_name(name)

    # Exact match first
    code = name_to_code.get(norm)
    if code is not None:
        return code

    # Prefix match: for "ben white", check if any key starts with "ben"
    # and ends with " white". Only used for multi-word names.
    parts = norm.split()
    if len(parts) >= 2 and _prefix_index is not None:
        last = parts[-1]
        code = _prefix_index.get(norm)
        if code is not None:
            return code

    return None


def _build_prefix_index(name_to_code: dict[str, int]) -> dict[str, int]:
    """Build a prefix-based index for fuzzy first-name matching.

    Maps "shortened_first last" → code by generating shortened variants
    of multi-word keys. E.g. "benjamin white" generates "ben white",
    "benj white", "benja white", etc.
    """
    prefix_idx: dict[str, int] = {}
    for full_key, code in name_to_code.items():
        parts = full_key.split()
        if len(parts) >= 2:
            first = parts[0]
            rest = " ".join(parts[1:])
            # Generate prefixes of length 3..len(first)-1
            for length in range(3, len(first)):
                short_key = f"{first[:length]} {rest}"
                prefix_idx.setdefault(short_key, code)
    return prefix_idx


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------


def compute_prior_season_features(
    data_dir: Path,
    season: str,
    id_resolver: IDResolver,
) -> pd.DataFrame:
    """Compute 12 prior-season features for all players in *season*.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``understat/league/``, ``fbref/``,
        and ``fotmob/`` sub-directories.
    season : str
        The target season (e.g. ``"2023-24"``). Features are drawn from
        the previous season ``S-1``.
    id_resolver : IDResolver
        Resolver providing code <-> understat/fbref ID mappings.

    Returns
    -------
    pd.DataFrame
        One row per player code, with columns ``code`` and 12 feature
        columns. Players without prior-season data get NaN.
    """
    codes = id_resolver.all_codes_for_season(season)

    prev = PREV_SEASON.get(season)
    if prev is None:
        logger.info("No prior season for %s — returning all NaN.", season)
        return _empty_df(codes, PRIOR_FEATURE_COLUMNS)

    # Load features from all sources
    us_df = _load_understat_features(data_dir, prev, id_resolver, codes)
    fb_df = _load_fbref_features(data_dir, prev, id_resolver, codes)
    fm_df = _load_fotmob_features(data_dir, prev, id_resolver, codes)

    # Merge on code
    merged = us_df.merge(fb_df, on="code", how="outer")

    # Use FotMob as fallback — only fill where FBref has NaN
    for col in _FOTMOB_FALLBACK_COLS:
        if col in merged.columns:
            fm_vals = fm_df.set_index("code")[col]
            mask = merged[col].isna()
            merged.loc[mask, col] = merged.loc[mask, "code"].map(fm_vals)

    # Ensure all expected columns exist
    for col in PRIOR_FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged[["code"] + PRIOR_FEATURE_COLUMNS].reset_index(drop=True)
