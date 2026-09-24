#!/usr/bin/env python3
"""Download FBref player season stats using undetected-chromedriver.

Bypasses Cloudflare automatically — no manual intervention needed.

Usage:
    python scripts/scrape_fbref.py
    python scripts/scrape_fbref.py --seasons 2023-24,2024-25
    python scripts/scrape_fbref.py --stat-types passing,defense --force
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fpl_optimizer.utils.constants import AVAILABLE_SEASONS

logger = logging.getLogger(__name__)

STAT_TYPES = ["stats", "passing", "defense", "shooting"]
STAT_FILE_NAMES = {
    "stats": "standard",
    "passing": "passing",
    "defense": "defense",
    "shooting": "shooting",
}

# Columns that should contain actual data for each stat type (not just metadata).
# Used to validate that scraped data has real content.
STAT_VALIDATION_COLS = {
    "stats": ["gls", "ast", "90s"],
    "passing": ["cmp", "att", "cmp%"],
    "defense": ["tkl", "int", "blocks"],
    "shooting": ["gls", "sh", "sot"],
}

FBREF_COMP_ID = 9

SEASON_TO_FBREF = {
    s: f"20{s.split('-')[0][2:]}-20{s.split('-')[1]}" for s in AVAILABLE_SEASONS
}

MIN_DELAY = 8


def build_url(fbref_season: str, stat_type: str) -> str:
    return (
        f"https://fbref.com/en/comps/{FBREF_COMP_ID}"
        f"/{fbref_season}/{stat_type}/{fbref_season}-Premier-League-Stats"
    )


def wait_for_page(driver, timeout: int = 60) -> bool:
    """Wait for Cloudflare to auto-solve and page to load."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        title = driver.title
        if "Just a moment" not in title:
            return True
        time.sleep(2)
    return False


def _wait_for_stat_cells(driver, timeout: int = 30) -> bool:
    """Poll until stat table <td data-stat> cells contain actual text.

    FBref tables with passing/defense stats have cell values populated by
    deferred JavaScript. This polls until those cells have content, or
    times out after *timeout* seconds.
    """
    from selenium.webdriver.common.by import By

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            cells = driver.find_elements(
                By.CSS_SELECTOR, "table.stats_table td[data-stat]"
            )
            # Check if a reasonable fraction of cells have text content
            if len(cells) >= 10:
                filled = sum(1 for c in cells[:50] if c.text.strip())
                if filled > len(cells[:50]) * 0.3:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _extract_from_page_source(driver, stat_type: str) -> pd.DataFrame | None:
    """Fallback: extract table from page source by stripping HTML comments.

    FBref wraps secondary tables in HTML comments (<!-- ... -->). If the
    primary Selenium element extraction yields empty stat columns, we fall
    back to getting the raw page source, stripping comment delimiters, and
    using pd.read_html to find the target table.
    """
    try:
        html = driver.page_source
    except Exception as exc:
        logger.warning("Failed to get page source: %s", exc)
        return None

    # Strip HTML comment delimiters so commented-out tables become visible
    html = re.sub(r"<!--", "", html)
    html = re.sub(r"-->", "", html)

    try:
        dfs = pd.read_html(io.StringIO(html))
    except Exception as exc:
        logger.warning("pd.read_html failed on page source: %s", exc)
        return None

    if not dfs:
        return None

    # Find the table that has stat-specific columns
    check_cols = STAT_VALIDATION_COLS.get(stat_type, [])
    best_df = None
    best_score = -1

    for df in dfs:
        if isinstance(df.columns, pd.MultiIndex):
            flat_cols = ["_".join(str(c) for c in col).strip("_") for col in df.columns]
        else:
            flat_cols = [str(c) for c in df.columns]

        # Must have a Player column
        has_player = any("player" in c.lower() for c in flat_cols)
        if not has_player:
            continue

        # Score by how many expected stat fragments appear
        score = 0
        for frag in check_cols:
            if any(frag.lower() in c.lower() for c in flat_cols):
                score += 1

        if score > best_score:
            best_score = score
            best_df = df

    if best_df is None:
        return None

    # Flatten MultiIndex columns
    if isinstance(best_df.columns, pd.MultiIndex):
        best_df.columns = [
            "_".join(str(c) for c in col).strip("_") for col in best_df.columns
        ]

    # Remove header rows repeated in the body
    rk_col = [c for c in best_df.columns if "Rk" in str(c)]
    if rk_col:
        best_df = best_df[best_df[rk_col[0]] != "Rk"]
        best_df = best_df.dropna(subset=[rk_col[0]])

    return best_df


def _validate_stat_columns(df: pd.DataFrame, stat_type: str) -> bool:
    """Check that stat-specific columns have >50% numeric (non-null) values.

    Returns True if the DataFrame passes validation. Checks that values are
    actually numeric, not just non-null strings like "None" or repeated headers.
    """
    check_frags = STAT_VALIDATION_COLS.get(stat_type, [])
    if not check_frags:
        return True

    cols = [str(c) for c in df.columns]

    found_any = False
    for frag in check_frags:
        matching = [c for c in cols if frag.lower() in c.lower()]
        for mc in matching:
            found_any = True
            # Convert to numeric to filter out string "None" and header text
            numeric_col = pd.to_numeric(df[mc], errors="coerce")
            numeric_frac = numeric_col.notna().mean()
            if numeric_frac > 0.5:
                return True

    if not found_any:
        logger.warning(
            "Validation: none of the expected stat columns %s found in %s",
            check_frags,
            cols[:10],
        )
        return False

    logger.warning(
        "Validation failed for %s: all stat columns have <=50%% numeric values",
        stat_type,
    )
    return False


def scrape_table(driver, url: str, stat_type: str) -> pd.DataFrame | None:
    """Navigate to a FBref page and extract the player stats table."""
    driver.get(url)

    if not wait_for_page(driver):
        logger.warning("Cloudflare did not resolve for %s", url)
        return None

    # Wait for stat cell content to be populated by deferred JS
    cells_ready = _wait_for_stat_cells(driver, timeout=30)
    if not cells_ready:
        logger.info(
            "Stat cells not populated after 30s for %s — trying extraction anyway",
            url,
        )

    # --- Primary extraction: Selenium element ---
    from selenium.webdriver.common.by import By

    df = None
    try:
        tables = driver.find_elements(By.CSS_SELECTOR, "table.stats_table")
        target_table = None
        for table in tables:
            headers = table.find_elements(By.TAG_NAME, "th")
            for h in headers:
                if h.text.strip() == "Player":
                    target_table = table
                    break
            if target_table:
                break

        if not target_table:
            if tables:
                target_table = tables[-1]
            else:
                logger.warning("No stats table found at %s", url)

        if target_table:
            table_html = target_table.get_attribute("outerHTML")
            dfs = pd.read_html(io.StringIO(table_html))
            if dfs:
                df = dfs[0]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [
                        "_".join(str(c) for c in col).strip("_")
                        for col in df.columns
                    ]
                rk_col = [c for c in df.columns if "Rk" in str(c)]
                if rk_col:
                    df = df[df[rk_col[0]] != "Rk"]
                    df = df.dropna(subset=[rk_col[0]])
    except Exception as exc:
        logger.warning("Primary extraction failed for %s: %s", url, exc)

    # Validate primary extraction
    if df is not None and not df.empty and _validate_stat_columns(df, stat_type):
        return df

    # --- Fallback: page source with comment stripping ---
    logger.info("Primary extraction empty/invalid for %s — trying comment fallback", url)
    df_fallback = _extract_from_page_source(driver, stat_type)

    if df_fallback is not None and not df_fallback.empty:
        if _validate_stat_columns(df_fallback, stat_type):
            return df_fallback
        else:
            logger.warning(
                "Fallback extraction also failed validation for %s", url
            )
            return None

    logger.warning("Both primary and fallback extraction failed for %s", url)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape FBref stats")
    parser.add_argument(
        "--seasons", type=str, default=None,
        help="Comma-separated seasons. Default: all",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Base data directory (default: data/)",
    )
    parser.add_argument(
        "--delay", type=int, default=MIN_DELAY,
        help=f"Seconds between requests (default: {MIN_DELAY})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if cached file exists",
    )
    parser.add_argument(
        "--stat-types", type=str, default=None,
        help="Comma-separated stat types to download (e.g. passing,defense). Default: all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    seasons = args.seasons.split(",") if args.seasons else list(AVAILABLE_SEASONS)
    fbref_dir = Path(args.data_dir) / "fbref"
    fbref_dir.mkdir(parents=True, exist_ok=True)

    # Filter stat types if specified
    if args.stat_types:
        requested_types = [s.strip() for s in args.stat_types.split(",")]
        stat_items = {
            k: v for k, v in STAT_FILE_NAMES.items() if v in requested_types
        }
        if not stat_items:
            logger.error(
                "No valid stat types found in %r. Available: %s",
                args.stat_types,
                ", ".join(STAT_FILE_NAMES.values()),
            )
            return
    else:
        stat_items = STAT_FILE_NAMES

    tasks: list[tuple[str, str, str, Path]] = []
    for season in seasons:
        fb_season = SEASON_TO_FBREF.get(season)
        if not fb_season:
            logger.error("Unknown season: %s", season)
            continue
        for stat_url_type, stat_file_name in stat_items.items():
            dest = fbref_dir / f"{season}_{stat_file_name}.parquet"
            if not args.force and dest.exists() and dest.stat().st_size > 0:
                logger.info("Cached: %s", dest.name)
                continue
            tasks.append((season, fb_season, stat_url_type, dest))

    if not tasks:
        print("All FBref data already cached!")
        return

    print(f"{len(tasks)} tables to download")
    print("Launching browser...\n")

    import undetected_chromedriver as uc

    options = uc.ChromeOptions()
    driver = uc.Chrome(options=options, version_main=145)

    try:
        ok = 0
        failed = 0
        for season, fb_season, stat_type, dest in tqdm(
            tasks, desc="FBref tables", unit="table"
        ):
            url = build_url(fb_season, stat_type)
            stat_name = STAT_FILE_NAMES[stat_type]
            logger.info("Fetching %s %s ...", season, stat_name)

            try:
                df = scrape_table(driver, url, stat_type)
                if df is not None and not df.empty:
                    df.to_parquet(dest)
                    logger.info("Saved %s (%d rows)", dest.name, len(df))
                    ok += 1
                else:
                    logger.warning("Empty/invalid table for %s %s — NOT saved", season, stat_name)
                    failed += 1
            except Exception as exc:
                logger.error("Failed %s %s: %s", season, stat_name, exc)
                failed += 1

            time.sleep(args.delay)

        print(f"\nDone: {ok} saved, {failed} failed out of {len(tasks)} total")

    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
