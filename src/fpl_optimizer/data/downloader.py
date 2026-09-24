"""Download CSV data files from the vaastav Fantasy-Premier-League GitHub repo."""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from fpl_optimizer.utils.constants import (
    AVAILABLE_SEASONS,
    SEASON_FILES,
    SEASON_FILES_REQUIRED,
    VAASTAV_BASE_URL,
)

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data") / "raw"


def download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a single file from URL to dest. Returns True on success."""
    if dest.exists():
        logger.debug("Already exists: %s", dest)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        logger.info("Downloaded: %s", dest)
        return True
    except requests.RequestException as e:
        logger.warning("Failed to download %s: %s", url, e)
        return False


def download_season(season: str, data_dir: Path = DEFAULT_DATA_DIR) -> bool:
    """Download all data files for a single season. Returns True if all succeeded."""
    if season not in AVAILABLE_SEASONS:
        raise ValueError(f"Unknown season: {season}. Available: {AVAILABLE_SEASONS}")

    season_dir = data_dir / season
    required_ok = True
    for file_path in SEASON_FILES:
        url = f"{VAASTAV_BASE_URL}/{season}/{file_path}"
        dest = season_dir / file_path
        ok = download_file(url, dest)
        if not ok and file_path in SEASON_FILES_REQUIRED:
            required_ok = False
    return required_ok


def download_all_seasons(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, bool]:
    """Download data for all available seasons. Returns dict of season->success."""
    results = {}
    for season in AVAILABLE_SEASONS:
        results[season] = download_season(season, data_dir)
    return results


def ensure_season_data(season: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Ensure season data is available, downloading if needed. Returns season dir."""
    season_dir = data_dir / season
    merged_gw = season_dir / "gws" / "merged_gw.csv"
    if not merged_gw.exists():
        logger.info("Downloading data for season %s...", season)
        download_season(season, data_dir)
    if not merged_gw.exists():
        raise FileNotFoundError(
            f"Could not obtain merged_gw.csv for season {season}. "
            f"Expected at: {merged_gw}"
        )
    return season_dir
