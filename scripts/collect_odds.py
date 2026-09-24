#!/usr/bin/env python3
"""Standalone script for collecting historical betting odds.

Fetches Pinnacle h2h odds from The Odds API for seasons 2020-21 onward.
Requires ODDS_API_KEY in .env or environment.

Usage:
    python scripts/collect_odds.py
    python scripts/collect_odds.py --seasons 2023-24,2024-25
    python scripts/collect_odds.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fpl_optimizer.data.collectors.odds import ODDS_SEASONS, OddsCollector


def _load_dotenv() -> None:
    """Load .env file from project root if it exists."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect historical Pinnacle betting odds for EPL."
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help=f"Comma-separated seasons (default: {','.join(ODDS_SEASONS)})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory (default: data/)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _load_dotenv()

    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        print("Error: ODDS_API_KEY not found in environment or .env file")
        sys.exit(1)

    seasons = None
    if args.seasons:
        seasons = [s.strip() for s in args.seasons.split(",")]
        invalid = [s for s in seasons if s not in ODDS_SEASONS]
        if invalid:
            print(f"Error: seasons {invalid} not in odds coverage: {ODDS_SEASONS}")
            sys.exit(1)

    collector = OddsCollector(
        data_dir=Path(args.data_dir),
        api_key=api_key,
    )

    print(f"Collecting odds for seasons: {seasons or ODDS_SEASONS}")
    print(f"Data directory: {args.data_dir}")
    print()

    results = collector.collect_all(seasons=seasons)

    print("\n--- Results ---")
    for season, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {season}: {status}")

    all_ok = all(results.values())
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
