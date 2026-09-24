#!/usr/bin/env python3
"""CLI entry point for data collection.

Usage:
    python scripts/collect_data.py
    python scripts/collect_data.py --sources vaastav,understat
    python scripts/collect_data.py --seasons 2022-23,2023-24
    python scripts/collect_data.py --sources understat --per-match
    python scripts/collect_data.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fpl_optimizer.data.collectors.orchestrator import ALL_SOURCES, DataOrchestrator
from fpl_optimizer.utils.constants import AVAILABLE_SEASONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect FPL data from multiple sources."
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=",".join(ALL_SOURCES),
        help=f"Comma-separated sources ({','.join(ALL_SOURCES)}). Default: all",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Comma-separated seasons (e.g. 2022-23,2023-24). Default: all",
    )
    parser.add_argument(
        "--per-match",
        action="store_true",
        help="Also collect Understat per-match player data (slow, ~2h)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory (default: data/)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Max concurrent workers for parallel downloads (default: 4)",
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

    sources = [s.strip() for s in args.sources.split(",")]
    invalid = [s for s in sources if s not in ALL_SOURCES]
    if invalid:
        print(f"Error: unknown sources: {invalid}. Valid: {list(ALL_SOURCES)}")
        sys.exit(1)

    seasons = None
    if args.seasons:
        seasons = [s.strip() for s in args.seasons.split(",")]
        invalid_seasons = [s for s in seasons if s not in AVAILABLE_SEASONS]
        if invalid_seasons:
            print(f"Error: unknown seasons: {invalid_seasons}. Valid: {AVAILABLE_SEASONS}")
            sys.exit(1)

    orchestrator = DataOrchestrator(
        data_dir=Path(args.data_dir),
        sources=sources,
        seasons=seasons,
        understat_per_match=args.per_match,
        max_workers=args.workers,
    )

    results = orchestrator.run()

    # Exit code: 0 if all succeeded, 1 if any failures
    all_ok = all(
        all(v for v in sr.values())
        for sr in results.values()
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
