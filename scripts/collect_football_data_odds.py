"""Download free h2h odds from football-data.co.uk for a season.

Converts E0.csv into the same JSON layout as data/odds/{season}.json
(dict keyed by GW -> list of {event_id, commence_time, home/away_team,
home/draw/away odds}).  Price preference per match: Pinnacle closing
(PSCH/PSCD/PSCA) > Pinnacle opening (PSH...) > market-average closing
(AvgCH...) > market average (AvgH...).  football-data dropped Pinnacle from
the 2026-27 files, so the live season uses the market average; after
normalising the overround the implied probabilities differ from Pinnacle's
by ~1-2pp.  GW mapping comes from the season's fixtures.csv kickoff dates.

With ``--include-upcoming`` the not-yet-played matches listed in
football-data's ``fixtures.csv`` (pre-match prices) are added too, so the
live season's upcoming GW gets odds features at the deadline.

Usage:
    python scripts/collect_football_data_odds.py --season 2025-26
    python scripts/collect_football_data_odds.py --season 2026-27 --include-upcoming
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{yy}/E0.csv"
UPCOMING_URL = "https://www.football-data.co.uk/fixtures.csv"


def season_to_code(season: str) -> str:
    """'2025-26' -> '2526'"""
    start, end = season.split("-")
    return start[2:] + end


def build_date_to_gw(fixtures_path: Path) -> dict[str, int]:
    """Map 'YYYY-MM-DD' -> most common GW among fixtures that day."""
    fx = pd.read_csv(fixtures_path)
    fx = fx.dropna(subset=["event", "kickoff_time"])
    date_events: dict[str, Counter] = {}
    for _, row in fx.iterrows():
        date = str(row["kickoff_time"])[:10]
        date_events.setdefault(date, Counter())[int(row["event"])] += 1
    return {d: c.most_common(1)[0][0] for d, c in date_events.items()}


def _download_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig", errors="replace")))


def _rows_to_gw_matches(
    df: pd.DataFrame, season: str, date_to_gw: dict[str, int]
) -> tuple[dict[str, list[dict]], int]:
    by_gw: dict[str, list[dict]] = {}
    unmapped = 0
    for _, row in df.iterrows():
        # football-data dates are DD/MM/YYYY (or DD/MM/YY in old files)
        try:
            date = pd.to_datetime(row["Date"], dayfirst=True).strftime("%Y-%m-%d")
        except Exception:
            unmapped += 1
            continue
        gw = date_to_gw.get(date)
        if gw is None:
            # fall back to nearest known fixture date within 3 days
            candidates = [
                (abs((pd.Timestamp(date) - pd.Timestamp(d)).days), g)
                for d, g in date_to_gw.items()
                if abs((pd.Timestamp(date) - pd.Timestamp(d)).days) <= 3
            ]
            if candidates:
                gw = min(candidates)[1]
            else:
                unmapped += 1
                continue

        def _odds(*cols: str) -> float | None:
            for col in cols:
                v = row.get(col)
                if pd.notna(v):
                    return float(v)
            return None

        home = _odds("PSCH", "PSH", "AvgCH", "AvgH")
        draw = _odds("PSCD", "PSD", "AvgCD", "AvgD")
        away = _odds("PSCA", "PSA", "AvgCA", "AvgA")
        if home is None or draw is None or away is None:
            continue
        by_gw.setdefault(str(gw), []).append(
            {
                "event_id": f"fd_{season}_{row['HomeTeam']}_{row['AwayTeam']}",
                "commence_time": f"{date}T00:00:00Z",
                "home_team": str(row["HomeTeam"]),
                "away_team": str(row["AwayTeam"]),
                "home_odds": home,
                "draw_odds": draw,
                "away_odds": away,
                "last_update": "",
            }
        )
    return by_gw, unmapped


def build_season_odds(
    season: str, data_dir: Path, include_upcoming: bool = False
) -> Path:
    """Write data/odds/{season}.json from football-data; returns the path.

    Played matches come from the season's E0.csv; with *include_upcoming*,
    E0 fixtures not yet played are added from fixtures.csv (pre-match
    prices). A played match always wins over an upcoming quote.
    """
    fixtures_path = data_dir / "raw" / season / "fixtures.csv"
    if not fixtures_path.exists():
        raise FileNotFoundError(f"fixtures.csv missing for {season} — collect it first")
    date_to_gw = build_date_to_gw(fixtures_path)

    played = _download_csv(URL_TEMPLATE.format(yy=season_to_code(season)))
    by_gw, unmapped = _rows_to_gw_matches(played, season, date_to_gw)
    n_played = sum(len(v) for v in by_gw.values())

    n_upcoming = 0
    if include_upcoming:
        up = _download_csv(UPCOMING_URL)
        if "Div" in up.columns:
            up = up[up["Div"] == "E0"]
        seen = {m["event_id"] for ms in by_gw.values() for m in ms}
        up_by_gw, up_unmapped = _rows_to_gw_matches(up, season, date_to_gw)
        unmapped += up_unmapped
        for gw, matches in up_by_gw.items():
            for m in matches:
                if m["event_id"] not in seen:
                    by_gw.setdefault(gw, []).append(m)
                    n_upcoming += 1

    out_path = data_dir / "odds" / f"{season}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {k: by_gw[k] for k in sorted(by_gw, key=int)}
    out_path.write_text(json.dumps(ordered, indent=1), encoding="utf-8")
    print(f"Odds {season}: {n_played} played + {n_upcoming} upcoming matches across "
          f"{len(by_gw)} GWs -> {out_path} ({unmapped} unmapped)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, help="e.g. 2025-26")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--include-upcoming", action="store_true",
                        help="also add not-yet-played fixtures (live season)")
    args = parser.parse_args()
    build_season_odds(args.season, args.data_dir, args.include_upcoming)


if __name__ == "__main__":
    main()
