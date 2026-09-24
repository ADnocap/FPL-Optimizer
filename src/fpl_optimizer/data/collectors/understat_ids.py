"""Understat ID supplement for players the master ID map doesn't cover.

The ChrisMusson master map lags new signings and promoted clubs (16.6% of the
2026-27 players had no understat ID at GW5, so their understat features were
NaN at every deadline). This matches the season's understat league table to
FPL players by club + name tokens + minutes, and writes
``data/id_maps/live_understat_ids_{season}.csv`` (code, understat), which
``IDResolver`` loads on top of the master map.

Matching (per unmapped understat player):
- candidate FPL players have no understat ID yet and played for (or are
  registered at) one of the understat player's clubs this season;
- rank by name-token overlap, then by |understat minutes - FPL minutes|;
- accept only a unique best candidate within 45 minutes.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# understat team_title -> FPL teams.csv name (seasons 2016-17..2026-27)
UNDERSTAT_TO_FPL_TEAM: dict[str, str] = {
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Tottenham": "Spurs",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom",
    "Sheffield United": "Sheffield Utd",
    "Coventry": "Coventry City",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
}
_MAX_MINUTES_GAP = 45


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s)
    return " ".join(s.split())


def _team_aliases(fpl_name: str) -> set[str]:
    """FPL names vary between seasons (Ipswich vs Ipswich Town): match loosely."""
    n = _norm(fpl_name)
    return {n, n.replace(" city", "").replace(" town", "")}


def build_understat_id_supplement(data_dir: Path, season: str) -> pd.DataFrame:
    """Match unmapped understat players to FPL codes; write and return the table."""
    data_dir = Path(data_dir)
    league_path = data_dir / "understat" / "league" / f"{season}.json"
    raw = data_dir / "raw" / season
    league = json.loads(league_path.read_text(encoding="utf-8"))
    master = pd.read_csv(data_dir / "id_maps" / "master_id_map.csv", encoding="utf-8-sig")
    players = pd.read_csv(raw / "players_raw.csv", encoding="utf-8")
    teams = pd.read_csv(raw / "teams.csv")
    tid2name = dict(zip(teams["id"], teams["name"]))

    mapped_us = {int(v) for v in master["understat"].dropna()}
    code_has_us = set(master.loc[master["understat"].notna(), "code"].astype(int))

    minutes: dict[int, float] = {}
    played_for: dict[int, set[str]] = {}
    mg_path = raw / "gws" / "merged_gw.csv"
    if mg_path.exists():
        mg = pd.read_csv(mg_path, encoding="utf-8")
        mg = mg[pd.to_numeric(mg["minutes"], errors="coerce").fillna(0) > 0]
        minutes = mg.groupby("element")["minutes"].sum().to_dict()
        for el, team in zip(mg["element"], mg["team"]):
            played_for.setdefault(int(el), set()).add(str(team))

    rows = []
    for p in league:
        uid = int(p["id"])
        if uid in mapped_us:
            continue
        us_teams = {
            alias
            for t in str(p.get("team_title", "")).split(",")
            for alias in _team_aliases(UNDERSTAT_TO_FPL_TEAM.get(t.strip(), t.strip()))
        }
        us_tokens = set(_norm(p.get("player_name", "")).split())
        cands = []
        for e in players.itertuples():
            if int(e.code) in code_has_us:
                continue
            fpl_teams = set()
            for t in played_for.get(int(e.id), set()) | {tid2name.get(e.team, "")}:
                fpl_teams |= _team_aliases(t)
            if not (us_teams & fpl_teams):
                continue
            tokens = set(_norm(f"{e.first_name} {e.second_name}").split()) | set(
                _norm(e.web_name).split()
            )
            overlap = len(us_tokens & tokens)
            if overlap == 0:
                continue
            gap = abs(float(p.get("time") or 0) - float(minutes.get(int(e.id), 0)))
            cands.append((overlap, -gap, int(e.code), gap))
        if not cands:
            continue
        cands.sort(reverse=True)
        best = cands[0]
        unique = len(cands) == 1 or cands[1][:2] < best[:2]
        if best[3] <= _MAX_MINUTES_GAP and unique:
            rows.append({"code": best[2], "understat": uid, "understat_name": p.get("player_name")})

    out = pd.DataFrame(rows, columns=["code", "understat", "understat_name"])
    out = out.drop_duplicates("code", keep=False)  # never guess between two
    dest = data_dir / "id_maps" / f"live_understat_ids_{season}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False, encoding="utf-8")
    logger.info("Understat ID supplement %s: %d players matched -> %s", season, len(out), dest)
    return out
