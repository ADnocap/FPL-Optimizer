"""Live-season collector: builds vaastav-format season files from the FPL API.

The vaastav/Fantasy-Premier-League repo stopped weekly updates after 2024-25,
so the current season's ``data/raw/{season}/`` files are built locally from the
official FPL API instead.  Every column the pipeline consumes maps 1:1 onto
API endpoints:

- ``gws/merged_gw.csv``   <- element-summary/{id}/ history (one row per fixture)
- ``players_raw.csv``     <- bootstrap-static elements
- ``teams.csv``           <- bootstrap-static teams
- ``fixtures.csv``        <- fixtures/
- ``cleaned_players.csv`` <- bootstrap-static elements (subset)
- ``xP`` column           <- ep_this of the snapshot taken while that GW was
                             current (vaastav semantics: post-GW recomputed;
                             only its previous-GW lag is a model feature)

Point-in-time rules:
- ``snapshot_predeadline()`` must run BEFORE each GW deadline: ``ep_this``
  (the previous GW's xP), set-piece orders, and chance_of_playing are
  unrecoverable afterwards.
- ``build_season_files(include_upcoming=True)`` appends synthetic rows for the
  upcoming GW (stats zeroed, price/ownership/fixture real) so FeaturePipeline
  can produce pre-deadline predictions.  The synthetic rows are replaced by
  real ones on the next rebuild after the GW is played.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fpl_optimizer.data.collectors.base import BaseCollector, RateLimiter, DEFAULT_DATA_DIR
from fpl_optimizer.data.collectors.fpl_api import FPLAPICollector, FPL_API_BASE
from fpl_optimizer.utils.constants import CURRENT_SEASON

logger = logging.getLogger(__name__)

ELEMENT_TYPE_TO_POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# merged_gw.csv stat columns copied verbatim from element-summary history rows.
_HISTORY_STAT_COLS = [
    "assists", "bonus", "bps", "clean_sheets", "creativity",
    "expected_assists", "expected_goal_involvements", "expected_goals",
    "expected_goals_conceded", "fixture", "goals_conceded", "goals_scored",
    "ict_index", "influence", "kickoff_time", "minutes", "opponent_team",
    "own_goals", "penalties_missed", "penalties_saved", "red_cards", "round",
    "saves", "selected", "starts", "team_a_score", "team_h_score", "threat",
    "total_points", "transfers_balance", "transfers_in", "transfers_out",
    "value", "was_home", "yellow_cards",
    # 2025-26+ DEFCON-era columns (present in API history; harmless extras)
    "clearances_blocks_interceptions", "defensive_contribution",
    "recoveries", "tackles",
]

MERGED_GW_COLUMNS = ["name", "position", "team", "xP"] + _HISTORY_STAT_COLS + ["GW"]

_CLEANED_PLAYER_COLS = [
    "first_name", "second_name", "id", "goals_scored", "assists",
    "total_points", "minutes", "goals_conceded", "creativity", "influence",
    "threat", "bonus", "bps", "ict_index", "clean_sheets", "red_cards",
    "yellow_cards", "selected_by_percent", "now_cost", "element_type",
]

_FIXTURE_COLS = [
    "code", "event", "finished", "finished_provisional", "id", "kickoff_time",
    "minutes", "provisional_start_time", "started", "team_a", "team_a_score",
    "team_h", "team_h_score", "team_h_difficulty", "team_a_difficulty",
    "pulse_id",
]

# From 2026-27 the FPL API publishes team ``strength`` as null and every
# ``strength_attack_*`` / ``strength_defence_*`` as 0; only a 1-5 rating in
# ``strength_overall_home`` / ``strength_overall_away`` survives.  The model's
# opponent features (opponent.py) were trained on the old ~1000-1400 scale, so
# reconstruct it: mean attack/defence values per 1-5 rating over the 2020-21 to
# 2025-26 teams.csv files (home rating -> *_home columns, away -> *_away).
_STRENGTH_SCALE = {
    #        att_h, att_a, def_h, def_a
    1: (975, 1000, 985, 1000),
    2: (1038, 1060, 1047, 1065),
    3: (1102, 1138, 1123, 1164),
    4: (1181, 1228, 1236, 1289),
    5: (1337, 1361, 1337, 1352),
}


def _deadline_selected(el: dict, last: tuple[int, int] | None, total_players: int) -> int:
    """Ownership count at the upcoming deadline, as training rows record it.

    History rows carry the exact ``selected`` count; bootstrap only exposes
    ``selected_by_percent`` rounded to 0.1% (~11k managers), which quantises
    low-owned players (combiner experiment 2026-09-25: -0.024 per-GW Spearman
    in simulation, -0.005 live). Reconstruct it instead: last exact count +
    this window's transfers in - out. Players without history fall back to the
    rounded percentage.
    """
    if last is not None:
        return max(0, last[1] + int(el.get("transfers_in_event") or 0)
                   - int(el.get("transfers_out_event") or 0))
    return int(float(el.get("selected_by_percent") or 0) / 100.0 * total_players)


def _backfill_team_strengths(team: dict) -> dict:
    """Return a copy of a bootstrap team dict with the legacy strength fields
    filled in when the API no longer provides them (2026-27 onwards)."""
    t = dict(team)
    legacy = [
        "strength_attack_home", "strength_attack_away",
        "strength_defence_home", "strength_defence_away",
    ]
    if t.get("strength") is not None and any(t.get(k) for k in legacy):
        return t  # old-format payload, nothing to do
    home = int(t.get("strength_overall_home") or 3)
    away = int(t.get("strength_overall_away") or 3)
    home, away = min(max(home, 1), 5), min(max(away, 1), 5)
    t["strength"] = int(round((home + away) / 2))
    t["strength_attack_home"] = _STRENGTH_SCALE[home][0]
    t["strength_defence_home"] = _STRENGTH_SCALE[home][2]
    t["strength_attack_away"] = _STRENGTH_SCALE[away][1]
    t["strength_defence_away"] = _STRENGTH_SCALE[away][3]
    return t



class LiveFPLCollector(BaseCollector):
    """Build and refresh the current season's data files from the FPL API."""

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        season: str = CURRENT_SEASON,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            rate_limiter=RateLimiter(calls_per_second=1.0),
        )
        self.season = season
        self.live_dir = self.data_dir / "live" / season
        self.snapshot_dir = self.live_dir / "snapshots"
        self.raw_season_dir = self.data_dir / "raw" / season
        self.api = FPLAPICollector(data_dir=data_dir)

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    def collect_season(self, season: str) -> bool:
        self.season = season
        return self.refresh(include_upcoming=True)

    def collect_all(self) -> dict[str, bool]:
        return {self.season: self.collect_season(self.season)}

    # ------------------------------------------------------------------
    # Fetch helpers
    # ------------------------------------------------------------------

    def _fetch_json(self, url: str) -> dict | list:
        resp = self._request_with_retry(url)
        return resp.json()

    def fetch_bootstrap(self) -> dict:
        return self._fetch_json(f"{FPL_API_BASE}/bootstrap-static/")

    def fetch_fixtures(self) -> list:
        return self._fetch_json(f"{FPL_API_BASE}/fixtures/")

    # ------------------------------------------------------------------
    # Pre-deadline snapshots
    # ------------------------------------------------------------------

    @staticmethod
    def _target_event(bootstrap: dict) -> tuple[int | None, str]:
        """Return (gw, ep_field) for the GW whose xP this snapshot captures.

        Before a deadline the upcoming GW is ``is_next`` and its EP lives in
        ``ep_next``; once the deadline passes it becomes ``is_current`` and the
        EP moves to ``ep_this`` (vaastav's xP was scraped in that window).
        """
        now = datetime.now(timezone.utc)
        for ev in bootstrap.get("events", []):
            deadline = datetime.fromisoformat(ev["deadline_time"].replace("Z", "+00:00"))
            if ev.get("is_next") and deadline > now:
                return ev["id"], "ep_next"
        for ev in bootstrap.get("events", []):
            if ev.get("is_current") and not ev.get("finished"):
                return ev["id"], "ep_this"
        # Post-deadline API lag: the deadline just passed but FPL hasn't
        # flipped is_current/is_next yet — target the first unfinished event.
        for ev in bootstrap.get("events", []):
            if not ev.get("finished"):
                deadline = datetime.fromisoformat(
                    ev["deadline_time"].replace("Z", "+00:00")
                )
                return ev["id"], "ep_next" if deadline > now else "ep_this"
        return None, "ep_next"

    def snapshot_predeadline(self) -> int | None:
        """Snapshot bootstrap + fixtures for the upcoming GW.  Returns the GW."""
        bootstrap = self.fetch_bootstrap()
        fixtures = self.fetch_fixtures()
        gw, ep_field = self._target_event(bootstrap)
        if gw is None:
            logger.warning("Live snapshot: no upcoming/current GW found")
            return None

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Never overwrite a pre-deadline snapshot after the deadline has
        # passed — the pre-deadline capture (ep, prices, ownership) is the
        # point-in-time record and is unrecoverable.
        meta_path = self.snapshot_dir / f"gw{gw}_meta.json"
        if meta_path.exists():
            try:
                old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                taken = datetime.fromisoformat(old_meta["taken_utc"])
                event = next(e for e in bootstrap["events"] if e["id"] == gw)
                deadline = datetime.fromisoformat(
                    event["deadline_time"].replace("Z", "+00:00")
                )
                now = datetime.now(timezone.utc)
                if taken <= deadline < now:
                    logger.info(
                        "Live snapshot: GW%d pre-deadline snapshot preserved "
                        "(deadline passed, not overwriting)", gw,
                    )
                    return gw
            except (KeyError, ValueError, StopIteration):
                pass  # unreadable meta — take a fresh snapshot
        meta = {
            "gw": gw,
            "ep_field": ep_field,
            "taken_utc": datetime.now(timezone.utc).isoformat(),
        }
        (self.snapshot_dir / f"gw{gw}_bootstrap.json").write_text(
            json.dumps(bootstrap), encoding="utf-8"
        )
        (self.snapshot_dir / f"gw{gw}_fixtures.json").write_text(
            json.dumps(fixtures), encoding="utf-8"
        )
        self._snapshot_set_piece_notes(gw)
        (self.snapshot_dir / f"gw{gw}_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info("Live snapshot: GW%d saved (%s)", gw, ep_field)
        return gw

    def _snapshot_set_piece_notes(self, gw: int) -> None:
        """Fetch team set-piece notes and save them with the GW snapshot.

        Endpoint shape: ``{"last_updated": ISO-8601, "teams": [{"id": int,
        "notes": [{"info_message": str, "external_link": bool,
        "source_link": str}]}]}``.  Non-fatal on failure — the notes are
        supplementary to the bootstrap/fixtures point-in-time record.
        """
        try:
            notes = self._fetch_json(f"{FPL_API_BASE}/team/set-piece-notes/")
        except Exception:
            logger.warning("Live snapshot: set-piece notes fetch failed", exc_info=True)
            return
        (self.snapshot_dir / f"gw{gw}_setpiece_notes.json").write_text(
            json.dumps(notes), encoding="utf-8"
        )

    def _load_snapshot_xp(self) -> dict[tuple[int, int], float]:
        """Build (element_id, gw) -> xP with vaastav's semantics.

        vaastav's ``xP`` for GW c is FPL's ``ep_this`` read while c is the
        *current* event, i.e. after its deadline — by then FPL has recomputed
        it from form that includes GW c's own points. The model therefore only
        uses the previous GW's value (``fpl_xp_lag``). To keep train/serve
        parity, GW c's xP here is the ``ep_this`` of the latest snapshot whose
        current event is c: our pre-deadline snapshot for GW c+1 is taken
        inside that window, so at every deadline the lag is the exact analog
        of the historical feature. Snapshots are ordered by capture time so a
        later capture in the same window wins.
        """
        xp: dict[tuple[int, int], float] = {}
        if not self.snapshot_dir.exists():
            return xp
        snaps = []
        for boot_path in self.snapshot_dir.glob("gw*_bootstrap.json"):
            gw_str = boot_path.stem.replace("gw", "").replace("_bootstrap", "")
            if not gw_str.isdigit():
                continue
            taken = ""
            meta_path = self.snapshot_dir / f"gw{gw_str}_meta.json"
            if meta_path.exists():
                try:
                    taken = json.loads(meta_path.read_text(encoding="utf-8")).get("taken_utc", "")
                except (json.JSONDecodeError, OSError):
                    pass
            snaps.append((taken, int(gw_str), boot_path))
        for _, _, boot_path in sorted(snaps):
            boot = json.loads(boot_path.read_text(encoding="utf-8"))
            current = next(
                (ev["id"] for ev in boot.get("events", []) if ev.get("is_current")), None
            )
            if current is None:  # pre-season: there is no current GW yet
                continue
            for el in boot.get("elements", []):
                val = el.get("ep_this")
                if val is None:
                    continue
                try:
                    xp[(el["id"], current)] = float(val)
                except (TypeError, ValueError):
                    pass
        return xp

    # ------------------------------------------------------------------
    # Element summaries (force refresh — history grows every GW)
    # ------------------------------------------------------------------

    def refresh_element_summaries(self, bootstrap: dict, max_workers: int = 4) -> None:
        """Re-download every element summary, atomically.

        Downloads into a staging directory and only swaps it in when every
        player succeeded, so a network failure can never leave the season
        half-refreshed (the old code deleted everything first and then ignored
        failed downloads). Raises RuntimeError on any failure; the previous
        files stay in place.
        """
        base = self.data_dir / "fpl_api" / "element_summaries"
        summary_dir = base / self.season
        tag = f"{self.season}.staging"
        staging = base / tag
        if staging.exists():
            shutil.rmtree(staging)
        # Reuse FPLAPICollector's threaded downloader via a staging season tag
        boot_dir = self.data_dir / "fpl_api" / "bootstrap"
        boot_dir.mkdir(parents=True, exist_ok=True)
        (boot_dir / f"{self.season}.json").write_text(json.dumps(bootstrap), encoding="utf-8")
        staging_boot = boot_dir / f"{tag}.json"
        staging_boot.write_text(json.dumps(bootstrap), encoding="utf-8")
        try:
            ok = self.api._collect_element_summaries(tag, max_workers=max_workers)
        finally:
            staging_boot.unlink(missing_ok=True)
        n_expected = len(bootstrap.get("elements", []))
        n_got = len(list(staging.glob("*.json"))) if staging.exists() else 0
        if not ok or n_got < n_expected:
            raise RuntimeError(
                f"element-summary refresh incomplete ({n_got}/{n_expected} players); "
                f"kept the previous files in {summary_dir}. Re-run, or use "
                "--skip-refresh if they are recent enough."
            )
        old = base / f"{self.season}.old"
        if old.exists():
            shutil.rmtree(old)
        if summary_dir.exists():
            summary_dir.rename(old)
        staging.rename(summary_dir)
        if old.exists():
            shutil.rmtree(old)
        (summary_dir / "_refreshed_utc.txt").write_text(
            datetime.now(timezone.utc).isoformat(), encoding="utf-8"
        )

    def summaries_refreshed_utc(self) -> datetime | None:
        """When the element summaries were last fully refreshed (None if unknown)."""
        stamp = self.data_dir / "fpl_api" / "element_summaries" / self.season / "_refreshed_utc.txt"
        if not stamp.exists():
            return None
        try:
            return datetime.fromisoformat(stamp.read_text(encoding="utf-8").strip())
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Season file builder
    # ------------------------------------------------------------------

    def build_season_files(
        self,
        bootstrap: dict | None = None,
        fixtures: list | None = None,
        include_upcoming: bool = True,
    ) -> Path:
        """Write vaastav-format files for the live season.  Returns raw dir."""
        bootstrap = bootstrap or self.fetch_bootstrap()
        fixtures = fixtures or self.fetch_fixtures()
        elements = bootstrap["elements"]
        teams = bootstrap["teams"]
        team_name = {t["id"]: t["name"] for t in teams}

        self.raw_season_dir.mkdir(parents=True, exist_ok=True)
        (self.raw_season_dir / "gws").mkdir(exist_ok=True)

        # --- players_raw.csv (bootstrap elements verbatim) ---
        pd.DataFrame(elements).to_csv(
            self.raw_season_dir / "players_raw.csv", index=False, encoding="utf-8"
        )

        # --- teams.csv ---
        pd.DataFrame([_backfill_team_strengths(t) for t in teams]).to_csv(
            self.raw_season_dir / "teams.csv", index=False, encoding="utf-8"
        )

        # --- fixtures.csv (drop nested stats) ---
        fx_rows = [{k: f.get(k) for k in _FIXTURE_COLS} for f in fixtures]
        pd.DataFrame(fx_rows).to_csv(
            self.raw_season_dir / "fixtures.csv", index=False, encoding="utf-8"
        )

        # --- cleaned_players.csv + player_idlist.csv ---
        cleaned = pd.DataFrame(elements)
        keep = [c for c in _CLEANED_PLAYER_COLS if c in cleaned.columns]
        cleaned[keep].to_csv(
            self.raw_season_dir / "cleaned_players.csv", index=False, encoding="utf-8"
        )
        cleaned[["first_name", "second_name", "id"]].to_csv(
            self.raw_season_dir / "player_idlist.csv", index=False, encoding="utf-8"
        )

        # --- id map supplement (element_id -> stable code from the API itself) ---
        id_map_dir = self.data_dir / "id_maps"
        id_map_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"element_id": el["id"], "code": el["code"], "web_name": el["web_name"]}
                for el in elements
            ]
        ).to_csv(
            id_map_dir / f"live_element_code_{self.season}.csv",
            index=False,
            encoding="utf-8",
        )

        # --- gws/merged_gw.csv from element summaries ---
        xp_map = self._load_snapshot_xp()
        el_meta = {
            el["id"]: {
                "name": f"{el['first_name']} {el['second_name']}".strip(),
                "position": ELEMENT_TYPE_TO_POSITION.get(el["element_type"], "MID"),
                "team": team_name.get(el["team"], str(el["team"])),
                "team_id": el["team"],
            }
            for el in elements
        }

        # Club at the time of each fixture: history rows carry the fixture id
        # and was_home, so the side is exact.  (Labelling past rows with the
        # player's CURRENT club puts a transferred player's old-club fixtures
        # under the new club -> phantom DGWs and mixed team stats.)
        fixture_sides = {
            f["id"]: (f.get("team_h"), f.get("team_a")) for f in fixtures
        }

        def _team_at_fixture(h: dict, fallback: str) -> str:
            sides = fixture_sides.get(h.get("fixture"))
            if not sides or h.get("was_home") is None:
                return fallback
            tid = sides[0] if h.get("was_home") else sides[1]
            return team_name.get(tid, fallback)

        summary_dir = self.data_dir / "fpl_api" / "element_summaries" / self.season
        rows: list[dict] = []
        # element -> (round, exact `selected` count) of its latest history row
        last_selected: dict[int, tuple[int, int]] = {}
        if summary_dir.exists():
            for summary_path in summary_dir.glob("*.json"):
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    logger.warning("Bad element summary: %s", summary_path.name)
                    continue
                for h in summary.get("history", []):
                    eid = h["element"]
                    meta = el_meta.get(eid)
                    if meta is None:
                        continue
                    row = {c: h.get(c) for c in _HISTORY_STAT_COLS}
                    row["name"] = meta["name"]
                    row["position"] = meta["position"]
                    row["team"] = _team_at_fixture(h, meta["team"])
                    row["xP"] = xp_map.get((eid, h["round"]), "")
                    row["element"] = eid
                    row["GW"] = h["round"]
                    rows.append(row)
                    if h.get("selected") is not None and (
                        eid not in last_selected or h["round"] >= last_selected[eid][0]
                    ):
                        last_selected[eid] = (h["round"], int(h["selected"]))

        # --- synthetic rows for the upcoming GW (pre-deadline prediction) ---
        if include_upcoming:
            gw, _ = self._target_event(bootstrap)
            if gw is not None:
                played = {(r["element"], r["fixture"]) for r in rows}
                gw_fixtures = [f for f in fixtures if f.get("event") == gw]
                total_players = bootstrap.get("total_players") or 1
                for el in elements:
                    meta = el_meta[el["id"]]
                    for f in gw_fixtures:
                        if f["team_h"] == meta["team_id"]:
                            was_home = True
                        elif f["team_a"] == meta["team_id"]:
                            was_home = False
                        else:
                            continue
                        if (el["id"], f["id"]) in played:
                            continue
                        row = {c: 0 for c in _HISTORY_STAT_COLS}
                        row.update(
                            {
                                "name": meta["name"],
                                "position": meta["position"],
                                "team": meta["team"],
                                # post-GW value doesn't exist yet; only
                                # the lag (previous row) is a feature
                                "xP": "",
                                "element": el["id"],
                                "GW": gw,
                                "round": gw,
                                "fixture": f["id"],
                                "opponent_team": f["team_a"] if was_home else f["team_h"],
                                "was_home": was_home,
                                "kickoff_time": f.get("kickoff_time") or "",
                                "team_h_score": "",
                                "team_a_score": "",
                                "value": el["now_cost"],
                                "selected": _deadline_selected(
                                    el, last_selected.get(el["id"]), total_players
                                ),
                                "transfers_in": el.get("transfers_in_event", 0),
                                "transfers_out": el.get("transfers_out_event", 0),
                                "transfers_balance": (
                                    el.get("transfers_in_event", 0)
                                    - el.get("transfers_out_event", 0)
                                ),
                            }
                        )
                        rows.append(row)

        merged = pd.DataFrame(rows, columns=["element"] + MERGED_GW_COLUMNS)
        # vaastav column order puts element inside; keep our canonical order
        merged = merged[MERGED_GW_COLUMNS[:4] + ["element"] + MERGED_GW_COLUMNS[4:]]
        merged = merged.sort_values(["GW", "element", "fixture"], kind="stable")
        merged.to_csv(
            self.raw_season_dir / "gws" / "merged_gw.csv",
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL,
        )
        logger.info(
            "Live season files built: %d merged_gw rows (%d players)",
            len(merged),
            len(elements),
        )
        return self.raw_season_dir

    # ------------------------------------------------------------------
    # One-call weekly refresh
    # ------------------------------------------------------------------

    def refresh(
        self, include_upcoming: bool = True, refresh_summaries: bool = True
    ) -> bool:
        """Full weekly refresh: snapshot -> summaries -> season files."""
        try:
            bootstrap = self.fetch_bootstrap()
            fixtures = self.fetch_fixtures()
            self.snapshot_predeadline()
            if refresh_summaries:
                self.refresh_element_summaries(bootstrap)
            self.build_season_files(
                bootstrap=bootstrap,
                fixtures=fixtures,
                include_upcoming=include_upcoming,
            )
            return True
        except Exception:
            logger.exception("Live refresh failed")
            return False
