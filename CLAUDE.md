# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"          # also: .[prediction] .[optimizer] .[data]

# Run all tests (305 as of 2026-09)
pytest

# Run one area
pytest tests/test_engine/ -v     # also: test_optimizer, test_prediction, test_data

# Run a single test class or method
pytest tests/test_engine/test_chips.py::TestActivateChip::test_one_chip_per_gw -v
```

## Live-season operation (2026-27)

The repo runs live for the 2026-27 season. Weekly loop, before each GW deadline:

```bash
python scripts/gameweek.py --team-id <FPL_ENTRY_ID>
```

This snapshots bootstrap-static pre-deadline (captures `ep_next`/prices/ownership —
unrecoverable later), refreshes element summaries, rebuilds `data/raw/2026-27/` in
vaastav format from the official FPL API (vaastav stopped weekly updates after
2024-25), synthesizes feature rows for the upcoming GW, predicts with the LightGBM
model, fetches the real team state (public entry API), and prints recommended
transfers / lineup / captain. `--fresh-squad` picks a full squad (GW1/wildcard).

Retrain the model of record with `python scripts/train_predictor.py`
(→ `models/prod_2026-27/`; the only committed reproducible training recipe).

## Architecture

`src/fpl_optimizer/` has 6 subpackages with a strict layering:

**`engine/`** — Pure game logic. Stateless:
`step(GameState, EngineAction) → (GameState, StepResult)`; never mutates inputs.
Scoring is a lookup of recorded `total_points` (historical replay — cannot
simulate an unplayed GW).

**`data/`** — `SeasonDataLoader` (pre-indexed `(element_id, gw)` lookups, DGW
aggregation, cross-season position/team backfill) + `collectors/` for 7 sources:
vaastav, understat, fpl_api, fbref, fotmob, odds, id_mapping, plus
**`fpl_live.py`** (`LiveFPLCollector`) which builds vaastav-format season files
directly from the FPL API for the current season.

**`prediction/`** — LightGBM point predictor: `feature_pipeline.py` orchestrates
feature modules (`features/vaastav|understat|prior_season|opponent|odds|players_raw`),
`id_resolver.py` (element_id ↔ stable code ↔ understat/fbref ids; auto-loads
`data/id_maps/live_element_code_*.csv` supplements built from bootstrap `code`),
`model.py` (4 boosters, one per position; NaN-tolerant; selects features by name),
`integration.py` pre-computes a season's `(element_id, gw) → xPts` lookup (used by the backtest),
`horizon.py` (multi-GW predictions as of a deadline: the as-of-t row with only
the fixture features swapped per future fixture — opponent/venue/FDR/DGW, team
stats rolled over GW < t, odds/props unknown → NaN/no-line; blanks = 0).

**`optimizer/`** — PuLP MILP suite: `squad_selection.py` (initial 15),
`transfer_optimizer.py` (single-GW transfers vs a GameState: bank, selling prices,
FTs, hit costs), `horizon_optimizer.py` (multi-period receding-horizon planner:
FT banking to 5, hits, budget per GW, a GIVEN chip schedule WC/FH/BB/TC, bench
value from the XI's DNP risk; execute GW t only, re-solve weekly),
`lineup_selector.py` (XI+captain), `backtest.py` (full-season replay; season
replays of the planner: `scripts/backtest_horizon.py`). Solver: PULP_CBC_CMD.
Chips are a human schedule (SEASON_GUIDE.md) — there is no chip *scheduler*.

**`live/`** — Live-season glue: `entry.py` (public entry API → GameState with
reconstructed purchase/selling prices, FT bank simulation, chips),
`pool.py` (candidates from bootstrap with availability filtering/scaling;
`build_live_horizon_candidates` for the planner), `predict.py` (upcoming-GW and
horizon model predictions).

`cluster/` — SLURM scripts for the LaRuche cluster.

### Key Data Flow (live season)

```
FPL API ──LiveFPLCollector──> data/raw/2026-27/ (vaastav format + synthetic
                              upcoming-GW rows, xP from pre-deadline snapshots)
        ──FeaturePipeline──> feature rows for upcoming GW
        ──PointPredictor──> element_id → xPts
entry API ──fetch_entry_state──> GameState (squad/bank/FTs/chips)
(GameState, candidates) ──optimize_transfers──> transfers/lineup/captain
```

## Important Conventions

- **Prices are in tenths**: `100 = £10.0m`. All price math is integer.
- **Lineup/bench are indices into `Squad.players`**, not element_ids.
- **Point-in-time discipline**: post-match features come from gw-1; pre-match
  (price, selected, was_home, xP) from the current gw. `ep_this`/`ep_next` is
  point-in-time safe but must be snapshotted in its GW window (see EP_FORMULA.md —
  xP is NOT a lookahead leak; MEMORY notes saying otherwise are outdated).
- **Live rebuilds replace synthetic rows**: `build_season_files()` regenerates
  merged_gw.csv each run; upcoming-GW synthetic rows (stats zeroed) are replaced
  by real rows after the GW completes.

## Data sources & keys

No paid keys required. FPL API, understat (understatapi ≥0.7.1 — the site moved
to AJAX endpoints in Dec 2025), FotMob CDN (`data.fotmob.com`, season IDs incl.
2025-26=27110, 2026-27=36781), football-data.co.uk (free Pinnacle/avg odds
backfill: `scripts/collect_football_data_odds.py`), ChrisMusson ID map. The Odds
API live tier is free (500 credits/mo; 1 credit/GW) — key in `.env` as
`ODDS_API_KEY` (optional; odds features add ~nothing per our ablation). FBref is
Cloudflare-blocked since 2026 — FotMob covers the fallback features.

## FPL Rules Encoded (verified for 2026/27)

- 8 valid formations, always 1 GK; 15-player squad; 3-per-club; £100.0m start
- 4 chips × 2 halves (GW1-19 / GW20-38); one chip per GW; first-half chips expire
  after GW19; **Wildcard & Free Hit have start_event=2 (not playable GW1)**;
  Free Hit cannot be used in both GW19 and GW20
- FT banking to max 5; WC/FH do NOT reset banked FTs; hits −4/extra transfer
- Selling price = purchase + floor(appreciation/2)
- Auto-subs walk bench in priority order respecting formations; captain failover
  when captain didn't play (0 min and no card) or left the lineup
- 2026/27 scoring unchanged from 2025/26 (incl. DEFCON defensive contribution:
  2 pts DEF/MID/FWD at thresholds, 0 for GK); BPS internals overhauled (bonus
  distribution only); GW scores now final 09:00 UK the day after the last match
  (don't scrape final stats before then)

## Test Data Pattern

Tests use hand-crafted CSVs in `tests/test_data/` (18 players, 2 GWs).
`SeasonDataLoader.__init__` is monkey-patched in `conftest.py` to skip downloads.
**Constraints**: max 3 players per team; the `team` column in `merged_gw.csv` and
`cleaned_players.csv` must be consistent. Add scenarios by adding CSV rows/files,
not by mocking loader methods.
