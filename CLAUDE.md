# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"          # also: .[prediction] .[optimizer] .[data]

# Run all tests (638 as of 2026-09-25)
pytest

# Run one area
pytest tests/test_engine/ -v     # also: test_optimizer, test_prediction, test_data, test_rules

# Run a single test class or method
pytest tests/test_engine/test_chips.py::TestActivateChip::test_one_chip_per_gw -v
```

## Live-season operation (2026-27)

The repo runs live for the 2026-27 season. Weekly loop, before each GW deadline:

```bash
python scripts/gameweek.py --team-id <FPL_ENTRY_ID> --horizon 3   # primary (planner)
python scripts/gameweek.py --team-id <FPL_ENTRY_ID>               # single-GW cross-check
```

This snapshots bootstrap-static pre-deadline (`ep_this` = the previous GW's xP,
prices, ownership, flags — unrecoverable later), refreshes element summaries
atomically, rebuilds `data/raw/2026-27/` in vaastav format from the official FPL
API, synthesizes feature rows for the upcoming GW, predicts with the model of
record, prints a DATA HEALTH block, fetches the real team state (public entry API;
with `--apply` the authenticated my-team state), prints transfers / lineup /
captain + a captaincy view (model vs FPL EP vs bookmakers), and writes a decision
log to `data/live/2026-27/decisions/`. The planner beat the single-GW optimizer in
all three leak-free season replays (2023-24..2025-26). `--fresh-squad` = GW1 only.

Chip decision points: `python scripts/chip_eval.py --team-id <ID>` prints the
expected gain of every remaining chip in every GW (numbers only — the chip plan
in SEASON_GUIDE.md is a human decision; offline: `--entry-dir/--bootstrap/--fixtures`;
refuses a model with the leaky same-GW `fpl_xp`).

Retrain the model of record with `python scripts/train_predictor.py
--features-cache <file> --rebuild-cache --second-fold` (→ `models/prod_2026-27/`,
outgoing model kept as `.prev`; RECIPE in the script is the only committed
reproducible recipe; `/retrain` skill). Pre-v5 models (trained on the leaky
`fpl_xp`) live in `models/archive/` and are refused by the live leak guard.

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
feature modules (`features/vaastav|understat|prior_season|opponent|odds|props|players_raw|minutes_history`;
`FEATURE_PIPELINE_VERSION` + `load_or_build_feature_cache`),
`id_resolver.py` (element_id ↔ stable code ↔ understat/fbref ids; auto-loads
`data/id_maps/live_element_code_*.csv` supplements built from bootstrap `code`),
`model.py` (`PointPredictor`: 4 boosters, one per position; NaN-tolerant; selects
features by name; optional monotone calibration), `minutes.py` (`MinutesModel`
P(0/1-59/60+) and `MinutesBlendPredictor` = the model of record's kind;
**`load_predictor(model_dir)`** loads either kind — use it, never a class's `load`),
`feature_sets.py` (`EXCLUDED_FEATURES` = unservable understat/FBref/legacy +
h2h odds (neutral, fragile to serve) + reconstructed team strengths (2026-27 API
only has a 1-5 rating) — excluded from training),
`integration.py` pre-computes a season's `(element_id, gw) → xPts` lookup (used by the backtest),
`horizon.py` (multi-GW predictions as of a deadline: the as-of-t row with only
the fixture features swapped per future fixture — opponent/venue/FDR/DGW, team
stats rolled over GW < t; live planner uses market="drop" so props don't favour
GW t over later GWs; blanks = 0).

**`optimizer/`** — PuLP MILP suite: `squad_selection.py` (initial 15),
`transfer_optimizer.py` (single-GW transfers vs a GameState: bank, selling prices,
FTs, hit costs), `horizon_optimizer.py` (multi-period receding-horizon planner:
FT banking to 5, hits, budget per GW, a GIVEN chip schedule WC/FH/BB/TC, bench
value from the XI's DNP risk; execute GW t only, re-solve weekly),
`lineup_selector.py` (XI+captain), `backtest.py` (full-season replay; season
replays of the planner: `scripts/backtest_horizon.py`), `chip_eval.py` (chip
values per GW: TC/BB by Monte Carlo on the planned squad, WC/FH as
optimize_horizon with vs without the chip, calibration split by season half).
Solver: PULP_CBC_CMD.
Chips are a human schedule (SEASON_GUIDE.md) — there is no chip *scheduler*.

**`live/`** — Live-season glue: `entry.py` (public entry API → GameState with
reconstructed purchase/selling prices, FT bank simulation, chips),
`pool.py` (candidates from bootstrap; availability enters ONCE here via the
calibrated `FLAG_MULTIPLIER` — 75% flag → 0.40x; `build_live_horizon_candidates`
for the planner), `predict.py` (upcoming-GW and horizon predictions,
`serving_health` data-health check vs `serving_reference.json`, leak guard
`check_not_leaky`), `chip_inputs.py` (chip_eval glue: leak guard, DGW
blend, optional P(play) classifier `prediction/play_model.py`, offline entry).

`cluster/` — SLURM scripts for the LaRuche cluster.

### Key Data Flow (live season)

```
FPL API ──LiveFPLCollector──> data/raw/2026-27/ (vaastav format, fixture-side
                              club labels, synthetic upcoming-GW rows with
                              deadline ownership, xP = post-GW ep_this)
        ──FeaturePipeline──> feature rows for upcoming GW
        ──load_predictor(models/prod_2026-27)──> element_id → xPts
entry API ──fetch_entry_state──> GameState (squad/bank/FTs/chips)
(GameState, candidates) ──optimize_transfers──> transfers/lineup/captain
  (--horizon N: predict_horizon_live ──> optimize_horizon, chip plan given)
```

## Important Conventions

- **Prices are in tenths**: `100 = £10.0m`. All price math is integer.
- **Lineup/bench are indices into `Squad.players`**, not element_ids.
- **Point-in-time discipline**: post-match features come from gw-1; pre-match
  (price, selected, was_home) from the current gw. **vaastav's `xP` for GW n is a
  same-GW leak** (FPL recomputes `ep_this` after the GW with its own points; EP_FORMULA.md
  is retracted): the model only uses `fpl_xp_lag` = the previous GW's xP. Live, the
  collector gives merged_gw `xP` the same post-GW semantics from the next deadline's
  snapshot `ep_this`, so the lag is identical in training and serving.
- **Understat per-match features compare calendar days** (a match on the GW's first
  kickoff day is that GW's outcome — the old timestamp filter leaked it).
- **Train only on what can be served**: a feature that is always NaN at a live deadline
  must not be in the model. `gameweek.py` prints a data-health block (live coverage vs
  the model's `serving_reference.json`) — read it every week.
- **Live rebuilds replace synthetic rows**: `build_season_files()` regenerates
  merged_gw.csv each run; upcoming-GW synthetic rows (stats zeroed) are replaced
  by real rows after the GW completes.

## Data sources & keys

The live model needs only the FPL API (+ FotMob prior-season stats, collected
once per season). Optional: The Odds API player props (`ODDS_API_KEY` in `.env`,
~40 credits/GW, needs a prop-capable plan; the office Zscaler network blocks it —
missing props are in-distribution and the data-health block flags them).
Research-only (not model features): understat (understatapi ≥0.7.1; the live
season can be refreshed with `UnderstatCollector.refresh_live_season` + the
`understat_ids` supplement), h2h odds (`scripts/collect_football_data_odds.py`,
football-data.co.uk + Odds API fallback). FBref is Cloudflare-blocked since 2026.

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
