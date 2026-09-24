# FPL-Optimizer

Predict-and-optimize system for Fantasy Premier League that combines LightGBM point predictions with MILP squad optimization, achieving backtest scores above the all-time human record.

Replays historical seasons (2016-17 to 2025-26) using real player data from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League), with the full FPL rules encoded in a stateless game engine used for backtesting.

**Now running live for the 2026-27 season** — current-season data is built directly
from the official FPL API (`LiveFPLCollector`), and the weekly loop is one command:

```bash
python scripts/gameweek.py --team-id <YOUR_FPL_ENTRY_ID>
# GW1 / wildcard from scratch:
python scripts/gameweek.py --fresh-squad
# Retrain the model of record (10 seasons through 2025-26):
python scripts/train_predictor.py
```

## Results (2024-25 holdout season)

| Strategy | Net Points | Context |
|----------|-----------|---------|
| No transfers (optimized GW1 squad) | 1,950 | Passive baseline |
| MILP optimizer, 1 transfer/GW | **2,918** | Human-realistic |
| MILP optimizer, 5 transfers/GW | **3,171** | Aggressive |
| Oracle (perfect foresight, 5 xfers) | 3,713 | Theoretical ceiling |
| Best human 2024-25 | ~2,810 | Lovro Budisin |
| Best human ever | ~2,844 | Jamie Pigott, 2021-22 |

Prediction model: 0.787 per-GW correlation with actuals (86 features, LightGBM). All features verified point-in-time safe (no lookahead bias).

## Architecture

```
src/fpl_optimizer/
├── data/          # Data loading + collectors (vaastav, FPL API, Understat, FotMob, odds, props)
├── engine/        # Pure game logic — stateless rules engine for historical replay
├── prediction/    # LightGBM point prediction (4 position-specific models)
├── optimizer/     # PuLP/CBC MILP optimizer (squad selection, transfers, lineup, backtest)
├── live/          # Live-season glue (entry API → GameState, candidate pool, predictions)
└── utils/         # Shared constants and helpers
```

**Engine** -- Stateless: `step(GameState, EngineAction) -> (GameState, StepResult)`. Never mutates input state. The MILP backtest drives it directly: scoring is a lookup of recorded points, so a season can be replayed decision by decision under the real rules.

### Prediction Pipeline

```
merged_gw.csv --> FeaturePipeline.build() --> 86 features per (player, GW)
                                                     |
                                              LightGBM (4 models: GK/DEF/MID/FWD)
                                                     |
                                              predicted_points per player
                                                     |
                                              MILP optimizer --> optimal squad/transfers
```

All rolling features use `.shift(1)` before `.rolling()` to prevent lookahead. Point-in-time safety verified for every feature source (vaastav, Understat, FBref, FotMob, odds, FPL xP).

## Installation

```bash
git clone <repo-url>
cd FPL-Optimizer
pip install -e ".[dev]"
```

Requires Python 3.11+.

**Extras:**
- `pip install -e ".[prediction]"` -- adds lightgbm, scikit-learn
- `pip install -e ".[optimizer]"` -- adds pulp
- `pip install -e ".[dev]"` -- adds pytest plus the prediction/optimizer extras

## Quick Start

### Evaluate the MILP optimizer

```python
from pathlib import Path
from fpl_optimizer.data.downloader import DEFAULT_DATA_DIR
from fpl_optimizer.data.loader import SeasonDataLoader
from fpl_optimizer.engine.engine import FPLGameEngine
from fpl_optimizer.prediction.integration import PredictionIntegrator
from fpl_optimizer.optimizer.squad_selection import select_squad
from fpl_optimizer.optimizer.transfer_optimizer import optimize_transfers
from fpl_optimizer.optimizer.types import build_candidate_pool, to_engine_action

# Load prediction model and run on a season
integrator = PredictionIntegrator.from_model(
    Path("models/point_predictor"), DEFAULT_DATA_DIR.parent, "2024-25"
)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/gameweek.py` | Weekly live loop: refresh data, predict, optimise transfers/lineup/captain |
| `scripts/train_predictor.py` | Retrain the model of record (committed recipe) |
| `scripts/backtest_season.py` | Leakage-free season backtest (eval model + MILP replay) |
| `scripts/benchmark_external.py` | Benchmark predictions against OpenFPL / theFPLkiwi |
| `scripts/oracle_comparison.py` | Compare model predictions vs oracle vs baselines |
| `scripts/collect_data.py` | Download all historical data sources |

## FPL Rules Encoded

The engine implements 2025/26+ FPL rules (verified unchanged for 2026/27 scoring) with a 305-test suite:

- **8 valid formations** (3-4-3 through 5-4-1), always 1 GK in starting XI
- **4 chips x 2 halves** (GW1-19, GW20-38) -- one chip per GW, unused first-half chips expire after GW19
- **Free transfer banking** up to 5 (Wildcard/Free Hit do NOT reset banked transfers)
- **Selling price** = purchase_price + floor(appreciation / 2)
- **Transfer hit** = 4 points per extra transfer beyond free allowance
- **Auto-substitution** walks bench in priority order, respects formation validity
- **Captain failover** -- if captain has 0 minutes, vice-captain gets the multiplier
- **Triple Captain** -- 3x multiplier instead of 2x
- **Bench Boost** -- all bench players' points count
- **Free Hit** -- unlimited transfers for one GW, squad reverts next GW

## Data Sources

| Source | Coverage | Features |
|--------|----------|----------|
| [vaastav](https://github.com/vaastav/Fantasy-Premier-League) | 2016-17 to 2024-25 | Points, minutes, goals, assists, xG/xA, ICT, prices, ownership, xP |
| [Understat](https://understat.com) | 2016-17 to 2024-25 | Per-match xG, xA, npxG, shots, key passes, xGChain, xGBuildup |
| [FBref](https://fbref.com) | 2016-17 to 2024-25 | Season-level passing, shooting, defense stats (prior-season features) |
| [FotMob](https://data.fotmob.com) | 2016-17 to 2024-25 | Pass completion, blocks, long balls (fills FBref gaps) |
| [The Odds API](https://the-odds-api.com) | 2020-21 to 2024-25 | Pinnacle pre-match odds (win/draw/loss implied probabilities) |
| [football-data.co.uk](https://football-data.co.uk) | 2016-17 to 2019-20 | Historical Pinnacle closing odds (fills Odds API gap) |

## Testing

```bash
pytest                            # All 305 tests
pytest tests/test_engine/ -v      # Engine unit tests
pytest tests/test_optimizer/ -v   # MILP + backtest tests
pytest tests/test_prediction/ -v  # Feature pipeline + model tests
```

Tests use hand-crafted CSVs in `tests/test_data/` (18 players, 2 GWs). The `SeasonDataLoader` is monkey-patched in test fixtures to skip downloads.

## Models

Pre-trained models are available in [GitHub Releases](../../releases):

- **point_predictor/** -- LightGBM point prediction (4 position-specific models)

Download and place in the project root:
```bash
# Models go in models/
models/point_predictor/  # LightGBM .lgb files + metadata
```

## Roadmap

- [x] **Stage 0:** Rules engine with full FPL rules and historical replay
- [x] **Stage 1:** LightGBM point prediction model (86 features, 0.787 per-GW correlation)
- [x] **Stage 2:** PuLP/CBC MILP optimizer for squad selection and transfers
- [ ] **Stage 3:** Multi-season holdout backtesting with confidence intervals
- [x] **Stage 4:** Live deployment (2026-27 season) — API-native data (`fpl_live.py`),
      live team state (`live/entry.py`), weekly driver (`scripts/gameweek.py`),
      committed training recipe (`scripts/train_predictor.py`)
- [ ] **Stage 5:** Multi-GW planning horizon + chip scheduler in the MILP
