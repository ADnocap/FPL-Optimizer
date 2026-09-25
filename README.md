# FPL-Optimizer

Predict-and-optimize system for Fantasy Premier League: a LightGBM point predictor
with a minutes model feeds a mixed-integer optimizer that picks transfers, lineup,
captain and bench under the full FPL rules — single-GW or planned over the next
three gameweeks. It runs live for the 2026-27 season.

```bash
python scripts/gameweek.py --team-id <ENTRY_ID> --horizon 3   # the weekly run
python scripts/gameweek.py --team-id <ENTRY_ID>               # single-GW cross-check
python scripts/chip_eval.py --team-id <ENTRY_ID>              # value of every remaining chip
python scripts/train_predictor.py --features-cache <f.parquet> --rebuild-cache --second-fold
```

## Results (honest, leak-free)

**Predictor** — mean per-gameweek Spearman correlation between predicted and actual
points, model trained only on earlier seasons:

| | 2024-25 holdout | 2025-26 holdout | 2026-27 live, GW1-5 |
|---|---|---|---|
| **Model of record (v5)** | **0.732** | **0.742** | **0.711** |
| Leak-fixed LightGBM baseline | 0.719 | 0.726 | 0.685 |
| FPL's own expected points (live) | — | — | 0.631 |

**Season replays** — chip-free, the evaluation model trained only on seasons before
the replayed one, every gameweek replayed through the rules engine (auto-subs,
captain failover, FT banking, hits). Net points:

| Strategy | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|
| Multi-GW planner (`--horizon 3`, hit margin 4) | **2,291** | **2,414** | **2,108** |
| Single-GW optimizer (4-point hit rule) | 2,174 | 2,303 | 2,086 |
| Single-GW, at most 1 transfer/GW | 2,126 | 2,153 | 2,069 |

Season-level differences under ~100 points are within the path noise of a single
replay.

> **Withdrawn numbers.** Earlier versions of this README reported 2,918 / 3,171 on
> 2024-25 and a 0.787 per-GW correlation. They came from a feature that leaked
> post-match information: vaastav's `xP` is FPL's `ep_this` read after the
> gameweek, when FPL has already recomputed it with that gameweek's own points.
> The model now uses the previous gameweek's value (`fpl_xp_lag`), which is exactly
> what is known at a deadline. Details: `reports/2026-27_gw1-5_review.md`.

## How it works

```
FPL API ──LiveFPLCollector──> data/raw/2026-27 (vaastav format; synthetic rows for the
                              upcoming GW; xP with vaastav's post-GW semantics from
                              the pre-deadline snapshots)
        ──FeaturePipeline──> 121 features per (player, GW), point-in-time
        ──MinutesBlendPredictor──> expected points
              = ½ · [P(60+)·E(pts|60+) + P(1-59)·E(pts|1-59)]  (minutes model)
              + ½ · stacked points model (features + P(play), P(60+))
              huber loss, monotone calibration
        ──live pool──> calibrated availability multipliers (75% flag → 0.40×)
        ──MILP──> transfers, XI, captain/vice, bench (single GW or 3-GW plan with
                  FT banking, hit margin and a chip schedule)
```

- **Point-in-time discipline.** Post-match features come from gameweeks before the
  one predicted; pre-match features (price, ownership, fixture) from the deadline.
  Features that cannot be served at a live deadline are not trained on
  (`prediction/feature_sets.py`). Each weekly run prints a data-health block that
  compares the live feature coverage against the model's training reference.
- **Rules engine.** Stateless `step(GameState, EngineAction)`; the backtests, the
  planner and a 207-test rules suite (checked against the API's own 2026-27 chip and
  game settings) all use it.
- **Human in the loop.** Every run writes a decision log; captaincy is shown as model
  vs FPL EP vs bookmaker probability; hits need a margin; chips are a human decision
  informed by `chip_eval.py`.

## Layout

```
src/fpl_optimizer/
├── data/        collectors (FPL API live season, vaastav, understat, FotMob, odds, props), loader
├── engine/      stateless rules engine (scoring lookup, transfers, chips, auto-subs)
├── prediction/  feature pipeline + modules, minutes model, MinutesBlendPredictor, horizon predictions
├── optimizer/   squad / transfer / lineup MILPs, multi-GW horizon planner, chip evaluation, backtests
├── live/        entry state, candidate pool, live predictions + data health, API executor
└── utils/
```

## Install and test

```bash
git clone https://github.com/ADnocap/FPL-Optimizer && cd FPL-Optimizer
pip install -e ".[dev]"      # Python 3.11+
pytest                       # 638 tests
```

Tests use hand-crafted CSVs in `tests/test_data/`; the rules suite
(`tests/test_rules/`) uses a trimmed copy of the 2026-27 API config and a real entry's
GW1-5 history.

## Data sources

| Source | Used for |
|---|---|
| Official FPL API | the live season (element histories, fixtures, pre-deadline snapshots) |
| [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) | 2016-17..2025-26 history |
| [FotMob](https://data.fotmob.com) | prior-season passing/defence stats |
| [The Odds API](https://the-odds-api.com) | player-prop odds (optional; live snapshots) |
| [Understat](https://understat.com), [football-data.co.uk](https://football-data.co.uk) | research only — their per-match / h2h features are not in the model (not reliably servable at a deadline; neutral once leak-free) |

## Roadmap

- [x] Rules engine, historical replay, 207-test rules suite
- [x] Leak-free predictor with a minutes model (v5)
- [x] MILP optimizer; multi-GW horizon planner with chip schedules
- [x] Live 2026-27 operation with data-health checks and decision logs
- [ ] Chip scheduler (chip timing is a human decision, `chip_eval.py` gives the numbers)
- [ ] Better multi-week predictions and a bonus-points model
