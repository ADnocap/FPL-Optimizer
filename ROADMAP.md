# Model & Optimizer Improvement Roadmap

From the 2026-08-21 research sweep (5 research agents: academic literature,
player-prop betting markets, open-source landscape, data/features, modeling
techniques + synthesis). Baseline being improved: LightGBM 107 feats,
2025-26 holdout MAE 0.825 / per-GW corr 0.631; single-GW MILP;
honest chipless backtest 2,261 net (@1 FT) / 2,566 (unconstrained).

## Top 10 (ranked by expected impact for this repo)

1. **Expected-minutes (xMins) model** — unanimous #1 across all reports.
   Two-stage: P(minutes band 0 / 1-59 / 60+) GBM × E[pts | band].
   Evidence: OpenFPL (arXiv 2508.09992) shows the commercial ceiling's only
   remaining edge over public models is minutes prediction (Zeros RMSE 0.689
   vs 0.818). Inputs already on disk (rolling starts, days-rest, congestion).
   Lift: MAE → ~0.78-0.80, +30-80 pts/season via rotation traps, bench order,
   per-fixture DGW EV. Effort: ~1 week + inference news-scrape wiring.
2. **Benchmark before building** — score OpenFPL's MIT pretrained models +
   theFPLkiwi archived projections on OUR 2025-26 holdout; add
   Zeros/Blanks/Tickers/Haulers conditional RMSE to evaluation.py as the
   permanent diagnostic. Tells us whether zeros (→ #1) or haulers (→ #7) is
   the real weakness. Effort: ~1 day. Do first.
3. **Multi-horizon forecasts (GW+1..GW+6)** — hold form features, swap
   fixture features per future GW; emit the community {gw}_Pts/{gw}_xMins CSV
   schema (enables cross-checking vs reference solvers). Prerequisite for #4.
4. **Multi-period transfer MILP** — port sertalpbilal/FPL-Optimization-Tools
   formulation onto our PuLP infra: decayed horizon objective (0.9^w), FT
   banking as integer state [1,5], terminal FT value (~1.5, nonlinear),
   ITB value (~0.08/£1m). Our own backtest brackets the headroom: 305 pts
   between 1-FT and myopic-unconstrained. Lift: +40-100 pts/season.
   **BUILT 2026-09-24** (`optimizer/horizon_optimizer.py`,
   `prediction/horizon.py`, `gameweek.py --horizon N --chip-plan ...`,
   `scripts/backtest_horizon.py`). Leak-free replays 2023-24..2025-26:
   H=3 + hit margin 4 = +19/season vs the single-GW MILP (6/9 replays won),
   +24 with a chip plan; unregularised H=4 = −160 (hit churn). The headroom
   above was mostly the 1-FT cap (−66/season vs unconstrained). Next lever:
   better future-GW predictions (fixture swap adds only +0.02-0.04 Spearman
   over persistence) — e.g. the xMins model (#1).
5. **Chips as MILP decision variables** — use_wc/fh/bb/tc binaries, FH as
   parallel one-week squad, BB extends lineup, TC ≤ captain; ingest Ben
   Crellin's DGW/BGW sheet. Backtests are chipless today: +60-150 pts/season.
   Effort: days once #4 exists. **Partly built**: the planner executes a
   GIVEN chip schedule (WC/FH squad revert/BB/TC, FT carry rules) and
   `backtest_horizon.py` replays chip plans; choosing chip weeks is not built.
6. **Zero-new-data feature bundle** — team attacking share, minutes-share in
   position group, venue-split form, opponent-adjusted rolling, per-90
   decomposition, congestion (days-rest, matches-in-14d, Euro/cup via
   FPL-Elo-Insights CSVs — also replaces dead FBref + adds team Elo).
   Lift: GW-corr +0.005-0.015. Effort: days, all data on disk.
7. **Loss/weighting experiments** — entropy-binned balanced sample weights
   (OpenFPL's trick), tweedie/poisson objective, per-GW Spearman as the CV
   selection metric (we consume rankings, not MAE). Effort: ~1 day A/B.
8. **Player-prop odds pipeline** — the only genuinely NEW market signal
   (props ≠ redundant h2h: finishing + team news + minutes in one number).
   Gate: probe The Odds API historical archive (~80 credits) → if props
   exist for 2023-25, one $30 month backfills 2025-26 (AGS/assists/SOT/cards,
   Shin devig, consensus across books); live = 40-80 credits/GW (free tier).
   Judge on hauler-MAE + captain accuracy, not global MAE. +20-50 pts plausible.
9. **Odds-derived clean-sheet features** — Dixon-Coles on the Pinnacle h2h
   already on disk → P(CS), E[goals against] for GK/DEF boosters. Our +0.07%
   null result was the wrong transformation, not a dead source; DEF was the
   only position odds helped. Effort: days.
10. **Decision quality under uncertainty** — (a) now: ~30-run sensitivity
    re-solves with minutes-scaled noise → "%-of-runs" buy/captain tables;
    (b) after #5: quantile/distributional head for TC/BB week choice (chip
    timing needs upside tails; plain captaincy stays argmax-mean — already
    EV-optimal under linear 2x).

## Before GW2 (realistic in <5 days)

- [ ] #2 benchmark + stratified metrics (~1 day)
- [ ] bench_weights {GK .03, B1 .21, B2 .06, B3 .002} + vcap 0.1 in the
      lineup/transfer objectives (hours, free EV)
- [ ] #7 loss/weighting A/B in TemporalCV (~1 day)
- [ ] Props archive probe (~80 credits, hours) → go/no-go for #8
- [ ] Set-piece features from live `api/team/set-piece-notes/` endpoint
      (removes the players_raw mild lookahead; catches in-season changes)
- [ ] Empirical-minutes stopgap in gameweek flow: scale EV by starts-share ×
      chance_of_playing until #1 lands (~half day)
- [ ] First two bundle features: venue-split form + team attacking share

## Evidence-based SKIP list (don't spend time here)

- **Deep sequence models (transformer/LSTM)**: every published "win" is vs
  crippled baselines; our holdout beats everything published. Revisit only
  as an ensemble member.
- **News-text/sentiment**: tested in literature — no signal (arXiv 2405.02412).
- **StatsBomb free data**: EPL open data = 2015/16 only. Verified dead end.
- **Full component decomposition** (goals/assists/bonus submodels): OpenFPL's
  monolithic beats component-based; AIrsenal (canonical decomposition) landed
  mid-pack→near-bottom live. Only minutes (#1) and CS (#9) survive.
- **Sofascore/FotMob ratings backfill**: 40+ hrs of rate-limited scraping for
  a composite correlated with features we have.
- **Betfair/oddschecker props scraping**: strictly dominated by one $30 Odds
  API month.
- **Weather, referee features**: effect ≈ 0 at player-points granularity.
- **Cross-league augmentation**: ~1.3% MSE in one paper, didn't transfer.
- **Transfermarkt values**: redundant with price/ownership/ICT.
- **Championship priors for promoted players**: right idea, wrong timing —
  in-season rolling covers them by ~GW6; do next pre-season.
- **Price-predictor scraping**: valuable only after #4's ITB machinery exists.
- **Pooled position-embedding model**: no published win vs per-position; at
  most an ensemble A/B during #7.

Full research reports: session task output (5 reports + synthesis, 2026-08-21).
