# 2026-27 Season Operations Guide

The repo runs live this season. This is the operational handbook.

## Weekly loop (before every deadline — calendar events are already set)

```bash
# The one command (computes transfers + lineup + captain for your real team):
python scripts/gameweek.py --team-id <YOUR_ENTRY_ID>

# Then either apply manually on the site, or fully via API:
python scripts/gameweek.py --team-id <YOUR_ENTRY_ID> --apply          # dry-run validation
python scripts/gameweek.py --team-id <YOUR_ENTRY_ID> --apply --yes    # commit for real
```

Timing: run it the evening before the deadline (post-press-conference news is in
`chance_of_playing`), sanity-check again 1-2h before. If very close to the
deadline, add `--skip-refresh` (saves ~10 min of element-summary downloads).

Chip evaluation: re-run with `--chip wildcard|free_hit|bench_boost|triple_captain`
and compare objective values. Sanity check vs FPL's own EP: `--ep`.

### Multi-GW planner (`--horizon`)

```bash
# plan GW t..t+2 jointly around the approved chip schedule; executes GW t only
python scripts/gameweek.py --team-id <ID> --horizon 3 --chip-plan "tc:7,wc:11,bb:12,fh:18"
```

Prints the plan for every horizon GW (FTs available, moves, hits, captain,
xPts, bank) — only the first GW is applied; re-run every week (receding
horizon). It knows the FT bank (max 5, WC/FH keep the count), selling prices,
blanks/doubles in the fixture list, and plays around the given chips
(WC/FH free transfers, FH squad reverts, BB bench counts, TC ×3). It does NOT
choose chip weeks — the table below does.

Backtests (leak-free models, 2023-24/2024-25/2025-26, rules engine,
`scripts/backtest_horizon.py`): season results swing ±100 pts on small
changes, so treat differences under ~50 as noise.
- `--horizon 3` (defaults: discount 0.85, hit margin 4) ≈ +19 pts/season vs the
  single-GW optimizer without chips (6/9 replays won) and +24 with a chip plan,
  with half the hits (65 vs 121 pts/season).
- Without the hit margin the planner overtrades: −160 pts/season.
- The "max 1 transfer" policy is −66/season vs the unconstrained optimizer.
- Don't forbid hits outright (`--max-hits 0`) without chips: it is fine in
  quiet seasons but lost 310 pts in 2023-24, spread over the season and worst
  in the spring doubles (GW34, a 7-team DGW: −57) where the single-GW run
  paid hits to field doublers.
Use it every week (it is at least as good as the single-GW run and better
around chip weeks); cross-check hits with the single-GW run.

## One-time setup remaining (you)

1. **Your FPL team ID**: after GW1, from the Points page URL
   (`/entry/<ID>/event/1`). Put it in `.env` as `FPL_TEAM_ID=<id>` for reference.
2. **API write access** (so you never open the app): log in at
   fantasy.premierleague.com, DevTools Console:
   ```js
   copy(JSON.parse(localStorage.getItem(Object.keys(localStorage).find(k=>k.startsWith('oidc.user:')))).refresh_token)
   ```
   Paste into `.env` as `FPL_REFRESH_TOKEN=...`. Token rotation is handled
   automatically (`src/fpl_optimizer/live/auth.py`). If refresh ever 400/401s,
   re-extract from a fresh browser login.
   *Caveat: PL competition T&Cs void "script-generated entries" (prize
   eligibility, not bans — no documented ban for automating your own account).
   Keep volumes tiny; the site UI always shows what was submitted.*
3. **(Optional) Odds**: free account at the-odds-api.com (Starter, 500
   credits/mo — a live EPL h2h snapshot costs 1 credit/GW). Key → `.env`
   `ODDS_API_KEY`. Your own ablation says odds add ~nothing; skip guilt-free.

## Data & retraining

- **2025-26 backfill: DONE** (vaastav complete season, understat league +
  per-match, FotMob 27110, football-data odds 380/380).
- **Model of record**: `models/prod_2026-27` — produced by
  `python scripts/train_predictor.py` (RECIPE in the script: minutes-blend
  predictor, huber + calibration, unservable features excluded, 10 complete
  seasons + the current season's completed GWs; eval report in
  `training_report.json`). `gameweek.py` picks it up automatically (any
  model kind, via `load_predictor`).
- **Mid-season retrain** (~GW10, GW19, GW30 — the in-season gain grows with
  the rows): `/retrain` skill, i.e. `scripts/train_predictor.py
  --features-cache <file> --rebuild-cache`. No understat collection needed.
- **Lockdown rule**: GW scores are final 09:00 UK the morning after the GW's
  last match — never rebuild training data before that.

## 2026/27 rules (verified against bootstrap + official articles)

- Scoring **unchanged** from 2025/26, incl. DEFCON (+2: DEF ≥10 CBIT; MID/FWD
  ≥12 CBIRT; GK never; capped at 2/match).
- **BPS internals changed** (bonus only): tackled −1 removed, CBI 1-per-3,
  GK saves 2 + inside-box +1 + big-chance +1, pen save 8→7. Bonus-sensitive
  historical patterns from ≤2025/26 are slightly off — retraining absorbs most.
- Chips 4×2: **WC/FH not playable GW1** (start GW2); BB/TC from GW1; one chip
  per GW; first set **expires at GW19 deadline (Fri 1 Jan 2027, 18:30 UTC —
  re-read from the API 2026-09-18)**; FH in GW19 blocks FH in GW20.
- FTs: bank to 5; on a WC/FH week the banked count carries **unchanged**;
  hits −4; max 20 transfers/GW (no cap on WC/FH). Sell = purchase +
  floor(appreciation/2). No AFCON FT top-up this season.
- Prices: locked until GW1 deadline, then ±£0.1m/day at 00:00 UK; NEW official
  Price Change Predictor page (updates every 15 min) — useful before buying.

## Chip plan (approved 2026-09-18, after GW5)

Facts behind it: **no blank or double GW anywhere in GW1-19** (0 unscheduled
fixtures on 2026-09-18), so first-half chips are single-GW plays worth roughly
TC +8, BB +10-15, WC +15-25 — the second set is where the season is decided.
Haaland's soft home fixtures: GW7 IPS, GW9 BHA, GW11 FUL, GW13 LEE, GW16 HUL.
Decision points are in Google Calendar (green events, like the deadlines).

| Chip | Plan | Decision point |
|------|------|----------------|
| Triple Captain 1 | **GW7, Haaland home v Ipswich (Sat 17 Oct)** — promoted side, close enough to carry little injury/expiry risk. Fallback GW16 v Hull (H) | Fri 16 Oct, GW7 prep |
| Wildcard 1 | **GW11 (Sat 21 Nov)**, after the November break and the ~GW8 retrain, built for the festive run (GW12-19 = 8 GWs in 5 weeks). Pull forward to any GW6-10 if ≥3 starters are flagged/out; until then 1 FT/week handles the drift | Fri 23 Oct (retrain + early-pull check), Fri 20 Nov (build) |
| Bench Boost 1 | **GW12 or GW13** — first GW after the WC where the bench has 4 starters; the WC squad is built for it | the WC week |
| Free Hit 1 | Reactive (injury pile-up / bad fixture week). Hard fallback **GW18 (Tue 29 Dec)**, festive midweek rotation. **Never GW19**: FH there blocks FH2 in GW20 | Sat 26 Dec, GW17 prep |
| Second set (GW20-38) | Hold all four for the spring BGW/DGW map, revealed by the FA Cup R5 draw (~mid-Feb): WC2 the GW before the big DGW, BB on it, TC on a Haaland DGW, FH on the big BGW | mid-Feb 2027 |

Rules that shaped it: one chip per GW (so TC GW7 and BB cannot share a week);
WC/FH keep the banked FT count unchanged (rolling into a WC week is free);
first-half chips die at the GW19 deadline.

## Meta notes for 2026/27

- **Salah has left Liverpool.** Haaland (£15.5m, ~70% owned) is the one
  near-mandatory premium; skipping him is a large rank-risk. Our model agrees:
  he's its highest-rated player (8.66 xPts GW1).
- Template GW1: Haaland + Bruno Fernandes + João Pedro; promoted teams
  (Coventry, Hull, Ipswich): attacking picks only (Leif Davis £4.0m the
  standout), never trust their clean sheets.
- Early season: no hits GW1-4, bank FTs, act after press conferences.

## Known gaps (next build targets)

1. **Chip scheduler**: the multi-GW planner (`--horizon`, built 2026-09-24)
   plans transfers around a GIVEN chip schedule but does not choose chip
   weeks. Its gains are modest because the model's future-GW predictions add
   little beyond this week's (fixture swap: +0.02-0.04 Spearman vs persistence)
   — better multi-week predictions are the lever now.
2. **Bonus/BPS module**: 2026-27 BPS overhaul not modeled explicitly (only via
   realized totals in training data).
3. **Price-change modeling**: selling-price management is reactive, not planned.
