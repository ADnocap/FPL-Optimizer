# 2026-27 Season Operations Guide

The repo runs live this season. This is the operational handbook.

## Weekly loop (before every deadline — calendar events are already set)

```bash
# The weekly run: 3-GW planner, only this GW's moves are executed
python scripts/gameweek.py --team-id <YOUR_ENTRY_ID> --horizon 3 --chip-plan "tc:7"
# Cross-check: single-GW optimizer
python scripts/gameweek.py --team-id <YOUR_ENTRY_ID> --skip-refresh
# Apply (plans on the authenticated my-team state; commits in one go):
python scripts/gameweek.py --team-id <YOUR_ENTRY_ID> --horizon 3 --chip-plan "tc:7" --apply --yes
```

`--chip-plan` lists only chips already decided that fall inside the next three
GWs (the planner plays around them: TC ×3, BB bench counts, WC/FH free moves, FH
revert). In a chip week the applied run plays the planned chip; with the
single-GW run pass `--chip <name>` instead.

Every run prints, in order:
1. **DATA HEALTH** — features the model will see empty/constant vs its training
   reference. Any line = fix the data first (or read the plan with that caveat).
   Expected before props are quoted (Odds API player props appear ~2 days before a
   weekend): the six `props_*` lines.
2. Team state (bank, FTs), the multi-GW plan, transfers, XI, bench.
3. **Captaincy view** — model xPts vs FPL's EP vs bookmaker P(goal). The call is yours.
4. The decision-log path (`data/live/2026-27/decisions/`).

Timing: run it the evening before the deadline (post-press-conference flags),
again 1-2 h before. `--skip-refresh` is refused until the summaries postdate the
last GW's final scores (09:00 UK the day after its last match).

Flags: availability enters once, as a calibrated multiplier (75% flag → 0.40×,
50% → 0.25×, 25% → 0.10×; a 75% flag meant ~25% chance of playing in GW1-5).

### Why the planner is the default

Leak-free season replays (evaluation model trained only on earlier seasons, every
GW through the rules engine, `scripts/backtest_horizon.py`), net points:

| | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|
| `--horizon 3` (hit margin 4) | **2,293** | **2,467** | **2,129** |
| single-GW, max 1 transfer | 2,208 | 2,342 | 2,048 |
| single-GW (4-pt hit rule) | 2,144 | 2,333 | 2,069 |

- The planner won every season (+60 to +149 vs single-GW). A 6-GW horizon did
  not help; neither did a hit margin in the single-GW run.
- Chip schedules in replays (TC7/BB12/FH18; a GW11 Wildcard) moved season totals
  by -145 to +84 and average ~0: within single-replay noise, and replays have no
  injury flags, so they cannot credit a Wildcard's real job (clearing unavailable
  starters). Chip timing stays the plan below, checked with
  `python scripts/chip_eval.py --team-id <ID>` at each decision point
  (its calibrated columns: 2025-26 realised/predicted ratios, split by season half).
- `--wc-move-cost C` makes each Wildcard-week move clear C planned points (off by
  default — no evidence either way yet); worth considering when building WC1.

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
3. **(Optional) Player props**: The Odds API key in `.env` (`ODDS_API_KEY`) with a
   prop-capable plan (~40 credits/GW). Props are a small input; if they are missing
   the model degrades gracefully and the data-health block says so. The office
   Zscaler network blocks the API — run from home. h2h match odds are NOT needed.

## Data & retraining

- **Model of record**: `models/prod_2026-27` — minutes-blend v5 (121 features),
  trained by `scripts/train_predictor.py` on 2016-17..2025-26 + 2026-27 GW1-5.
  Per-GW Spearman 0.742 / 0.732 on the 2025-26 / 2024-25 holdouts, 0.711 live GW1-5.
  Rollback: `models/prod_2026-27.prev`. Pre-v5 models (trained on the leaky
  same-GW xP) are in `models/archive/` and refused by the live leak guard.
- **Retrain** at ~GW10, GW19 (before the chip expiry), GW30: `/retrain` skill, i.e.
  `python scripts/train_predictor.py --features-cache <file> --rebuild-cache
  --second-fold` (promotion gate in `.claude/skills/retrain/SKILL.md`). Nothing
  beyond the FPL API is needed.
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

1. **Chip scheduler**: chip weeks are a human decision; `chip_eval.py` prices them,
   the planner plays around a given schedule.
2. **Haulers**: every model under-predicts 5+ point returns (~3 predicted vs ~7.8
   actual); captaincy relies on the captaincy view, not the point estimate alone.
3. **Bonus/BPS module** and **price-change planning**: not modelled.
