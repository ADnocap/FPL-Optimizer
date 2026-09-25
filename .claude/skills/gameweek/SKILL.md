---
name: gameweek
description: Run the weekly FPL pre-deadline routine — refresh live data, predict the upcoming GW, optimize transfers/lineup/captain for the user's real team, and present the recommendation. Use before each GW deadline or when the user asks "what should I do this gameweek".
---

# Weekly gameweek routine (live 2026-27 season)

## Steps

1. **Check the deadline first.** `GET https://fantasy.premierleague.com/api/bootstrap-static/`
   → events → the `is_next` event's `deadline_time`. Tell the user how long is left.
   Run the evening before (post-press-conference flags) AND again 1-2 h before the
   deadline; snapshots taken far from the deadline carry stale ownership/flags.
   `--skip-refresh` only if the element summaries were refreshed after the last GW's
   results (the driver refuses stale summaries).

2. **Run the driver** (`FPL_TEAM_ID` = entry 8737706 is in `.env`):
   ```
   python scripts/gameweek.py --team-id 8737706 --horizon 3 --chip-plan "<decided chips>"
   python scripts/gameweek.py --team-id 8737706 --skip-refresh          # single-GW cross-check
   ```
   The 3-GW planner is the primary recommendation: it beat the single-GW
   optimizer in every leak-free season replay (2023-24..2025-26, +60 to +149).
   `--chip-plan` = chips from SEASON_GUIDE.md that are decided and fall within
   the next 3 GWs (e.g. `"tc:7"` from GW5-7). Never change the plan without the user.
   - Chip decision points: `python scripts/chip_eval.py --team-id 8737706` (the
     calibrated columns are the ones to trust; replays can't credit clearing
     injured players, so weigh the squad's actual problems).
   - Sanity check: `--ep` runs the single-GW optimizer on FPL's own EP.

3. **Read the output before presenting**:
   - **DATA HEALTH block**: any line there means the model is seeing a feature
     empty/constant that it had in training (the silent failure that wrecked
     GW1-5 2026-27). Fix the data first (usually a failed odds/props/understat
     refresh or a missing snapshot) or present the plan with that caveat.
   - **Hits**: the single-GW optimizer takes a hit when it promises > 4 points
     (`--hit-margin`, default 0 — a margin of 2 lost in all three model-v5 season
     replays); the planner uses 4 + 4. Still mention every hit explicitly and
     compare with `--max-transfers <FTs>`.
   - **Captaincy view**: model xPts vs FPL EP vs bookmaker P(goal). The call is the
     user's — in GW1-5 the human/market captain beat the (then leaky) model's
     every time. Never captain a flagged player.
   - **Flags**: availability enters once, through the pool's calibrated
     multiplier (75% flag → 0.40x, 50% → 0.25x, 25% → 0.10x; in 2026-27 a 75%
     flag meant ~25% chance of playing). Point out flagged starters and whether
     the bench can cover them (0-minute bench players = no auto-sub).
   - DGW/BGW: fixture counts in `data/raw/2026-27/fixtures.csv`.

4. **Present**: transfers (out→in with prices), XI + formation, captain/vice,
   bench order, expected points, hit cost if any, the data-health status, and the
   decision-log path. Apply only when the user says so, with the SAME flags as the
   recommended run: `python scripts/gameweek.py --team-id 8737706 --horizon 3
   --chip-plan "<chips>" --apply --yes` (the planned chip for this GW is played; with
   the single-GW run pass `--chip <name>` in a chip week). It plans on the
   authenticated my-team state and refuses if transfers were already made on the
   site or a different chip is active there. Auth is `FPL_REFRESH_TOKEN` in `.env` (single-use rotating
   token — see the live-season memory for re-extraction; close the site tab after).
   When recommending players by name, ALWAYS disambiguate with club + position
   (two Palmers exist: Cole Palmer CHE MID vs Alex Palmer IPS GK).

5. **After the run**: the decision log is at
   `data/live/2026-27/decisions/gw{N}_*.json` (predictions, plan, overrides, data
   health). Keep it — post-mortems depend on it.

## Failure modes

- Model predictions empty → season files not rebuilt; run without `--skip-build`.
- `element-summary refresh incomplete` → network hiccup; re-run (old files kept).
- `--skip-refresh but the element summaries predate GW n` → run without it.
- Understat/odds refresh WARNING → the data-health block will show the gap;
  odds come from football-data.co.uk (upcoming fixtures appear ~2 days before
  the weekend), props from The Odds API (paid plan; office Zscaler blocks it).
- After the GW completes, scores are final 09:00 UK the NEXT day (2026-27
  lockdown rule) — don't rebuild training data before that.

## Weekly cadence (already in the user's Google Calendar, green events)

Deadlines vary: Fri 19:30, Sat 11:00-15:30, some midweek — always read
`deadline_time` from the API, never assume.
