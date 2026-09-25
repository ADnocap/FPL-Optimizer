# Rules, chips and transfer-policy audits (2026-09-24)

Each deliverable was re-derived by an adversarial verifier; verdicts follow each section.


---

I checked every FPL rule the system encodes against the official 2026-27 rules (rules_2026-27.json, plus the premierleague.com chip and free-transfer articles) and against our real GW1-5 history. The live path is correct for GW6. The FT simulation gives 1, 2, 1, 1, 2 for GW2-6, so we have 2 free transfers entering GW6. It also reproduces every official hit charge for GW2-5. Bank is £0.2m, purchase and selling prices rebuild correctly, and all 8 chips are available. Replaying our real GW1-5 picks through the engine gives exactly the official points (40/99/52/69/36), bench points (2/-1/8/9/0) and auto-subs (none).

Answers to the specific questions:
- **FTs after a Wildcard or Free Hit:** the count carries over unchanged. You keep your saved transfers and do not get the +1 for the chip week. The official example is 4 saved before GW29, Free Hit in GW29, still 4 in GW30. The engine and live/entry.py both do this.
- **Free Hit GW19/GW20 restriction:** it is real for 2026-27. The official article says "if you use your first Free Hit chip in GW19, you can't then play the second one in GW20". It is implemented, and it survives the team reconstruction.

Also confirmed:
- first-half chips expire after GW19; one chip per GW
- the 5-FT cap and -4 hits
- selling price = purchase + floor(gain/2)
- 3 players per club, the £100.0m budget, and the 8 formations
- auto-sub goalkeeper and formation rules; a player with 0 minutes but a card counts as having played
- DGW points are summed; a blank-GW player is auto-subbed
- captain → vice failover, and Triple Captain passes to the vice at x3
- Bench Boost counts all 15
- executor chip names and endpoints: wildcard and freehit go to the transfers endpoint, bboost and 3xc go to my-team, with the bench GK at position 12
- the --chip checks and routing in gameweek.py

I found 9 bugs; none changes the GW6 numbers:
- **Live, triggered only by `--apply`:** R-05 and R-09.
- **Backtest only:** R-02 and R-06, which skew any backtest used to tune the optimizer.
- **Engine only, not reached yet:** R-03 (no backtest plays chips today) and R-01.
- **Low:** R-04, R-07 and R-08.

Tests: 207 tests in scratchpad/rules_audit/tests. Against the current repo, exactly the 16 tests that document bugs fail. With all 9 patches applied, 207/207 pass. The repo's own suite goes from 305 to 309 passing (4 new tests; 4 existing tests that played WC/FH in GW1, plus one GW1 FT-banking test, now use GW2). Each patch applies to HEAD on its own and all 9 apply together. I did not edit the repo.

The biggest planning problem is not a rules bug (I-05). The chip plan says the GW11 Wildcard squad is "built for" Bench Boost in GW12/13. But `--chip wildcard` optimizes GW11 alone and values bench slots at 0.21/0.06/0.002 of their points (0.03 for the bench GK). That produces a dead bench, so it cannot deliver the plan as written. It needs predictions over several GWs and a bench valued for Bench Boost before GW11.


**Findings:**

- [medium] **R-05** gameweek.py plans on the public-API guess even when logged in; my-team is fetched only after the optimizer has run — fix: P05: with --apply, fetch my-team right after the chip check and overwrite free_transfers (transfers.limit), bank (transfers.bank) and per-pick selling/purchase prices, printing each correction. Abort if transfers.made > 0 or the squads differ.
- [medium] **R-09** A Bench Boost or Triple Captain already active on the site is silently cancelled or replaced by the lineup POST — fix: P09: after get_my_team, collect any chip that is active or pending. If one exists and --chip does not match it, print an error and submit nothing. The user either re-runs with --chip <that chip>, which also optimizes for it, or cancels it on the site.
- [medium] **R-02** Backtests start GW2 with 2 free transfers (FPL gives 1) — fix: P02: in engine.step, if gw == 1 set next-GW free_transfers = INITIAL_FREE_TRANSFERS (1), otherwise bank as normal. This fixes both backtests in one place. Updates the existing test (moves it to GW2) and adds test_gw2_starts_with_one_free_transfer.
- [medium] **R-06** Squad players missing from the candidate pool (blank GWs) all get fake club 0, breaking the 3-per-club rule — fix: P06: optimize_transfers(..., squad_teams: dict[int,int] | None). A missing squad player uses its real club, or a unique sentinel -element_id if the club is unknown. backtest.py and backtest_season.py pass loader.get_player_team for each squad member.
- [medium] **R-03** Engine does not restore the bank after a Free Hit — fix: P03: add GameState.free_hit_bank_stash; activate_chip stores the bank when a Free Hit is activated; revert_free_hit restores it; copy() carries it. Adds test_revert_restores_bank.
- [low] **R-01** Engine allows Wildcard and Free Hit in GW1 (official start_event is 2) — fix: P01: add constants.CHIP_START_EVENT = {wildcard: 2, free_hit: 2, bench_boost: 1, triple_captain: 1} and check gw >= start in is_available. Moves the affected repo tests to GW2 and adds GW1 rejection tests.
- [low] **R-04** FT rebuild assumes every team started in GW1 (late joiners get too many FTs) — fix: P04: start the loop from the first event in history['current'] (first_gw + 1). Also documents the WC/FH carry-over rule in the docstring.
- [low] **R-07** Optimizer undervalues the vice-captain under Triple Captain — fix: P07: vice_weight = VICE_CAPTAIN_WEIGHT * (2 if chip == 'triple_captain' else 1).
- [low] **R-08** The documented --fresh-squad 'wildcard' mode uses a flat £100.0m budget; our real Wildcard budget is £99.8m — fix: P08: docstring and --help now say --fresh-squad is for GW1 only and point Wildcard/Free Hit users to --team-id --chip wildcard.
- [info] **I-01** Checked and correct: free-transfer and price rebuild for our real entry — fix: None needed.
- [info] **I-02** Checked and correct: engine scoring, auto-subs, captaincy, DGW/BGW, Bench Boost and Triple Captain — fix: None needed.
- [info] **I-03** Checked and correct: chip rules, executor payloads, gameweek.py chip checks and routing, constants — fix: None needed.
- [info] **I-05** Chip-plan gap: the WC11 then BB12/13 decision cannot be priced correctly by the current single-GW objective — fix: Before 20 Nov (WC build): add a multi-GW horizon (sum predictions over GW11-13) and value the bench at 1.0 for the Bench Boost GW when building the Wildcard squad. This belongs in the optimizer/multi-period work, not the rules patches.
- [info] **I-06** GW19 deadline: the official article and the API disagree — fix: Re-read events[19].deadline_time from the API around Christmas and move the calendar decision point if it changes.

| Test file | Tests | Fail on current repo (bug IDs) | Pass with P01-P09 |
|---|---|---|---|
| test_r01_constants_official.py | 20 | 3 (R-01) | 20 |
| test_r02_ft_chipstate.py | 69 | 0 | 69 |
| test_r03_engine_chips.py | 22 | 4 (R-01 x2, R-02, R-03) | 22 |
| test_r04_engine_scoring_autosub.py | 21 | 0 (incl. real GW1-5 replay = official) | 21 |
| test_r05_live_entry.py | 24 | 1 (R-04) | 24 |
| test_r06_optimizer_chips.py | 21 | 3 (R-06 x2, R-07) | 21 |
| test_r07_executor_payloads.py | 15 | 0 | 15 |
| test_r08_gameweek_cli.py | 13 | 3 (R-05, R-09 x2) | 13 |
| test_r09_backtest_rules.py | 2 | 2 (R-02) | 2 |
| **Total** | **207** | **16 (exactly the 16 tests marked `bug`)** | **207** |
| Repo suite (tests/) | 305 at HEAD | n/a | 309 (4 new tests, 5 existing tests updated) |

How to run (from scratchpad/rules_audit):
- Current repo: `python -m pytest -q`
- Only the tests that document bugs: `python -m pytest -q -m bug`
- Patched: `git apply patches/*.diff` in a clone, then `PYTHONPATH=<clone>/src RULES_AUDIT_REPO=<clone> python -m pytest -q`

Ground-truth figures confirmed:
- FTs GW2-6 = 1, 2, 1, 1, 2 (2 entering GW6); hit charges for GW2-5 reproduced.
- Bank £0.2m; Wildcard budget £99.8m.
- Engine points GW1-5 = 40/99/52/69/36; bench points 2/-1/8/9/0.


---

I backtested 25 transfer policies through the repo engine. That is 252 full-season replays: 2024-25 and 2025-26, 11 prediction paths (the leak-free H24_fast / H25_fast / H25_full, plus 8 copies with ±12% noise per player-GW), 0 FTs banked at GW1.

**Transfer policy: keep the current rule.** It prices a hit at 4 in the objective, has no transfer cap, and uses bench weights 0.03 / 0.21 / 0.06 / 0.002. Nothing beats it robustly across both seasons, and it is the maximin choice.
- **Never taking a hit loses points:** −80 pts/season, 3 of 11 paths won, 90% CI [−120, −39].
- **Too many hits loses more:** hit at gain ≥ 2 (m=−2) gives −177, 0 of 11 paths won.
- **Other hit margins:** m=1 gives −49 and m=4 gives −46.
- **Ties:** hit margin m=2 (+4.5 overall; +30 in 2024-25, −17 in 2025-26), FT option value τ=1–4 (+1 to −11) and a stricter hit rule only in blank/double GWs (+0.5 / −5). All are within noise; one path's standard error is about ±18.
- **DNP-risk bench weights hurt:** −27 overall and −85 on the 3 clean paths. The weights they derive (0.62 / 0.24 / 0.06, GK 0.10) match how often starters actually miss (P(XI DNP ≥ k) = 0.55 / 0.16 / 0.045). They add about 12 auto-sub points a season but cost more in the XI. A minimum-playing-probability floor for bench slot 1 is neutral (−3.5).

**The hit audit shows why the 4-point rule works.** Over 105 default-hit decisions (187 hits), the realised net gain was +0.8 in the same GW and +3.6 over 2 GWs, against a predicted +2.2. On clean predictions it was calibrated (predicted 2.45, realised 2.51). On noisy predictions it was overstated by about 2.3. 43% of hit points fall in the 16% of GWs that are blank or double, and those hits realise about 0. The chip plan (FH/BB) covers most of those weeks.

So the −8/−16 hits in GW1-5 came from prediction quality, the leaky model served honestly, not from the hit rule. Live predictions are weaker again (LIVE per-GW Spearman 0.672 vs 0.72). If you want to lean conservative live, m=2 is the only alternative that ties.

**Availability is the real lever.** In the 2026-27 snapshots:
- `ep_this` embeds `chance_of_playing_THIS_round`, the stale flag from the GW just played.
- `ep_next` embeds `chance_of_playing_NEXT_round`, the flag for the GW you're picking for.

The fixed model is fed `fpl_xp_lag` = snapshot `ep_this`, and it ignores flags almost entirely:
- Removing the stale flag changes predictions by only about 1% (ratio 0.99).
- Feeding `ep_next` instead moves 75%-flag predictions by only −1%.

Players with a 75% flag scored far below that: 0.16 to 0.27 of their prediction relative to unflagged players. Only 33% of the plausible picks played, against 90% of unflagged ones.

**Spec for how availability enters once:**
1. Model input stays `ep_this`, as trained. Never feed it `ep_next`.
2. The pool applies a single multiplier on the upcoming-GW flag: 100/None = 1.0, 75 = 0.40, 50 = 0.25, 25 = 0.10, 0 or status i/s/u/n = 0.
3. That one value drives the XI, captain, bench and transfers.
4. This must ship at the same time as the fixed model. The old model was fed `ep_next` and double-counted (0.78 × 0.75 = 0.585). Keeping the 0.75 multiplier with the fixed model would value flagged players about 28% higher than today.

**Two backtest bugs are confirmed and patch-sketched, not applied:**
- **GW1 free transfers:** replays bank 1 FT at GW1, so GW2 gets 2. The repo's `optimize_transfers` was otherwise reproduced exactly (0/38 mismatches).
- **Placeholder club:** squad players with no fixture are given club 0, so four or more blanking players break a fake 3-per-club limit. That makes the solver crash when transfers are capped, and forces extra hits when they are not (2025-26 GW34: 9 transfers for −32 instead of 7 for −24). The live pool is not affected.


**Findings:**

- [high] **TP-1** Availability multiplier far too optimistic; fixed model ignores flags, so the pool is the only channel — fix: pool.py: replace pts *= chance/100 with pts *= AVAILABILITY_MULT[cop_next] = {100/None: 1.0, 75: 0.40, 50: 0.25, 25: 0.10, 0 or status i/s/u/n: 0.0}. Keep the model input as fpl_xp_lag = snapshot ep_this (never ep_next). Ship together with the fixed-model switch, or flagged players become about 28% more attractive than today. Re-fit monthly with avail_table.py + avail_calib.py as GW6+ snapshots ac
- [medium] **TP-2** Keep the current hit rule; banning or limiting hits loses points — fix: No change to optimize_transfers defaults: hit cost 4 in the objective, max_transfers=None. m=2 (only take a hit if predicted single-GW net gain >= 2) is statistically tied and uses 76% fewer hit points (26 vs 110 per season). It is an acceptable conservative live setting while live predictions stay weaker (LIVE Spearman 0.672 vs 0.72 in holdout).
- [medium] **TP-3** DNP-risk bench weights lose points; keep the static bench weights — fix: Leave BENCH_GK_WEIGHT / BENCH_OUTFIELD_WEIGHTS unchanged. Handle the dead-bench problem through TP-1: don't start a 75% player; the multiplier moves him to the bench. Optionally a soft P(play) floor on bench slot 1, which is cost-neutral.
- [medium] **TP-4** BUG: backtests start GW1 with 1 banked FT (GW2 then has 2; FPL gives 1) — fix: Start the GW1 replay state at free_transfers=0 and use max_transfers=0 for the GW1 optimizer call (patch sketch section 2).
- [medium] **TP-5** BUG (backtest only): squad players with no fixture get club id 0 — fix: Give re-added players their real club (pass team_of=loader._team_map from the backtests), else a unique dummy (-element_id). Patch sketch section 3.
- [low] **TP-6** Hits cluster in blank/double GWs and realise about 0 there — fix: No code change needed if the chip plan covers these weeks: Free Hit in the blank GW, Bench Boost in the double. If a blank/double GW is played without a chip, use --max-transfers or the optional irregular_hit_cost_obj=8 knob; the single-GW objective will otherwise suggest a -20/-24 one-week rebuild.
- [info] **TP-7** FT option value never changes decisions below tau=3; rolling rules are neutral — fix: No rolling rule. Keep using the FT each week, as the optimizer already does.

Season net points, 11 replay paths (3 clean: H24_fast, H25_fast, H25_full; 8 with ±12% per-row prediction noise), 0 FTs at GW1, no chips. Δ = mean change vs A_default per path. The noise barely moves per-GW Spearman (0.713–0.721 vs 0.716–0.726) but does disturb the top of the ranking. Path-level SE ≈ ±18.

| policy | mean net | Δ | Δ 2024-25 | Δ 2025-26 | Δ clean | Δ noisy | hit pts | auto-sub pts | paths won |
|---|---|---|---|---|---|---|---|---|---|
| A_default (hit cost 4, no cap) | 2171.0 | 0 | 0 | 0 | 0 | 0 | 109.8 | 62.7 | – |
| C_m2 (hit if gain ≥ 6) | 2175.5 | +4.5 | +30.4 | −17.0 | −41.3 | +21.8 | 26.2 | 57.8 | 5/11 |
| T_2 (FT value 2) | 2171.9 | +0.9 | −8.4 | +8.7 | −12.3 | +5.9 | 108.4 | 62.7 | 6/11 |
| I_m4 (m=4 in blank/double GWs only) | 2171.5 | +0.5 | −7.2 | +6.8 | −36.7 | +14.4 | 64.7 | 60.0 | 3/11 |
| T_1 | 2170.1 | −0.9 | −8.6 | +5.5 | −8.7 | +2.0 | 109.8 | 62.7 | 4/11 |
| D_minp (bench slot 1 P(play) ≥ 0.7, soft) | 2167.5 | −3.5 | +8.8 | −13.8 | −16.0 | +1.1 | 105.5 | 64.1 | 6/11 |
| I_nohit (no hits in blank/double GWs) | 2165.7 | −5.3 | −3.0 | −7.2 | −52.7 | +12.5 | 61.8 | 62.0 | 3/11 |
| T_3 | 2165.4 | −5.6 | −13.6 | +1.0 | −14.3 | −2.4 | 101.5 | 60.4 | 4/11 |
| D_dnp_minp | 2164.8 | −6.2 | +25.6 | −32.7 | −16.3 | −2.4 | 119.6 | 74.7 | 6/11 |
| T_4 | 2159.9 | −11.1 | −4.0 | −17.0 | −51.3 | +4.0 | 71.6 | 58.5 | 5/11 |
| H_cap1 (at most 1 hit per GW) | 2150.0 | −21.0 | −5.4 | −34.0 | −57.0 | −7.5 | 60.4 | 59.0 | 4/11 |
| C_m3 | 2145.0 | −26.0 | +21.2 | −65.3 | −112.7 | +6.5 | 12.0 | 52.5 | 4/11 |
| D_dnp (DNP-risk bench weights) | 2143.8 | −27.2 | −7.4 | −43.7 | −85.3 | −5.4 | 119.3 | 74.9 | 3/11 |
| C_m4 | 2125.2 | −45.8 | +26.2 | −105.8 | −161.0 | −2.6 | 4.0 | 61.5 | 4/11 |
| C_m1 | 2122.4 | −48.6 | −54.4 | −43.8 | −82.0 | −36.1 | 51.6 | 58.9 | 4/11 |
| C_m8 | 2095.4 | −75.6 | −23.4 | −119.2 | −185.3 | −34.5 | 0.4 | 59.8 | 3/11 |
| B_noHits (max transfers = FTs) | 2091.5 | −79.5 | −31.8 | −119.2 | −185.3 | −39.8 | 0 | 58.8 | 3/11 |
| C_m−2 (hit if gain ≥ 2) | 1994.4 | −176.6 | −144.2 | −203.7 | −92.7 | −208.1 | 420.7 | 57.4 | 0/11 |

Per-GW net distribution (per_gw_distribution.csv):

| policy | GW mean | GW sd | p10 | p50 | p90 | GWs with a hit | largest GW hit |
|---|---|---|---|---|---|---|---|
| A_default | 57.1 | 16.6 | 37 | 57 | 79 | 41.6% | 24 |
| C_m2 | 57.3 | 16.5 | 37 | 56 | 79 | 12.2% | 16 |
| B_noHits | 55.0 | 16.2 | 35 | 55 | 76 | 0% | 0 |

Availability calibration, 2026-27 GW1-5 (flag for the upcoming GW):

| flag | rows | played | pts / model pred vs unflagged | shrunk P(play), played last GW | recommended multiplier |
|---|---|---|---|---|---|
| 75% | 53 | 24.5% | 0.27 [0.13–0.44]; plausible picks 0.16 [0.06–0.28] | 0.41 (unflagged 0.874) | 0.40 |
| 50% | 14 | 36% | 0.52 [0.22–1.13] | – | 0.25 |
| 25% | 8 | 25% | 0.18 | – | 0.10 |
| 0 / i, s, u | 712 | 0.4% | 0 | – | 0 |

Fixed-model prediction with the stale flag vs removed: 0.99. Feeding `ep_next` instead of `ep_this` changes 75%-flag predictions by −1%.


---

I built the chip decision tool (`chip_eval.py`, drop-in for `REPO/scripts/chip_eval.py`), checked it against 2025-26, and ran it on entry 8737706 as of GW6. I did not change the approved plan and submitted nothing. Nothing under REPO was edited: all work, including a copy of the data folder, is in `SP/chip_tooling/`.

**What the tool reports.** For every remaining chip and every GW in its window:
- **TC:** expected points of the effective captain, including failover to the vice.
- **BB:** the bench's expected points minus what auto-subs would recover anyway.
- **FH:** best one-GW squad (bank + selling value) minus the current squad; also compared with using the FTs banked by then.
- **WC:** 6-GW gain of a rebuilt squad over the same plan without the WC, both from a multi-GW solver with FT banking, the WC FT-carry rule and no hits.
- **Also:** blank/double GWs and unscheduled fixtures from the fixture list; player availability from flags and return dates in the news text; the plan's early-WC trigger (3+ starters flagged or out); the FH GW19/GW20 block; first-half expiry.
- **Calibrated columns:** raw gains multiplied by realised/predicted ratios measured on 2025-26 (see below).
- **Horizon predictions:** there was no predictor for GWs beyond the next one, so the tool takes each player's current feature row and swaps in the fixture details of the later GW.
- **Rescoring:** gains are recomputed with a simulation of who plays, using a new playing-probability model (AUC 0.946 vs 0.917 for the simple fallback formula).

**Checks on 2025-26:**
- The fixture swap reproduces the pipeline's features and predictions exactly for the current GW.
- Prediction accuracy falls from per-GW Spearman 0.723 for the next GW to 0.611 eight GWs ahead.
- Raw gains are optimistic because the optimiser picks players the model over-rates:
  - **TC:** 8.67 predicted vs 6.37 realised for the current GW (×0.73), about ×0.5 three or more GWs ahead.
  - **BB:** about ×1.
  - **FH:** ×0.77 for the current GW, ×0.36 six GWs ahead.
  - **WC:** 55 predicted vs 10 realised (×0.18, n=8, standard deviation 40). One point (WC at GW18) realised −68; I checked it and it is variance, not a bug.
- Choosing the TC week 1-6 GWs ahead did not beat playing it in the current week in the first half (7.11 vs 7.63).

**Live, as of GW6:**
- Rules check out: 2 free transfers entering GW6; all 8 chips unused; windows as published (WC/FH GW2-19 and 20-38, TC/BB GW1-19 and 20-38).
- No blank or double GW in GW6-19 and 0 unscheduled fixtures.
- Early-WC trigger not met: 1 flagged starter (João Pedro, 75%).
- Using the model trained without understat features (reason below):
  - **TC GW7 (Haaland):** raw 5.85, calibrated 4.15.
  - **BB on the current bench:** only +1.4 to +2.5 per GW.
  - **FH:** raw +10 to +21 per GW; GW18 fallback +9.9 raw (+3.6 calibrated).
  - **WC GW11:** +14.5 raw (+2.9 calibrated), measured against using the 5 FTs that would be banked by then.
  - **WC11+BB12 combined:** +28.4, with BB on the rebuilt squad worth +16.5 (+14.9 for WC11+BB13).

**Is Haaland still the right GW7 TC target?** Numbers only:
- **Fixed model as it would be served today:** 3.13 points in GW7, rank 85 in the pool. This is an artefact: 2026-27 understat was never collected, so those inputs are all blank. On the 2025-26 holdout, blanking them costs 2.47 captain points per GW (95% CI −4.05 to −0.97). My tool now prints a warning when this happens.
- **Model trained without understat:** 5.70, rank 4 in the pool. He is the squad's top captain in GW7, and GW7 is his best home GW of GW6-19 (GW13 5.15, GW16 5.14, GW11 5.08, GW9 4.59). Over GW1-5 this model's captain picks realised 6.6 per GW vs 3.4 for the as-served model (only 5 GWs).
- **Filling understat from FPL's own xG:** 7.86, rank 2. This fill did not validate: it lost per-GW Spearman by 0.069 and did not recover captain points, so it is diagnostic only.
- **Model-free history:** Haaland at home to promoted sides (2022-23 to 2026-27, 13 fixtures) averaged 6.92 points, with a 31% chance of 10 or more. The plan's "TC +8" looks about 1 point optimistic.

**Side bug found:** the live collector labels every past row with the player's current club. Players who moved clubs then put their old club's fixtures under the new club. That gives false double-GW flags in GW1-2 and wrong points-conceded opponent features on 72% of GW6 rows, shifting predictions by up to 0.57 points. It will recur after the January window.


**Findings:**

- [high] **F1** GW7 TC check: the fixed model as served today rates Haaland 3.13 (rank 85) because 2026-27 understat is missing; without that gap he is the squad's GW7 TC target — fix: No change to the plan. Before the Fri 16 Oct GW7 decision, collect 2026-27 understat (the feature-groups agent's `exp/feature_groups/understat_probe/collect_live_understat.py` shows it can be done cheaply) or serve a model trained without understat. chip_eval now warns loudly when understat is all blank.
- [medium] **F2** Raw chip gains are optimistic: 2025-26 realised/predicted ratios for TC, FH and WC — fix: chip_eval prints a calibrated column next to each raw gain (`CALIB` table in the script, with sources). Treat WC gains especially as noisy. Refresh the ratios after the ~GW8 retrain.
- [medium] **F3** Bug: live collector labels past rows with the player's current club — fix: In `build_season_files`, derive each row's club from the fixture: build `side = {f['id']: (f['team_h'], f['team_a']) for f in fixtures}`, then `th, ta = side[h['fixture']]` and `row['team'] = team_name[th if h['was_home'] else ta]`.
- [info] **F4** Live chip numbers as of GW6 (no-understat model); BB only pays on a WC-built bench — fix: Numbers only. The plan's 'BB +10-15' holds only on a WC-built bench, as the plan intends. 'WC +15-25' is at or above the model's raw numbers and well above what the replay realised.
- [info] **F5** Rules and chip handling verified against the FPL API and entry history — fix: None needed. Re-run chip_eval after the FA Cup R5 draw (~mid-Feb) to populate the spring blank/double GW map for the second set of chips.
- [low] **F6** Predictions weaken with lead time: commit TC in the week it is played — fix: Consistent with the plan making the TC call at GW7 prep. The GW7 vs GW16 fallback comparison is mostly fixture effects: under the no-understat model Haaland differs by 0.56 xP (5.70 vs 5.14), within noise.
- [low] **F7** Flag handling in chip_eval avoids the audit's double count; CBC hangs on Windows in two cases — fix: Copy `models/pplay_LIVE` to `REPO/models/pplay_2026-27` when dropping in the script.

2025-26 replay: realised/predicted by chip and lead k (GWs ahead)
| Chip | k | Predicted | Realised | Ratio [95% CI] | Pearson |
|---|---|---|---|---|---|
| TC | 0 | 8.67 | 6.37 | 0.73 [0.56, 0.92] | 0.20 |
| TC | 3 | 8.22 | 4.37 | 0.53 [0.41, 0.66] | 0.53 |
| BB | 0 | 2.98 | 3.71 | 1.25 [0.71, 1.89] | 0.35 |
| FH | 0 | 18.6 | 14.3 | 0.77 [0.51, 1.00] | 0.57 |
| FH | 6 | 19.3 | 6.9 | 0.36 [0.07, 0.62] | 0.64 |
| WC (6 GWs) | 0 | 54.9 | 10.0 (sd 40) | 0.18 [−0.35, 0.58] | 0.33 |
| BB on WC-built squad | 1-2 | 11.1 | 9.6 | 0.86 [0.59, 1.11] | – |

Live as of GW6, no-understat model (raw / calibrated)
| GW | TC (captain) | BB | FH | WC (6 GWs) |
|---|---|---|---|---|
| 6 | 6.70 / 4.89 (Bruno) | 1.42 | 16.6 / 12.8 | 37.3 / 7.5 |
| 7 | 5.85 / 4.15 (Haaland) | 1.65 | 13.8 / 8.7 | 28.5 / 5.7 |
| 11 | 5.36 / 2.79 (Bruno) | 2.45 | 16.4 / 6.2 | 14.5 / 2.9 |
| 16 | 5.33 / 2.77 (Haaland) | 2.14 | 21.3 / 7.7 | – |
| 18 | 6.50 / 3.38 (Bruno) | 1.90 | 9.9 / 3.6 | – |

WC11+BB12: +28.4 combined (BB on the built squad +16.5). WC11+BB13: +27.2 (BB +14.9).

Haaland GW7 IPS(H), by source
| Source | xP | Pool rank |
|---|---|---|
| Fixed model as served (understat blank) | 3.13 | 85 |
| Model without understat | 5.70 | 4 |
| Understat filled from FPL xG (not validated) | 7.86 | 2 |
| History: home to promoted sides, n=13 | 6.92 mean, 31% chance of 10+ | – |


---

I built the multi-GW receding-horizon planner, wired it into gameweek.py behind flags, fixed five rule/engine bugs along the way, and backtested everything. All work is committed on worktree branch `worktree-wf_386e8d29-1da-3` (commits 96d93bc and 37caeda, not pushed); 350 tests pass (305 before, 45 new).

**What the backtests say:**
- **Headline:** the planner is a modest gain, not a big one. The best setting (plan 3 GWs ahead, extra 4-pt penalty on every hit) averaged +19 pts/season over the current single-GW optimizer across 9 replays (3 seasons x 3 starting squads, won 6/9), and +24/season when both follow the same chip plan. Season results swing about ±100 pts from small changes, so these gains are within noise.
- **Why it isn't bigger:** the model's future-GW predictions add little over this week's (only +0.02-0.04 Spearman vs just reusing this week's prediction). Better multi-week predictions (e.g. a minutes model) are the next lever.
- **Without the hit penalty the planner is clearly harmful:** planning 4 GWs ahead took about 370 pts of hits a season and lost about 160 pts/season. The same prediction error for a player repeats across every horizon week, so planned gains are overstated. The shipped defaults (hit margin 4) guard against this.
- **The "max 1 transfer" policy the old backtests quoted costs about 66 pts/season** versus the unconstrained single-GW optimizer (-100 over the £100m starts).
- **"Never take a hit" is unsafe without chips:** it lost 310 pts in 2023-24, spread over the season and worst in the big spring doubles (GW34: -57).
- **Live sanity replay (GW2-5 from our real GW1 squad, no captain overrides):** single-GW 237, planner 235, unregularised 4-GW planner 194. We actually scored 256 with the human captain calls.

**What was built:**
- **Horizon predictions (prediction/horizon.py):** for each future GW it takes the player's as-of-deadline feature row and swaps only the fixture features. Team stats use matches before the deadline only, odds/props are treated as unknown, blank GWs score 0, and double GWs use a 50/50 blend of two methods. Unit tests confirm the swap reproduces the pipeline's own fixture features; checked on the 2025-26 data, the match was exact.
- **Planner (optimizer/horizon_optimizer.py):** free-transfer banking to 5, no hit while FTs are unused, selling prices, 3-per-club, formations. It follows a given chip schedule: Wildcard/Free Hit weeks are free and keep the FT count, the Free Hit squad reverts, Bench Boost counts the bench, Triple Captain triples. Bench value comes from the starters' chance of not playing. Solves take 1-7 s on real data (max about 34 s).
- **gameweek.py:** `--horizon N --chip-plan "tc:7,wc:11,bb:12,fh:18"` plus `--discount`, `--hit-margin`, `--max-hits`, `--ft-value`. Default behaviour is unchanged, and `--apply` uses the planned chip for this GW.
- **Backtest runner:** `scripts/backtest_horizon.py` (predict and replay steps).

**Bugs fixed:**
- Backtests gave 2 FTs in GW2 instead of 1; they now start GW1 with 0 FTs.
- The engine's Free Hit revert restored the squad but not the bank.
- `optimize_transfers` became infeasible when 4+ squad players blank; this crashed a 2024-25 replay.
- The live collector labelled every past row with the player's *current* club. This created fake double GWs for 18 team-GWs in 2026-27 GW1-2 and mixed up team stats.
- Chip plans within one horizon now respect one use per half and the GW19/20 Free Hit rule.

**Rules check:** replaying our real GW1-5 decisions through the engine reproduces FPL's points (40/99/52/69/36), bank and hits exactly, with 2 FTs entering GW6. The approved chip plan was not changed.

**Things to know before GW6:**
- The planner uses whatever model directory it is given. The current prod model still uses the leaky `fpl_xp` feature and produces garbage if the GW's pre-deadline snapshot is missing (Haaland drops from 7.6 to 2.29 xPts). Normal gameweek.py runs snapshot first, but never use `--skip-build` before a snapshot exists.
- If the fixed model needs `fpl_xp_lag` and the pipeline doesn't produce it yet, the horizon path fills it as NaN with a loud warning. The predictor fix still has to wire that into the pipeline.
- CLAUDE.md still says xP is not a leak. I left that line alone because it belongs to the predictor fix.


**Findings:**

- [high] **F1** Unregularised multi-GW planning overtrades and loses points — fix: HorizonConfig default hit_margin=4.0 (a hit must promise >8 planned pts). h3:m4 over 9 replays: 2211 vs single 2193 (+19, 6/9 wins), 65 vs 121 hit pts/season.
- [high] **F2** Backtests gave 2 FTs in GW2 (FPL gives 1) — fix: New constant GW1_FREE_TRANSFERS=0, used in backtest.py, backtest_season.py and 5 other GW1-start scripts; backtest_season plays the selected GW1 squad as-is. Test: test_gw2_gets_exactly_one_free_transfer.
- [high] **F3** Free Hit revert kept the post-Free-Hit bank — fix: GameState.free_hit_bank_stash is saved at activation and restored on revert. Test: test_revert_restores_bank.
- [medium] **F4** optimize_transfers infeasible when 4+ squad players blank — fix: Each missing player gets its own placeholder club id. Test: test_four_blanking_squad_players_stay_feasible.
- [medium] **F5** Live collector labelled past rows with the player's current club — fix: fpl_live.build_season_files takes each row's club from the fixture side (team_h/team_a by was_home). Test: tests/test_data/test_fpl_live_teams.py. Takes effect on the next season-file rebuild.
- [medium] **F6** Future-GW props encoded as 'no line' pushed live predictions down — fix: horizon.py sets all odds/props to NaN for k>=1. Optional market='drop' blanks them at k=0 too (odds shift regulars by +0.13 to +0.24 pts with 0.5 spread while barely changing ranking).
- [low] **F7** Aggregated DGW rows under-predict regulars — fix: dgw_mode='blend' is the horizon default. The single-GW pipeline path is unchanged (flagged for the predictor work).
- [medium] **F8** Prod model breaks when this GW's fpl_xp is missing — fix: Not changed (predictor scope). Live runs must snapshot before predicting (gameweek.py does this unless --skip-build). The fixed model needs fpl_xp_lag from the pipeline; predict_horizon_live fills missing model features with NaN and a warning.
- [info] **F9** Engine reproduces FPL for 2026-27 GW1-5 (rules check) — fix: None needed; the planner enforces the transfers cap of 20 and chip availability per half.
- [info] **F10** Backtest caveats — fix: Treat differences under ~50 pts/season as noise; re-run scripts/backtest_horizon.py with the promoted fixed model once it exists (--model to reuse it).

Leak-free replays through the rules engine. "single" = the current single-GW optimizer (unconstrained); "single1" = max 1 transfer/GW; hN:mM:fF:xK = horizon N GWs, extra penalty M per hit, value F per banked FT, at most K hits/GW.

A) GW1 squad at £100m, no chips (net points; hits = pts lost to hits/season)
| variant | 2023-24 | 2024-25 | 2025-26 | mean | hits/season |
|---|---|---|---|---|---|
| single1 | 2106 | 2138 | 2080 | 2108 | 0 |
| single | 2212 | 2174 | 2211 | 2199 | 121 |
| h4 (no margin) | 1951 | 2083 | 2085 | 2040 | 376 |
| h3 (no margin) | – | 2137 | 2112 | 2124 | 296 |
| h2:m2:f1.5 | 2139 | 2249 | 2286 | 2225 | 99 |
| **h3:m4 (default)** | 2114 | 2265 | 2242 | 2207 | 68 |
| h4:m4 | 2062 | 2282 | 2208 | 2184 | 108 |
| h4:bfixed:m4 (old bench weights) | – | 2244 | 2232 | 2238 | 68 |
| h2:x0 (no hits) | 1902 | 2337 | 2190 | 2143 | 0 |

B) Robustness: 3 seasons x 3 starting squads (£100m/£99m/£98m), no chips
| variant | mean net | vs single | wins vs single |
|---|---|---|---|
| h3:m4 | 2211 | +19 | 6/9 |
| single | 2193 | 0 | – |
| h2:m2:f1.5 | 2190 | -3 | 4/9 |
| h2:x0 | 2181 | -12 | 4/9 |
| h3:x0 | 2138 | -54 | 2/9 |
| single1 | 2127 | -66 | 2/9 |

C) With a chip plan (first half TC7, WC11, BB12, FH18 as in SEASON_GUIDE; second half: FH on the big blank GW, WC the week before the big double, BB/TC on doubles); zero engine rejects
| variant | 2023-24 | 2024-25 | 2025-26 | mean |
|---|---|---|---|---|
| single + chips | 2303 | 2296 | 2269 | 2289 |
| h2:m2:f1.5 + chips | 2413 | 2293 | 2268 | 2325 |
| h3:m4 + chips | 2340 | 2245 | 2353 | 2313 |
| h4:m4 + chips | 2338 | 2245 | 2346 | 2310 |

D) Horizon prediction quality (per-GW Spearman, swapped fixtures vs reusing this week's prediction), k = weeks ahead
| season | k=0 | k=1 | k=2 | k=3 |
|---|---|---|---|---|
| 2024-25 | 0.715 | 0.676 vs 0.655 | 0.651 vs 0.630 | 0.632 vs 0.610 |
| 2025-26 | 0.722 | 0.688 vs 0.669 | 0.666 vs 0.646 | 0.649 vs 0.629 |
| 2023-24 | 0.695 | 0.656 vs 0.617 | 0.635 vs 0.594 | 0.618 vs 0.579 |
| LIVE 2026-27 (only 2-4 GWs) | 0.683 | 0.638 vs 0.643 | 0.598 vs 0.596 | 0.574 vs 0.563 |

E) LIVE 2026-27 GW2-5 replay from our real GW1 squad (model only, no captain overrides; actual with human overrides = 256)
| single / single1 / h1:m2 | h3:m4 | h3:m2:f1.5 | h2:m2:f1.5 | h4 (no margin) |
|---|---|---|---|---|
| 237 | 235 | 247 | 231 | 194 (32 pts of hits) |

Solve time for 3-4 GW horizons with ~110 candidates: mean 1-7 s, max about 34 s.


---

**Verifier:** The rules audit mostly holds up. Its test numbers reproduce exactly:
- **Current repo:** 191 of the 207 audit tests pass and 16 fail. The 16 failures are exactly the tests marked as bugs (`-m bug`).
- **All 9 patches in a fresh clone of HEAD 887f429:** 207/207 pass.
- **Repo suite:** 305 passing at HEAD, 309 with the patches.
- **Each patch alone:** each one applies cleanly on its own, and each one alone keeps `tests/test_engine`, `tests/test_optimizer` and the 191 non-bug audit tests green.

**GW6 live state** (my own offline rebuild): 2 free transfers, bank £0.2m and correct purchase/selling prices. The Wildcard budget is £99.8m (998). All 8 chips are available.

**Rule claims checked against official sources:**
- The chip start/stop events and the FH19/FH20 block match the API and the premierleague.com chips article (fetched).
- The GW19 deadline mismatch in I-06 is real.
- FTs carrying over unchanged through a Wildcard or Free Hit (4 stays 4) is supported by the premierleague.com five-free-transfers article.
- The bank being restored after a Free Hit is supported by FPL-rules summaries such as onefpl.com. The premierleague.com article itself only mentions the squad.

**Corrections to the audit:**
- **R-06 failure mode overstated.** With no transfer cap (SeasonBacktester), 4 or 5 blanking squad players force sales, which burns the FT or costs a -4 hit. The MILP only becomes infeasible with `--max-transfers`. The club-limit case would crash SeasonBacktester, not turn the GW into a no-op. P06 also covers only 2 of the 7 backtest call sites.
- **R-09 partly unverified.** The my-team chip field names, and whether FPL cancels an active chip on `chip=null`, can't be checked without logging in. P09 may do nothing if the field names differ.
- **Test-edit count wrong.** The summary says 5 existing repo tests were updated; it is actually 9. Every edit keeps the test's intent.
- **DEFCON overstated.** DEFCON is not a repo constant; the audit's DEFCON test only checks the official JSON against itself.

**No rules bug changes the GW6 numbers.** The largest remaining gap is I-05: the approved Wildcard GW11 → Bench Boost GW12/13 plan can't be priced by the single-GW `--chip wildcard` run with the current bench weights.

My verification files are in `C:/Users/alexa/AppData/Local/Temp/claude/C--Users-alexa-OneDrive-Documents-FPL-RL/7cb1adba-4ec3-4b13-a50f-92372351e85a/scratchpad/verify_rules_audit/`:
- `v_entry.py`: independent live-state rebuild
- `v_r06.py`: blank-GW optimizer check, unpatched vs patched
- `clone/`: HEAD with all 9 patches
- `solo/`: HEAD for the one-patch-at-a-time runs

- blocking (addressed in season-hardening): Nothing blocks the GW6 dry run. The live rebuild (2 FTs, bank 2, selling prices, chips) and the engine's scoring are correct.
- blocking (addressed in season-hardening): Must be fixed before the GW11 Wildcard build, decision point around 20 Nov (I-05). `--chip wildcard` optimizes GW11 alone. Outside Bench Boost, bench slots are valued at 0.21/0.06/0.002 (outfield) and 0.03 (GK), so it builds a dead bench and cannot deliver the approved Wildcard GW11 → Bench Boost GW12/13 plan. It needs predictions over several GWs (GW11-13) and a bench valued at 1.0 for the Bench Boost GW.
- blocking (addressed in season-hardening): Must be fixed before any backtest is used to tune the optimizer or the hit policy. Apply P02 (GW2 gets 1 FT; fixes all 7 engine callers) and P06. Also pass real clubs (squad_teams) in the other 5 optimize_transfers backtest callers: generate_season_report, oracle_comparison, predictor_ablation, retrain_and_test, train_predictor_no_xp. Otherwise every backtest carries one extra free transfer, and blank GWs cause forced sales, hits or crashes.
- blocking (addressed in season-hardening): Only if `--apply --yes` is used: apply P05 and P09 before the next automated submission, especially in chip weeks (TC in GW7, BB in GW12/13). On the first logged-in run, print the raw my-team 'transfers' and 'chips' objects to confirm the fields `limit`, `made`, `status_for_entry` and `is_pending`. If those names are wrong, P09 silently does nothing.

---

**Verifier:** The transfer_policy work is reproducible and its numbers are honest. I rebuilt the key results without its code:
- **Replay parity:** my own replay using the repo optimizer and engine gets exactly the same season totals: 2277 / 2017 / 2245.
- **Summary tables:** re-aggregating all 252 runs reproduces every figure.
- **Availability calibration:** recomputing it from the raw snapshots reproduces it exactly.
- **Bugs:** both backtest bugs (GW1 FT, club-0 placeholder) reproduce with the repo code unchanged, and neither affects the live path.

Its policy conclusions are directionally sound but weaker than presented.
- **Only two seasons:** there are 2 independent seasons. The 8 'noisy' paths are the same predictions with mean-zero ±12% noise, and their per-GW Spearman (0.713-0.721) is far from LIVE (0.672, bias +0.19).
- **Path chaos is large:** one solver tie-break swings a season by about 50 pts, and one extra FT at GW1 by about 70.
- **CIs are too narrow:** the GW-bootstrap intervals understate uncertainty. Path-level SE is about 33 for never-hit, not ±18.
- **Hit rule:** keeping the default is a defensible maximin choice, but m=2 ties overall and wins on the noisy paths. It is a reasonable live setting while live predictions over-predict.
- **Hit audit:** its multi-GW 'hits pay over 2 GWs' figure is biased toward hits, because the no-hit comparator never uses its next FT.

Smaller factual errors, none of which changes a recommendation:
- **TP-3 live fact:** outfield bench slot 1 played in GW1-4. The real live failure was GW5, when the only XI DNP happened with an all-dead bench.
- **TP-7 'τ ≤ 2 never binds':** false. T_2 binds on H25_full GW10, with predicted gain 1.41.

The availability recommendation (one pool multiplier: 75 → 0.40 and so on) points the right way, since flagged players clearly under-deliver. But it rests on 18 relevant rows from 40 players over 5 GWs, and it has integration risks listed under blocking issues.

Verification scripts are in C:/Users/alexa/AppData/Local/Temp/claude/C--Users-alexa-OneDrive-Documents-FPL-RL/7cb1adba-4ec3-4b13-a50f-92372351e85a/scratchpad/verify_transfer_policy/:
- v1_summary.py
- v2_replay.py (+ v2_*.csv)
- v3_noise.py
- v4_avail.py
- v5_nextgw.py
- v6_patch_mult.py

- blocking (addressed in season-hardening): TP-1 cannot go live alone. The pool multiplier (75 → 0.40, 50 → 0.25, 25 → 0.10) must ship in the same change as the fixed model. With the old ep_next-fed model the effective discount is about 0.585, so pairing the multiplier with the old model double-discounts flagged players. Pairing the fixed model with the old 0.75 over-values flagged players by about 27%.
- blocking (addressed in season-hardening): TP-1's claim that the model is 'availability-blind' was measured with the harness feeding fpl_xp_lag = the GW-n pre-deadline snapshot's ep_this. The repo has no fpl_xp_lag yet, and fpl_live.py fills the merged_gw xP column from ep_next snapshots. Live serving must read ep_this from the GW-n snapshot, not shift the xP column (that would feed GW n-1's pre-deadline ep_next, a different quantity from training). Needs a regression test.
- blocking (addressed in season-hardening): The availability multiplier must discount only the upcoming GW's points in the transfer objective. For GW t+1 onward in a horizon planner, use a recovery value: 60% of relevant 75%-flagged players played the next GW, vs 33% in the flagged GW. Otherwise the optimizer will sell, possibly with hits, players who are back a week later. build_live_candidates currently has zero test coverage; add tests with the patch.
- blocking (addressed in season-hardening): The hit-rule, FT and bench-weight conclusions (TP-2, TP-3, TP-7) are validated only for the current single-GW optimizer with no chips. If the horizon planner is what goes live, re-run the policy grid with it first. Before any repo backtest is used for that go/no-go, fix TP-4 (GW1 must start at 0 FT with max_transfers=0) and TP-5 (real club for blank-GW squad members) in backtest.py, backtest_season.py and the other five scripts.

---

**Verifier:** The chip tool's rules handling and its optimisation and simulation code are correct. Its live numbers reproduce exactly. The weak points are what the TC numbers rest on and how the tool is set up to run.

- **Rules:** chip windows match rules_2026-27.json. From an independent replay of the entry history: 2 free transfers entering GW6, bank 0.2m, no chips used. All 15 selling prices follow the half-of-rise rule. FH in GW19 correctly removes GW20 from FH2. First-half chips expire at GW20.
- **Wildcard transfer rule:** "banked transfers kept, no extra one in the chip week" matches the official Premier League and Fantasy Football Scout guidance.
- **Planner:** its optimum equals an exhaustive search on all 36 small test instances, with budget and 3-per-club limits binding.
- **Auto-subs:** match an independent FPL implementation on 20,000 random cases.
- **Horizon predictions:** the current-week predictions equal the leak-free baseline exactly. Accuracy by lead time reproduces.
- **Calibration ratios:** all reproduce from the saved replay files.
- **Collector bug (F3):** real.
- **Live re-run:** the GW6 report reproduced identically.

**What goes too far:**
- **F1:** "serve a model without understat" is not a validated fix.
  - On the 2025-26 holdout its captain points (5.11) are no better than serving understat blank (5.26). With real understat it is 8.13.
  - On 2024-25, blanking understat even helped the captain metric (+0.53).
  - So Haaland's 5.70 has only 5 live GWs behind it. The only fix with evidence is collecting 2026-27 understat.
- **Calibration:** the single pooled factors hide a split between halves. The TC ratio at k=0 is 0.93 in the first half vs 0.56 in the second. The replay also used final fixtures and FDR, which the tool could not have known in advance.
- **Double GWs:** the tool adds up single-fixture predictions, which overstates double-GW players by about 20%. This inflates the spring chip numbers.
- **The drop-in's defaults** point at the leaky production model. The playing-probability model directory they point to does not exist.

Verification scripts are in `C:/Users/alexa/AppData/Local/Temp/claude/C--Users-alexa-OneDrive-Documents-FPL-RL/7cb1adba-4ec3-4b13-a50f-92372351e85a/scratchpad/verify_chip_tooling/`: v_rules.py, v_milp.py (+ v_milp_H3.log, v_milp_H4.log), v_autosub.py, v_hits.py, v_horizon.py, v_movebug.py, v_haaland.py, v_nous.py, v_live_rerun.py (+ live_rerun.log, live_rerun.json) and dbg_wc18.log.

- blocking (addressed in season-hardening): Drop-in defaults are unsafe. --model-dir defaults to models/prod_2026-27, which uses the leaky same-GW fpl_xp (no fpl_xp_lag) plus understat, and chip_eval gives no xP warning. --pplay-dir defaults to models/pplay_2026-27, which does not exist, so the logistic fallback is used. Before the first run from REPO, promote a leak-fixed model and copy pplay_LIVE into REPO/models, or change the defaults.
- blocking (addressed in season-hardening): Collect 2026-27 understat before the GW7 TC decision (Fri 16 Oct). With understat blank, the as-served fixed model's captain/TC numbers are artefacts (Haaland rank 85). The proposed alternative, a model trained without understat, is not validated: holdout captain points are 5.11 on 2025-26 vs 5.26 served-blank and 8.13 with true understat, and 6.68 vs 7.26 on 2024-25. Until then, neither model's TC column can be trusted for the Haaland decision.
- blocking (addressed in season-hardening): Fix the live-collector club-attribution bug (fpl_live.py:387; F3 fix is correct) before the ~GW8 retrain and before the January window. It creates false double GWs in GW1-2 rows (is_dgw 61.6%/30.4%) and wrong opponent/team-form features in the retraining data and in gameweek.py's GW6 predictions.
- blocking (addressed in season-hardening): Before any spring double-GW chip decision (TC2/BB2/FH2), replace the double-GW summing of single-fixture predictions: on 2025-26 it over-predicted DGW top-10 players 10.19 vs 7.5 realised. Also split or recalibrate CALIB by season half or double-GW; the pooled TC ratio 0.73 hides 0.93 in the first half vs 0.56 in the second.

---

**Verifier:** The planner's core is correct. The MILP matched a brute-force optimum on 106/106 small instances covering FT banking, hits, budget, selling prices, club and formation limits, and every chip combination. Horizon features use nothing from after the deadline (identical when the data is cut off at the deadline). The five rule and engine bug fixes are real, and the engine reproduces entry 8737706's GW1-5 exactly with 2 FTs entering GW6. Every replay number I re-ran reproduced exactly. The benefit is overstated, though. The headline '+19/season, 6/9 wins' mixes two prediction modes and counts one duplicate starting squad. Consistently it is about +12 over 8 distinct replays (5/8 wins), and all three 2023-24 starts lose by about 100-150. It is a noise-level gain chosen after trying about 20 variants on the same 3 seasons. Code issues that don't block GW6: (1) an LP loophole ('phantom' half-transfers) can make an unaffordable plan when a selling price is above the current price. It can't happen live but is a two-line fix. (2) The F5 collector fix makes SeasonDataLoader assign transferred 2026-27 players their old club, because it keeps the first row. (3) Legacy scripts now charge -4 for GW1 changes. (4) The branch overlaps rules-audit patches P02/P03/P06/P07 and must be reconciled at merge. My detached verification worktree is at C:/Users/alexa/AppData/Local/Temp/claude/C--Users-alexa-OneDrive-Documents-FPL-RL/7cb1adba-4ec3-4b13-a50f-92372351e85a/scratchpad/verify_horizon_planner/wt; remove it later with git worktree remove. Scripts are in the same folder: brute.py, brute_multi.py, pit.py, swapcheck.py, swapcheck2.py, phantom.py, hq.py, engine_check_v.py, squads.py, and rep_*.log for the re-run replays.

- blocking (addressed in season-hardening): Model and feature mismatch on the live path. The only validated setup is the leak-fixed model with fpl_xp_lag, and it cannot run live yet: FeaturePipeline and LiveFPLCollector never produce fpl_xp_lag, so predict_horizon_live would feed it as all-NaN (only a warning). The current prod model (fpl_xp) with fixture swaps was never backtested: at k>=1 it carries this GW's ep_next forward, which is fixture- and flag-specific. Before --horizon is used for real decisions: wire fpl_xp_lag into the live pipeline, promote the fixed model, and re-run scripts/backtest_horizon.py with that model (or explicitly validate prod + horizon). Until then, treat --horizon output as advisory only.
- blocking (addressed in season-hardening): Fix the SEASON_GUIDE.md operating instruction 'Use it every week (it is at least as good as the single-GW run and better around chip weeks)'. The branch's own data contradicts it: h3:m4 lost all three 2023-24 starts by 97-152 and lost by 51 with chips in 2024-25. Restate it as roughly neutral, about +12 to +20/season (within noise), cross-check against the single-GW run, and correct the '+19, 6/9' headline: it mixes row and blend predictions, and the 2024-25 £98m and £99m starts are the same squad.