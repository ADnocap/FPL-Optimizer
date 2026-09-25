# FPL 2026-27, GW1-5: performance, prediction quality and soundness audit

Entry 8737706. Written 2026-09-25, during the international break. GW6 deadline: **2026-10-10 10:00Z**.

This report combines the work of 7 investigators (prediction_quality, train_serve_skew, synthetic_parity, decision_audit, gw5_deep_dive, live_code_review, historical_context). Their claims were then checked twice: adversarial verifiers (verify_*) and impact verifiers (impact_*).

Paths are relative to:
- SP = C:/Users/alexa/AppData/Local/Temp/claude/C--Users-alexa-OneDrive-Documents-FPL-RL/7cb1adba-4ec3-4b13-a50f-92372351e85a/scratchpad
- REPO = C:/Users/alexa/OneDrive/Documents/FPL-RL

Nothing under REPO was modified.

How claims were treated:
- A claim is **established** only if a verifier confirmed or partially confirmed it. Where a verifier corrected the wording or the numbers, the correction is what appears here.
- **Context** means a "sound" or "variance" finding, or an unverified low-severity item.
- Where agents disagreed on a number, Appendix A says which value was used and why.

---

## 1. Bottom line

- **Result: 296 vs 299 for the average manager (-3; overall rank 4.84M). That is noise.**
  - The SD of a 5-GW total vs the field is about 25-32, so z is about -0.1. GW5's -12 is about -1 SD.
  - About 55% of the field played chips, worth +3 to +4 on the average. We have played none, so chip-adjusted we are level.
  - This is median-manager territory. Last season's top-10% managers are +30 (+15 to +19 among those who also played no chips).
- **The points came from the human layer, not the model.**
  - Following the model unsupervised, with uncapped transfers as `gameweek.py` defaults, would have scored **239**, 57 fewer.
  - The 57 breaks down as -24 in hits, about -21 from model captains the human overrode, and about -10 from the GW2 João Pedro to Gyökeres swap.
  - The human overrides were worth **+48 measured one GW at a time, and +57 over the full path**. Every one of them beat the model.
  - Luck vs FPL's ep_next was about zero (+2). Vs the model's expectation it was +24, almost entirely Bruno's 23-point GW2 captaincy.
- **The model has a critical train/serve leak in its most important feature, `fpl_xp`.** Six investigators confirmed it with five different methods.
  - **In training:** the historical vaastav `xP` for GW n (n >= 2) is FPL's `ep_this` after FPL recomputed it once GW n had finished, so it already contains GW n's own points.
  - **In serving:** the model gets the honest pre-deadline `ep_next`.
  - **What the model learned:** roughly "4 x xP - 3 x recent form = this GW's points". Live, that arithmetic returns stale form.
  - **Consequences:** the model's response to form is inverted, its top end is hugely inflated (predictions of 6 or more averaged 7.11 and returned 2.75), and it proposes plans with hits.
  - `EP_FORMULA.md` is wrong. Its "ep_this is immutable" check was made mid-GW.
- **The advertised performance is a leak artifact.**
  - The holdout per-GW Pearson of 0.632 = 0.87 on the 11 leaky 2025-26 GWs + 0.54 on the 27 GWs with no xP.
  - Served the way live serves it, the same model scores **0.42 in GW1-5**. Live it scored **0.475**, against 0.462 for FPL's ep_next.
  - The README's 2,918 in 2024-25 becomes **about 1,850-2,020 when the model is served as it runs live**. The field average was 2,008.
  - So the live model is doing exactly what should be expected. This is not variance, and it is not a live-pipeline bug.
  - The old `no_xp` model (per-GW corr 0.591 on 2024-25), which memory calls "the overcorrection", was the honest one.
- **The fix is cheap and large.** Retrain with a leak-free `fpl_xp` (lag vaastav xP by one GW, or drop it).
  - Live GW1-5 per-GW Pearson rises by **+0.07 (95% CI +0.046 to +0.096)**, and Spearman goes from 0.651 to 0.685-0.695.
  - The honest holdout level is about 0.59-0.60.
  - The 2024-25 backtest gains **+240 to +370 pts per season at 1 transfer/GW**, and +350 to +715 with free transfers.
  - Caveats: MAE does not improve, and captaincy does not improve. No model variant captained better than the human did in GW1-5.
- **The rest of the pipeline is largely sound.**
  - Checks that passed: the engine reproduces the 296 exactly; the transfer MILP matches brute force in 45 of 45 cases; lineup selection matches exhaustive search in 25 of 25; entry state is correct; synthetic deadline rows match the later real rows on 120 of 122 features with no rolling-feature lookahead; the strength fix is correct.
  - Secondary defects:
    - Understat was never collected for 2026-27.
    - The summary refresh deletes the old files first and then ignores download failures.
    - `--skip-refresh` has no staleness guard.
    - History rows are stamped with the player's current club.
    - A 75% flag actually meant about a 27% chance of playing.
    - Bench slots 2 and 3 are dead.
- **What matters before GW6:**
  1. Do **not** act on the current prod GW6 output (Bruno captain at 12.45 vs ep_next 7.2, plus a -8 plan).
  2. Retrain leak-free with servable features. Change the promotion gate first, because the current gate would reject the correct model.
  3. Until the retrained model has been backtested, keep the no-hit rule and the market/EP captain rule.
  4. Use one free transfer on the dead bench slot (Drakes-Thomas).
  5. Reset expectations: an honest system is worth about **+2 to +5 pts/GW over the average** (top 10-50%), not top 1%. The current model, followed unsupervised, is below the field.
  6. Any portfolio or README presentation of "the current model" must stop quoting 0.632, 0.787, 2,918 or "beats OpenFPL/FPL Review".

---

## 2. Season scorecard GW1-5

Sources: decision_audit/step1_per_gw.csv, step1a_captain.csv, step2_expected_vs_realized.csv, and GT history.json. The engine and an independent scorer both reproduce 40/99/52/69/36.

| GW | Ours | Avg | Diff | Captain (raw pts) | Bench pts | Transfers / hits | Model exp. (our XI+C) | ep_next exp. | Luck vs model | Luck vs ep_next | Avg-proxy template: exp. model / exp. ep / realized |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 40 | 50 | -10 | Haaland 2 (model) | 2 | fresh squad (full_pregame model) / 0 | 55.6 | 34.2* | -15.6 | +5.8* | 50.8 / 33.2* / 52 |
| 2 | 99 | 81 | +18 | B.Fernandes 23 (model) | -1 | 0, rolled / 0 | 44.8 | 34.2* | +54.2 | +64.8* | 38.9 / 33.3* / 102 |
| 3 | 52 | 51 | +1 | Haaland 9 (override of Bruno) | 8 | 2 free (James→Calafiori, Tzolis→Stach) / 0 | 63.5 | 79.0 | -11.5 | -27.0 | 71.1 / 98.0 / 46 |
| 4 | 69 | 69 | 0 | João Pedro 12 (override of Bruno) | 9 | 1 free (Lucky→Mendy) / 0 | 56.2 | 76.2 | +12.8 | -7.2 | 55.3 / 74.9 / 82 |
| 5 | 36 | 48 | -12 | Haaland 6 (override of Stach) | 0 | 0, rolled (model wanted 4-5 for -12/-16) / 0 | 51.2 | 70.4 | -16.2 | -34.4 | 56.9 / 79.2 / 33 |
| **Total** | **296** | **299** | **-3** | armband 52 | 18 | 3 transfers / 0 hits | **272.2** | **294.0** | **+23.8** (z +0.9) | **+2.0** | **273.0 / 318.6 / 315** |

\* GW1-2 ep_next was still the pre-season prior: 526 of 600 GW2 values equalled GW1. It carries no information about those weeks.

Notes on the scorecard:
- The GW1 squad was picked by `models/full_pregame`, because prod_2026-27 was saved after the GW1 deadline.
- The avg-proxy template is the most-owned £100m-feasible squad with Haaland captained every week.
  - Our expected edge over it was **-0.7 by the model and -24.6 by ep_next**; the realized edge was -19. No lens gives us a positive ex-ante edge.
  - Against the actual average manager we realized -3.
- Hindsight regret: captain 9 (all GW1); lineup 16 (2/0/7/7/0).
- Auto-subs: zero all season.

---

## 3. GW5 deep dive: 36 vs 48

Sources: gw5_deep_dive/out/{verdict,field_exante,bench_ev}.json, captain_table.csv, gw5_eo_swing.csv; verify_gw5_deep_dive/v0*.

**Before the deadline we were at or above the field.** The field is a sample of 300 random Overall-league managers (net mean 48.40, SD 11.46).
- Model XI+C: ours 51.22 vs field 51.13 (50th percentile).
- EP: ours 70.4 vs 66.2 (61st percentile).
- Realized: 11.7th percentile (z = -1.08).

**What happened:**
- **João Pedro:**
  - Flagged 75% ("unspecified injury").
  - His match kicked off 90 minutes after the deadline, so there was no lineup news.
  - He played 0 minutes.
- **The whole bench also played 0:**
  - Palmer (GK2).
  - Greaves: unflagged, with 90/90/90/87 minutes in GW1-4.
  - Drakes-Thomas: 0 minutes all season.
  - Mendy: 0%, concussion.
- **The field benefited from auto-subs:** 52% of the field started João Pedro, and 89% of them got an auto-sub worth 5.69.
- **Template players we did not own scored:**
  - Groß 14, Semenyo 17, Isak 8, Cherki 8.
  - The template players we did own all blanked at 1-2 points.
- **Our differentials paid:** Van Hecke +5.8, Stach +4.9, and Haaland captain +4.0.

**Decision checks (all sound):**
- **Starting João Pedro was correct.** The choice only matters if both he and Greaves play, and then he is worth more (model ~5.1 vs 1.96; EP 8.2 vs 0.8). Both XIs score 36 on actuals.
- **Captaining Haaland was correct (+1).** He led on EP (8.2) and anytime probability (0.45 devigged). The team total is 36 with Haaland vs 35 with Stach.
- **Rejecting the model's hit plan was correct (+13 to +16).** Replays give 5 transfers for -16 (Palmer captain) scoring 20, or 4 for -12 scoring 22-23. The recorded plan (4 for -12, Stach captain) could not be reproduced; the 4- and 5-transfer objectives are within 0.08. EP valued these plans below rolling (64.0-64.7 vs 67.7).
- **Rolling the free transfer was about neutral.**
  - Across 1,426 legal single transfers, the median gain was -1 and 23.7% beat rolling.
  - The model's best single move would have added +4.
  - The model's top-20 single moves predicted +4.14 and realized +1.00.

**Attribution A: realized decomposition vs the field (sums to -12.00)**

| Component | Pts |
|---|---|
| Captaincy (our extra 6 vs field 5.25) | +0.75 |
| Hits (we took 0; field 0.77) | +0.77 |
| Field bench boosts | -0.10 |
| Auto-subs (ours 0 vs field 3.88) | -3.88 |
| Base XI (ours 30 vs field 39.94) | -9.94 |
| Sample average 48.40 vs official 48 | +0.40 |
| **Total** | **-12.00** |

**Attribution B: luck vs process (sums to -12.00)**

The bench figure uses the verifier's formation-aware Monte Carlo (-2.3; range -2.0 to -3.7) instead of the -2.02 ad hoc rescale. The realized auto-sub gap of -3.88 is fixed; only the structure/luck split changes.

| Component | Pts | Type |
|---|---|---|
| Thin bench: ex-ante auto-sub EV gap vs field | -2.30 | Process / structure |
| João Pedro's flag scaled at 0.75, not the empirical ~0.27 (ex-ante; realized 0 because Greaves also played 0) | -0.87 | Process |
| Ex-ante XI+C edge vs field (model) | +0.96 | Process (+) |
| No hits vs field | +0.77 | Process (+) |
| Auto-sub outcome worse than its expectation | -1.58 | Variance |
| XI+C outcomes vs model expectation, relative to field | -9.38 | Variance |
| Sampling | +0.40 | Sampling |
| **Total** | **-12.00** | |

**Verdict:** about **-11 of the -12 is variance**. Net process is about -1.4: about -3.2 of structure and flag-scaling costs, offset by +1.7 of good decisions. The things to act on are the bench structure and the flag calibration.

---

## 4. Prediction quality: live vs holdout vs FPL ep_next

Sources:
- prediction_quality/metrics_by_gw.csv: a decision-time replica, 90% within 0.05 of the printed xPts (correlation 0.9992).
- prediction_quality/deleak_comparison.csv.
- impact_live_code_review/i10_bootstrap.out.
- impact_historical_context/i2*, i6*.
- train_serve_skew/xp_retrain_eval.csv.

### 4.1 Live per GW

| GW | n | Pearson model / EP | Spearman model / EP | MAE model / EP |
|---|---|---|---|---|
| 1 | 600 | 0.426 / 0.347 | 0.614 / 0.481 | 1.494 / 1.598 |
| 2 | 616 | 0.391 / 0.436 | 0.504 / 0.519 | 1.544 / 1.463 |
| 3 | 652 | 0.495 / 0.451 | 0.727 / 0.706 | 1.136 / 1.290 |
| 4 | 656 | 0.557 / 0.543 | 0.707 / 0.706 | 1.185 / 1.234 |
| 5 | 659 | 0.508 / 0.535 | 0.721 / 0.744 | 1.165 / 1.158 |
| **Mean per GW** | 3183 | **0.475 / 0.462** | **0.654 / 0.631** | pooled **1.299 / 1.343** |
| GW2-5 | 2583 | 0.488 / 0.491 | 0.665 / 0.669 | 1.253 / 1.283 |

- The model beats EP clearly only in GW1, the one GW where training xP is not leaky.
- Its edge over EP is +0.013, against the roughly +0.14 the holdout promised.

### 4.2 Live vs holdout vs leak-free

| Metric | Live prod | Live ep_next | Live leak-free retrain | Holdout 2025-26 as reported | Holdout served live-like | Holdout leak-free |
|---|---|---|---|---|---|---|
| Mean per-GW Pearson | 0.475 (0.462-0.477 other reconstructions) | 0.462 | **0.536-0.541**; lag + servable features 0.552 | 0.632 (0.867 xP GWs / 0.536 no-xP GWs) | 0.515-0.541; **0.422 in GW1-5** | **0.589-0.598** |
| Mean per-GW Spearman | 0.654 | 0.631 | 0.685-0.695 | 0.746 | 0.709 | 0.729 |
| MAE | 1.299 | 1.343 | 1.30-1.34 | 0.826 | 0.975 (GW1-5 1.32) | 0.943 |
| Top-20 mean actual pts | 3.95 | 4.88 | 4.35-4.97 (not robust) | 6.24 | 4.43 | 4.84 |

- **2024-25 fold:**
  - as trained: 0.835-0.848;
  - served live-like: 0.472-0.479;
  - leak-free: 0.588-0.596 (MAE 0.983-0.996).
- This is the same honest level the repo's own `no_xp` record shows (0.591 / MAE 0.993 on 2024-25; SKILL.md:34).
- **Player-bootstrap gains over prod on live rows (95% CI):**
  - lag-xP +0.072 [0.046, 0.096];
  - no-xP +0.069 [0.043, 0.093];
  - lag + servable features +0.084 [0.059, 0.108];
  - ep_next -0.006 (not significant).
- Spearman gain for lag-xP: [+0.033, +0.059].

### 4.3 Top end and captaincy

- **Calibration of the model's high predictions:**
  - Predictions of 5 or more: 6.28 predicted vs **3.91** actual (n=74).
  - Predictions of 6 or more: 7.11 vs **2.75** (n=36; all status 'a', all played).
  - The 6-7, 7-8 and 8+ bins returned 2.73, 3.71 and 2.29 actual.
- **Top-20 precision:** model 0.13 (3.95 pts), EP 0.20 (4.88), price 0.22 (4.56).
- **Captain pool:** the model's top 15 averaged 3.64 actual vs 4.97 for EP's top 15. The model's #1 was the best of its top 15 in 1 of 5 GWs. Spearman within the model's top 15 was -0.14.
- **fpl_xp drives the top end:** its SHAP contribution to the top 20 is +3.33 (+5.45 in GW3).
  - Bruno GW3: predicted 10.22, of which +8.2 comes from fpl_xp. With xP zeroed he would be 4.24. He scored 2.
  - Stach GW5: fpl_xp SHAP +4.1.
- **After the leak-free retrain:**
  - Live: predictions of 5 or more return 4.25 (n=170). The 7-8 and 8+ bins return 5.62 and 6.36, so they are ordered.
  - Holdout: predictions of 5 or more return 5.8-6.1, vs 3.3-4.1 for the leaky model served live-like.
  - 2024-25 #1 pick: 8.58 vs 6.00 for the leaky model and 7.82 for EP.

### 4.4 Mechanism

- **FPL recomputes ep_this after a GW finishes.** GW2-5 values changed for 79%, 53%, 57% and 46% of players.
  - 96-98% of the changed values equal post-GW form.
  - Correlation with GW points: 0.44-0.54 before the recompute, 0.74-0.83 after (e.g. Semenyo 4.0 → 7.8 after scoring 17).
  - GW1 is not recomputed (94% unchanged).
- **Historical vaastav xP carries the outcome of its own GW:**
  - Same-GW OLS weight is 0.29-0.37 in every season from 2020-21 to 2025-26, and 0 of 213 GWs fall below 0.10.
  - The change xP_g - xP_{g-1} correlates +0.45 to +0.49 with GW g's own points in 100% of GWs. The live control shows the reverse: +0.16 with the same GW, +0.60 with the previous GW.
  - A GW29 bootstrap taken after all 10 matches matches vaastav xP29 for only 60.5% of players.
- **The learned decode `4*fpl_xp - 3*pts_rolling_3`** correlates 0.52-0.66 with the target on training rows but only 0.068 live.
  - Among GW5 nailed starters, prod loads +0.36 on **GW1** points, and its correlation with the GW5 outcome is 0.006.
- **Symptoms:**
  - Adding +2 to pts_rolling_3 lowers 93-99% of MID/FWD predictions with fpl_xp >= 3.
  - Holding fpl_xp fixed, a higher team win probability lowers predictions (-0.11 to -0.60).
- **Gain share of fpl_xp:** 30.2% overall. It ranks #1 for DEF, MID and FWD (35%, 33%, 25%) and #2 for GK (15%).

### 4.5 Other findings

- **GW2 prior-EP quirk** (medium; a special case of the leak):
  - About 88% of players still had the pre-season prior EP at the GW2 deadline.
  - Players with 0 GW1 minutes averaged 1.04 live, against 0.08-0.16 in training.
  - Effect: Martinez 4.97 and Pope 4.91 were predicted vs Raya 2.70, and Gyökeres 4.65, which led to the rejected swap that saved 20 points.
  - The lag retrain fixes it.
- **Haulers:** predicted 2.73 vs 7.87 actual, which is typical (holdout 3.03 vs 7.78). Fixing this needs an upside or quantile head.
- **Regular forwards are under-predicted** (2.84 vs 3.92) because understat is missing. The understat repair gives 3.45.
- **Context (unverified):** no systematic bias for promoted clubs, new players or DEFCON profiles.

---

## 5. Soundness audit

### 5.1 Confirmed defects, ranked by impact

**1. fpl_xp leak** (skew, CRITICAL; players_raw.py:79-88 unshifted; fpl_live.py:251-280)
- Impact:
  - Live Pearson -0.07.
  - Inflated top end.
  - Hit-chasing.
  - 2024-25 backtest -240 to -370 pts at 1 transfer/GW, and -350 to -715 with free transfers.
- Fix:
  - For seasons up to 2025-26, set fpl_xp(g) = xP(g-1) for the same player, scaled by fixture count.
  - Keep GW1, set all-zero GWs to NaN, and do not shift 2026-27. Or drop the feature.
  - Retrain. Serving-time zero or NaN is **not** a fix.

**2. Metrics, README, benchmark claims and the retrain gate all rest on the leak** (process, HIGH)
- Evidence:
  - 0.632 = (11 x 0.868 + 27 x 0.536)/38.
  - 2,918 reproduces as 2,899-2,916 with leaky xP, and becomes 1,849-2,019 served live-like.
  - SKILL.md:36-39 and :43 would block the leak-free model.
- Fix:
  - Use a live-like or leak-free holdout, split by xP presence.
  - Re-baseline the gate: about 0.54 for the current model served live-like, about 0.59 for a leak-free model.
  - Retract the README and benchmark claims.

**3. Uncapped single-GW MILP converts the inflated top end into hits** (process, MEDIUM; counterfactual)
- Evidence:
  - The pure-model path scores 239 with -24 in hits.
  - GW3 took -8 for +0.87 of objective; GW5 took -16 for +4.3.
  - The model's 11 proposed buys were predicted 77.7 and scored 28.
  - 2024-25 simulation of the model as served live: -2.5/GW capped, -9.8/GW uncapped.
- Fix:
  - Default --max-transfers to the free-transfer count.
  - Require a hit margin, and give a rolled free transfer an explicit value.
  - Longer term, plan over multiple GWs.

**4. Understat never collected for 2026-27** (skew, MEDIUM)
- 15 features are 100% NaN, and 70 of 421 players have no understat ID.
- Impact on its own is small: Spearman +0.009, Pearson +0.017.
- Forwards are under-predicted, and this data is needed alongside fix 1: otherwise Haaland comes out at 2.7-3.2.
- Fix: a weekly refresh in gameweek.py plus ID supplements, or drop the features.

**5. 2025-26 missing xP stored as 0** (data, MEDIUM)
- Covers 27 GWs (21,045 rows), including GW31-37 of the early-stopping set.
- Fix: set these to NaN as part of fix 1, and choose validation GWs that have xP.

**6. refresh_element_summaries deletes everything, then ignores failures** (bug, MEDIUM; fpl_live.py:286-296)
- Evidence: with 3 summaries missing, Haaland goes 5.21 → 10.35 and Bruno 4.49 → 10.13.
- Fix: download to a temporary directory, swap only on full success, and abort otherwise.

**7. --skip-refresh has no staleness guard** (bug, MEDIUM)
- Evidence: Stach 8.35 → 3.74; mean absolute change 0.48, max 7.96.
- Fix: compare the latest round in the summaries with the last finished event.

**8. Current-club stamping** (bug, MEDIUM, latent; fpl_live.py:387, :416)
- Evidence:
  - 25 rows from 17 players who moved club.
  - is_dgw is set on 376 GW1 rows and 190 GW2 rows.
  - opp_pts_conceded_r5 is inflated by +17 to +36.
- Impact: small on live predictions, but it corrupts the rows the ~GW8 retrain would use.
- Fix: take the team from the fixture itself.

**9. Flag calibration** (MEDIUM; pool.py:58-59)
- Evidence: players flagged 75% did not play 73-76% of the time (40 of 53; regulars 8 of 11).
- The flag is also applied twice: once inside fpl_xp and again by the chance/100 scaling.
- Impact: about 0.9 pts ex-ante in GW5.
- Fix: use a single empirical P(play) table.

**10. Bench weights fixed and risk-blind** (LOW-MEDIUM; lineup_selector.py:12-14)
- Evidence:
  - 10 of 15 outfield bench-slot weeks had a player with 0 minutes, and there were 0 auto-subs.
  - The implied slot-1 weight is about 0.35.
- Impact: about 3 pts in GW5, and roughly 7-18 pts over the rest of the season.
- Fix: risk-aware weights, plus a penalty for bench players who have not played.

**11. h2h odds NaN live** (LOW)
- Filling them hurts under the leaky model (-0.042) and is neutral under lag-xP.
- Fix: collect them only together with the retrain.

**12. Minor skews** (LOW)
- FBref prev_* are NaN.
- Props book count and timing differ from training.
- The synthetic `selected` column is rounded.
- ep_next = form with no fixture offset; restoring the offset does not help.

**Unverified low-severity items (context):**
- F4 `_ensure_team_column`.
- LCR-07: select_squad has no vice or bench terms.
- LCR-08: printed xPts vs scaled values, and the objective is not recomputed after overrides.
- LCR-09: double-gameweek over-prediction of 7-22%.
- DA-11: the backtest starts GW1 with free_transfers=1.
- TSS-11: end-of-season set-piece flags.

**Analysis note:** SP/cache/preds_2026-27_prod.parquet is not what was served; it differs by 0.10 on average. Use train_serve_skew/preds/A0_asserved_gw{k}.parquet instead.

### 5.2 Checked and sound

- **Engine replay:** reproduces 296 exactly.
- **MILPs:** the transfer MILP matches brute force in 45 of 45 cases and the lineup MILP in 25 of 25; the pre-filter loses nothing.
- **Entry state:** 2 free transfers into GW6, bank £0.2m, selling prices correct, all chips available.
- **Synthetic rows:** match the real rows on 120 of 122 features, with no lookahead, and the GW5 deadline file reproduces byte for byte.
- **Live-loop mechanics:** deadline logic, captain override and submission payload, pool filters, and ID mapping (667 of 667).
- **Strength backfill:** the scale matches training, and the GW1-4 bug was negligible (mean change 0.07-0.08).
- **Replicas validated:** GW3 matches 23 of 23 players and GW4 17 of 17, both within 0.01.
- **Leak check across features:** no feature other than fpl_xp shows a same-GW excess above 0.056.

---

## 6. Decision process

### 6.1 Captaincy

| GW | Chosen | Model | Market | EP | Most-captained | Best |
|---|---|---|---|---|---|---|
| 1 | Haaland 2 (model) | Haaland 2 | – | Bruno 2 | Haaland 2 | João Pedro 11 |
| 2 | Bruno 23 (model) | Bruno 23 | Haaland 13 | Bruno 23 | Haaland 13 | Bruno 23 |
| 3 | Haaland 9 (override) | Bruno 2 (pred 10.22) | Haaland 9 | Bruno 2 | Haaland 9 | Haaland 9 |
| 4 | João Pedro 12 (override) | Bruno 2 (pred 8.72) | João Pedro 12 | Bruno 2 | Haaland 9 | João Pedro 12 |
| 5 | Haaland 6 (override) | Stach 5 (pred 8.35) | Haaland 6 | Haaland 6 | Haaland 6 | Van Hecke 6 |
| **Armband** | **52** | 34 | 42 | 35 | 39 | 61 |

- The overrides gained +7, +10 and +1 for the team. The notes' "+14" is the captain-slot gap, not the team gain.
- The leak-free retrain would not fix captaincy. On the squads actually held, totals with each variant's captain were:
  - prod 278;
  - lag-xP 257-263;
  - no-xP 265-276;
  - lag + servable features 290;
  - the human 296.

### 6.2 Transfers

| Transfer | Expected gain | Realized gain |
|---|---|---|
| James→Calafiori | +10.1 | +7 |
| Tzolis→Stach | +12.4 | 0 |
| Lucky→Mendy | +5.9 | +1 |
| João Pedro→Gyökeres (rejected) | +2.24 | would have been -20 over GW2-5 |

Transfers made realized about 30% of their expected gain.

### 6.3 Lineup and bench

- **Lineup regret was 16.**
  - GW3: the Van Hecke under-prediction (0.82, driven by fpl_xp = form = 1.0) cost +5 to +7.
  - GW4: the regret came from Mendy's over-prediction (5.44 predicted, 1 point).
- **Dead bench:**
  - Drakes-Thomas played 0 minutes in all 5 GWs, and his prediction went 0.15 → 0.02.
  - Lucky played 0 minutes in all 3 GWs he was owned.
  - Slot 1 always held a nailed player; slots 2-3 were dead.
- **Cost:** about 3 pts in GW5, since no starter blanked in GW1-4.
- All four de-leaked GW6 plans sell Drakes-Thomas.

### 6.4 Counterfactual paths (from the actual GW1 squad)

| Path | GW1 | GW2 | GW3 | GW4 | GW5 | Total | Hits |
|---|---|---|---|---|---|---|---|
| **Actual decisions** | 40 | 99 | 52 | 69 | 36 | **296** | 0 |
| Actual squads, model lineup + captain | 40 | 99 | 45 | 59 | 35 | 278 | 0 |
| Pure model, 1 transfer/GW cap | 40 | 89 | 51 | 57 | 35 | 272 | 0 |
| Set-and-forget GW1 squad | 40 | 99 | 47 | 54 | 28 | 268 | 0 |
| Pure ep_next MILP | 40 | 95 | 27 | 47 | 39 | 248 | -20 |
| Pure model + market captain | 40 | 79 | 26 | 70 | 30 | 245 | -24 |
| **Pure model uncapped (default)** | 40 | 89 | 20 | 60 | 30 | **239** | -24 |

**Model-variant paths are noisy** (5-GW SD about 22-33):
- Pure-model paths: prod 247; lag-xP 239-304; no-xP 231-233; lag + servable features 301; ep_next 272.
- The one robust signal is hits: prod recommends 24 points of hits, de-leaked variants 0-16.

**Override ledger: +48 measured one GW at a time (+57 over the path).**

| Override | Gain |
|---|---|
| GW2 roll instead of João Pedro→Gyökeres | +10 |
| GW3 2-free-transfer plan instead of the -8 plan | +5 |
| GW3 Haaland captain | +7 |
| GW4 João Pedro captain | +10 |
| GW5 roll instead of the -16 plan | +15 |
| GW5 Haaland captain | +1 |

---

## 7. Variance vs process verdict

**Is -3 over 5 GWs informative? No.**
- The 5-GW SD is about 25-32, so z is about -0.1.
- P(total ≤ -3) is 46% with zero edge, and 19% even with a +4.5/GW edge.

**Is GW5's -12 informative? No.**
- z = -1.08, and about 11 of the 12 points were variance. About 2.3 was bench structure.
- 19.6% of all manager-GWs finish at -12 or worse.

**Were we lucky? About neutral.**
- Luck vs ep_next was +2.
- Luck vs the model was +24, mostly GW2.

**Did we have an ex-ante edge? No, by any lens.**
- Expected edge vs the template: -0.7 (model), -24.6 (ep_next).
- Realized: -19 vs the template, -3 vs the average.
- The gap is structural.

**Is the model as good as advertised? No: process/bug.**
- Live 0.475, against the 0.632 advertised.
- Its edge over EP was +0.013, against +0.14 promised.
- The live-like replay predicted 0.422, which is exactly what we observed.

**Was the decision process sound? The human layer was.**
- Overrides were worth +48 to +57.
- 0 hits were taken, and the GW5 roll saved 13-16.
- The optimizer was the weak link: -24 in hits, plus wrong captains.

**What to expect from here:**
- Current prod followed unsupervised: -2.5/GW capped, -9.8/GW uncapped (2024-25 simulation).
- Leak-free model: about +2 to +5/GW, varying by season from -1 to +8.
- That means retraining and targeting a finish somewhere in the top 10-50%.

**Summary:**
- The points outcome is variance: after 5 GWs the standard error of our per-GW edge is about 5.8, and about 33 GWs are needed to show a +4.5/GW edge.
- The model's quality is a real process and bug problem, but a fixable one. The human veto has so far hidden it.
- The realized cost so far is small. The expected-value cost going forward is material, and would grow if the ~GW8 retrain kept the leak.

---

## 8. Recommended actions

### (a) Before the GW6 deadline (2026-10-10 10:00Z)

1. **Do not act on the prod GW6 output.**
   - Prod recommends Bruno as captain (12.45, against an EP of 7.2) and a -8 plan: Stach→Davis, Van Hecke→Groß, Szoboszlai→Scott, Drakes-Thomas→Schade.
   - The de-leaked variants rate Bruno at 4.4-5.4, take no hit, and sell Drakes-Thomas. They disagree on everything else, so lean on EP and the market.
   - The EP roll plan captains Haaland (EP 9.2). João Pedro is still flagged at 75%.
2. **Change the eval and the gate first.**
   - Use a live-like holdout, split by xP presence.
   - Re-baseline SKILL.md:36-39 and delete line :43.
3. **Retrain leak-free** (lag xP for seasons up to 2025-26; all-zero GWs to NaN; keep GW1; no shift for 2026-27).
   - Accept only if:
     - the holdout is about 0.59;
     - it gains at least +0.05 over prod on the A0 as-served 2026-27 rows;
     - the 2024-25 backtest at 1 transfer/GW is at or above 2,008.
   - Existing recipes: prediction_quality/08_deleak_experiment.py, impact_live_code_review/i03_train_arms.py, impact_historical_context/i2_train_variants.py.
   - A run takes about 45-50 minutes on 4 threads.
4. **Ship with servable features.** Collect understat for 2026-27 and add the 70 missing IDs, or drop the 22 features that are all-NaN live. Do not ship the xP fix alone.
5. **Run GW6 with guard rails:**
   - `--max-transfers 2`;
   - captain by the market/EP rule;
   - one free transfer on Drakes-Thomas, replacing him with a nailed player at £4.0-4.5m;
   - treat João Pedro's 75% flag as about 27% chance of playing.
6. **Add cheap guards:**
   - make the refresh atomic;
   - add a staleness check;
   - add a NaN-coverage warning;
   - fix current-club stamping now.
7. **Log every run** to runs/.
8. **Correct the record:** EP_FORMULA.md, the memory note "xP verdict", the "+14" in the notes, the fpl_live.py docstring, the README, and reports/season_report_2024-25.md. **This includes the portfolio page the user wants focused on "the current model".** Use the honest figures: live 0.475 against ep_next 0.462, and about 0.59 leak-free on the holdout.

### (b) Model improvements to test, ranked by expected value

**1. Leak-free fpl_xp**
- Expected value: +0.07 live; +240 to +370 pts per season at 1 transfer/GW and +350 to +715 with free transfers (2024-25 backtest).
- Evaluate with:
  - a live-like holdout on both folds;
  - a paired per-GW Wilcoxon (37 of 38 GWs won);
  - the A0 rows;
  - backtests averaged over at least 4 seeds.

**2. Servable feature set**
- Expected value: +0.012 on top of item 1, and it closes the forward under-prediction.
- Evaluate with the same A/B plus forward calibration.

**3. Top-end guard (a stopgap)**
- Cap predictions at 6, or shrink them above 4, or blend them with EP.
- Expected value: one-step 214 → 234-242.
- Evaluate with calibration plots and one-step replays.

**4. Hit discipline and free-transfer value, then multi-GW planning**
- Expected value: 1,634 vs 1,913 for the leaky model; not yet measured for the retrained model.
- Evaluate with a grid of hit margins.

**5. Empirical availability model**
- Expected value: about 0.9 pts per flagged decision.
- Evaluate against 2026-27 flags, leaving one GW out.

**6. Risk-aware bench**
- Expected value: 7-18 pts over the rest of the season; a prerequisite for Bench Boost.
- Evaluate with a formation-aware Monte Carlo and a backtest.

**7. Captain rule or model, market first**
- Expected value: 52 vs 34 armband points in GW1-5.
- Evaluate with a captain backtest on the holdout.

**8. Hygiene**
- Validation GWs that have xP, props filters, the `selected` estimator, FBref prev_*, and a double-gameweek check.
- Each is worth ≤0.02.

**9. The ~GW8 retrain with 2026-27 rows**
- Do not mix leaky xP with honest snapshots, and fix current-club stamping first.
- Evaluate walk-forward within 2026-27.

xMins modelling was not evaluated.

### (c) Process changes

- **Formalise the human veto:**
  - default `--max-transfers` to the number of free transfers;
  - require a hit to show a positive EP gain and a model gain of more than 4 + margin;
  - captain market/EP first, never a flagged player;
  - apply the same rule to TC1 in GW7.
- **Judge by diagnostics, not rank,** until at least half the season: weekly Pearson/Spearman vs ep_next, calibration of predictions ≥5, realized vs predicted transfer gains, and expected edge vs the template.
- **Make the gate like-for-like:** live-like holdout plus as-served rows.
- **Run close to the deadline,** and never use `--skip-refresh` without the staleness guard.
- **Weekly regression checks:** engine replay and harness.py.
- **Save every decision artifact.**

---

## 9. Claims that did not survive verification

1. **DA-04 impact: "the top-end collapse is a season-level form drop; EP collapsed too (0.58 vs 0.87-0.98)."** REFUTED. The historical baseline is post-GW xP. A pre-deadline proxy's EP>7 bin scores 4.38, the same as live's 4.35 (i4b_calib_control.log).
2. **"The stale GW2 ep_next is new this season."** UNVERIFIABLE. The historical comparison uses the post-GW2 xP.
3. **"Don't drop fpl_xp; retrain with 2026-27 rows so the model learns the new semantics."** SUPERSEDED. That was tested only with serve-time NaN; five leak-free retrains improve by +0.07.
4. **"Missing fixture offset causes the miscalibration."** NOT SUPPORTED. Restoring the offset gives no gain.
5. **"Serve xP=0 as a stopgap."** PARTIAL. It hurts calibration and leads to goalkeeper captains; use it only as an emergency fallback.
6. **Live top-20 gains from the retrain.** NOT ROBUST. The gain comes from GW2, and one verifier's retrain got 3.78. The holdout does support a +1.6 gain.
7. **"The retrain removes the hits and changes the captain in 4 of 5 GWs."** CORRECTED. Hits fall to 0-16 (not 0), and each variant changes the captain in 3 of 5 GWs.
8. **"+14/+20 captain points."** CORRECTED to +7/+10 team points.
9. **"Backup keepers" (PQ-3).** CORRECTED. Martinez and Suzuki started; the staleness affected about 88% of all players.
10. **PQ-5 OLS coefficients.** POORLY IDENTIFIED (collinearity 0.967). The conclusion stands on the SHAP evidence.
11. **TSS-3 severity high.** CORRECTED to medium.
12. **TSS-5 "odds 0.539→0.551" and "+3 pts".** CORRECTED. That was the whole bundle, and the mean is +0.18.
13. **LCR-03 "-0.018".** CORRECTED to -0.005 to -0.018.
14. **LCR-02 "silent", high.** CORRECTED. WARNINGs are printed; medium.
15. **DA-03 "-57", high.** CORRECTED to medium, since it is counterfactual. Hits account for only 24 of the 57.
16. **DA-07 "-14 from Van Hecke."** CORRECTED to +5 to +7.
17. **DA-08 "0.52 weight; 15-50 pts."** CORRECTED to about 0.35 and 7-18 pts; the dead slots were 2-3.
18. **GW5-02 "one live bench player", "never pays", "-2/GW".** CORRECTED:
    - GW4 had two live bench players.
    - The Ajayi transfer scored +4.63 in the objective.
    - The typical cost is 0.35-0.75/GW.
19. **GW5-03 "-0.87 lost", pool.py:62-63.** CORRECTED. It is an ex-ante figure with 0 realized, and the lines are 58-59.
20. **GW5-06 "unique to the model."** PARTIAL. EP also overshot, but the model's top 15 was worse in 5 of 5 GWs.
21. **F2 "15-75 min", medium.** CORRECTED to 28-374 min, low.
22. **HC2 "honest system +2 to +8, top 10-35%."** CORRECTED. That applies to a retrained model with ±100-250 of noise; current prod served live is below the field. Realistic range: top 10-50%.
23. **HC3 "258 vs 289" as proof.** WEAK. It is policy-dependent.
24. **HC4 "0 vs NaN causes the collapse."** NOT ATTRIBUTABLE.
25. **HC6 "+30".** QUALIFIED. Excluding chips, the like-for-like gap is +15 to +19.
26. **HC1 "all other features within 0.05."** CORRECTED. Five features exceed that, but they are not leaks.
27. **Recorded GW5 plan "4 transfers -12, Stach captain."** NOT REPRODUCIBLE. The +13 to +16 saving holds for every variant.
28. **DA-04: the "honest" benchmark calibration used as a baseline.** INVALID. The benchmark predictions contain leaky xP.

---

## 10. Gaps

- **Only 5 GWs.** All counterfactuals are single-path and noisy.
- **GW1 squad not re-audited.** It came from the full_pregame model, which also contains fpl_xp.
- **Missing printouts.** The GW5 printout does not exist, and the GW2 printout is a stale test run.
- **The retrain has not been run in the repo.** The best variant (lag + servable features) was trained by only one agent.
- **Leak-free models have flat MAE and a +0.2 mean over-prediction.** Their effect on hit decisions under free transfers has not been characterised.
- **Backtest limitations:** no chips, no availability filter, and the DA-11 free-transfer bug is unverified.
- **Not evaluated:** xMins, multi-GW MILP, chips.
- **Entry state was checked only against formula plus the public API.** The authenticated endpoint was not queried.
- **FPL's future EP semantics are unknown,** including how double gameweeks are handled.
- **Unverified low-severity items** are listed as context only.
- **The relayed user request was not actioned here** (the synthesis is read-only). The request was to delete the RL content from the portfolio page and present the current model. That page is not in REPO. The inflated figures appear in README.md, reports/season_report_2024-25.md, scripts/make_figures.py, ROADMAP.md, notebooks/train_predictor.ipynb and .claude/skills/retrain/SKILL.md.

---

## Appendix A: number reconciliation

| Quantity | Values reported | Used | Why |
|---|---|---|---|
| Live prod per-GW Pearson | 0.475, 0.466, 0.464-0.477 | 0.475 (range 0.462-0.477) | PQ's replica is validated against the printouts and its per-GW file is on disk |
| Leak-free retrain, live | 0.529 (≤2024-25 training), 0.536-0.541 | 0.536-0.541, i.e. +0.07 | Like-for-like training seasons |
| "Honest holdout" | 0.47-0.55, 0.48-0.52, 0.589-0.598 | Two separate quantities | Current prod served live-like: 0.515-0.541 (2025-26), 0.472-0.479 (2024-25). Leak-free model: about 0.59 |
| 2024-25 backtest, leaky model served live, 1 transfer/GW | 1,967 / 1,849 / 2,019 / 1,881-1,938 | About 1,850-2,020, central 1,913 | Mean over 4 seeds; the other figures fall inside that range |
| 2024-25 backtest, leak-free, 1 transfer/GW | 1,967-2,320 | Central about 2,160 | Path noise is ±65-250 |
| 2024-25 backtest, free transfers | Leaky 1,515-1,728; leak-free 2,003-2,249 | Gain +350 to +715 | All agents agree on the direction |
| De-leaked hits over GW2-5 | 0 to 16 | 0-16, against prod 24 | Depends on which rows were used |
| Captain overrides | +14/+20 or +7/+10/+1 | +7/+10/+1 | Team total, not captain-slot gap |
| GW5 bench EV gap | -2.02 / -2.48 / -2.3 to -3.7 | -2.3 (range given) | Formation-aware Monte Carlo; the attribution still sums to -12.00 |
| Bench slot-1 weight | 0.52 / 0.35 | About 0.35 | Flag-aware estimate |
| Players flagged 75% who did not play | 0.755 / 0.727; LCR 37.5% played | 73-76% did not play | Verified; LCR used a different filter |
