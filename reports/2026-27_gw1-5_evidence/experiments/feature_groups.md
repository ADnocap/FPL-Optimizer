# feature_groups: leave-one-group-out, serve-sim and pruning on the leak-fixed baseline

Agent `feature_groups`, 2026-09-24/25. The reference is the `baseline_fixed` harness (xP leak fixed).
The primary metric is per-GW Spearman. Every number below is a paired per-GW delta vs a reference, with a
bootstrap 95% CI and the count of GWs won. "Pooled" means the 76 GWs of H25+H24 taken together.

## TL;DR

1. **New leak found in the understat per-match features. It is in the current baseline and in prod.**
   `features/understat.py` keeps matches with `date < gw_date`. Here `gw_date` is the GW's first-kickoff
   timestamp, but understat `date` is a date with no time (midnight). So every match played on the first
   calendar day of a GW counts toward that same GW's `xg_/xa_/npxg_/shots_/key_passes_/xgchain_/xgbuildup_rolling_*`
   and `goals_vs_xg_5`/`assists_vs_xa_5`. Example: Haaland 2023-24 `xg_rolling_3` at GW1 is 0.882, which is the
   xG of his own GW1 match. His GW5, GW6 and GW7 values also include that week's own match. About 14.5% of all
   rows (19-26% of rows with understat data, depending on the season) are affected. A vectorised rebuild
   (`understat_leak.py`) reproduces the cached features 100% in leaky mode. The fixed rebuild (strictly
   earlier calendar days) changes those rows, and the Spearman of FWD/MID `xg_rolling_3` with the GW's
   points drops from 0.334 to 0.302. Fix: `gw_date.normalize()`.
   Removing the leak lowers the baseline's per-GW Spearman by 0.0012 on H25 and 0.0052 on H24 (fast), and by
   0.0001 and 0.0040 (full). top10 and cap drop even more: full H25 top10 goes from 5.66 to 4.99. This matters
   for other agents: part of `baseline_fixed`'s top10 and cap is understat leak.
2. **Understat is worthless once honest, so drop it. Do not collect it.** A model trained without understat
   vs the honest-understat baseline: pooled +0.0007 [-0.0003, +0.0017], 41/76 GWs (H25 +0.0009, H24 +0.0005).
   A 2026-27 collection path works: `understatapi` 0.7.1 plus a cheap roster+shots path with about 20 calls/GW,
   verified to reconcile exactly. It would still buy nothing.
3. **Today's live model is mis-served on 26 features that are 100% NaN in 2026-27.** These are 15 understat
   per-match features, the 4 **h2h odds** features, 3 FBref-only prior features and 4 legacy 2016-19 stats.
   The h2h odds were not previously flagged: `gameweek.py` only collects props, and `data/odds/2026-27.json`
   does not exist. In the LIVE-protocol model these features carry about 20% of the total gain. Serve-sim
   (baseline model, those features set to NaN at test time, which is how live runs today) costs pooled
   -0.0061 per-GW Spearman fast, top10 -0.78. In full mode it costs -0.0025 on H25 and -0.0092 on H24.
4. **WINNER: "train what you serve" (`drop:livedead`), i.e. retrain without those 26 features.** Compared with
   the baseline served as live, it wins on both holdouts in both modes:
   fast H25 +0.0028 (29/38), H24 +0.0044 (28/38); full H25 +0.0029 [+0.0018, +0.0042] (27/38),
   H24 +0.0038 [+0.0018, +0.0059] (25/38). top10, cap and MAE hold or improve.
   On LIVE (the real 2026-27 GW1-5 serve): fast +0.0078 (4/5), full +0.0026 (4/5), top10 +0.66, cap +3.0,
   MAE -0.039. Compared with the honest all-features-available baseline it is neutral (full pooled -0.0005).
   It needs no new data. It only removes features.
5. **Most groups do not pay their way, individually or jointly.** 20 of 27 group ablations fall within seed
   noise (|delta| <= 0.0024). Dropping the 20 or 40 lowest-importance features together is neutral
   (+0.0002 and +0.0005 pooled). Stacking 40 more pruned features on top of the winner is also neutral
   (56 features left; +0.0002 vs the winner). Only ownership (-0.014 pooled, 4/76 GWs won), price (-0.0035),
   transfers (-0.0026), opp_form (-0.0021) and minutes (-0.0019; H24 -0.0026) clearly pay their way.
   Borderline, at the noise floor: venue_split (-0.0017), odds_h2h (-0.0017; H24 -0.0029) and xp_lag
   (H24 -0.0043 but H25 +0.0009).
6. **Methodology caveat (seed noise).** Retraining the baseline with seeds 43 and 44 "loses" pooled -0.0019,
   with the CI excluding 0, in all 4 runs. Seed 42 is a lucky draw by about 0.0015-0.002, and the harness's
   GW-bootstrap CI does not capture training randomness. A single-seed delta inside +/-0.0025 is noise, even
   when the CI excludes 0.

## 1. Feature -> group map (all 122 features, `groups.py`)

The columns are: group size; non-NaN coverage for 2022-26 training rows vs 2026-27 live; gain share in the
H25 model (mean over the 4 position boosters); and the leave-one-group-out delta vs `baseline_fixed`
(fast; per protocol, then pooled with CI; pooled top10, cap and MAE).

| group | n | cov train/live | gain | H25 dS (won) | H24 dS (won) | pooled dS [CI] | dTop10 | dCap | dMAE |
|---|---|---|---|---|---|---|---|---|---|
| understat_rolling (leaky as cached) | 15 | 0.61/0.00 | 0.124 | -0.0003 (18/38) | -0.0047 (14/38) | -0.0025 [-0.0048,-0.0004] | -0.318 | -1.803 | +0.0116 |
| odds_h2h | 4 | 1.00/**0.00** | 0.047 | -0.0004 (16/38) | -0.0029 (9/38) | -0.0017 [-0.0028,-0.0006] | +0.070 | -1.447 | +0.0036 |
| legacy_detail (2016-19 only) | 4 | 0.00/0.00 | 0.007 | +0.0002 (21/38) | -0.0011 (16/38) | -0.0004 [-0.0015,+0.0005] | -0.078 | +0.184 | +0.0007 |
| prior_fbref_dead | 3 | 0.45/**0.00** | 0.014 | -0.0018 (14/38) | -0.0013 (20/38) | -0.0015 [-0.0025,-0.0005] | +0.150 | -0.066 | +0.0009 |
| set_pieces (end-of-season snapshot) | 4 | 1.00/1.00 | 0.003 | -0.0010 (13/38) | -0.0021 (14/38) | -0.0015 [-0.0025,-0.0006] | -0.099 | +0.553 | +0.0026 |
| opp_ratings (fdr, strengths) | 4 | 1.00/1.00 | 0.028 | +0.0002 (19/38) | -0.0014 (14/38) | -0.0006 [-0.0016,+0.0004] | -0.143 | -0.566 | +0.0096 |
| prior_understat | 6 | 0.51/0.59 | 0.032 | +0.0005 (19/38) | -0.0012 (16/38) | -0.0003 [-0.0013,+0.0006] | +0.179 | +0.132 | -0.0008 |
| prior_fotmob | 3 | 0.30/0.37 | 0.012 | +0.0006 (21/38) | +0.0002 (21/38) | +0.0004 [-0.0006,+0.0014] | +0.042 | +0.737 | -0.0029 |
| discipline | 2 | 0.97/0.79 | 0.001 | -0.0009 (15/38) | -0.0001 (18/38) | -0.0005 [-0.0013,+0.0003] | -0.084 | -0.105 | -0.0026 |
| attack_returns | 4 | 0.97/0.79 | 0.001 | -0.0007 (17/38) | +0.0007 (22/38) | +0.0000 [-0.0008,+0.0009] | +0.093 | -0.355 | -0.0011 |
| defcon (+tackles/recoveries) | 7 | 0.26/0.79 | 0.009 | -0.0000 (17/38) | -0.0019 (11/38) | -0.0010 [-0.0018,-0.0001] | -0.008 | -0.329 | -0.0031 |
| fpl_xstats (+team_xgi_share) | 8 | 0.96/0.79 | 0.020 | -0.0004 (17/38) | +0.0001 (20/38) | -0.0001 [-0.0011,+0.0008] | -0.050 | -0.368 | +0.0016 |
| derived_form (incl. nailedness == starts_rolling_5) | 6 | 0.97/0.79 | 0.039 | -0.0003 (15/38) | -0.0010 (18/38) | -0.0007 [-0.0017,+0.0003] | +0.112 | -0.079 | -0.0025 |
| venue_split | 2 | 0.95/0.69 | 0.014 | -0.0018 (12/38) | -0.0016 (17/38) | -0.0017 [-0.0026,-0.0007] | +0.084 | -0.013 | +0.0018 |
| def_returns | 4 | 0.97/0.79 | 0.010 | +0.0003 (19/38) | -0.0003 (16/38) | -0.0000 [-0.0010,+0.0009] | -0.126 | -0.092 | +0.0003 |
| xp_lag | 1 | 0.74/0.80 | 0.010 | +0.0009 (24/38) | -0.0043 (7/38) | -0.0017 [-0.0030,-0.0005] | +0.104 | -0.026 | -0.0024 |
| price | 2 | 0.99/0.90 | 0.016 | -0.0032 (16/38) | -0.0037 (7/38) | -0.0035 [-0.0057,-0.0018] | +0.074 | -0.211 | +0.0071 |
| opp_form | 2 | 0.98/0.81 | 0.018 | -0.0012 (12/38) | -0.0031 (12/38) | -0.0021 [-0.0030,-0.0012] | -0.124 | -0.263 | +0.0024 |
| fixture_meta (was_home, is_dgw, gw_phase) | 3 | 1.00/1.00 | 0.030 | +0.0013 (24/38) | +0.0004 (21/38) | +0.0009 [-0.0001,+0.0018] | +0.053 | -0.947 | +0.0089 |
| transfers | 6 | 0.97/0.79 | 0.041 | -0.0016 (17/38) | -0.0035 (6/38) | -0.0026 [-0.0034,-0.0017] | +0.084 | +0.316 | +0.0054 |
| ownership | 2 | 0.99/0.90 | 0.056 | **-0.0152 (1/38)** | **-0.0126 (3/38)** | **-0.0139 [-0.0178,-0.0108]** | -0.029 | -1.053 | +0.0131 |
| bonus_bps | 4 | 0.97/0.79 | 0.019 | -0.0012 (13/38) | -0.0004 (17/38) | -0.0008 [-0.0018,+0.0001] | +0.007 | +0.145 | +0.0015 |
| ict | 6 | 0.97/0.79 | 0.074 | -0.0009 (15/38) | -0.0002 (18/38) | -0.0006 [-0.0014,+0.0003] | +0.191 | -0.171 | -0.0010 |
| core_points | 4 | 0.97/0.79 | 0.028 | +0.0003 (18/38) | -0.0014 (14/38) | -0.0006 [-0.0016,+0.0004] | +0.103 | +0.579 | -0.0028 |
| minutes | 7 | 0.97/0.79 | 0.172 | -0.0013 (14/38) | -0.0026 (12/38) | -0.0019 [-0.0030,-0.0009] | -0.018 | -0.105 | +0.0018 |
| synth_ep | 3 | 0.98/0.86 | 0.173 | -0.0002 (19/38) | +0.0004 (22/38) | +0.0001 [-0.0008,+0.0010] | -0.041 | -0.434 | -0.0017 |
| props | 6 | 0.18/0.56 | 0.000 | not testable: all NaN in H25/H24 training; LIVE-model gain 0.005 | | | | | |

Reading the table: the model is highly redundant. The big-gain groups (synth_ep 17%, minutes 17%, ict 7%)
cost nothing when removed alone, because the other groups carry the same information. The noise floor is
about 0.0024 (seed reruns: -0.0017, -0.0021, -0.0014, -0.0024).

Risk notes. `set_pieces` comes from `players_raw.csv`, an end-of-season snapshot, so training and holdout
rows see set-piece roles assigned later in the season. That is a mild lookahead, and the group is neutral
anyway. `opp_ratings` (fdr from fixtures.csv and strengths from teams.csv) are also end-of-season snapshots,
and live they need the strength reconstruction (the GW5 bug in memory). This group is neutral too. The
"robust" variant (winner plus dropping set_pieces and opp_ratings) is neutral vs the winner: pooled -0.0003.

## 2. Serve-sim: baseline model with features NaN at test time only (= live today)

| test-time NaN | H25 fast | H24 fast | pooled fast [CI] (won) | pooled dTop10 | H25 full | H24 full |
|---|---|---|---|---|---|---|
| understat_rolling | -0.0043 | -0.0077 | -0.0060 [-0.0082,-0.0039] (21/76) | -0.601 | -0.0028 | -0.0073 |
| odds_h2h | -0.0000 | -0.0026 | -0.0013 [-0.0021,-0.0005] (28/76) | +0.070 | -0.0010 | -0.0026 |
| prior_fbref_dead | +0.0004 | +0.0005 | +0.0004 [+0.0001,+0.0007] (44/76) | -0.086 | +0.0002 | +0.0004 |
| all 26 live-dead (**today's live**) | -0.0026 | -0.0095 | -0.0061 [-0.0083,-0.0039] (22/76) | -0.780 | -0.0025 | -0.0092 |

IS25 and IS24 in-season analogs point the same way (ss_livedead -0.0031 and -0.0071, 2/5 and 0/5 GWs won).

## 3. Understat decision

| comparison (fast) | H25 | H24 | pooled [CI] (won) |
|---|---|---|---|
| honest-understat baseline (`usfix+base`) vs leaky baseline | -0.0012 | -0.0052 | -0.0032 [-0.0054,-0.0012] (30/76) |
| trained without understat vs honest baseline | +0.0009 (21/38) | +0.0005 (20/38) | +0.0007 [-0.0003,+0.0017] (41/76) |
| honest model served NaN vs honest model served with understat | -0.0020 | +0.0000 | -0.0010 [-0.0018,-0.0002] |
| trained without understat vs leaky baseline served NaN (today) | +0.0040 (30/38) | +0.0030 (27/38) | +0.0035 [+0.0022,+0.0049] (57/76) |
| full: honest baseline vs leaky baseline | -0.0001 | -0.0040 | -0.0020 [-0.0041,-0.0001] |
| full: honest model served NaN vs served with understat | -0.0012 | -0.0009 | -0.0010 [-0.0016,-0.0005] |

LIVE (2026-27 GW1-5), with understat actually collected for 2026-27 (scratch, fixed date filter):
- honest model with collected understat: +0.0082 vs base.
- the same model served NaN: +0.0103.
- trained without understat: +0.0032.

Collecting understat therefore did not help on the 5 live GWs.

**Verdict:** dropping honest understat costs less than noise (+0.0007 pooled, i.e. slightly better). Drop it.
Do not build weekly collection.

**Collection feasibility, done anyway in a scratch dir, never REPO/data** (`understat_probe/`):
- `understatapi` 0.7.1 serves season "2026": 421 EPL players, 50 played matches up to 2026-09-20.
- The repo collector (`UnderstatCollector._collect_player_matches`) would take about 420 calls (about 10 min),
  and `_is_cached()` never refreshes a file, so it cannot run weekly as written.
- The cheap path in `collect_live_understat.py`: `league.get_match_data` (1 call), then per played match
  `match.get_roster_data` plus `match.get_shot_data`. That is 2 calls per match, about 20 per GW.
  npxG = xG minus the player's penalty-shot xG.
- Output format is identical to the files `features/understat.py` reads. Per-match values match the
  per-player endpoint exactly (checked on Haaland). Season totals match the league aggregates for all 421
  players (xG, npxG including 8 penalty takers, xA, xGChain, xGBuildup, games).
- ID coverage: 547/667 2026-27 players have an understat id in `master_id_map.csv`.

## 4. Low-importance pruning

The ranking is the max over positions of gain share, taken from THIS protocol's baseline model (no test
information). Zero-gain features, which are all-NaN in training, are excluded from K.

| variant | H25 | H24 | pooled [CI] (won) | vs |
|---|---|---|---|---|
| lowimp:20 | +0.0008 | -0.0004 | +0.0002 [-0.0008,+0.0011] (40/76) | baseline |
| lowimp:30 | -0.0012 | -0.0015 | -0.0014 [-0.0023,-0.0005] (28/76) | baseline (within seed noise) |
| lowimp:40 | +0.0001 | +0.0009 | +0.0005 [-0.0004,+0.0013] (39/76) | baseline |
| winner + 20 lowest | -0.0010 | -0.0005 | -0.0008 [-0.0016,+0.0000] | winner |
| winner + 40 lowest (56 features left) | -0.0008 | +0.0013 | +0.0002 [-0.0011,+0.0016] | winner |

The bottom-30 lists of H25 and H24 overlap on 27/30. The bottom features are: nailedness (a duplicate of
starts_rolling_5), reds/yellows, goals/assists_rolling_*, value_momentum, set-piece flags, legacy detail,
tackles/cbi*, fpl_xg*/xa*/xgi* rolling, prev_blocks_per90, cs_rolling_5, fixture_offset, opp_strength.
Pruning is safe but optional. The recommendation is to ship the winner first.

## 5. Winner: `drop:livedead` = train what you serve (96 features)

Dropped (26): `xg_rolling_3/5/10, xa_rolling_3/5/10, npxg_rolling_5/10, shots_rolling_5,
key_passes_rolling_5, xgchain_rolling_5/10, xgbuildup_rolling_5, goals_vs_xg_5, assists_vs_xa_5,
odds_team_win_prob, odds_team_draw_prob, odds_team_loss_prob, odds_team_strength, prev_sot_per90,
prev_tkl_int_per90, prev_gls_per90, fpl_key_passes_rolling_5, completed_passes_rolling_5,
big_chances_created_rolling_5, dribbles_rolling_5`.

| comparison | mode | H25 dS [CI] (won) | H24 dS [CI] (won) | pooled dS (won) | dTop10 | dCap | dMAE |
|---|---|---|---|---|---|---|---|
| vs baseline served as live (`ss_livedead`) | fast | +0.0028 [+0.0012,+0.0045] (29/38) | +0.0044 [+0.0023,+0.0064] (28/38) | +0.0036 (57/76) | +0.184 | +0.105 | -0.0057 |
| vs baseline served as live | **full** | +0.0029 [+0.0018,+0.0042] (27/38) | +0.0038 [+0.0018,+0.0059] (25/38) | +0.0034 (52/76) | +0.058 | +0.474 | -0.0061 |
| vs honest all-available baseline (`usfix+base`) | fast | +0.0014 (25/38) | +0.0001 (20/38) | +0.0007 [-0.0003,+0.0017] | -0.262 | -0.618 | +0.0015 |
| vs honest all-available baseline | full | +0.0005 (19/38) | -0.0015 (13/38) | -0.0005 [-0.0012,+0.0003] | -0.265 | -0.026 | +0.0034 |
| vs `baseline_fixed` (leaky understat, odds available) | fast | +0.0002 | -0.0052 | -0.0025 | -0.596 | -1.868 | +0.0147 |
| vs `baseline_fixed` | full | +0.0004 | -0.0054 | -0.0025 | -0.743 | -2.487 | +0.0168 |
| LIVE 2026-27 GW1-5 (sanity) | fast | +0.0078 [+0.0009,+0.0132] (4/5) | | | +0.94 | +4.4 | -0.062 |
| LIVE 2026-27 GW1-5 (sanity) | full | +0.0026 [-0.0012,+0.0063] (4/5) | | | +0.66 | +3.0 | -0.039 |

Absolute per-GW Spearman, full mode:

| protocol | baseline | baseline served as live | winner |
|---|---|---|---|
| H25 | 0.7258 | 0.7233 | **0.7263** |
| H24 | 0.7189 | 0.7097 | **0.7135** |
| LIVE (5 GWs) | 0.6849 | = base (inputs already NaN) | **0.6875** |

The row vs `baseline_fixed` is not the relevant comparison. It credits the baseline with understat, which is
leaky and unavailable live, and with odds, which are unavailable live. The H24 gap there is the understat leak:
the honest baseline itself is -0.0052 fast and -0.0040 full.

**Follow-up variant: keep the h2h odds and collect them live** (`KO` = drop understat, fbref_dead and legacy only):
- Full: +0.0008 [+0.0002,+0.0015] vs the winner, pooled (H24 +0.0020, 28/38).
- vs served-as-live: +0.0042 pooled (58/76).
- Fast: -0.0006 vs the winner (noise).

This is worth doing only together with live h2h collection: 1 Odds API credit/GW, or football-data.co.uk
`fixtures.csv` pre-match odds for free. Never ship KO without that collection, because it would be
mis-served exactly as today.

## 6. Patch sketch (REPO untouched; for the maintainer)

1. `scripts/train_predictor.py`, right after the synthetic-row drop, and the `/retrain` skill:
   ```python
   # Features that are 100% NaN at a 2026-27 deadline (train/serve skew; understat
   # per-match also leaked same-GW matches). See exp/feature_groups/RESULTS.md.
   LIVE_DEAD_FEATURES = [
       "xg_rolling_3", "xg_rolling_5", "xg_rolling_10", "xa_rolling_3", "xa_rolling_5",
       "xa_rolling_10", "npxg_rolling_5", "npxg_rolling_10", "shots_rolling_5",
       "key_passes_rolling_5", "xgchain_rolling_5", "xgchain_rolling_10", "xgbuildup_rolling_5",
       "goals_vs_xg_5", "assists_vs_xa_5",
       "odds_team_win_prob", "odds_team_draw_prob", "odds_team_loss_prob", "odds_team_strength",
       "prev_sot_per90", "prev_tkl_int_per90", "prev_gls_per90",
       "fpl_key_passes_rolling_5", "completed_passes_rolling_5",
       "big_chances_created_rolling_5", "dribbles_rolling_5",
   ]
   df = df.drop(columns=[c for c in LIVE_DEAD_FEATURES if c in df.columns])
   ```
   `PointPredictor` takes its features from the training columns and saves `feature_names.json`.
   `live/predict.py` then selects only those names, so the live pipeline needs no change. The ignored
   columns may stay in the frame. Better: put the list in `fpl_optimizer/prediction/feature_sets.py` and
   import it in `train_predictor.py` and in the experiment harness.
2. `features/understat.py::compute_understat_features`: `gw_date = pd.Timestamp(gw_dates[gw]).normalize()`.
   This is the leak fix, and it matters if understat is ever re-enabled. Optionally drop the understat
   per-match step from `FeaturePipeline._build_season` entirely: it is the slowest step (a per-player Python
   loop), and its output is no longer used.
3. Optional follow-up (KO): add live h2h odds collection to `gameweek.py` next to `collect_live_props`,
   writing `data/odds/2026-27.json` in the existing `{gw: [match...]}` schema. Then keep the 4 odds features,
   i.e. remove them from the drop list.
4. Optional pruning: `lowimp_live:40` is neutral. Drop the 40 lowest LIVE-model-gain features on top of the
   winner only if a leaner model is wanted.

## 7. Point-in-time and servability

- The winner only removes features. The remaining 96 columns and their construction are unchanged from
  `baseline_fixed`: shifted per-player rolling, pre-deadline `value`/`selected`, fixtures, and `fpl_xp_lag`.
  No new information enters the model, and one leak (understat same-day matches) leaves it.
- Every remaining input is served at a deadline by data the live pipeline already builds:
  - `data/raw/2026-27` from LiveFPLCollector (FPL API history, bootstrap and fixtures) feeds the vaastav
    rolling, transfers, ownership, price, opponent, fixture_meta, derived, synthetic EP and set pieces.
  - The pre-deadline snapshot `ep_this` feeds `fpl_xp_lag`.
  - `data/understat/league/2025-26.json` and `data/fotmob/2025-26.json` feed the prior-season features.
  - `collect_live_props` feeds props (NaN if not collected, as in 82% of training rows).
- Live coverage of every kept group is above 0 (table in section 1).

## Files

- `groups.py`: feature to group map, plus LIVE_DEAD.
- `run.py`: every training. Includes serve-sim, `--usfix`, logo, drop, lowimp and seeds.
- `analyze.py`: all tables.
- `understat_leak.py`: leak proof and the fixed rebuild.
- `understat_probe/collect_live_understat.py`: cheap 2026-27 collector (scratch).
- `reproduce_winner.sh`: reproduces the winner.
- `preds/*.parquet`: per-row predictions, same schema as the baseline plus `pred`.
- `results_{fast,full}.jsonl`, `tables_{fast,full}.md`, `analyze_{fast,full}.txt`: results and tables.
- `importance/*.csv`: per-position gain shares.
- `cache_understat_{leaky,fixed,fixed_all}.parquet`: understat feature rebuilds.
