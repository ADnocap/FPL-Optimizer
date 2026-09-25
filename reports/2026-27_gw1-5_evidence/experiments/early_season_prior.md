# early_season_prior: prior-season FPL features against early-season over-reaction

**Verdict: NO WINNER.** None of the 12 variants improves per-GW Spearman over GW1-8 on either
holdout. The two families that stay non-negative overall (`shrink`, `career_roll`, both checked
with 3 seeds against 3 baseline seeds) gain only +0.0003 to +0.0006, with CIs that include 0. Those
small gains come from GW9-38, not the early season. Full mode was not run because nothing passed
the fast screen.

All numbers are FAST mode against `EXP/baseline_fixed/{H25,H24}_fast.parquet` (the leak-fixed set).
Format: `delta [bootstrap 95% CI] GWs won/lost`.

## 1. What was built (all point-in-time, all servable live)

| family | features | source / as-of logic |
|---|---|---|
| A. prior-season FPL aggregates (`pv_*`, `pv2_*`) | pts per row / per GW / per appearance / per 90, mins share, mins per row, 60-min rate, start rate, bonus/bps/ict/xGI/GI per 90, CS rate (60+ min), last-10-GW pts & mins, first/end price, price change vs last season's end/start, S-2 pts-per-GW / mins share / ppg | season S-1 (and S-2) `merged_gw.csv`, complete before GW1 of S; linked by stable `code` (`players_raw.csv` id->code, identical to the cache's IDResolver mapping on 100% of rows). The current row's own price is the pre-deadline price. |
| B. career rolling (`cr_*`) | pts r3/5/10/20, mins r3/5/10, ict r5/10, bps r5/10, bonus r10, cs r10 over the concatenated cross-season per-GW sequence | `groupby(code).shift(1).rolling(w)`. Verified: at GW1 of 2025-26, `cr_pts_r5` equals the mean of the player's last 5 cached 2024-25 targets (512/512), and from GW10 on it equals `pts_rolling_5` (99.3%). |
| C. shrinkage (`sh_*`) | `(sum_pts_this_season_before_GW + m*prior)/(rows_before + m)` for m=3,6,12, plus a mins version and a last-10-anchored version; the fallback prior is the S-1 median | cumulative sums strictly before the row, per (code, season) |
| D. context (`ctx_*`, `pv_new_to_pl`) | new-to-PL, no previous minutes, promoted team (team_code not in S-1), changed club, rows this season | team id from the row's fixture (`was_home` + `opponent_team`), then mapped to team_code via players_raw. Correct promoted sets: 2024-25 Ipswich/Leicester/Southampton, 2025-26 Burnley/Leeds/Sunderland, 2026-27 Coventry/Hull/Ipswich. |
| E. pseudo-history (`ph_*`, build_features2.py) | career windows. A player with NO earlier PL season gets the missing part of each window filled with the per-row mean that newly arrived players of the same position x start-price band scored in their first 10 rows of S-1 | S-1 only. Example, 2026-27 Hull DEF Mendy (GBP4.0m): GW2 `ph_pts_r3` = (15 + 2*0.83)/3 = 5.55 vs a raw `pts_rolling_3` of 15. |

Servability: every input comes from `data/raw/2025-26/{gws/merged_gw.csv,players_raw.csv}` (on disk)
and the weekly-rebuilt `data/raw/2026-27/`. The synthetic upcoming-GW rows carry `fixture`,
`was_home`, `opponent_team` and `value`, which is all the current-row logic needs. Their zeroed
stats are never read because of the shift. Coverage on the 2026-27 rows: pv_* 74%, cr_* 96%, ph_* 100%.

## 2. Screen results (fast, single seed 42, vs seed-42 baseline)

### H25 (test 2025-26), base per-GW Spearman 0.7234
| variant | d_spearman | d_sp GW1-8 | d_sp GW9-38 | d_top10 | d_cap | d_top10 GW1-10 | d_cap GW1-10 | d_mae |
|---|---|---|---|---|---|---|---|---|
| prior_all | -0.0006 [-0.0018,+0.0006] 17/21 | -0.0006 5/3 | -0.0006 | +0.163 | -0.658 | -0.040 | -2.000 | +0.0036 |
| prior_core | -0.0008 [-0.0019,+0.0002] 16/22 | -0.0028 1/7 | -0.0003 | -0.013 | -0.921 | -0.050 | -0.700 | +0.0061 |
| career_roll | +0.0002 [-0.0012,+0.0016] 22/16 | -0.0014 4/4 | +0.0006 | -0.042 | +0.079 | -0.380 | -0.200 | -0.0032 |
| shrink | -0.0001 [-0.0015,+0.0014] 20/18 | +0.0007 5/3 | -0.0003 | -0.137 | +0.000 | -0.120 | +0.200 | -0.0017 |
| career_replace | -0.0010 [-0.0020,-0.0000] 14/24 | -0.0004 5/3 | -0.0011 | -0.163 | -0.632 | -0.530 | +0.100 | +0.0013 |
| ph_replace | +0.0011 [-0.0000,+0.0022] 21/17 | -0.0014 2/6 | +0.0018 | -0.121 | -0.158 | +0.040 | +1.000 | -0.0059 |
| ph_replace_ep | -0.0013 [-0.0025,+0.0000] 14/24 | -0.0052 0/8 | -0.0002 | +0.032 | -0.289 | -0.110 | -0.300 | -0.0009 |
| indicators | -0.0003 [-0.0016,+0.0009] 18/20 | +0.0000 3/5 | -0.0004 | +0.103 | -1.026 | -0.290 | -0.100 | +0.0006 |
| spec_base_k10u8 | -0.0037 [-0.0067,-0.0013] 0/8 | -0.0177 0/8 | 0 | -0.021 | -0.658 | -0.080 | -2.500 | +0.0101 |
| spec_prior_k10u8 | -0.0033 [-0.0062,-0.0010] 1/7 | -0.0156 1/7 | 0 | -0.053 | -1.289 | -0.200 | -4.900 | +0.0108 |

### H24 (test 2024-25), base per-GW Spearman 0.7162
| variant | d_spearman | d_sp GW1-8 | d_sp GW9-38 | d_top10 | d_cap | d_top10 GW1-10 | d_cap GW1-10 | d_mae |
|---|---|---|---|---|---|---|---|---|
| prior_all | -0.0020 [-0.0037,-0.0005] 16/22 | -0.0034 3/5 | -0.0016 | -0.203 | -0.895 | -0.470 | -0.400 | +0.0070 |
| prior_core | -0.0025 [-0.0043,-0.0008] 13/25 | -0.0042 3/5 | -0.0020 | +0.116 | +0.316 | -0.330 | +0.100 | +0.0046 |
| career_roll | -0.0009 [-0.0030,+0.0009] 19/19 | -0.0057 3/5 | +0.0004 | +0.082 | +0.421 | +0.110 | +0.400 | -0.0007 |
| shrink | +0.0005 [-0.0014,+0.0023] 20/18 | -0.0040 3/5 | +0.0018 | -0.021 | +0.447 | -0.170 | +0.900 | -0.0068 |
| career_replace | -0.0009 [-0.0030,+0.0011] 16/22 | -0.0030 3/5 | -0.0003 | +0.124 | +0.447 | -0.320 | +1.800 | +0.0001 |
| ph_replace | -0.0028 [-0.0046,-0.0012] 12/26 | -0.0069 2/6 | -0.0017 | -0.103 | +0.237 | -0.310 | +1.100 | +0.0063 |
| ph_replace_ep | -0.0025 [-0.0053,-0.0002] 14/24 | -0.0100 1/7 | -0.0006 | +0.074 | +1.553 | -0.100 | +0.600 | +0.0060 |
| indicators | -0.0019 [-0.0033,-0.0004] 14/24 | -0.0019 2/6 | -0.0019 | -0.126 | +1.368 | -0.490 | +4.100 | +0.0008 |
| spec_base_k10u8 | -0.0027 [-0.0049,-0.0009] 0/8 | -0.0127 0/8 | 0 | +0.003 | +0.658 | +0.010 | +2.500 | +0.0092 |
| spec_prior_k10u8 | -0.0039 [-0.0065,-0.0016] 0/8 | -0.0184 0/8 | 0 | -0.203 | -0.237 | -0.770 | -0.900 | +0.0158 |

Variant definitions (esp_common.py): prior_all = A + D. prior_core = 12 core A/D features.
career_roll = B + rows_this_season. shrink = C + new_to_pl. career_replace / ph_replace =
**overwrite** the 11 within-season pts/mins/ict/bps/bonus/cs rolling features with B / E (derived
deltas recomputed). ph_replace_ep = ph_replace + synthetic_ep/playing_prob recomputed on the
replaced windows. indicators = D only. spec_* = a separate booster set trained only on GW<=10 rows
(val = the val season's GW<=10) that serves GW<=8, with the baseline for GW>8.

## 3. Seed noise: the single-seed screen is biased against every variant

The baseline alone, retrained with seed 43 or 44 (same features):

| | H25 d_sp | H25 GW1-8 | H24 d_sp | H24 GW1-8 |
|---|---|---|---|---|
| base_seed43 | -0.0017 [-0.0028,-0.0005] 13/25 | -0.0017 2/6 | -0.0021 [-0.0036,-0.0006] 9/29 | -0.0048 1/7 |
| base_seed44 | -0.0014 [-0.0028,+0.0001] 14/24 | +0.0004 5/3 | -0.0024 [-0.0037,-0.0010] 12/26 | -0.0047 1/7 |

The cached seed-42 fast baseline is a favourable draw, especially for H24 GW1-8. A seed change
alone produces "significant" deltas of about 0.002. Any new feature re-randomises the
feature_fraction/bagging draws, so single-seed deltas of +/-0.002 are not evidence. I therefore
reran the two least-bad variants with 3 seeds and compared the per-GW metrics averaged over seeds
against the baseline averaged over seeds 42/43/44 (seed_avg.py):

| variant (3v3 seeds) | H25 d_sp | H25 GW1-8 | H24 d_sp | H24 GW1-8 | H25 top10 / cap | H24 top10 / cap | GW1-10 top10 H25 / H24 | MAE H25 / H24 |
|---|---|---|---|---|---|---|---|---|
| shrink | +0.0006 [-0.0008,+0.0018] 24/14 | -0.0010 4/4 | +0.0006 [-0.0012,+0.0020] 25/13 | -0.0022 5/3 | +0.09 / +0.38 | -0.05 / -0.47 | +0.08 / -0.24 | -0.0042 / -0.0046 (CI<0) |
| career_roll | +0.0003 [-0.0005,+0.0011] 22/16 | -0.0016 2/6 | +0.0003 [-0.0011,+0.0016] 23/15 | -0.0009 4/4 | +0.20 / -0.25 | +0.02 / +0.52 | -0.04 / -0.24 | -0.0025 / -0.0020 |

Neither passes the winner criteria: the pooled CIs include 0, H25 shrink wins 63% of GWs, and the
early-season deltas are negative on both holdouts. The single-seed variants measured against the
3-seed baseline mean are in `logs/seedavg_final.txt`. The best-looking one, ph_replace (H25 +0.0022,
27/11), flips to -0.0013 on H24, so it is not a winner.

## 4. Why prior-season information does not fix the early season (diagnostics.py)

1. **One game of current-season form carries more information than a whole prior season.** Univariate
   per-GW Spearman with the target at GW2: `pts_rolling_3` (a single game) 0.688 (H25) and 0.740
   (H24). The best prior-season aggregate (`pv_pts_per_row`, `pv_last10_mins`) reaches 0.52-0.60.
   All-player Spearman is dominated by who is playing this season, and the first match reveals
   that. At GW1, the prior aggregates (0.53-0.56) already beat value (0.40-0.44) and ownership
   (0.47-0.49). Even so, the model's GW1 score (0.62-0.63) does not improve when they are added:
   GW1 is where most variants lose most (H24 -0.017 to -0.035; H25 up to -0.012), and the seed
   replicas show GW1 is also the noisiest GW (seed-only shifts of -0.010 to +0.005).
2. **There is no systematic early-season over-reaction in the holdouts.** Per-GW Spearman at
   GW2-8 (0.72-0.77) matches GW9-38. For hot starters (`pts_rolling_3 >= 6`), baseline bias at
   GW2-4 is -0.53 (H25) and +0.01 (H24) for players with PL history, i.e. no over-prediction. The
   only over-predicted pocket is **new-to-PL hot starters at GW2-4**: bias +0.97 (n=28, H25) and
   +1.15 (n=7, H24). That group is too small to move any metric. New-to-PL players picked into the
   early top-10 actually scored more than the rest in H25 (6.86 vs 5.59 over 14 picks).
3. **The live 2026-27 GW2 blow-up is a GW-wide level shift, not a missing prior.** LIVE GW2 mean
   prediction is 2.09 against an actual 1.42 across all players. GW1/3/4/5 are within about 0.1.
   The Hull defenders (Mendy and Egan, both GBP4.0m, predicted about 8.8-8.9) are the visible tip.
   Prior features damp them: prior_all 6.3-6.9, ph_replace_ep 4.5-5.4. They do not remove the GW2
   level shift (prior_all 1.91, shrink 2.03, ph_replace_ep 2.09). LIVE deltas are 5 GWs and only
   a sanity check: prior_all +0.0045 (4/1), shrink +0.0035 (4/1), ph_replace_ep -0.0091 (2/3).

## 5. Side findings for the orchestrator (outside this agent's scope)

- **Odds are 0% populated live.** `odds_team_*` covers 92-100% of rows in every training season
  and 0% of 2026-27. There is no `data/odds/2026-27.json`, and `gameweek.py` collects props but not
  h2h odds. odds_gap_check.py trained the H25 fast model and predicted 2025-26 with and without the
  odds columns set to NaN. Ranking is unaffected (d_spearman -0.00002), but every prediction goes
  up: bias +0.067 pts per row, MAE +0.021, top-20 inflated. This explains part, not all, of the
  LIVE upward bias. FBref `prev_*` (about 45-65% in training) and understat rolling features
  (about 60-80%) are also 0% live.
- **Seed noise is about 0.002 pooled per-GW Spearman.** The fast baselines (seed 42) sit at the
  lucky end, especially H24 GW1-8. Other agents' single-seed "winners" with deltas at or below
  0.002 should be rechecked with seed averaging. Replica preds are at
  `preds/base_seed43_*` and `preds/base_seed44_*` (same schema as the baseline).

## 6. Implementation

There is no winner, so no repo change is recommended. If a later full retrain wants the small but
consistent MAE gain from `shrink` (-0.004 on both holdouts, 3 seeds), the patch would go in a new
`features/fpl_prior.py`, called from `FeaturePipeline._build_season`:

```python
# load S-1 merged_gw via _read_csv, aggregate per (element, GW), map element->code with
# id_resolver.code_from_element_id(prev_season, eid); per code: pts_per_row = pts.sum()/rows,
# mins_per_row; then per current-season (element, GW) row, sorted by GW:
n  = groupby(element).cumcount();  s = groupby(element).total_points.cumsum() - total_points
sh_pts_m{m} = (s + m*prior_pts_per_row.fillna(S-1 median)) / (n + m)   # m in 3,6,12
```

It is not recommended on current evidence.

## Files
- build_features.py -> prior_features.parquet (families A-D); build_features2.py -> prior_features2.parquet (E)
- esp_common.py (variant definitions, build_df), run_batch.py, early_specialist.py, odds_gap_check.py
- analyze.py -> results_fast.{json,csv}; pergw.py; seed_avg.py -> results_seedavg.json; diagnostics.py
- preds/*.parquet (per-row predictions, baseline schema + pred); logs/*.txt (final tables)
