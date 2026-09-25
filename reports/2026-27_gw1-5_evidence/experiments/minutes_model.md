# minutes_model: a point-in-time minutes model for the points predictor

Agent `minutes_model`. Everything is compared against the leak-fixed baselines in `EXP/baseline_fixed/`
(the `fpl_xp` feature is replaced by `fpl_xp_lag`). The primary metric is per-GW Spearman. Deltas are paired per GW
with bootstrap 95% CIs, from `harness.compare`.

## Verdict

**WINNER: `blend_fact_mh+stack`.** Each row's prediction is the average of two models:

* **factorized**: `P60·E[pts|60+] + P(1-59)·E[pts|1-59]`.
* **stacked**: the points model plus the minutes model's `P(play)` and `P(60+)` as features.

Results:

| mode | H25 Δspearman | H24 Δspearman | LIVE Δspearman |
|---|---|---|---|
| fast | **+0.0167** [+0.0145, +0.0187], 38-0 GWs | **+0.0166** [+0.0142, +0.0189], 38-0 | +0.0162, 5-0 |
| full | **+0.0147** [+0.0127, +0.0166], 38-0 | **+0.0142** [+0.0120, +0.0165], 38-0 | not run |

* **Other metrics:** Spearman on relevant rows improves by +0.03, MAE by −0.02 to −0.04 and hauler MAE by −0.02 to −0.07. These improvements hold in both modes and both holdouts.
* **Top-10:** noise-level: fast −0.03/+0.41, full −0.17/−0.04. All CIs include 0 except fast H24, which is +.
* **Captain (weakest point):** full −1.00/−0.82. Pooled over 76 GWs the full-mode delta is −0.91 [−1.95, 0.00]. That is inside the baseline's own seed noise. Re-running the unchanged baseline with seed 7 gives pooled cap −0.78 and cap_top3 −0.56 (CI excludes 0). The full baseline's H24 cap of 10.68, against 7.26 in fast mode, is an outlier driven by picking Salah 9 times in his 2024-25 season. Captaincy should stay human-overridable (`--captain`), as it is today.
* **Ensembling control:** part of any blend's gain is plain ensembling. Averaging the baseline with a second seed of itself gives +0.0018/+0.0019. The winner's gain is about 8x that. The single-model components alone are already large: factorized +0.0148/+0.0134 and stacked +0.0130/+0.0125 (fast).

## What was built

1. **Minutes labels.** Source is `REPO/data/raw/<season>/gws/merged_gw.csv` (read-only), with DGW rows aggregated per
   (season, element, GW): `min_total` (sum), `min_max` (max single fixture) and `mcls` = 0 / 1-59 / 60+ (from the total).
   The join is on (season, element, GW) with codes taken from the feature cache. 212,100/212,100 rows matched, and the
   summed `total_points` equals the cache `target` on 100% of rows.
2. **Minutes-history features** (`prep.py`, 28 columns, all shifted):
   * **Lags:** minutes lags 1-3.
   * **60+ shares:** share of the last 3/5/10 rows at 60+ minutes; share played (last 5); share as a sub (last 5);
     season-to-date 60+ share.
   * **Streaks and history:** zero-minute and full-game streaks; EWM of minutes/90; rows registered this season;
     in-season club move; starts share (2022-23 onward).
   * **Cross-season:** EWM and last-5 60+ share across seasons, plus previous-season share / 60+ rate / last-6 share /
     rows. These help GW1.
   * **Team depth chart:** rank of cross-season EWM minutes and of price within (GW, team, position), plus slot gap.
   * **Availability proxy:** `fpl_xp_lag == 0` and `fpl_xp_lag / (pts_rolling_3 + 0.5)`. FPL's ep_this is multiplied
     by chance_of_playing, and `fpl_xp_lag` is already point-in-time.
3. **Minutes model** (`mm.py`): one LightGBM multiclass P(0) / P(1-59) / P(60+).
   * **Features:** 35 minutes-relevant columns from the fixed feature set, plus the 28 features above, plus position.
   * **Rounds:** 223, early-stopped once with train ≤ 2022-23 and val = 2023-24. That split is point-in-time for both
     holdouts.
   * **Temporally nested predictions:** season S is predicted by a model trained only on seasons < S (2016-17 = NaN).
     The same file is therefore leak-free for every protocol. The test-season predictions come from exactly the model a
     live deployment would use.

   Quality:

| season predicted (model trained on all earlier seasons) | logloss (3-class) | AUC play | AUC 60+ | mean P(play) vs rate | with only the 35 base features: logloss / AUC play | naive `playing_prob` AUC |
|---|---|---|---|---|---|---|
| 2024-25 | 0.465 | 0.950 | 0.946 | 0.419 vs 0.425 | 0.471 / 0.949 | 0.899 |
| 2025-26 | 0.419 | 0.959 | 0.955 | 0.395 vs 0.387 | 0.429 / 0.957 | 0.909 |
| 2026-27 GW1-5 | 0.542 | 0.926 | 0.940 | 0.484 vs 0.478 | 0.556 / 0.922 | 0.805 |

The model is well calibrated. The new history features lower logloss by 1.4-2.4%. Top gain features: `mh_lag1` (36%),
`mh_zero_streak`, `mh_ewm`, `mins_rolling_3`, `xs_ewm`, `mh_sub_r5`, `selected_momentum`, `mh_full_streak`.

## Variants (FAST, vs `baseline_fixed/{P}_fast`)

| variant | proto | spearman | Δspearman [95% CI] | GWs won-lost | Δspearman_rel | Δtop10 [CI] | Δcap [CI] | ΔMAE | Δhauler_MAE |
|---|---|---|---|---|---|---|---|---|---|
| mh (features only) | H25 | 0.7278 | +0.0044 [+0.0024, +0.0065] | 28-10 | +0.0107 | +0.22 [-0.13, +0.58] | -1.16 [-2.63, +0.42] | -0.0120 | -0.0045 |
| mh | H24 | 0.7223 | +0.0060 [+0.0040, +0.0082] | 30-8 | +0.0119 | -0.13 [-0.58, +0.32] | -0.13 [-2.13, +2.08] | -0.0095 | -0.0272 |
| mh | LIVE | 0.6784 | +0.0060 [+0.0024, +0.0085] | 4-1 | +0.0053 | +0.00 | -0.40 | -0.0433 | +0.0666 |
| stack (base + mm_pplay, mm_p60) | H25 | 0.7364 | +0.0130 [+0.0109, +0.0151] | 36-2 | +0.0254 | +0.14 [-0.26, +0.54] | -0.97 [-2.29, +0.40] | -0.0283 | -0.0503 |
| stack | H24 | 0.7287 | +0.0125 [+0.0104, +0.0146] | 38-0 | +0.0237 | +0.21 [-0.16, +0.59] | -0.42 [-3.08, +2.42] | -0.0154 | -0.1001 |
| stack | LIVE | 0.6858 | +0.0133 [+0.0052, +0.0194] | 4-1 | +0.0134 | -0.58 | -0.60 | +0.0214 | -0.2080 |
| stack_mh | H25 | 0.7347 | +0.0113 [+0.0093, +0.0132] | 37-1 | +0.0220 | +0.20 [-0.18, +0.59] | -0.97 [-2.29, +0.26] | -0.0237 | -0.0769 |
| stack_mh | H24 | 0.7277 | +0.0115 [+0.0091, +0.0140] | 36-2 | +0.0225 | -0.36 [-0.76, +0.01] | -0.76 [-2.76, +1.29] | -0.0150 | -0.1015 |
| fact (conditionals on base feats) | H25 | 0.7377 | +0.0143 [+0.0117, +0.0168] | 37-1 | +0.0319 | -0.18 [-0.52, +0.17] | -0.79 [-2.08, +0.47] | -0.0323 | -0.0013 |
| fact | H24 | 0.7301 | +0.0139 [+0.0105, +0.0170] | 35-3 | +0.0324 | -0.14 [-0.57, +0.29] | -0.89 [-3.08, +1.18] | -0.0250 | +0.0156 |
| fact_mh (conditionals on base+mh) | H25 | 0.7382 | +0.0148 [+0.0119, +0.0175] | 36-2 | +0.0326 | +0.20 [-0.19, +0.59] | -1.79 [-3.63, +0.21] | -0.0365 | +0.0188 |
| fact_mh | H24 | 0.7297 | +0.0134 [+0.0099, +0.0167] | 34-4 | +0.0310 | +0.20 [-0.22, +0.61] | +0.50 [-1.87, +2.97] | -0.0274 | -0.0215 |
| fact_mh | LIVE | 0.6851 | +0.0126 [+0.0039, +0.0211] | 4-1 | +0.0130 | -0.38 | +1.20 | -0.0134 | +0.0672 |
| **blend_fact_mh+stack** | H25 | 0.7401 | **+0.0167 [+0.0145, +0.0187]** | **38-0** | +0.0341 | -0.03 [-0.40, +0.37] | -0.97 [-2.50, +0.68] | -0.0362 | -0.0203 |
| **blend_fact_mh+stack** | H24 | 0.7329 | **+0.0166 [+0.0142, +0.0189]** | **38-0** | +0.0332 | +0.41 [+0.01, +0.82] | +0.68 [-1.84, +3.26] | -0.0257 | -0.0675 |
| **blend_fact_mh+stack** | LIVE | 0.6886 | +0.0162 [+0.0086, +0.0230] | 5-0 | +0.0162 | -0.56 | +0.40 | -0.0035 | -0.0844 |
| blend_fact_mh+base (no stacking needed) | H25 | 0.7355 | +0.0120 [+0.0106, +0.0132] | 38-0 | +0.0253 | +0.04 | -0.29 | -0.0230 | +0.0052 |
| blend_fact_mh+base | H24 | 0.7290 | +0.0128 [+0.0112, +0.0143] | 38-0 | +0.0258 | +0.19 | +1.32 | -0.0207 | -0.0165 |
| control: base_s7 (baseline, seed 7) | H25 | 0.7230 | -0.0004 [-0.0016, +0.0008] | 18-20 | -0.0006 | +0.15 | **-2.08 [-3.55, -0.63]** | +0.0018 | -0.0209 |
| control: base_s7 | H24 | 0.7147 | -0.0015 [-0.0031, +0.0001] | 14-24 | -0.0019 | +0.12 | +0.53 | -0.0014 | +0.0139 |
| control: blend base+base_s7 | H25 | 0.7252 | +0.0018 [+0.0012, +0.0025] | 30-8 | +0.0045 | +0.08 | -0.24 | -0.0022 | -0.0151 |
| control: blend base+base_s7 | H24 | 0.7181 | +0.0019 [+0.0011, +0.0026] | 30-8 | +0.0051 | +0.08 | **+1.39 [+0.05, +3.08]** | -0.0051 | -0.0001 |

On noise: a seed change alone moves `cap` by −2.08 (H25) and a two-seed average moves it by +1.39 (H24), both with
CIs that exclude 0. The per-protocol cap CI does not capture model variance, so it cannot be used as a gate on its own.

### FULL mode (vs `baseline_fixed/{P}_full`)

| variant | proto | spearman | Δspearman [95% CI] | GWs | Δspearman_rel | Δtop10 [CI] | Δcap [CI] | ΔMAE | Δhauler_MAE |
|---|---|---|---|---|---|---|---|---|---|
| stack | H25 | 0.7379 | +0.0120 [+0.0100, +0.0141] | 38-0 | +0.0239 | -0.24 [-0.49, +0.01] | -1.05 [-2.61, +0.68] | -0.0270 | -0.0591 |
| stack | H24 | 0.7315 | +0.0126 [+0.0103, +0.0149] | 38-0 | +0.0235 | +0.01 [-0.29, +0.32] | -1.66 [-3.82, +0.47] | -0.0128 | -0.0809 |
| fact_mh | H25 | 0.7395 | +0.0136 [+0.0111, +0.0162] | 37-1 | +0.0301 | -0.24 [-0.52, +0.01] | -1.11 [-2.32, -0.03] | -0.0358 | +0.0147 |
| fact_mh | H24 | 0.7302 | +0.0113 [+0.0081, +0.0142] | 35-3 | +0.0268 | +0.12 [-0.25, +0.47] | -3.61 [-5.39, -2.03] | -0.0216 | -0.0010 |
| **blend_fact_mh+stack** | H25 | 0.7406 | **+0.0147 [+0.0127, +0.0166]** | **38-0** | +0.0297 | -0.17 [-0.41, +0.05] | -1.00 [-2.13, -0.05] | -0.0333 | -0.0243 |
| **blend_fact_mh+stack** | H24 | 0.7331 | **+0.0142 [+0.0120, +0.0165]** | **38-0** | +0.0281 | -0.04 [-0.33, +0.26] | -0.82 [-2.47, +0.76] | -0.0198 | -0.0441 |

Full-mode baseline spearman: H25 0.7258, H24 0.7189. Full-mode baseline cap: H25 7.61, H24 **10.68**. In fast mode the
H24 baseline cap is 7.26, and the seed-7 baseline gets 7.79.

### Captain / top-k, pooled over 76 GWs (H25+H24): delta, 95% CI, GWs won-lost (`captain.py`)

| variant, mode | cap | cap_top3 (mean of 3 best relevant) | rel_top5 | top10 | top20 |
|---|---|---|---|---|---|
| control base_s7, fast | -0.78 [-1.92, +0.51] | **-0.56 [-1.12, -0.03]** | +0.03 | +0.14 | -0.05 |
| blend_fact_mh+stack, fast | -0.15 [-1.58, +1.41] | -0.19 [-0.81, +0.44] | +0.21 | +0.19 | +0.13 |
| blend_fact_mh+stack, full | -0.91 [-1.95, 0.00] 5-15 | -0.29 [-0.85, +0.24] | -0.30 [-0.64, +0.05] | -0.11 | +0.10 |
| fact_mh, full | **-2.36 [-3.51, -1.34] 6-26** | -0.48 | -0.45 [-0.85, -0.07] | -0.06 | +0.06 |
| stack, full | -1.36 [-2.69, 0.00] | -0.68 [-1.22, -0.15] | -0.26 | -0.12 | +0.05 |
| (exploratory, post hoc) blend3 fact_mh+stack+base, full | -0.42 [-1.28, +0.45] | -0.00 | -0.14 | -0.06 | +0.11 (CI>0) |

The captain changes come almost entirely from the choice between Haaland (223094) and Salah (118748). In full mode the
baseline captains Salah 9 of 38 times in 2024-25 (cap 10.68), while fact_mh picks Haaland 10 times and Salah 7 (cap 7.08).
Neither component should be shipped alone as a captain picker. The 2-way blend is the most captain-neutral of the
single-structure options. A 3-way blend with the old model gives up about 0.002 Spearman for a neutral captain (post hoc,
not claimed as a win).

### Where the gain comes from

* **By GW:**
  * **Early/late split** (fast Δspearman, H25 | H24):
    * GW1-8: stack +0.015 | +0.011; fact_mh +0.007 | +0.001; blend +0.015 | +0.012.
    * GW9-38: stack +0.012 | +0.013; fact_mh +0.017 | +0.017; blend +0.017 | +0.018.
  * Factorization is weak before in-season minutes history exists. Stacking holds up in early GWs. The blend gets both.
* **By position:** the blend gains within every outfield position (DEF +0.013 to +0.019, MID +0.018 to +0.021,
  FWD +0.018 to +0.024). Factorization hurts **GK** within-position ranking (−0.03 / −0.01); the blend softens this
  (−0.027 / −0.004). A follow-up could use the stack component alone for GK. It was not tested, to avoid post-hoc
  selection.
* **Among players who scored non-zero:** the blend still gains +0.014 / +0.013. The improvement is not only "separating
  non-players".
* **Features vs structure:** adding the history features directly (`mh`) gives only +0.004 to +0.006. Most of the gain
  comes from the dedicated classifier and the factorized/stacked structure. Adding MH features next to the stacked
  probabilities slightly *hurts* (stack_mh < stack).

## Point-in-time proof

* **Features** (`python prep.py test`): all minutes/starts/points labels from (season S, GW ≥ g0) onward, and every
  later season, are replaced with random values, then all 28 features are rebuilt. Features for every row with GW ≤ g0
  (including the cut GW itself, whose own labels were scrambled) must be identical.
  * Result: **0 violations** across 6 cuts: 2024-25 GW1, 2024-25 GW12, 2025-26 GW25, 2023-24 GW38, 2026-27 GW3,
    2019-20 GW30. The check covered 48k-211k rows per cut, while 1.3k-26k later rows did change.
  * The inputs `fpl_xp_lag` and `pts_rolling_3` come from the already point-in-time cache.
* **Minutes model / stacking:** season S's predictions come from a booster trained only on seasons < S. The fixed
  round count was chosen with data ≤ 2023-24. So:
  * no training row of the points model sees its own label or any label from its own season or later;
  * the test-season probabilities are what a deployment trained on all completed seasons would output.
* **Factorized conditionals:** these are trained on rows selected by the realized minutes class. That is only the
  training-target structure. At test time they are weighted by *predicted* class probabilities and never by realized
  minutes.

## Live servability

| input | live source at the GW-n deadline |
|---|---|
| mh_* (lags, streaks, shares, EWM, rows, team change) | `data/raw/2026-27/gws/merged_gw.csv` real rows GW < n (rebuilt weekly from element-summary history; `minutes`, `starts` present). The synthetic GW-n row contributes only `team`; its zeroed stats are never read (shift). |
| xs_*, ps_* | previous seasons' `data/raw/<season>/gws/merged_gw.csv` (on disk through 2025-26) + current rows |
| dp_* | team (bootstrap, synthetic row), position (`cleaned_players.csv`), price `value` (synthetic row = now_cost) |
| av_* | `fpl_xp_lag` = pre-deadline snapshot `ep_this` (harness fix), `pts_rolling_3` (vaastav) |
| MM_BASE features | already served by the current pipeline (understat-based `prev_minutes` is the prior season → available) |

The whole LIVE protocol (2026-27 GW1-5) runs on data built exactly this way and gives +0.0162 Spearman, 5-0 GWs. None of
the new inputs uses understat, which is NaN live.

## Availability at serve time (pool.py): keep the multiplicative scaling, no double count

`live/pool.py` multiplies each prediction by `chance_of_playing_next_round/100` and excludes or zeroes status
i/s/u/n. The concern was that a minutes model that already reacts to injuries (through lagged zeros and the
ep-based `av_*` proxy) would be penalized twice.

This was tested on LIVE 2026-27 using the pre-deadline snapshots (`avail.py`). Of the 5 GWs, 75 player-GWs were flagged
at 25, 50 or 75%:

| rule for P(play) on flagged rows | mean | Brier vs actual play (rate 0.267) |
|---|---|---|
| model P(play) alone (no scaling) | 0.423 | 0.278 |
| chance c alone | 0.650 | 0.378 |
| **P(play) × c (current pool.py)** | **0.286** | **0.226** |
| min(P(play), c) (cap rule) | 0.370 | 0.251 |

The model's P(play) and the FPL flag are close to independent, and the product is the best calibrated. 75%-flagged
players played only 24.5% of the time (53 rows). On whole-pool ranking, `mult` beats `cap` and `raw` for every model.
Blend: raw 0.7136, mult **0.7165**, cap 0.7154 per-GW Spearman. Unflagged status-'a' rows are calibrated (P 0.601 vs
0.618 actual).

**Serving recommendation:** keep `availability_scaling=True`. It multiplies the final E[pts]. For the factorized part
this equals scaling P60 and P(1-59) by c. Do not pass a separate `chance_of_playing` into the model. This matters only
for the 25-75% flags, since a flag of 0 zeroes the player under any rule.

## Implementation (patch sketch — tested code in `repo_patch/`)

1. **New `src/fpl_optimizer/prediction/features/minutes_history.py`** = `repo_patch/minutes_history.py`.
   `verify_patch.py` drives it the way `FeaturePipeline._build_season` would. It reproduces `mh.parquet` with **0
   mismatches** on 2018-19, 2021-22, 2024-25 and 2026-27.
2. **`feature_pipeline.py::_build_season`:**
   * After step 2 (code mapping):
     `mh_df = compute_minutes_history_features(merged_gw, self.data_dir, season, self.id_resolver)`.
   * After the position/target merges: `result = result.merge(mh_df, on=["element","GW"], how="left")`, then
     `result = add_depth_and_availability(result)`.
   * `av_*` expect `fpl_xp_lag`, i.e. the xP leak fix implemented in `players_raw.py`. Without it they are NaN
     (harmless).
   * This also emits the labels `min_total`, `min_max`, `mcls`.
3. **`model.py`:** `_NON_FEATURE_COLS |= {"min_total", "min_max", "mcls"}`. This is required; otherwise the labels
   become features. `minutes.py` asserts it at import.
4. **New `src/fpl_optimizer/prediction/minutes.py`** = `repo_patch/minutes.py`. It provides `MinutesModel`, which is a
   multiclass model with `MM_BASE`, the MH features and `pos_code` at 223 rounds. It also provides:
   * `expanding_minutes_oof`;
   * `MinutesBlendPredictor`, with the same `train(train_df, val_df)` / `predict(df)` / `save` / `load` interface as
     `PointPredictor`, plus `predict_components` (p_play, p60, e60, esub);
   * `load_predictor(model_dir)`, which dispatches on `metadata.json["kind"]`.

   `smoke_patch.py` trains it with tiny params and checks that the save/load round trip is exact (max abs diff 0.0).
   It also checks that the stack sees `mm_*` but no MH columns, and that no labels are used as features.
5. **`scripts/train_predictor.py`:** replace both `PointPredictor(params=PARAMS, ...)` calls (EVAL and PROD) with
   `MinutesBlendPredictor(params=PARAMS, early_stopping_rounds=50)`. Nothing else changes. The labels come out of the
   pipeline. Training runs 1 minutes model, 10 expanding minutes models (about 1-3 min each), 2 conditional
   PointPredictors on row subsets and 1 stacked PointPredictor, about 3x today's time.
6. **`live/predict.py`:** `predictor = load_predictor(model_dir)` instead of `PointPredictor.load(model_dir)`. Nothing
   else changes; DGW rows are still summed per element.
7. **`live/pool.py`:** no change (see the availability section). `scripts/gameweek.py`: no change.
   `predict_components` makes P(play) available for future bench-order and rotation-risk logic.

## Not done / caveats

* **LOSO stacking not run:** `python mm.py loso H25|H24` exists but was not run because of machine load. The expanding
  scheme is strictly point-in-time and already wins 38-0.
* **Not tested:** `fact_stack` (conditionals that also take the stacked probabilities) and IS25/IS24.
* **Machine load:** the minutes model uses a 35-column subset of the base features to keep the 10 expanding fits cheap
  on an overloaded machine (16 GB RAM with about 30 agent processes). A full-feature minutes model (tried first,
  aborted as too slow) had similar 2017-19 fold quality.
* **Captain:** see above. Treat the model's captain as advisory.

## Files (all under this directory)

* **Pipeline scripts:**
  * `prep.py` (labels and features, plus the truncation test);
  * `mm.py` (minutes models) and `mm_expanding.parquet` (nested P0/Psub/P60 for every row);
  * `mm_report.json`, `mm_full_live.lgb`;
  * `run.py` (variants), `post.py` (fill/blend/table), `captain.py`, `avail.py`, `mdtable.py`, `reproduce.sh`.
* **Outputs:**
  * `mh.parquet`;
  * `preds/{PROTO}_{mode}_{variant}.parquet`, per-row with the baseline schema plus `pred` (fact variants also carry
    `e60`, `esub`, `mm_p60`, `mm_psub`);
  * `preds/*.json`, with the summary and the paired comparison;
  * `avail_report.json`.
* **Repo patch:** `repo_patch/minutes_history.py`, `repo_patch/minutes.py`, `verify_patch.py`, `smoke_patch.py`.
