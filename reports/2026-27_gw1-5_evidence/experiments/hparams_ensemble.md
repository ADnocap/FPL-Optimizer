# hparams_ensemble: LightGBM tuning, seed bagging and blending on the leak-fixed feature set

Agent: `hparams_ensemble`. Feature set: exactly `harness.load_features()` (the leak-fixed `fpl_xp_lag`).
No features were added or changed, so this experiment adds nothing new that has to be served live.
Baselines: `baseline_fixed/{H24,H25,LIVE}_{fast,full}.parquet`, which is the recipe params with seed 42.

## TL;DR

* **Winner: switch the objective to `huber` (alpha 0.9, the LightGBM default) and raise
  `min_child_samples` from 10 to 50. Then apply a monotone quadratic calibration fitted on
  the early-stopping val split.** Every other recipe parameter stays the same (num_leaves 127,
  ff 0.8, lambda_l2 1).
  * Per-GW Spearman goes up in **every protocol and mode**: H24 +0.0148 (38/38 GWs), H25 +0.0129 (38/38),
    H23 (an extra season never used for anything) +0.0170 (37/38), LIVE +0.022 (5/5), full-mode H24 +0.0127
    (37/38), full-mode H25 +0.0116 (38/38). All bootstrap CIs exclude 0.
  * spearman_rel, the MILP-relevant pool, goes up +0.023 to +0.033 everywhere.
  * top10 and cap stay within noise once you account for the baseline's seed luck (section 6).
    Pooled over H23+H24+H25 fast (114 GWs), the 5-seed bag gives top10 +0.06 [-0.17,+0.28] and cap +0.40 [-0.62,+1.45].
  * MAE after calibration is -0.01 to -0.06 on every holdout (LIVE -0.11). Raw huber MAE is -0.09 to -0.14, but raw huber predictions are too low to feed the MILP (see below).
* Almost the whole gain comes from the **loss**. The tree-structure grid spans only 0.7281 to 0.7310 on H24, which is 2 to 5x the
  seed noise (sd 0.0006). L1 and L2 do not help. Early-stopping on L2 instead of MAE does not help.
* **Huber shrinks predictions toward the median.** On H24 the raw top-500 predictions average 3.8 against an actual 5.9, and bias is -0.4.
  For the MILP (hits cost 4, bench EV) the scale has to be restored. **The quadratic calibration**
  `y = c0 + c1*p + c2*p^2` is fitted on the val split. It is strictly increasing, so every ranking metric is unchanged. It is well calibrated
  in the per-GW top 20: pred/actual is 5.34/5.33 on H24, 5.11/4.92 on H25, 5.15/5.45 on H23 and 5.16/4.89 on full H25.
  That beats the L2 baseline, which over-predicts its top 5 (H25 7.02/5.94, H23 7.72/6.28).
  Isotonic calibration is **unusable**: its clipped top step ties about 15 of each GW's top 20.
* **Seed bagging** (5 seeds) is neutral: +0.001 Spearman in fast mode and +0.0001 to 0.0003 in full mode, at 3 to 5x the training cost.
* **Blend** (0.5 x calibrated huber + 0.5 x current L2) is the conservative alternative. It keeps about 70% of the
  Spearman gain (+0.009 to +0.012, 38/38 in every protocol) and has the best top10 in all 5 evaluations
  (pooled fast top10 +0.25 [+0.09,+0.41]). It costs two model sets. w=0.5 is the H24 argmax of both top10 and cap.
  I noticed it first on H25, and it was then confirmed on the untouched H23 (section 7).

## 1. Protocol (clean)

1. **Tune on H24 only**: train 2016-17..2023-24, early-stop on the last 8 GWs of 2023-24, test 2024-25.
   Coordinate descent, FAST params, seed 42. Selection = mean per-GW Spearman (harness.per_gw).
   22 tuning fits plus 4 seed-bag fits, 26 H24 fits in total:
   objective (3) -> num_leaves x min_child_samples (11 new) -> feature_fraction x lambda_l2 (3) -> objective
   re-check at the new structure (2) -> huber alpha (2) -> early-stopping-metric diagnostic (1); then 4 more seeds for the bag.
2. **Confirm on H25 with no further tuning**: the chosen config, seed 42, plus a 5-seed bag.
3. Extra checks run **after** selection: H23 (train <= 2022-23, test 2023-24; 2023-24 was never looked at before this),
   LIVE (sanity check only, 5 GWs), and full mode on H24 and H25 against `baseline_fixed/*_full`.
4. Reproduction check: `o=l2-nl=127-mcs=10-ff=0.8-l2=1-s=42` reproduces `baseline_fixed/H24_fast.parquet` bit-for-bit
   (sp 0.7162 / top10 5.905 / cap 7.263 / mae 0.9965).

## 2. H24 tuning (fast, seed 42): all fits vs baseline_fixed/H24_fast

Per-GW Spearman, huber, ff=0.8, lambda_l2=1:

| num_leaves \ min_child_samples | 10 | 50 | 200 |
|---|---|---|---|
| 31 | 0.7292 | 0.7296 | 0.7281 |
| 63 | 0.7301 | 0.7302 | 0.7292 |
| 127 | 0.7295 | **0.7310** | 0.7287 |
| 255 | 0.7294 | 0.7300 | 0.7287 |

Same grid, cap (mcs=50 is the best column for every num_leaves): 31: 7.82/9.00/9.68, 63: 8.32/9.47/9.61, 127: 7.26/9.95/9.74, 255: 8.00/9.50/8.18.

| config (H24 fast) | spearman | d_sp [95% CI] | GWs won/lost | d_sp_rel | top10 (d) | cap (d) | d_mae (raw) |
|---|---|---|---|---|---|---|---|
| **huber nl127 mcs50 ff0.8 l2=1** (chosen) | 0.7310 | +0.0148 [+0.0128,+0.0167] | 38/0 | +0.0279 | 6.055 (+0.150) | 9.947 (+2.68) | -0.121 |
| huber ... alpha 0.5 | 0.7309 | +0.0146 | 38/0 | +0.0274 | 6.008 (+0.103) | 8.105 (+0.84) | -0.131 |
| huber ... ff 0.5 | 0.7305 | +0.0143 | 38/0 | +0.0270 | 5.884 (-0.021) | 8.395 (+1.13) | -0.119 |
| huber ... lambda_l2 10 | 0.7304 | +0.0142 | 38/0 | +0.0276 | 5.939 (+0.034) | 8.921 (+1.66) | -0.121 |
| huber ... ff 0.5 + l2 10 | 0.7300 | +0.0138 | 38/0 | +0.0259 | 5.858 (-0.047) | 8.789 (+1.53) | -0.118 |
| huber nl127 mcs10 (recipe structure) | 0.7295 | +0.0132 [+0.0113,+0.0152] | 38/0 | +0.0256 | 5.961 (+0.055) | 7.263 (0.00) | -0.120 |
| huber ... alpha 2.0 | 0.7255 | +0.0093 | 36/2 | +0.0201 | 6.139 (+0.234) | 8.316 (+1.05) | -0.093 |
| l1 nl127 mcs50 | 0.7193 | +0.0031 [-0.0004,+0.0064] | 22/16 | +0.0190 | 5.889 (-0.016) | 9.605 (+2.34) | -0.145 |
| l1 nl127 mcs10 | 0.7172 | +0.0010 [-0.0021,+0.0041] | 23/15 | +0.0168 | 5.763 (-0.142) | 9.789 (+2.53) | -0.142 |
| l2 (= baseline) | 0.7162 | 0 | - | 0 | 5.905 | 7.263 | 0 |
| l2 nl127 mcs50 | 0.7156 | -0.0006 [-0.0019,+0.0008] | 15/23 | +0.0006 | 6.003 (+0.097) | 8.447 (+1.18) | -0.007 |
| l2, early-stop metric l2 | 0.7151 | -0.0011 [-0.0022,-0.0001] | 17/21 | +0.0033 | 5.921 (+0.016) | 8.237 (+0.97) | +0.011 |

(All grid rows are in `analysis/H24_fast_runs.json`; `python grid_tables.py` prints the full table.)
Seed noise of the chosen config on H24 (seeds 42,1,2,3,4): Spearman 0.7310/0.7299/0.7302/0.7296/0.7310 (sd 0.0006),
cap 9.95/8.71/8.47/8.53/9.32, top10 6.06/5.93/6.00/5.83/6.16. The top of the grid is inside that noise band, so nl127/mcs50 is
"best" partly by seed luck. The robust facts are huber >> l2 ~ l1, and mcs 50 >= 10 > 200.

## 3. Confirmation (no re-tuning): fast mode vs baseline_fixed

| protocol | variant | spearman | d_sp [95% CI] | won/lost | d_sp_rel | d_top10 [CI] | d_cap [CI] | d_mae raw / quad-cal | d_hauler_mae (quad-cal) |
|---|---|---|---|---|---|---|---|---|---|
| H24 (tuning) | huber s42 | 0.7310 | +0.0148 [+0.0128,+0.0167] | 38/0 | +0.0279 | +0.150 [-0.24,+0.57] | +2.684 [+0.50,+4.92] | -0.121 / -0.019* | |
| H24 (tuning) | huber bag5 | 0.7318 | +0.0156 [+0.0138,+0.0174] | 38/0 | +0.0294 | +0.176 [-0.23,+0.60] | +1.237 [-0.53,+3.05] | -0.122 / -0.022 | +0.052 |
| **H25** | **huber s42** | 0.7363 | **+0.0129 [+0.0110,+0.0146]** | **38/0** | +0.0287 | +0.203 [-0.18,+0.58] | -1.237 [-2.76,+0.21] | -0.094 / -0.028 | +0.08 |
| **H25** | huber bag5 | 0.7373 | +0.0139 [+0.0122,+0.0155] | 38/0 | +0.0307 | +0.124 [-0.29,+0.54] | -1.763 [-3.11,-0.50] | -0.096 / -0.032 | +0.068 |
| H23 (untouched) | huber s42 | 0.7124 | +0.0170 [+0.0145,+0.0196] | 37/1 | +0.0313 | +0.061 [-0.37,+0.43] | +1.684 [-0.16,+3.79] | -0.140 / -0.057 | |
| H23 (untouched) | huber bag3 | 0.7135 | +0.0181 [+0.0156,+0.0206] | 37/1 | +0.0334 | -0.134 [-0.54,+0.25] | +1.737 [-0.39,+3.95] | -0.142 / -0.058 | +0.28 |
| LIVE (sanity) | huber s42 | 0.6942 | +0.0217 [+0.0151,+0.0305] | 5/0 | +0.0362 | -0.880 [-2.46,+0.70] | +0.200 [-2.20,+2.40] | -0.222 / -0.110 | +0.29 |

**LIVE watch item:** top10 is lower for every variant that includes huber. Pure huber gives -0.88; the blends give -0.28 at w=0.25,
-0.84 at w=0.5 and -0.98 at w=0.75. Per GW (L2 vs huber): GW1 2.5 vs 3.4, GW2 7.0 vs 4.2, GW3 4.2 vs 2.7, GW4 7.8 vs 5.2, GW5 2.5 vs 4.1,
with only 2-5 of the 10 players shared. That is 5 GWs, so do not decide on it, but re-check it once 2026-27 has about 10 played GWs.

\* H24 s42 has no saved val predictions (it ran before val-saving was added), so the seed-1 calibrated value is shown.
Baselines: H24 0.7162 / top10 5.905 / cap 7.263; H25 0.7234 / 5.332 / 8.132; H23 (own recipe run) 0.6954 / 5.879 / 6.237; LIVE 0.6725 / 4.80 / 3.40.

Pooled H23+H24+H25 fast (114 GWs, paired per-GW deltas):

| variant | d_sp [CI] won/lost | d_sp_rel | d_top10 [CI] won/lost | d_cap [CI] won/lost | d_mae (quad-cal) |
|---|---|---|---|---|---|
| huber s42 | +0.0149 [+0.0136,+0.0162] 113/1 | +0.0293 | +0.14 [-0.09,+0.36] 54/55 | +1.04 [-0.01,+2.19] 42/29 | - |
| huber bag | +0.0158 [+0.0146,+0.0170] 113/1 | +0.0312 | +0.06 [-0.18,+0.28] 57/53 | +0.40 [-0.62,+1.50] 34/37 | -0.0374 |
| blend50 | +0.0114 [+0.0107,+0.0120] 114/0 | +0.0231 | **+0.25 [+0.09,+0.41] 66/40** | +0.85 [-0.12,+1.90] 21/15 | -0.0247 |

## 4. Full mode (PARAMS_FULL: lr 0.01, 2000 rounds) vs baseline_fixed/*_full

| protocol | variant | spearman | d_sp [95% CI] | won/lost | d_sp_rel | d_top10 [CI] | d_cap [CI] | d_mae quad-cal |
|---|---|---|---|---|---|---|---|---|
| H25 | huber s42 | 0.7374 | +0.0116 [+0.0102,+0.0131] | 38/0 | +0.0234 | -0.150 [-0.52,+0.22] | -1.605 [-3.08,-0.26] | -0.027 |
| H25 | huber bag3 | 0.7376 | +0.0117 [+0.0103,+0.0132] | 38/0 | +0.0232 | -0.203 [-0.57,+0.14] | -1.211 [-2.53,-0.05] | -0.028 |
| H25 | blend50 | 0.7343 | +0.0085 [+0.0077,+0.0093] | 38/0 | +0.0179 | +0.045 [-0.18,+0.26] | -0.447 [-1.42,+0.32] | -0.018 |
| H24 | huber s42 | 0.7316 | +0.0127 [+0.0107,+0.0148] | 37/1 | +0.0224 | -0.318 [-0.72,+0.09] | -1.868 [-3.95,-0.03] | -0.012 |
| H24 | huber bag3 | 0.7319 | +0.0130 [+0.0110,+0.0151] | 37/1 | +0.0230 | -0.195 [-0.55,+0.18] | -2.000 [-3.97,-0.26] | -0.015 |
| H24 | blend50 | 0.7283 | +0.0094 [+0.0084,+0.0105] | 38/0 | +0.0172 | +0.063 [-0.20,+0.34] | -2.184 [-3.92,-0.68] | -0.012 |
| H24 | *L2 recipe itself, seed 43* | 0.7189 | +0.0000 | - | - | -0.116 | **-2.316** | 0 |
| H25 | *L2 recipe itself, seed 43* | 0.7269 | +0.0011 | - | - | -0.211 | -0.526 | 0 |

Full baselines: H25 0.7258 / top10 5.663 / cap 7.605; H24 0.7189 / 6.124 / **10.684**.
The **baseline itself with seed 43** loses 2.3 cap points on H24 and 0.5 on H25, and 0.12/0.21 top10.
The full-mode seed-42 baseline had an unusually lucky run of captain picks (section 6).

## 5. Why huber, and why calibration is mandatory

FPL points are 0/1/2-heavy with rare 10-20 point hauls. With L2, the model chases hauls: it over-reacts to one-off returns
(the known early-season weakness). Its top-5 predictions run 1 to 3 points above realized points
(H25 7.02 vs 5.94, H23 7.72 vs 6.28, LIVE 7.07 vs 4.12). Huber (delta 0.9 point) ranks the bulk better (who plays, who
blanks), and the relevant-row Spearman improves the most (+0.03). Its raw output, however, sits near the conditional
median: top-500 3.8 vs 5.9 actual, bias -0.4.

Calibrators were fitted on the val split only. Results are in `calib_tail.py` and `analysis/calib_*.json`. Top-k values are per-GW mean pred / mean actual.

| protocol | model | top1 | top5 | top20 | top50 | ties in top-20 (all GWs) | MAE | bias |
|---|---|---|---|---|---|---|---|---|
| H24 | L2 baseline | 7.65/7.26 | 6.80/6.37 | 5.59/5.25 | 4.66/4.39 | 1 | 0.994 | +0.04 |
| H24 | huber raw | 4.48/8.50 | 3.90/7.07 | 3.30/5.33 | 2.83/4.50 | 6 | 0.875 | -0.39 |
| H24 | huber iso | 8.00/8.50 | 6.62/7.07 | 5.69/5.33 | 4.54/4.50 | **596** | 0.969 | +0.02 |
| H24 | **huber quad** | 7.58/8.50 | 6.46/7.07 | 5.34/5.33 | 4.51/4.50 | 7 | 0.972 | +0.02 |
| H25 | L2 baseline | 8.32/8.13 | 7.02/5.94 | 5.58/4.89 | 4.64/4.33 | 3 | 0.977 | -0.04 |
| H25 | **huber quad** | 6.89/6.37 | 6.09/5.92 | 5.11/4.92 | 4.41/4.36 | 5 | 0.947 | -0.07 |
| H23 | L2 baseline | 9.17/6.24 | 7.72/6.28 | 6.20/5.39 | 5.15/4.39 | 6 | 0.980 | +0.10 |
| H23 | **huber quad** | 6.83/7.97 | 6.06/6.23 | 5.15/5.45 | 4.46/4.58 | 5 | 0.923 | -0.02 |
| H25 full | L2 baseline | 8.06/7.61 | 6.84/6.16 | 5.44/4.98 | 4.55/4.40 | 4 | 0.974 | -0.04 |
| H25 full | **huber quad** | 7.13/6.00 | 6.21/6.05 | 5.16/4.89 | 4.45/4.34 | 4 | 0.947 | -0.07 |
| LIVE | L2 baseline | 7.97/3.40 | 7.07/4.12 | 6.18/4.48 | 5.41/4.22 | 1 | 1.341 | +0.19 |
| LIVE | **huber quad** | 6.34/3.60 | 5.92/3.72 | 5.26/4.28 | 4.68/4.17 | 1 | 1.233 | -0.01 |

The quadratic coefficients are stable across seeds and seasons: c2 in [+0.006,+0.16], c1 in [1.20,1.46], c0 about 0.
The vertex is always far below the prediction range (<= -3.6 vs min pred about -0.4), so the map is strictly increasing.
Linear calibration under-predicts the top, and PWL/PAV behaves like quad but has more moving parts.
The isotonic clip saturates, e.g. LIVE GW2: seven players tied at 8.5.

## 6. Seed luck: why the cap CIs against a single-seed baseline mislead

`compare()` bootstraps GWs but holds both models fixed. cap is one player per GW, so a different LightGBM seed of the
**same** baseline already moves it "significantly":

| comparison (pooled H24+H25, paired per-GW) | d_spearman | d_top10 [CI] | d_cap [CI] |
|---|---|---|---|
| fast: L2 seed 43 vs L2 seed 42 (the baseline) | -0.0019 | -0.055 [-0.30,+0.17] | -0.763 [-1.62,+0.01] |
| full: L2 seed 43 vs L2 seed 42 (the baseline) | +0.0005 | -0.163 [-0.34,+0.02] | **-1.421 [-2.53,-0.32]** |
| fast: huber bag5 vs L2 bag3 (42,43,44) | +0.0131 [+0.012,+0.014] 76/0 | +0.057 [-0.22,+0.32] | -0.421 [-1.63,+0.83] |
| fast: huber bag5 vs L2 seed 43 | +0.0166 76/0 | +0.205 [-0.10,+0.51] | +0.500 [-0.72,+1.80] |
| full: huber bag3 vs L2 bag2 (42,43) | +0.0116 [+0.010,+0.013] 75/1 | -0.104 [-0.34,+0.14] | -0.882 [-1.97,+0.20] |
| full: huber bag3 vs L2 seed 43 | +0.0118 76/0 | -0.036 [-0.28,+0.21] | -0.184 [-1.34,+1.05] |
| fast: blend50(L2 bag) vs L2 bag3 | +0.0096 76/0 | +0.161 [-0.01,+0.33] | -0.250 [-1.32,+0.82] |
| full: blend50(L2 bag) vs L2 bag2 | +0.0084 76/0 | +0.143 [-0.01,+0.31] | -0.671 [-1.57,+0.13] |

(L2 seeds 43/44 in fast mode are read-only from `../early_season_prior/preds/base_seed{43,44}_*`. The full L2 seed 43 is my run
of the exact recipe params.) Compared like for like, top10 and cap are neutral for huber and slightly positive (top10) for the blend.
The huber and L2 captain picks differ in about 62% of GWs. Huber did not simply switch to premiums: on H24 its captain's average price was
11.3 vs 9.3, but on H25 10.4 vs 10.5.

## 7. Blend (ensemble of the two losses)

pred = w * quad-calibrated huber + (1-w) * L2. Both are on the points scale, so the blend is too.

| w | H24 fast sp / top10 / cap | H25 fast sp / top10 / cap | H23 fast sp / top10 / cap | H24 full sp / top10 / cap | H25 full sp / top10 / cap |
|---|---|---|---|---|---|
| 0 (baseline) | 0.7162 / 5.905 / 7.26 | 0.7234 / 5.332 / 8.13 | 0.6954 / 5.879 / 6.24 | 0.7189 / 6.124 / 10.68 | 0.7258 / 5.663 / 7.61 |
| 0.25 | 0.7229 / 6.005 / 9.03 | 0.7292 / 5.416 / 8.66 | 0.7022 / 6.013 / 6.26 | 0.7243 / 6.050 / 9.34 | 0.7307 / 5.626 / 7.50 |
| **0.5** | 0.7279 / **6.166** / **9.66** | 0.7335 / **5.613** / 7.79 | 0.7077 / **6.079** / 6.74 | 0.7283 / **6.187** / 8.50 | 0.7343 / **5.708** / 7.16 |
| 0.75 | 0.7308 / 6.111 / 8.63 | 0.7364 / 5.537 / 6.29 | 0.7116 / 6.071 / 7.13 | 0.7308 / 6.121 / 9.26 | 0.7367 / 5.621 / 6.90 |
| 1 (huber) | 0.7318 / 6.082 / 8.50 | 0.7373 / 5.455 / 6.37 | 0.7135 / 5.745 / 7.97 | 0.7319 / 5.929 / 8.68 | 0.7376 / 5.461 / 6.40 |

Spearman is monotone in w, so a Spearman-only rule picks w=1. On H24 alone, w=0.5 maximizes both top10 and cap.
w=0.5 is also the top10 argmax in all 4 later evaluations (H25 fast, H23, and both full-mode runs).

## 8. Verdicts

| variant | verdict | reason |
|---|---|---|
| huber + mcs50 (+ quad calibration) | **WINNER** | Spearman up on H24 and H25 (38/38 each, CIs exclude 0); replicated on H23, LIVE and full mode; top10/cap neutral seed-for-seed |
| huber seed bag (3-5 seeds) | neutral | +0.001 fast / +0.0002 full Spearman over single seed; 3-5x training cost |
| blend50 (0.5 huber_cal + 0.5 L2) | winner (alternative) | +0.009..+0.012 Spearman everywhere, best top10 in every evaluation; 2x training; w picked with H25 in view, then confirmed on H23 |
| huber at the recipe structure (mcs10) | winner-subsumed | +0.0132 H24; mcs50 adds a little Spearman and cap |
| num_leaves 31/63/255, mcs 10/200, ff 0.5, lambda_l2 10, huber alpha 0.5 | neutral | within about 0.0015 of the chosen config (seed sd 0.0006) |
| huber alpha 2.0 | loser | H24 +0.0093, clearly below alpha 0.9 (+0.0148); better top10 but the blend beats it |
| regression_l1 | loser/neutral | H24 +0.001..+0.003, CIs include 0 |
| L2 with mcs50 | neutral | -0.0006 |
| L2 early-stopped on l2 metric | loser | -0.0011 [-0.0022,-0.0001]; the huber gain is not an early-stopping artefact |
| L2 seed bag3 | neutral | +0.0016 fast / +0.0008 full |

## 9. Implementation (patch sketch; nothing under REPO was edited)

**`src/fpl_optimizer/prediction/model.py`**: optional post-hoc calibration inside `PointPredictor`, backward compatible.
```python
class PointPredictor:
    def __init__(self, params=None, early_stopping_rounds=50):
        ...
        self.calibration: dict | None = None   # {"type": "quad", "coef": [c2, c1, c0]}

    def _predict_raw(self, df) -> np.ndarray:        # = current predict() body
        ...

    def predict(self, df) -> np.ndarray:
        return self._apply_calibration(self._predict_raw(df))

    def fit_calibration(self, val_df: pd.DataFrame) -> dict:
        """Monotone quadratic map raw->points fitted on the early-stopping val split
        (robust losses like huber predict near the conditional median)."""
        p = self._predict_raw(val_df); y = val_df["target"].to_numpy(float)
        c2, c1, c0 = np.polyfit(p, y, 2)
        if c2 <= 0 or -c1 / (2 * c2) >= p.min() - 1.0:      # not increasing on range -> linear
            c2 = 0.0; c1, c0 = np.polyfit(p, y, 1)
        assert c1 > 0
        self.calibration = {"type": "quad", "coef": [float(c2), float(c1), float(c0)],
                            "vertex": float(-c1 / (2 * c2)) if c2 else None}
        return self.calibration

    def _apply_calibration(self, raw):
        if not self.calibration:
            return raw
        c2, c1, c0 = self.calibration["coef"]
        v = self.calibration.get("vertex")
        x = raw if v is None else np.maximum(raw, v)         # strictly monotone guard
        return c0 + c1 * x + c2 * x * x
    # save(): metadata["calibration"] = self.calibration
    # load(): predictor.calibration = metadata.get("calibration")   # None for old models -> identity
```
**`scripts/train_predictor.py`**:
```python
PARAMS = {
    "objective": "huber", "alpha": 0.9,          # was "regression"
    "metric": "mae", "num_leaves": 127, "learning_rate": 0.01,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "min_child_samples": 50,                      # was 10
    "n_estimators": 2000, "verbose": -1, "lambda_l1": 1.0, "lambda_l2": 1.0, "seed": 42,
}
...
eval_model.train(train_df, val_df); eval_model.fit_calibration(val_df)   # before predict(holdout)
...
prod.train(train_df, val_df); cal = prod.fit_calibration(val_df)
metrics["calibration"] = cal
```
Also: add a `fit_calibration` step to the `/retrain` skill checklist. Report `stratified` metrics and a
"top-20 pred/actual" line in `_report`. Expect about 1.5x the training time: huber early-stops at 650-1260 trees in full mode vs 500-750 for L2.
The live code path (`live/predict.py`, `prediction/integration.py`) needs **no change**, because `PointPredictor.load` restores
the calibration and `predict()` applies it.
Tests (`tests/test_prediction/`): calibration save/load round-trip; `predict()` is identity when `calibration is None`;
the fitted map is strictly increasing on [min,max] of the val preds.

Optional **blend50** (if top10 matters more than all-row Spearman): train a second `PointPredictor` with the old L2
PARAMS on the same split and save it to `<out>/l2/`. Save the huber model to `<out>/huber/` and write
`<out>/ensemble.json = {"members": [["huber", 0.5], ["l2", 0.5]]}`. Add an `EnsemblePredictor.load()` that
`live/predict.py`/`integration.py` call when `ensemble.json` exists; its `predict()` returns the weighted sum of member predictions.

## 10. Point-in-time and live servability

* **Inputs:** identical to the current fixed model: the FeaturePipeline row for the upcoming GW, built from `data/raw/2026-27`
  (rebuilt weekly from the FPL API), plus `fpl_xp_lag` = the pre-deadline snapshot's `ep_this`. No new column, no new source.
* **Loss / params:** a training-time choice; it uses no information from any row outside the training split.
* **Calibration:** fitted only on the early-stopping val split, which is the last 8 GWs of the final training season. That split is already used
  for early stopping, so no new data dependency is added. It is strictly before the test season in every protocol. In prod it is 2025-26 GW31-38,
  which precedes every 2026-27 deadline. For an in-season `/retrain`, `_split_val` again takes the last 8 already-played GWs.
  The 3 coefficients ship in `metadata.json`.

## Files

* `hp.py`: config-name-driven trainer (writes `runs/{PROTO}_{MODE}_{cfg}.parquet` + `.val.parquet` + `.json`); adds the H23 protocol.
* `analyze.py` (per-run / bag tables), `calibrate.py` (iso/lin/quad/pwl), `calib_tail.py` (top-k calibration and ties),
  `blend.py`, `final_tables.py` (per-protocol + pooled tables, `analysis/final_{fast,full}.json`), `seed_robust.py`,
  `grid_tables.py`, `reproduce.sh` (every command, in order).
* `runs/`: per-row predictions (baseline schema + `pred`) for every fit, bag, calibration and blend. `logs/`: training logs.
