# in_season_recency: training window, recency weighting, early stopping for the retrain

Agent `in_season_recency`, 2026-09-24/25. Leak-fixed feature set (`fpl_xp` -> `fpl_xp_lag`),
harness metrics. Primary metric: mean per-GW Spearman. All deltas are paired per GW against
`EXP/baseline_fixed/*` (fast: `PARAMS_FAST`, full: `PARAMS_FULL`).

## Recommendation for the retrain we do now

| choice | recommendation | evidence |
|---|---|---|
| window | **all seasons 2016-17..2025-26 + every completed 2026-27 GW (GW1-5)** | in-season rows help; dropping old seasons hurts |
| weighting | **none (uniform)**: no season decay, no last-season or DEFCON-era upweight | decay 0.9 neutral, 0.8/0.7 lose; last-season x2 loses; in-season/DEFCON x3/x6 lose |
| early stopping | **last 8 GWs of 2025-26 (last COMPLETE season)**, MAE, patience 50, per position, as today | early-stopping on 2026-27 GW1-5 means NOT training on them (`valIS`), which throws away the in-season gain |
| final fit | **refit on train+val (incl. 2025-26 GW31-38) for each position's best_iter** (optional) | neutral to slightly positive (+0.0007 full / +0.0009 fast pooled Spearman, MAE -0.002), within the seed noise floor; costs nothing and keeps the most recent DEFCON-era rows |

**Pitfall (would silently lose the in-season gain):** the retrain skill says "extend PROD_SEASONS"
with 2026-27. `scripts/train_predictor.py::_split_val(df, PROD_SEASONS)` takes the LAST season in the
list as the val season. With 2026-27 at GW5, "last 8 GWs" means GW > -3, i.e. ALL of GW1-5 become
early-stopping rows and none are trained on. That is exactly the `valIS` variant below:
H25 GW6-10 -0.0044 Spearman vs training on them, bias -0.13 pts/row.

Reference implementation (runs on the shared cache): `recommended_retrain.py`. Fast smoke run:
best_iter GK 106 / DEF 152 / MID 231 / FWD 170, 205,848 ES-train rows, 6,252 val rows, and a final fit on
all 212,100 rows (2016-17..2026-27 GW1-5). Log: `recommended_retrain_fast.log`.

## Method notes (read before trusting any small delta)

* `common.py::masks()` generalizes `harness.split`. `sanity.py` checked that the masks are identical to
  `harness.split` for H25/H24/LIVE/IS25/IS24 and that `run_variant(H25)` reproduces the cached H25 fast
  baseline exactly (Spearman 0.7234161307205286). `sanity_refit.py` checked that the refit trainer
  reproduces an early-stopped model bit-for-bit (max |diff| 0.0).
* **LightGBM 4.6 gotcha:** `lgb.train(params_with_n_estimators, ds, num_boost_round=n)` IGNORES `n`,
  because the `n_estimators` alias in params wins (`_choose_num_iterations`). Any refit/fixed-rounds code
  must strip the alias and set `num_iterations` (`common._refit`).
* **Seed noise floor (new, applies to every agent's results):** the SAME recipe retrained with
  `seed=7` instead of 42 gives these "deltas":

  | mode | pooled Spearman | H25 cap | H24 cap | pooled top10 | pooled MAE |
  |---|---|---|---|---|---|
  | fast | -0.0010 [-0.0019,+0.0000] | **-2.08 [-3.55,-0.68]** | +0.53 | +0.14 | +0.0002 |
  | full | -0.0001 [-0.0007,+0.0005] | -0.55 | -1.50 | -0.13 | **+0.0017 [+0.0003,+0.0032]** |

  The GW-bootstrap CI covers only the choice of GWs, not training randomness. A cap/top10/MAE CI that
  excludes 0 is NOT proof of an effect on its own. Fast-mode Spearman effects below about 0.001-0.0015
  are indistinguishable from a seed change. Per-GW seed SD is 0.004-0.005, so a 5-GW window (the IS
  protocols) carries seed noise of about ±0.002-0.006.

## (1) Value of training on the current season's completed GWs

**The literal question (IS vs H, same GW6-10 rows, fast):** H25 +0.0049 [-0.0019,+0.0118] 3/5;
H24 +0.0006 [-0.0037,+0.0047] 3/5; pooled +0.0028 6/10. spearman_rel +0.0104 (CI excludes 0), hauler MAE
-0.126, bias +0.056. On its own this is inconclusive, because 5 GWs sit inside the seed noise. The recommended
recipe on the same rows (`refitIS`: +GW1-5, early stop on the prior season's last 8, refit all) vs the frozen
pre-season model: **H25 +0.0067 (5/5), H24 +0.0029 (5/5), pooled +0.0048 [+0.0023,+0.0076], 10/10 GWs.**

**Longer tests settle it:**

| test (fast unless noted) | H25 | H24 | pooled | top10 / cap pooled |
|---|---|---|---|---|
| expanding retrain every 5 GWs (k=5,10,20,30; tests GW6-10, 11-15, 21-25, 31-35) | +0.0086, 18/20 | +0.0014, 13/20 | **+0.0050 [+0.0028,+0.0073], 31/40** | +0.07 / -0.62 |
| half season: +GW1-19, test GW20-38 | +0.0104, 19/19 | +0.0018, 10/19 | **+0.0061 [+0.0040,+0.0083], 29/38** | +0.28 / +0.58 |
| half season, **FULL mode** | **+0.0107, 18/19** | **+0.0015 [+0.0002,+0.0028], 12/19** | **+0.0061 [+0.0040,+0.0082], 30/38** | -0.15 [-0.43,+0.14] / -0.74 (seed floor -1.03) |

Per 5-GW window on H25 the gain grows as in-season rows accumulate: +0.0049 (GW6-10), +0.0036 (11-15),
+0.0144 (21-25), +0.0115 (31-35). H24 windows: +0.0006, -0.0006, +0.0047, +0.0010.

**Why:** 2025-26 was the first DEFCON season. Regular DEFs averaged +0.58 pts/row more than in 2024-25
(2.48 -> 3.06; 2026-27 GW1-5: 3.25). The frozen pre-DEFCON model under-predicts DEF by -0.18 pts/row on
2025-26 GW20-38; with in-season rows the DEF bias becomes +0.12 and cross-position ranking improves. In the
no-rule-change season (H24) the gain is at noise level (+0.0015-0.0018) but never negative. For 2026-27
(second DEFCON season with the same scoring, plus the BPS overhaul) expect a small-positive effect,
somewhere between the two. In no test did adding unweighted in-season rows hurt the primary metric.

Caveat: in full mode on H25 the in-season model puts about 3x more DEFs in the overall top-10
(7% -> 22% of picks) and the top10 metric fell -0.43 on H25 (fast mode: +0.37). Pooled across both seasons,
top10 and cap stay within noise. Worth watching the captain/top-end picks after the retrain.

**Verdict: WINNER.** Both holdouts improve; pooled CI excludes 0; 76-79% of GWs won; top10/cap within noise; confirmed in full mode.

## (2) Recency weighting: all losers or neutral

| variant | H25 | H24 | pooled | verdict |
|---|---|---|---|---|
| season decay 0.9 | -0.0014 (18/38) | +0.0010 (20/38) | -0.0002 [-0.0012,+0.0007] 38/76 | neutral |
| season decay 0.8 | -0.0022 (12/38) | -0.0011 (18/38) | -0.0017 [-0.0026,-0.0007] 30/76 | loser |
| season decay 0.7 | -0.0035 (9/38) | -0.0032 (11/38) | -0.0033 [-0.0045,-0.0022] 20/76 | loser |
| last full season x2 | -0.0026 (8/38) | -0.0006 (17/38) | -0.0016 [-0.0025,-0.0007] 25/76 | loser |
| in-season rows x3 (half test) vs x1 | -0.0039 (4/19) | -0.0030 (5/19) | -0.0034 [-0.0047,-0.0021] 9/38 | loser |
| in-season rows x6 (half test) vs x1 | -0.0026 (5/19) | -0.0039 (4/19) | -0.0032 [-0.0047,-0.0017] 9/38 | loser |
| in-season GW1-5 x3 vs IS (GW6-10) | -0.0066 (1/5) | +0.0019 (3/5) | -0.0023 4/10 | loser |
| LIVE: 2025-26 x2 (5 GWs, sanity) | | | +0.0048 3/5 (noise) | not decisive |

Upweighting the DEFCON era is the most tempting option, and it is tested directly on H25: 2025-26 GW1-19
x3 or x6 vs x1, tested on 2025-26 GW20-38. It is worse. It raises the level bias (+0.01 to +0.03) and MAE and
over-fits the few new-era rows. The model gets the regime shift from unweighted in-season rows. More
weight only adds variance. Stronger decay hurts monotonically.

## (3) Dropping the oldest seasons: losers

| train from | H25 | H24 | pooled |
|---|---|---|---|
| 2018-19 (drop 2016-18) | -0.0030 (14/38) | -0.0041 (12/38) | -0.0036 [-0.0047,-0.0024] 26/76 |
| 2020-21 (drop 2016-20: no xP, no FPL xG) | -0.0027 (15/38) | -0.0062 (4/38) | -0.0044 [-0.0056,-0.0032] 19/76 |

LightGBM handles those seasons' missing xP and FPL advanced stats natively (NaN). Their rows still teach
the form, minutes and fixture structure, and removing them costs rank quality. Keep all 10 seasons.

## Early stopping / final fit

| variant | test rows | vs | H25 | H24 | pooled | note |
|---|---|---|---|---|---|---|
| refit on train+val (fast) | full season | H | +0.0006 (21/38) | +0.0012 (24/38) | +0.0009 [-0.0001,+0.0019] 45/76 | MAE -0.0035, spearman_rel +0.0040 |
| refit on train+val (**FULL**) | full season | H full | +0.0006 (19/38) | +0.0007 (23/38) | +0.0007 [+0.0001,+0.0012] 42/76 | MAE -0.0019; cap -1.38 is within the seed floor (-1.03 full; seed-only H24 -1.50) |
| refitIS (recommended) | GW6-10 | IS | +0.0018 (3/5) | +0.0022 (2/5) | +0.0020 5/10 | |
| valIS (early-stop on GW1-5, not trained on) | GW6-10 | IS | **-0.0044** (2/5) | +0.0018 (3/5) | -0.0013 5/10 | H25 bias -0.134: loses the regime signal |
| valISrefit (early-stop on GW1-5, then refit all) | GW6-10 | IS | +0.0011 (2/5) | +0.0038 (3/5) | +0.0025 5/10 | similar to refitIS |
| LIVE refit (sanity, 5 GWs) | 2026-27 GW1-5 | LIVE | | | +0.0073 3/5 | noise-level |

Early stopping on 2026-27 GW1-5 only makes sense if those rows are refit afterwards (`valISrefit`, which
behaves like `refitIS`). 3.2k early-season rows (rolling features NaN at GW1-2) give a noisier stopping
signal than the 6.3k rows of 2025-26 GW31-38, and the chosen rounds are similar either way (H25: early-stopping on GW1-5 picks GK 82 / DEF 142 / MID 165 / FWD 90;
early-stopping on 2024-25 GW31-38 picks 133 / 121 / 141 / 98). So: **early stop on 2025-26 GW31-38, then refit on everything.**
Refit's gain is within noise, so skipping it is acceptable if the code change is unwanted. The part that
matters is that 2026-27 GW1-5 are in the training set.

Not done here (another agent covers it): the early-stopping METRIC. It is `mae` on an L2 objective, which
stops early at about 100-150 rounds in fast mode. `hparams_ensemble/hp.py` sweeps `m=l2|mae|huber`.

## Implementation (patch sketch; nothing in REPO was edited)

`src/fpl_optimizer/prediction/model.py`, add two methods to `PointPredictor`:

```python
_NUM_ITER_ALIASES = {"n_estimators", "num_iterations", "num_iteration", "n_iter", "num_tree",
                     "num_trees", "num_round", "num_rounds", "nrounds", "num_boost_round", "max_iter"}

    def best_iterations(self) -> dict[str, int]:
        return {p: int(b.best_iteration or b.current_iteration()) for p, b in self._models.items()}

    def train_fixed_rounds(self, train_df, rounds: dict[str, int], sample_weight=None) -> None:
        """Train each position booster for exactly rounds[pos] iterations (no validation).
        LightGBM 4.x: an n_estimators alias in params OVERRIDES num_boost_round -> strip it."""
        import lightgbm as lgb
        self._feature_names = [c for c in train_df.columns if c not in _NON_FEATURE_COLS]
        base = {k: v for k, v in self.params.items() if k not in _NUM_ITER_ALIASES}
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        for pos in POSITIONS:
            m = (train_df["position"] == pos).to_numpy()
            if not m.any() or pos not in rounds:
                continue
            ds = lgb.Dataset(train_df.loc[m, self._feature_names], label=train_df.loc[m, "target"],
                             weight=None if w is None else w[m])
            self._models[pos] = lgb.train({**base, "num_iterations": int(rounds[pos])}, ds)
```

`scripts/train_predictor.py`:

```python
CURRENT_SEASON = "2026-27"        # its COMPLETED GWs are trained on; never used as the val season
PROD_SEASONS = EVAL_TRAIN_SEASONS + [EVAL_HOLDOUT]   # unchanged: complete seasons only
...
parser.add_argument("--no-refit", action="store_true")
df = FeaturePipeline(args.data_dir, resolver, PROD_SEASONS + [CURRENT_SEASON]).build()
... existing synthetic-GW drop (keeps only completed 2026-27 GWs) ...
# EVAL phase: unchanged (train <=2024-25 -> holdout 2025-26); use `df[df.season.isin(PROD_SEASONS)]`
# PROD phase:
train_df, val_df = _split_val(df, PROD_SEASONS)                   # val = 2025-26 GW31-38 (NOT 2026-27)
cur = df[df["season"] == CURRENT_SEASON]
train_df = pd.concat([train_df, cur])                             # + 2026-27 GW1-5, weight 1
prod = PointPredictor(params=PARAMS, early_stopping_rounds=50)
prod.train(train_df, val_df)
if not args.no_refit:
    best = prod.best_iterations()
    prod = PointPredictor(params=PARAMS)
    prod.train_fixed_rounds(pd.concat([train_df, val_df]), best)  # final model sees everything
metrics["train_seasons"] = PROD_SEASONS + [f"{CURRENT_SEASON} GW1-{int(cur['GW'].max())}"]
metrics["best_iter"] = best
```

Guard to add: `assert _split_val` never receives the in-progress season (`seasons[-1] != CURRENT_SEASON`).
Update `.claude/skills/retrain/SKILL.md` step 2: replace "extend PROD_SEASONS" with "set CURRENT_SEASON;
completed GWs are trained on automatically". Suggested cadence: re-run the same retrain at about GW10, GW19
and GW30. The expanding-window test shows the in-season gain growing as rows accumulate.
(Separately, the fpl_xp -> fpl_xp_lag leak fix from `EXP/xp_fix` must be in the feature pipeline
before this retrain. All numbers here assume it.)

## Point-in-time / live servability

No new features and no new live inputs. The retrain changes only which rows are trained on and how
rounds are chosen. 2026-27 GW1-5 targets are final (GW5 scores final 09:00 UK the next day; synthetic
unplayed rows are dropped by the `gmax > 0` filter). Their features come from the same FeaturePipeline
invariants (post-match stats from GW < row GW; pre-match price/selected/was_home from the row GW;
`fpl_xp_lag` = snapshot `ep_this`). Every evaluation here trains on GW <= k and tests GW > k
(`common.masks`). Weights depend only on the season label or the in-season flag. The 2026-27 rows carry
exactly what live serving has, including understat NaN (not collected) and odds_team_* NaN (no
`data/odds/2026-27.json`). Training on them also aligns the model with the live NaN pattern.

## Files

`common.py` (splits, weights, fixed-rounds refit), `jobs.py` (variant grid, lock-file queue),
`analyze.py` (paired per-GW deltas, pooled bootstrap; writes `results_{fast,full}.{md,json}`),
`sanity.py`, `sanity_refit.py`, `recommended_retrain.py` (+ `recommended_retrain_fast.log`),
`preds/<variant>.parquet` (baseline schema + `pred`) and `preds/<variant>.json` (params, best_iter,
row counts, summary). Deferred, not run: `roll_*_k15/25/35`, `from2022_*`, `esl2_*` (their lock files say DEFERRED).

---
# Appendix: generated tables

# in_season_recency results (fast mode)

Cells: mean per-GW delta (new - base) [bootstrap 95% CI] GWs-won/GWs. For mae/hauler_mae lower is better (won = new lower).

### (1) Value of in-season training data

| variant | protocol | spearman | spearman_rel | mae | hauler_mae | top10 | cap | bias |
|---|---|---|---|---|---|---|---|---|
| IS (train +GW1-5) vs H, GW6-10 | H25 | +0.0049 [-0.0019,+0.0118] 3/5 | +0.0182 [+0.0058,+0.0306] 4/5 | +0.0355 [+0.0159,+0.0552] 0/5 | -0.2732 [-0.3287,-0.2246] 5/5 | -0.14 [-0.88,+0.46] 2/5 | -0.60 [-3.60,+2.20] 1/5 | +0.1255 [+0.0955,+0.1555] 5/5 |
| IS (train +GW1-5) vs H, GW6-10 | H24 | +0.0006 [-0.0037,+0.0047] 3/5 | +0.0025 [-0.0113,+0.0143] 3/5 | -0.0112 [-0.0292,+0.0062] 3/5 | +0.0215 [-0.0494,+0.0847] 2/5 | -0.74 [-2.04,+0.84] 1/5 | -3.20 [-4.80,-1.20] 0/5 | -0.0137 [-0.0260,-0.0010] 1/5 |
| IS (train +GW1-5) vs H, GW6-10 | pooled | +0.0028 [-0.0013,+0.0073] 6/10 | +0.0104 [+0.0001,+0.0207] 7/10 | +0.0121 [-0.0084,+0.0323] 3/10 | -0.1258 [-0.2274,-0.0295] 7/10 | -0.44 [-1.28,+0.42] 3/10 | -1.90 [-3.70,+0.00] 1/10 | +0.0559 [+0.0093,+0.1030] 6/10 |
| RECOMMENDED: +GW1-5, ES last-8 prior season, refit all; vs H | H25 | +0.0067 [+0.0027,+0.0108] 5/5 | +0.0221 [+0.0149,+0.0309] 5/5 | +0.0387 [+0.0212,+0.0530] 0/5 | -0.3377 [-0.4183,-0.2571] 5/5 | -0.04 [-0.66,+0.62] 2/5 | -0.40 [-1.20,+0.00] 0/5 | +0.1391 [+0.1126,+0.1664] 5/5 |
| RECOMMENDED: +GW1-5, ES last-8 prior season, refit all; vs H | H24 | +0.0029 [+0.0010,+0.0052] 5/5 | +0.0101 [-0.0030,+0.0244] 4/5 | -0.0176 [-0.0306,-0.0045] 5/5 | -0.0253 [-0.0956,+0.0451] 3/5 | -0.48 [-1.38,+0.44] 2/5 | +3.20 [-3.00,+11.40] 2/5 | -0.0151 [-0.0278,-0.0024] 2/5 |
| RECOMMENDED: +GW1-5, ES last-8 prior season, refit all; vs H | pooled | +0.0048 [+0.0023,+0.0076] 10/10 | +0.0161 [+0.0071,+0.0249] 9/10 | +0.0105 [-0.0108,+0.0308] 5/10 | -0.1815 [-0.2973,-0.0730] 8/10 | -0.26 [-0.84,+0.34] 4/10 | +1.40 [-1.60,+6.00] 2/10 | +0.0620 [+0.0120,+0.1132] 7/10 |
| +GW1-5, ES on GW1-5, refit all; vs H | H25 | +0.0061 [+0.0019,+0.0102] 4/5 | +0.0207 [+0.0131,+0.0300] 5/5 | +0.0406 [+0.0220,+0.0562] 0/5 | -0.3369 [-0.4201,-0.2542] 5/5 | -0.10 [-0.80,+0.70] 2/5 | -0.40 [-1.20,+0.00] 0/5 | +0.1381 [+0.1068,+0.1694] 5/5 |
| +GW1-5, ES on GW1-5, refit all; vs H | H24 | +0.0045 [+0.0015,+0.0067] 4/5 | +0.0133 [+0.0028,+0.0251] 4/5 | -0.0201 [-0.0328,-0.0074] 5/5 | -0.0133 [-0.0713,+0.0502] 3/5 | -0.62 [-1.44,+0.34] 1/5 | +5.00 [-1.40,+12.80] 3/5 | -0.0215 [-0.0345,-0.0081] 0/5 |
| +GW1-5, ES on GW1-5, refit all; vs H | pooled | +0.0053 [+0.0026,+0.0079] 8/10 | +0.0170 [+0.0096,+0.0249] 9/10 | +0.0103 [-0.0117,+0.0320] 5/10 | -0.1751 [-0.2919,-0.0663] 8/10 | -0.36 [-0.94,+0.26] 3/10 | +2.30 [-1.10,+6.90] 3/10 | +0.0583 [+0.0065,+0.1125] 5/10 |
| ES on GW1-5, NOT trained on them (naive PROD_SEASONS); vs H | H25 | +0.0005 [-0.0044,+0.0061] 3/5 | +0.0062 [-0.0052,+0.0176] 3/5 | -0.0029 [-0.0156,+0.0108] 3/5 | +0.0156 [-0.0334,+0.0631] 1/5 | -0.72 [-1.50,+0.12] 1/5 | +0.80 [+0.00,+2.40] 1/5 | -0.0090 [-0.0225,+0.0046] 2/5 |
| ES on GW1-5, NOT trained on them (naive PROD_SEASONS); vs H | H24 | +0.0025 [-0.0003,+0.0049] 4/5 | +0.0085 [-0.0081,+0.0231] 3/5 | -0.0138 [-0.0260,-0.0038] 4/5 | +0.0542 [-0.0113,+0.0987] 1/5 | +0.00 [-0.70,+0.74] 2/5 | +3.00 [+0.60,+5.60] 3/5 | -0.0217 [-0.0327,-0.0090] 1/5 |
| ES on GW1-5, NOT trained on them (naive PROD_SEASONS); vs H | pooled | +0.0015 [-0.0016,+0.0046] 7/10 | +0.0074 [-0.0023,+0.0167] 6/10 | -0.0084 [-0.0178,+0.0011] 7/10 | +0.0349 [-0.0087,+0.0732] 2/10 | -0.36 [-0.91,+0.24] 3/10 | +1.90 [+0.40,+3.60] 4/10 | -0.0153 [-0.0247,-0.0051] 3/10 |
| expanding in-season retrain every 5 GWs vs frozen H | H25 | +0.0086 [+0.0054,+0.0120] 18/20 | +0.0180 [+0.0101,+0.0266] 17/20 | +0.0177 [+0.0032,+0.0313] 6/20 | -0.3152 [-0.3757,-0.2608] 20/20 | -0.12 [-0.70,+0.44] 9/20 | -2.50 [-4.65,-0.55] 4/20 | +0.1153 [+0.0983,+0.1322] 20/20 |
| expanding in-season retrain every 5 GWs vs frozen H | H24 | +0.0014 [-0.0007,+0.0035] 13/20 | +0.0048 [-0.0020,+0.0114] 12/20 | -0.0133 [-0.0206,-0.0062] 16/20 | +0.0154 [-0.0263,+0.0605] 10/20 | +0.26 [-0.48,+1.03] 9/20 | +1.25 [-1.75,+4.95] 4/20 | -0.0159 [-0.0223,-0.0096] 3/20 |
| expanding in-season retrain every 5 GWs vs frozen H | pooled | +0.0050 [+0.0028,+0.0073] 31/40 | +0.0114 [+0.0060,+0.0173] 29/40 | +0.0022 [-0.0066,+0.0116] 22/40 | -0.1499 [-0.2115,-0.0879] 30/40 | +0.07 [-0.41,+0.53] 18/40 | -0.62 [-2.53,+1.62] 8/40 | +0.0497 [+0.0274,+0.0720] 23/40 |

Per window (spearman delta vs frozen pre-season model):

| proto | k | test_gws | base_spearman | d_spearman | won | n | d_mae | d_bias | d_top10 | d_cap |
|---|---|---|---|---|---|---|---|---|---|---|
| H25 | 5 | 6-10 | 0.7523 | 0.0049 | 3 | 5 | 0.0355 | 0.1255 | -0.1400 | -0.6000 |
| H25 | 10 | 11-15 | 0.7391 | 0.0036 | 5 | 5 | 0.0334 | 0.1394 | -0.1000 | -6.2000 |
| H25 | 20 | 21-25 | 0.7163 | 0.0144 | 5 | 5 | 0.0180 | 0.1343 | -0.4200 | -2.6000 |
| H25 | 30 | 31-35 | 0.7196 | 0.0115 | 5 | 5 | -0.0161 | 0.0619 | 0.2000 | -0.6000 |
| H24 | 5 | 6-10 | 0.7280 | 0.0006 | 3 | 5 | -0.0112 | -0.0137 | -0.7400 | -3.2000 |
| H24 | 10 | 11-15 | 0.7118 | -0.0006 | 3 | 5 | -0.0189 | -0.0165 | 1.3600 | 1.2000 |
| H24 | 20 | 21-25 | 0.7119 | 0.0047 | 5 | 5 | -0.0166 | -0.0169 | 0.3000 | 3.4000 |
| H24 | 30 | 31-35 | 0.7149 | 0.0010 | 2 | 5 | -0.0065 | -0.0167 | 0.1200 | 3.6000 |

### Half-season test: train + test-season GW1-19, test GW20-38 (H25 in-season rows = first DEFCON-era data)

| variant | protocol | spearman | spearman_rel | mae | hauler_mae | top10 | cap | bias |
|---|---|---|---|---|---|---|---|---|
| +GW1-19 (w_in=1) vs frozen H | H25 | +0.0104 [+0.0079,+0.0131] 19/19 | +0.0186 [+0.0120,+0.0254] 17/19 | +0.0132 [+0.0056,+0.0209] 4/19 | -0.3164 [-0.3734,-0.2603] 19/19 | +0.37 [-0.23,+0.92] 10/19 | +0.26 [-1.68,+2.42] 4/19 | +0.1129 [+0.1048,+0.1214] 19/19 |
| +GW1-19 (w_in=1) vs frozen H | H24 | +0.0018 [-0.0002,+0.0039] 10/19 | +0.0020 [-0.0045,+0.0087] 10/19 | -0.0059 [-0.0123,+0.0004] 12/19 | -0.0380 [-0.0740,-0.0042] 12/19 | +0.20 [-0.37,+0.79] 9/19 | +0.89 [-1.05,+3.58] 3/19 | +0.0022 [-0.0040,+0.0085] 11/19 |
| +GW1-19 (w_in=1) vs frozen H | pooled | +0.0061 [+0.0040,+0.0083] 29/38 | +0.0103 [+0.0049,+0.0156] 27/38 | +0.0036 [-0.0023,+0.0095] 16/38 | -0.1772 [-0.2325,-0.1235] 31/38 | +0.28 [-0.12,+0.69] 19/38 | +0.58 [-0.84,+2.34] 7/38 | +0.0575 [+0.0394,+0.0758] 30/38 |
| +GW1-19 (w_in=3) vs frozen H | H25 | +0.0065 [+0.0032,+0.0101] 14/19 | +0.0122 [+0.0044,+0.0207] 13/19 | +0.0284 [+0.0179,+0.0386] 3/19 | -0.4093 [-0.4771,-0.3437] 19/19 | -0.14 [-0.74,+0.43] 9/19 | -0.16 [-3.11,+3.05] 7/19 | +0.1424 [+0.1309,+0.1549] 19/19 |
| +GW1-19 (w_in=3) vs frozen H | H24 | -0.0012 [-0.0031,+0.0007] 7/19 | -0.0052 [-0.0098,-0.0002] 5/19 | +0.0036 [-0.0036,+0.0105] 6/19 | -0.0235 [-0.0556,+0.0108] 14/19 | -0.15 [-0.75,+0.44] 7/19 | +0.42 [-1.84,+3.42] 4/19 | +0.0101 [+0.0024,+0.0180] 13/19 |
| +GW1-19 (w_in=3) vs frozen H | pooled | +0.0026 [+0.0005,+0.0051] 21/38 | +0.0035 [-0.0016,+0.0093] 18/38 | +0.0160 [+0.0086,+0.0235] 9/38 | -0.2164 [-0.2909,-0.1472] 33/38 | -0.15 [-0.56,+0.26] 16/38 | +0.13 [-1.79,+2.26] 11/38 | +0.0763 [+0.0543,+0.0986] 32/38 |
| +GW1-19 (w_in=6) vs frozen H | H25 | +0.0078 [+0.0047,+0.0111] 17/19 | +0.0189 [+0.0127,+0.0256] 17/19 | +0.0194 [+0.0094,+0.0290] 4/19 | -0.4081 [-0.4752,-0.3443] 19/19 | -0.34 [-1.05,+0.33] 9/19 | -0.47 [-3.32,+2.42] 8/19 | +0.1352 [+0.1228,+0.1479] 19/19 |
| +GW1-19 (w_in=6) vs frozen H | H24 | -0.0021 [-0.0048,+0.0006] 7/19 | -0.0040 [-0.0108,+0.0029] 8/19 | +0.0020 [-0.0059,+0.0096] 11/19 | -0.0289 [-0.0612,+0.0046] 12/19 | -0.44 [-1.02,+0.17] 6/19 | +1.84 [-1.26,+5.47] 6/19 | +0.0064 [-0.0025,+0.0151] 12/19 |
| +GW1-19 (w_in=6) vs frozen H | pooled | +0.0028 [+0.0003,+0.0054] 24/38 | +0.0075 [+0.0014,+0.0134] 25/38 | +0.0107 [+0.0036,+0.0177] 15/38 | -0.2185 [-0.2900,-0.1501] 31/38 | -0.39 [-0.86,+0.06] 15/38 | +0.68 [-1.50,+3.00] 14/38 | +0.0708 [+0.0492,+0.0925] 31/38 |
| w_in=3 vs w_in=1 | H25 | -0.0039 [-0.0059,-0.0018] 4/19 | -0.0064 [-0.0120,-0.0005] 5/19 | +0.0152 [+0.0079,+0.0223] 4/19 | -0.0930 [-0.1318,-0.0553] 16/19 | -0.51 [-1.03,-0.03] 9/19 | -0.42 [-2.42,+1.74] 5/19 | +0.0295 [+0.0227,+0.0365] 19/19 |
| w_in=3 vs w_in=1 | H24 | -0.0030 [-0.0048,-0.0012] 5/19 | -0.0073 [-0.0119,-0.0027] 3/19 | +0.0096 [+0.0055,+0.0135] 4/19 | +0.0145 [-0.0135,+0.0397] 6/19 | -0.35 [-0.68,-0.05] 8/19 | -0.47 [-1.47,+0.53] 1/19 | +0.0079 [+0.0030,+0.0129] 15/19 |
| w_in=3 vs w_in=1 | pooled | -0.0034 [-0.0047,-0.0021] 9/38 | -0.0068 [-0.0104,-0.0032] 8/38 | +0.0124 [+0.0081,+0.0165] 8/38 | -0.0392 [-0.0691,-0.0116] 22/38 | -0.43 [-0.75,-0.15] 17/38 | -0.45 [-1.53,+0.68] 6/38 | +0.0187 [+0.0133,+0.0242] 34/38 |
| w_in=6 vs w_in=1 | H25 | -0.0026 [-0.0045,-0.0005] 5/19 | +0.0003 [-0.0064,+0.0067] 10/19 | +0.0063 [-0.0011,+0.0140] 8/19 | -0.0917 [-0.1281,-0.0552] 17/19 | -0.71 [-1.33,-0.11] 5/19 | -0.74 [-3.53,+1.68] 8/19 | +0.0223 [+0.0158,+0.0289] 18/19 |
| w_in=6 vs w_in=1 | H24 | -0.0039 [-0.0059,-0.0017] 4/19 | -0.0060 [-0.0120,-0.0003] 7/19 | +0.0079 [+0.0035,+0.0122] 3/19 | +0.0091 [-0.0260,+0.0469] 10/19 | -0.64 [-0.97,-0.36] 2/19 | +0.95 [-1.42,+4.05] 4/19 | +0.0042 [-0.0016,+0.0105] 10/19 |
| w_in=6 vs w_in=1 | pooled | -0.0032 [-0.0047,-0.0017] 9/38 | -0.0029 [-0.0074,+0.0017] 17/38 | +0.0071 [+0.0028,+0.0114] 11/38 | -0.0413 [-0.0713,-0.0119] 27/38 | -0.68 [-1.02,-0.35] 7/38 | +0.11 [-1.63,+2.00] 12/38 | +0.0133 [+0.0080,+0.0187] 28/38 |

### Seed noise floor: SAME recipe, LightGBM seed 7 vs 42 (any real effect must clear this)

| variant | protocol | spearman | spearman_rel | mae | hauler_mae | top10 | cap | bias |
|---|---|---|---|---|---|---|---|---|
| seed7 | H25 | -0.0004 [-0.0017,+0.0008] 18/38 | -0.0006 [-0.0055,+0.0043] 16/38 | +0.0018 [-0.0023,+0.0057] 15/38 | -0.0209 [-0.0481,+0.0054] 22/38 | +0.15 [-0.16,+0.46] 20/38 | -2.08 [-3.55,-0.68] 6/38 | +0.0084 [+0.0038,+0.0131] 30/38 |
| seed7 | H24 | -0.0015 [-0.0031,+0.0000] 14/38 | -0.0019 [-0.0065,+0.0028] 16/38 | -0.0014 [-0.0061,+0.0031] 23/38 | +0.0139 [-0.0107,+0.0379] 16/38 | +0.12 [-0.31,+0.54] 21/38 | +0.53 [-1.34,+2.53] 9/38 | -0.0059 [-0.0112,-0.0003] 12/38 |
| seed7 | pooled | -0.0010 [-0.0019,+0.0000] 32/76 | -0.0013 [-0.0046,+0.0021] 32/76 | +0.0002 [-0.0028,+0.0033] 38/76 | -0.0035 [-0.0221,+0.0144] 38/76 | +0.14 [-0.12,+0.39] 41/76 | -0.78 [-1.92,+0.50] 15/76 | +0.0012 [-0.0027,+0.0052] 42/76 |

### (2) Recency weighting (full holdout season, vs fixed H baseline)

| variant | protocol | spearman | spearman_rel | mae | hauler_mae | top10 | cap | bias |
|---|---|---|---|---|---|---|---|---|
| dec09 | H25 | -0.0014 [-0.0025,-0.0004] 18/38 | -0.0017 [-0.0054,+0.0020] 19/38 | -0.0004 [-0.0040,+0.0035] 20/38 | +0.0375 [+0.0111,+0.0618] 14/38 | -0.18 [-0.55,+0.20] 15/38 | -1.47 [-2.71,-0.37] 5/38 | -0.0091 [-0.0133,-0.0050] 9/38 |
| dec09 | H24 | +0.0010 [-0.0005,+0.0025] 20/38 | +0.0020 [-0.0029,+0.0071] 18/38 | -0.0038 [-0.0083,+0.0008] 26/38 | +0.0239 [-0.0006,+0.0462] 12/38 | +0.10 [-0.26,+0.48] 18/38 | +1.26 [-0.92,+3.58] 11/38 | -0.0090 [-0.0144,-0.0036] 11/38 |
| dec09 | pooled | -0.0002 [-0.0012,+0.0007] 38/76 | +0.0001 [-0.0029,+0.0032] 37/76 | -0.0021 [-0.0050,+0.0009] 46/76 | +0.0307 [+0.0134,+0.0472] 26/76 | -0.04 [-0.30,+0.22] 33/76 | -0.11 [-1.37,+1.24] 16/76 | -0.0091 [-0.0125,-0.0056] 20/76 |
| dec08 | H25 | -0.0022 [-0.0036,-0.0009] 12/38 | -0.0069 [-0.0111,-0.0028] 10/38 | -0.0031 [-0.0078,+0.0016] 24/38 | +0.0688 [+0.0389,+0.0985] 9/38 | -0.12 [-0.54,+0.27] 17/38 | -1.18 [-2.61,+0.18] 4/38 | -0.0215 [-0.0270,-0.0159] 5/38 |
| dec08 | H24 | -0.0011 [-0.0024,+0.0002] 18/38 | -0.0038 [-0.0080,+0.0004] 14/38 | -0.0031 [-0.0074,+0.0012] 22/38 | +0.0182 [-0.0083,+0.0418] 13/38 | -0.09 [-0.48,+0.31] 16/38 | +0.37 [-1.37,+2.18] 9/38 | -0.0096 [-0.0150,-0.0040] 12/38 |
| dec08 | pooled | -0.0017 [-0.0026,-0.0007] 30/76 | -0.0054 [-0.0083,-0.0024] 24/76 | -0.0031 [-0.0061,+0.0002] 46/76 | +0.0435 [+0.0229,+0.0632] 22/76 | -0.11 [-0.40,+0.19] 33/76 | -0.41 [-1.55,+0.74] 13/76 | -0.0155 [-0.0196,-0.0113] 17/76 |
| dec07 | H25 | -0.0035 [-0.0052,-0.0018] 9/38 | -0.0074 [-0.0120,-0.0030] 12/38 | -0.0039 [-0.0095,+0.0016] 21/38 | +0.0673 [+0.0432,+0.0910] 8/38 | -0.05 [-0.38,+0.29] 14/38 | -1.18 [-2.47,+0.11] 8/38 | -0.0248 [-0.0300,-0.0197] 1/38 |
| dec07 | H24 | -0.0032 [-0.0046,-0.0016] 11/38 | -0.0078 [-0.0129,-0.0027] 10/38 | +0.0002 [-0.0043,+0.0045] 20/38 | +0.0212 [-0.0048,+0.0456] 12/38 | -0.01 [-0.38,+0.37] 18/38 | +0.84 [-0.95,+2.76] 12/38 | -0.0069 [-0.0131,-0.0008] 13/38 |
| dec07 | pooled | -0.0033 [-0.0045,-0.0022] 20/76 | -0.0076 [-0.0111,-0.0041] 22/76 | -0.0018 [-0.0056,+0.0018] 41/76 | +0.0442 [+0.0263,+0.0626] 20/76 | -0.03 [-0.29,+0.23] 32/76 | -0.17 [-1.28,+1.03] 20/76 | -0.0159 [-0.0203,-0.0114] 14/76 |
| last2 | H25 | -0.0026 [-0.0038,-0.0014] 8/38 | -0.0035 [-0.0075,+0.0004] 13/38 | +0.0040 [+0.0004,+0.0075] 10/38 | -0.0259 [-0.0519,+0.0002] 21/38 | +0.03 [-0.31,+0.38] 16/38 | -0.29 [-1.79,+1.26] 6/38 | +0.0088 [+0.0045,+0.0129] 28/38 |
| last2 | H24 | -0.0006 [-0.0019,+0.0007] 17/38 | -0.0020 [-0.0058,+0.0018] 17/38 | -0.0013 [-0.0054,+0.0026] 21/38 | +0.0200 [-0.0007,+0.0395] 15/38 | +0.16 [-0.19,+0.53] 18/38 | -0.11 [-2.11,+1.87] 10/38 | -0.0045 [-0.0098,+0.0006] 14/38 |
| last2 | pooled | -0.0016 [-0.0025,-0.0007] 25/76 | -0.0028 [-0.0056,+0.0001] 30/76 | +0.0013 [-0.0014,+0.0040] 31/76 | -0.0029 [-0.0205,+0.0144] 36/76 | +0.10 [-0.16,+0.35] 34/76 | -0.20 [-1.43,+1.05] 16/76 | +0.0022 [-0.0016,+0.0058] 42/76 |

### (3) Dropping the oldest seasons (vs fixed H baseline)

| variant | protocol | spearman | spearman_rel | mae | hauler_mae | top10 | cap | bias |
|---|---|---|---|---|---|---|---|---|
| from2018 | H25 | -0.0030 [-0.0046,-0.0015] 14/38 | -0.0042 [-0.0092,+0.0007] 15/38 | +0.0065 [+0.0019,+0.0108] 9/38 | +0.0149 [-0.0107,+0.0406] 15/38 | -0.11 [-0.47,+0.27] 17/38 | -1.08 [-2.47,+0.29] 4/38 | +0.0070 [+0.0027,+0.0116] 25/38 |
| from2018 | H24 | -0.0041 [-0.0059,-0.0023] 12/38 | -0.0099 [-0.0153,-0.0045] 13/38 | +0.0054 [+0.0005,+0.0103] 16/38 | -0.0006 [-0.0261,+0.0249] 19/38 | +0.05 [-0.29,+0.41] 18/38 | +1.50 [-0.79,+3.97] 11/38 | +0.0048 [-0.0009,+0.0103] 24/38 |
| from2018 | pooled | -0.0036 [-0.0047,-0.0024] 26/76 | -0.0071 [-0.0108,-0.0033] 28/76 | +0.0060 [+0.0026,+0.0094] 25/76 | +0.0072 [-0.0108,+0.0255] 34/76 | -0.03 [-0.28,+0.24] 35/76 | +0.21 [-1.18,+1.67] 15/76 | +0.0059 [+0.0024,+0.0095] 49/76 |
| from2020 | H25 | -0.0027 [-0.0046,-0.0009] 15/38 | -0.0038 [-0.0091,+0.0014] 15/38 | -0.0078 [-0.0135,-0.0022] 26/38 | +0.1347 [+0.1027,+0.1681] 3/38 | -0.11 [-0.46,+0.24] 15/38 | -1.66 [-3.13,-0.29] 7/38 | -0.0398 [-0.0463,-0.0329] 3/38 |
| from2020 | H24 | -0.0062 [-0.0076,-0.0048] 4/38 | -0.0104 [-0.0155,-0.0055] 10/38 | +0.0072 [+0.0022,+0.0119] 12/38 | -0.0036 [-0.0368,+0.0275] 21/38 | +0.11 [-0.35,+0.52] 23/38 | +0.21 [-1.42,+1.97] 11/38 | +0.0056 [-0.0014,+0.0123] 26/38 |
| from2020 | pooled | -0.0044 [-0.0056,-0.0032] 19/76 | -0.0071 [-0.0109,-0.0034] 25/76 | -0.0003 [-0.0043,+0.0037] 38/76 | +0.0656 [+0.0381,+0.0937] 24/76 | -0.00 [-0.29,+0.28] 38/76 | -0.72 [-1.79,+0.41] 18/76 | -0.0171 [-0.0240,-0.0100] 29/76 |

### Early-stopping / final-fit strategy (full season, vs fixed H baseline)

| variant | protocol | spearman | spearman_rel | mae | hauler_mae | top10 | cap | bias |
|---|---|---|---|---|---|---|---|---|
| refit | H25 | +0.0006 [-0.0006,+0.0018] 21/38 | +0.0021 [-0.0019,+0.0060] 25/38 | -0.0016 [-0.0052,+0.0020] 23/38 | -0.0004 [-0.0261,+0.0258] 18/38 | +0.13 [-0.21,+0.48] 21/38 | -0.29 [-1.71,+1.13] 7/38 | +0.0007 [-0.0033,+0.0046] 22/38 |
| refit | H24 | +0.0012 [-0.0005,+0.0029] 24/38 | +0.0060 [+0.0012,+0.0108] 25/38 | -0.0053 [-0.0100,-0.0005] 27/38 | -0.0076 [-0.0294,+0.0146] 23/38 | +0.02 [-0.37,+0.43] 19/38 | +0.55 [-1.26,+2.26] 15/38 | -0.0028 [-0.0084,+0.0027] 18/38 |
| refit | pooled | +0.0009 [-0.0001,+0.0019] 45/76 | +0.0040 [+0.0008,+0.0072] 50/76 | -0.0035 [-0.0064,-0.0004] 50/76 | -0.0040 [-0.0213,+0.0131] 41/76 | +0.08 [-0.19,+0.35] 40/76 | +0.13 [-1.01,+1.29] 22/76 | -0.0011 [-0.0045,+0.0023] 40/76 |

### In-season protocol variants (train +GW1-5, test GW6-10) vs IS baseline

| variant | protocol | spearman | spearman_rel | mae | hauler_mae | top10 | cap | bias |
|---|---|---|---|---|---|---|---|---|
| refitIS | H25 | +0.0018 [-0.0019,+0.0055] 3/5 | +0.0039 [-0.0095,+0.0149] 3/5 | +0.0031 [-0.0078,+0.0153] 3/5 | -0.0645 [-0.1372,+0.0070] 4/5 | +0.10 [-0.58,+0.72] 3/5 | +0.20 [-2.40,+3.40] 1/5 | +0.0137 [+0.0056,+0.0204] 4/5 |
| refitIS | H24 | +0.0022 [-0.0010,+0.0063] 2/5 | +0.0075 [-0.0033,+0.0239] 2/5 | -0.0063 [-0.0148,+0.0004] 4/5 | -0.0468 [-0.1459,+0.0240] 4/5 | +0.26 [-0.48,+0.78] 4/5 | +6.40 [+1.20,+14.80] 4/5 | -0.0014 [-0.0097,+0.0077] 2/5 |
| refitIS | pooled | +0.0020 [-0.0005,+0.0048] 5/10 | +0.0057 [-0.0033,+0.0156] 5/10 | -0.0016 [-0.0087,+0.0062] 7/10 | -0.0556 [-0.1146,-0.0010] 8/10 | +0.18 [-0.30,+0.60] 7/10 | +3.30 [-0.20,+8.80] 5/10 | +0.0061 [-0.0012,+0.0131] 6/10 |
| valIS | H25 | -0.0044 [-0.0091,-0.0000] 2/5 | -0.0120 [-0.0234,-0.0008] 1/5 | -0.0384 [-0.0459,-0.0307] 5/5 | +0.2888 [+0.2551,+0.3163] 0/5 | -0.58 [-1.36,+0.20] 2/5 | +1.40 [+0.00,+3.80] 2/5 | -0.1344 [-0.1546,-0.1078] 0/5 |
| valIS | H24 | +0.0018 [-0.0024,+0.0074] 3/5 | +0.0060 [-0.0102,+0.0220] 4/5 | -0.0026 [-0.0145,+0.0152] 4/5 | +0.0327 [-0.0233,+0.1142] 2/5 | +0.74 [-0.26,+1.48] 4/5 | +6.20 [+3.20,+10.20] 5/5 | -0.0079 [-0.0159,+0.0042] 1/5 |
| valIS | pooled | -0.0013 [-0.0050,+0.0027] 5/10 | -0.0030 [-0.0137,+0.0083] 5/10 | -0.0205 [-0.0334,-0.0042] 9/10 | +0.1608 [+0.0748,+0.2473] 2/10 | +0.08 [-0.62,+0.77] 6/10 | +3.80 [+1.50,+6.50] 7/10 | -0.0712 [-0.1128,-0.0297] 1/10 |
| valISrefit | H25 | +0.0011 [-0.0026,+0.0052] 2/5 | +0.0025 [-0.0119,+0.0146] 3/5 | +0.0051 [-0.0064,+0.0171] 2/5 | -0.0637 [-0.1389,+0.0115] 4/5 | +0.04 [-0.88,+0.96] 3/5 | +0.20 [-2.40,+3.40] 1/5 | +0.0126 [+0.0073,+0.0198] 5/5 |
| valISrefit | H24 | +0.0038 [-0.0019,+0.0095] 3/5 | +0.0108 [-0.0035,+0.0275] 3/5 | -0.0089 [-0.0200,+0.0013] 4/5 | -0.0348 [-0.1207,+0.0266] 2/5 | +0.12 [-0.64,+0.80] 3/5 | +8.20 [+2.80,+16.00] 4/5 | -0.0078 [-0.0194,+0.0037] 2/5 |
| valISrefit | pooled | +0.0025 [-0.0009,+0.0062] 5/10 | +0.0066 [-0.0035,+0.0176] 6/10 | -0.0019 [-0.0107,+0.0073] 6/10 | -0.0493 [-0.1074,+0.0027] 6/10 | +0.08 [-0.52,+0.65] 6/10 | +4.20 [+0.30,+9.60] 5/10 | +0.0024 [-0.0073,+0.0110] 7/10 |
| win3_IS | H25 | -0.0066 [-0.0107,-0.0025] 1/5 | -0.0144 [-0.0250,-0.0039] 1/5 | +0.0096 [-0.0054,+0.0250] 3/5 | -0.1277 [-0.1956,-0.0832] 5/5 | +0.12 [-0.60,+1.14] 2/5 | +0.40 [-2.40,+3.60] 1/5 | +0.0282 [+0.0152,+0.0407] 5/5 |
| win3_IS | H24 | +0.0019 [-0.0031,+0.0069] 3/5 | +0.0013 [-0.0143,+0.0174] 3/5 | -0.0090 [-0.0207,+0.0027] 4/5 | +0.0489 [-0.0445,+0.1400] 2/5 | +0.50 [+0.08,+1.00] 4/5 | +2.80 [+0.80,+4.80] 3/5 | -0.0199 [-0.0334,-0.0070] 0/5 |
| win3_IS | pooled | -0.0023 [-0.0065,+0.0019] 4/10 | -0.0065 [-0.0168,+0.0049] 4/10 | +0.0003 [-0.0103,+0.0124] 7/10 | -0.0394 [-0.1187,+0.0438] 7/10 | +0.31 [-0.20,+0.86] 6/10 | +1.60 [-0.20,+3.60] 4/10 | +0.0042 [-0.0140,+0.0219] 5/10 |

### LIVE sanity (train <=2025-26, test 2026-27 GW1-5) vs LIVE baseline

| variant | protocol | spearman | spearman_rel | mae | hauler_mae | top10 | cap | bias |
|---|---|---|---|---|---|---|---|---|
| refit_LIVE | LIVE | +0.0073 [-0.0011,+0.0182] 3/5 | +0.0078 [-0.0011,+0.0215] 4/5 | -0.0154 [-0.0360,+0.0011] 4/5 | +0.0693 [+0.0190,+0.1196] 1/5 | -1.06 [-2.70,+0.84] 1/5 | +5.60 [+1.80,+9.20] 4/5 | -0.0400 [-0.0619,-0.0188] 0/5 |
| last2_LIVE | LIVE | +0.0048 [-0.0033,+0.0145] 3/5 | +0.0046 [-0.0070,+0.0180] 2/5 | +0.0013 [-0.0244,+0.0224] 2/5 | -0.0413 [-0.1014,+0.0356] 4/5 | -0.22 [-1.18,+0.68] 3/5 | +1.60 [+0.20,+3.40] 3/5 | +0.0150 [-0.0117,+0.0398] 3/5 |

### Early-stopping rounds per position (best_iter)

| variant | GK | DEF | MID | FWD | n_train | n_val | secs | spearman |
|---|---|---|---|---|---|---|---|---|
| dec07_H24 | 96 | 200 | 132 | 132 | 145827 | 6800 | 233 | 0.7131 |
| dec07_H25 | 148 | 116 | 147 | 95 | 173461 | 6085 | 389 | 0.7199 |
| dec08_H24 | 120 | 155 | 135 | 106 | 145827 | 6800 | 264 | 0.7152 |
| dec08_H25 | 125 | 118 | 121 | 138 | 173461 | 6085 | 1296 | 0.7212 |
| dec09_H24 | 107 | 157 | 115 | 101 | 145827 | 6800 | 366 | 0.7172 |
| dec09_H25 | 118 | 153 | 121 | 110 | 173461 | 6085 | 567 | 0.7220 |
| from2018_H24 | 118 | 144 | 161 | 118 | 125561 | 6800 | 151 | 0.7122 |
| from2018_H25 | 90 | 136 | 105 | 90 | 153195 | 6085 | 149 | 0.7204 |
| from2020_H24 | 84 | 129 | 133 | 101 | 93018 | 6800 | 341 | 0.7100 |
| from2020_H25 | 89 | 128 | 97 | 75 | 120652 | 6085 | 311 | 0.7207 |
| half_H24_w1 | 138 | 217 | 132 | 113 | 158570 | 6800 | 448 | 0.7153 |
| half_H24_w1_full | 532 | 628 | 625 | 700 | 158570 | 6800 | 412 | 0.7183 |
| half_H24_w3 | 97 | 116 | 147 | 114 | 158570 | 6800 | 389 | 0.7123 |
| half_H24_w6 | 116 | 218 | 131 | 113 | 158570 | 6800 | 143 | 0.7114 |
| half_H25_w1 | 75 | 114 | 100 | 107 | 187636 | 6085 | 311 | 0.7277 |
| half_H25_w1_full | 486 | 613 | 687 | 489 | 187636 | 6085 | 1039 | 0.7302 |
| half_H25_w3 | 106 | 181 | 97 | 95 | 187636 | 6085 | 362 | 0.7238 |
| half_H25_w6 | 132 | 117 | 130 | 110 | 187636 | 6085 | 101 | 0.7251 |
| last2_H24 | 140 | 165 | 136 | 106 | 145827 | 6800 | 234 | 0.7156 |
| last2_H25 | 104 | 158 | 162 | 111 | 173461 | 6085 | 366 | 0.7208 |
| last2_LIVE | 125 | 96 | 235 | 165 | 202632 | 6252 | 123 | 0.6773 |
| refit_H24 | 100 | 178 | 131 | 103 | 145827 | 6800 | 441 | 0.7174 |
| refit_H24_full | 556 | 694 | 639 | 522 | 145827 | 6800 | 856 | 0.7196 |
| refit_H25 | 91 | 119 | 145 | 109 | 173461 | 6085 | 558 | 0.7240 |
| refit_H25_full | 525 | 621 | 641 | 671 | 173461 | 6085 | 686 | 0.7264 |
| refit_LIVE | 96 | 258 | 215 | 162 | 202632 | 6252 | 437 | 0.6798 |
| refitIS_H24 | 122 | 212 | 133 | 140 | 149038 | 6800 | 702 | 0.7308 |
| refitIS_H25 | 133 | 121 | 141 | 98 | 177049 | 6085 | 440 | 0.7591 |
| roll_H24_k10 | 140 | 118 | 153 | 106 | 152379 | 6800 | 108 | 0.7111 |
| roll_H24_k20 | 94 | 159 | 127 | 135 | 159281 | 6800 | 116 | 0.7166 |
| roll_H24_k30 | 97 | 240 | 123 | 91 | 166661 | 6800 | 116 | 0.7159 |
| roll_H25_k10 | 114 | 155 | 185 | 141 | 180772 | 6085 | 126 | 0.7426 |
| roll_H25_k20 | 118 | 183 | 132 | 106 | 188426 | 6085 | 104 | 0.7307 |
| roll_H25_k30 | 112 | 106 | 140 | 97 | 196547 | 6085 | 110 | 0.7311 |
| sanity_H25 | 91 | 119 | 145 | 109 | 173461 | 6085 | 157 | 0.7234 |
| seed7_H24 | 86 | 183 | 153 | 118 | 145827 | 6800 | 117 | 0.7147 |
| seed7_H24_full | 568 | 827 | 744 | 490 | 145827 | 6800 | 503 | 0.7191 |
| seed7_H25 | 143 | 193 | 147 | 84 | 173461 | 6085 | 119 | 0.7230 |
| seed7_H25_full | 479 | 569 | 629 | 586 | 173461 | 6085 | 383 | 0.7255 |
| valIS_H24 | 131 | 156 | 125 | 70 | 152627 | 3211 | 347 | 0.7304 |
| valIS_H25 | 82 | 142 | 165 | 90 | 179546 | 3588 | 253 | 0.7528 |
| valISrefit_H24 | 131 | 156 | 125 | 70 | 152627 | 3211 | 460 | 0.7324 |
| valISrefit_H25 | 82 | 142 | 165 | 90 | 179546 | 3588 | 453 | 0.7584 |
| win3_IS_H24 | 95 | 218 | 134 | 110 | 149038 | 6800 | 266 | 0.7305 |
| win3_IS_H25 | 96 | 155 | 103 | 111 | 177049 | 6085 | 270 | 0.7507 |

---

# in_season_recency results (full mode)

Cells: mean per-GW delta (new - base) [bootstrap 95% CI] GWs-won/GWs. For mae/hauler_mae lower is better (won = new lower).

### Half-season test: train + test-season GW1-19, test GW20-38 (H25 in-season rows = first DEFCON-era data)

| variant | protocol | spearman | spearman_rel | mae | hauler_mae | top10 | cap | bias |
|---|---|---|---|---|---|---|---|---|
| +GW1-19 (w_in=1) vs frozen H | H25 | +0.0107 [+0.0078,+0.0135] 18/19 | +0.0183 [+0.0111,+0.0249] 17/19 | +0.0115 [+0.0048,+0.0179] 5/19 | -0.3313 [-0.3717,-0.2932] 19/19 | -0.43 [-0.78,-0.12] 5/19 | +0.16 [-1.68,+2.00] 2/19 | +0.1118 [+0.1021,+0.1204] 19/19 |
| +GW1-19 (w_in=1) vs frozen H | H24 | +0.0015 [+0.0002,+0.0028] 12/19 | +0.0024 [-0.0020,+0.0075] 9/19 | -0.0072 [-0.0124,-0.0025] 14/19 | -0.0209 [-0.0516,+0.0055] 12/19 | +0.14 [-0.25,+0.59] 8/19 | -1.63 [-3.68,+0.05] 2/19 | -0.0041 [-0.0075,-0.0009] 5/19 |
| +GW1-19 (w_in=1) vs frozen H | pooled | +0.0061 [+0.0040,+0.0082] 30/38 | +0.0103 [+0.0054,+0.0153] 26/38 | +0.0021 [-0.0030,+0.0072] 19/38 | -0.1761 [-0.2311,-0.1219] 31/38 | -0.15 [-0.43,+0.14] 13/38 | -0.74 [-2.16,+0.61] 4/38 | +0.0539 [+0.0347,+0.0729] 24/38 |

### Seed noise floor: SAME recipe, LightGBM seed 7 vs 42 (any real effect must clear this)

| variant | protocol | spearman | spearman_rel | mae | hauler_mae | top10 | cap | bias |
|---|---|---|---|---|---|---|---|---|
| seed7 | H25 | -0.0003 [-0.0009,+0.0003] 18/38 | -0.0014 [-0.0030,+0.0001] 16/38 | +0.0021 [+0.0003,+0.0039] 14/38 | +0.0190 [+0.0073,+0.0307] 10/38 | -0.18 [-0.43,+0.08] 14/38 | -0.55 [-1.55,+0.29] 5/38 | -0.0003 [-0.0024,+0.0017] 21/38 |
| seed7 | H24 | +0.0002 [-0.0008,+0.0011] 19/38 | +0.0017 [-0.0003,+0.0038] 22/38 | +0.0014 [-0.0006,+0.0037] 17/38 | -0.0124 [-0.0261,+0.0015] 24/38 | -0.08 [-0.32,+0.15] 17/38 | -1.50 [-3.68,+0.63] 5/38 | +0.0048 [+0.0021,+0.0074] 29/38 |
| seed7 | pooled | -0.0001 [-0.0007,+0.0005] 37/76 | +0.0001 [-0.0012,+0.0015] 38/76 | +0.0017 [+0.0003,+0.0032] 31/76 | +0.0033 [-0.0067,+0.0132] 34/76 | -0.13 [-0.30,+0.04] 31/76 | -1.03 [-2.18,+0.16] 10/76 | +0.0022 [+0.0004,+0.0040] 50/76 |

### Early-stopping / final-fit strategy (full season, vs fixed H baseline)

| variant | protocol | spearman | spearman_rel | mae | hauler_mae | top10 | cap | bias |
|---|---|---|---|---|---|---|---|---|
| refit | H25 | +0.0006 [-0.0001,+0.0013] 19/38 | +0.0003 [-0.0018,+0.0025] 17/38 | -0.0021 [-0.0039,-0.0003] 24/38 | +0.0222 [+0.0117,+0.0331] 14/38 | -0.02 [-0.26,+0.22] 16/38 | -0.84 [-2.26,+0.42] 4/38 | -0.0052 [-0.0075,-0.0031] 8/38 |
| refit | H24 | +0.0007 [-0.0001,+0.0015] 23/38 | +0.0029 [+0.0008,+0.0049] 27/38 | -0.0017 [-0.0042,+0.0007] 26/38 | -0.0136 [-0.0269,+0.0001] 23/38 | -0.26 [-0.52,+0.02] 11/38 | -1.92 [-3.55,-0.53] 4/38 | +0.0013 [-0.0011,+0.0036] 19/38 |
| refit | pooled | +0.0007 [+0.0001,+0.0012] 42/76 | +0.0016 [+0.0001,+0.0031] 44/76 | -0.0019 [-0.0034,-0.0004] 50/76 | +0.0043 [-0.0052,+0.0136] 37/76 | -0.14 [-0.31,+0.04] 27/76 | -1.38 [-2.41,-0.43] 8/76 | -0.0020 [-0.0038,-0.0002] 27/76 |

### Early-stopping rounds per position (best_iter)

| variant | GK | DEF | MID | FWD | n_train | n_val | secs | spearman |
|---|---|---|---|---|---|---|---|---|
| dec07_H24 | 96 | 200 | 132 | 132 | 145827 | 6800 | 233 | 0.7131 |
| dec07_H25 | 148 | 116 | 147 | 95 | 173461 | 6085 | 389 | 0.7199 |
| dec08_H24 | 120 | 155 | 135 | 106 | 145827 | 6800 | 264 | 0.7152 |
| dec08_H25 | 125 | 118 | 121 | 138 | 173461 | 6085 | 1296 | 0.7212 |
| dec09_H24 | 107 | 157 | 115 | 101 | 145827 | 6800 | 366 | 0.7172 |
| dec09_H25 | 118 | 153 | 121 | 110 | 173461 | 6085 | 567 | 0.7220 |
| from2018_H24 | 118 | 144 | 161 | 118 | 125561 | 6800 | 151 | 0.7122 |
| from2018_H25 | 90 | 136 | 105 | 90 | 153195 | 6085 | 149 | 0.7204 |
| from2020_H24 | 84 | 129 | 133 | 101 | 93018 | 6800 | 341 | 0.7100 |
| from2020_H25 | 89 | 128 | 97 | 75 | 120652 | 6085 | 311 | 0.7207 |
| half_H24_w1 | 138 | 217 | 132 | 113 | 158570 | 6800 | 448 | 0.7153 |
| half_H24_w1_full | 532 | 628 | 625 | 700 | 158570 | 6800 | 412 | 0.7183 |
| half_H24_w3 | 97 | 116 | 147 | 114 | 158570 | 6800 | 389 | 0.7123 |
| half_H24_w6 | 116 | 218 | 131 | 113 | 158570 | 6800 | 143 | 0.7114 |
| half_H25_w1 | 75 | 114 | 100 | 107 | 187636 | 6085 | 311 | 0.7277 |
| half_H25_w1_full | 486 | 613 | 687 | 489 | 187636 | 6085 | 1039 | 0.7302 |
| half_H25_w3 | 106 | 181 | 97 | 95 | 187636 | 6085 | 362 | 0.7238 |
| half_H25_w6 | 132 | 117 | 130 | 110 | 187636 | 6085 | 101 | 0.7251 |
| last2_H24 | 140 | 165 | 136 | 106 | 145827 | 6800 | 234 | 0.7156 |
| last2_H25 | 104 | 158 | 162 | 111 | 173461 | 6085 | 366 | 0.7208 |
| last2_LIVE | 125 | 96 | 235 | 165 | 202632 | 6252 | 123 | 0.6773 |
| refit_H24 | 100 | 178 | 131 | 103 | 145827 | 6800 | 441 | 0.7174 |
| refit_H24_full | 556 | 694 | 639 | 522 | 145827 | 6800 | 856 | 0.7196 |
| refit_H25 | 91 | 119 | 145 | 109 | 173461 | 6085 | 558 | 0.7240 |
| refit_H25_full | 525 | 621 | 641 | 671 | 173461 | 6085 | 686 | 0.7264 |
| refit_LIVE | 96 | 258 | 215 | 162 | 202632 | 6252 | 437 | 0.6798 |
| refitIS_H24 | 122 | 212 | 133 | 140 | 149038 | 6800 | 702 | 0.7308 |
| refitIS_H25 | 133 | 121 | 141 | 98 | 177049 | 6085 | 440 | 0.7591 |
| roll_H24_k10 | 140 | 118 | 153 | 106 | 152379 | 6800 | 108 | 0.7111 |
| roll_H24_k20 | 94 | 159 | 127 | 135 | 159281 | 6800 | 116 | 0.7166 |
| roll_H24_k30 | 97 | 240 | 123 | 91 | 166661 | 6800 | 116 | 0.7159 |
| roll_H25_k10 | 114 | 155 | 185 | 141 | 180772 | 6085 | 126 | 0.7426 |
| roll_H25_k20 | 118 | 183 | 132 | 106 | 188426 | 6085 | 104 | 0.7307 |
| roll_H25_k30 | 112 | 106 | 140 | 97 | 196547 | 6085 | 110 | 0.7311 |
| sanity_H25 | 91 | 119 | 145 | 109 | 173461 | 6085 | 157 | 0.7234 |
| seed7_H24 | 86 | 183 | 153 | 118 | 145827 | 6800 | 117 | 0.7147 |
| seed7_H24_full | 568 | 827 | 744 | 490 | 145827 | 6800 | 503 | 0.7191 |
| seed7_H25 | 143 | 193 | 147 | 84 | 173461 | 6085 | 119 | 0.7230 |
| seed7_H25_full | 479 | 569 | 629 | 586 | 173461 | 6085 | 383 | 0.7255 |
| valIS_H24 | 131 | 156 | 125 | 70 | 152627 | 3211 | 347 | 0.7304 |
| valIS_H25 | 82 | 142 | 165 | 90 | 179546 | 3588 | 253 | 0.7528 |
| valISrefit_H24 | 131 | 156 | 125 | 70 | 152627 | 3211 | 460 | 0.7324 |
| valISrefit_H25 | 82 | 142 | 165 | 90 | 179546 | 3588 | 453 | 0.7584 |
| win3_IS_H24 | 95 | 218 | 134 | 110 | 149038 | 6800 | 266 | 0.7305 |
| win3_IS_H25 | 96 | 155 | 103 | 111 | 177049 | 6085 | 270 | 0.7507 |
