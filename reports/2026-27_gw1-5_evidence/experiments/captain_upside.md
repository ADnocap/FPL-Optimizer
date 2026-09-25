# captain_upside: captain rules and the top of the ranking

Agent `captain_upside`. Baseline: the leak-fixed-xP set (`fpl_xp_lag`) from `harness.load_features()`.
Every number below is a paired per-GW delta against **argmax(pred)** of the matching baseline, shown as:
`delta [95% bootstrap CI over GWs] GWs won/lost`.

## Bottom line

1. **Keep the point prediction as it is. Keep argmax(pred) as the captain rule.** No upside model (LightGBM
   quantile q80/q90, P(pts>=6) / P(pts>=10) classifiers), LightGBM lambdarank score, FPL EP, ownership, price or
   bookmaker-props rule beats argmax(pred) robustly on both H25 and H24 once the confound is removed. The
   confound is either the understat leak (the "honest" baseline) or fast-mode undertraining (FULL mode). In
   FULL mode every candidate captain blend is within +0.02 to +0.05 captain-points/GW of argmax(pred), and no
   CI excludes 0.
2. **Bookmaker props are *worse* captain signals than the model on 2025-26.** This holds on every baseline
   (fast, honest and full), for z-blends, shortlist rules and an OLS stack. It holds even though the 2025-26
   props backfill was snapshotted at kickoff-2h, which is after the FPL deadline for 82% of matches and so
   flatters the props. Props, ownership, price and EP only help when the model is **starved of the
   features it was trained on**. That is exactly the live situation today. At every 2026-27 deadline, 22
   features are 100% NaN:
   - understat per-match (never collected)
   - FBref `prev_*` (FBref is blocked)
   - h2h `odds_team_*`: `data/odds/2026-27.json` does not exist, and `gameweek.py` only collects props, not h2h.

   Serving the H25 model that way drops the captain pick from 8.13 to 4.87 pts/GW. That matches LIVE's 3.4.
3. The LIVE gains of the market/EP/ownership rules are a symptom of that serving gap. They are not
   evidence that those signals are better. **The fix is to serve or retrain the features** (the
   feature_groups agent's scope, see `ss_livedead`). The one cheap captain-only stop-gap that is never
   significantly negative in any regime is `epp@0.25`:
   `captain_points = 0.75*xP + 0.25*FPL ep_this`.
   - LIVE: +0.16 [+0.04,+0.29], 4/0 GWs
   - live-like H25+H24: +0.38 [+0.20,+0.58]
   - honest fast: +0.21 [+0.07,+0.37]
   - FULL: +0.02 [-0.14,+0.18]

   It is **not a WINNER by the rules**: on H25 the EP value exists for only 10 of 38 GWs, and there it is not
   significant. So verdict = inconclusive, and winner = none.

## Metrics and protocol

- Candidates = harness "relevant" rows (value>=50 or top-150 selected_norm).
  - `cap` = actual points of the rule's #1. This is the harness definition.
  - `top3` / `top10` = mean actual points of the rule's top-3 / top-10.
  - **`sim` (primary for captaincy)**: per GW, 2000 random 11-man squads are drawn from a rule-independent
    pool (top-40 ownership ∪ top-40 baseline pred). The captain is the rule's argmax inside each squad, and
    the score is the mean actual points.
- `cap` is too noisy to decide on. The *same* H24 season gives a base `cap` of 7.26 with the fast model and
  10.68 with the full model.
- Blend families:
  - `q80@w` and `epp@w` are points-unit blends, `(1-w)*pred + w*signal`.
  - The other `X@w` rules are z-blends over the GW's candidates: `(1-w)*z(pred) + w*z(X)`.
  - `slK:X` = among the top-K by pred (inside each squad), pick the max of X.
- Weights are chosen honestly:
  - Model and EP families: chosen on the other holdout season (the "cross" rows in `report.json`).
  - Props (2025-26 and 2026-27 only): 2-fold inside 2025-26 (GW1-19 and GW20-38).
- Upside models, trained per protocol with the same splits and features:
  - Per position, early stopping on the val split: q80, q90 (quantile), p6 = P(pts>=6), p10 = P(pts>=10).
  - One lambdarank model over all positions. Query = (season, GW). Relevance = 8 point bins.
    `label_gain` = bin-mean points. `truncation_level` = 40.
  - `rank_all` trains on all rows; `rank_rel` trains on candidate rows only.
  - Early stopping on 8 val queries stopped the ranker at iteration 4 (noise). The ranker therefore uses a fixed,
    a-priori 300 rounds (fast) / 1500 rounds (full).

## Baselines (the four regimes)

| regime | what | H25 sim / cap / top10 | H24 sim / cap / top10 |
|---|---|---|---|
| main fast | `baseline_fixed/*_fast` (official) | 5.62 / 8.13 / 5.33 | 6.08 / 7.26 / 5.85 |
| honest fast (usfix) | + understat same-day leak fixed (`usfix_baseline.py`) | 5.28 / 6.61 / 4.99 | 5.69 / 6.29 / 5.55 |
| full | `baseline_fixed/*_full` | 5.74 / 7.61 / 5.58 | 6.55 / 10.68 / 6.08 |
| live-like fast | 22 live-dead features NaN at test time (`livelike.py`) | 4.75 / 4.87 / 4.62 | 5.27 / 6.58 / 5.08 |
| LIVE (2026-27 GW1-5) | as served | sim 4.35 / cap 3.40 / top10 4.60 | |

**Understat leak.** The feature_groups agent found it, and I re-verified it:
- The cached `xg_rolling_3` equals their leaky rebuild exactly.
- 18% of 2023-26 rows differ from the fixed version.
- Spearman(target, `xg_rolling_3`) on 2024-25 is 0.288 leaky vs 0.270 fixed.

The leak costs almost nothing in overall Spearman (H25 -0.0012, H24 -0.0052). It inflates **the top**: vs
the official baseline, the honest baseline has top10 -0.33 / -0.34 and cap -1.53 / -0.97 on H25 / H24.
Leak-free signals (EP, props, ownership) were therefore competing against a model that peeks at the GW's
own xG.

## Results: captain rules (`sim` = captain pts/GW; pooled = H25+H24)

| rule | main fast pooled | honest fast H25 / H24 | FULL pooled | live-like pooled | LIVE |
|---|---|---|---|---|---|
| q80@0.25 | +0.14 [+0.05,+0.24] 45/31 | +0.03 (22/16) / **+0.21** [+0.08,+0.34] | +0.04 [-0.02,+0.10] 41/35 | n/a | +0.09 (3/2) |
| q80@0.5 | +0.14 [-0.00,+0.30] | -0.07 / +0.26 | +0.05 [-0.07,+0.16] | n/a | -0.26 |
| q90@0.25 (H25 only) | H25 +0.05 | n/a | n/a | n/a | n/a |
| p6@0.25 | +0.11 [+0.00,+0.20] | -0.05 / +0.06 | n/a | n/a | -0.33 |
| p10@0.25 (H25 only) | H25 -0.09; @1.0 -0.45 | n/a | n/a | n/a | n/a |
| rank_all@0.25 | +0.08 [+0.04,+0.13] 53/23 | +0.02 / +0.13 | +0.03 [-0.01,+0.07] 48/27 | n/a | -0.01 |
| rank_all@0.5 | +0.16 [+0.07,+0.26] 48/28 | +0.06 (22/15) / **+0.21** | +0.03 [-0.04,+0.10] 44/32 | n/a | +0.22 (4/1) |
| rank_all@0.75 | +0.23 [+0.06,+0.40] | -0.11 / +0.22 | -0.00 [-0.12,+0.12] | n/a | +0.44 |
| rank_rel@0.5 | +0.13 [+0.02,+0.24] | +0.05 / +0.22 | n/a | n/a | -0.11 |
| ep@0.25 (z) | +0.15 [+0.03,+0.28] | +0.05 (6/4) / **+0.36** [+0.13,+0.61] | +0.04 [-0.09,+0.17] | **+0.38** [+0.21,+0.57] | +0.24 (4/0) |
| **epp@0.25** (points) | +0.16 [+0.01,+0.32] | +0.07 (5/5) / **+0.36** | +0.02 [-0.14,+0.18] | **+0.38** [+0.20,+0.58] | +0.16 [+0.04,+0.29] 4/0 |
| own@0.25 | +0.02 [-0.19,+0.23] | +0.01 / +0.21 | -0.15 [-0.36,+0.07] | **+0.45** [+0.18,+0.72] | +0.91 (5/0) |
| price@0.25 | -0.06 | **-0.26** / +0.28 | **-0.29** [-0.46,-0.11] | +0.35 [+0.12,+0.60] | +0.98 |
| props_pts@0.25 (H25) | -0.07 (12/26) | -0.06 | -0.13 | +0.26 [+0.01,+0.54] | +0.48 (3/1) |
| props_xg@0.25 (H25) | -0.14 | -0.18 | **-0.29** | +0.22 | +0.47 (4/0) |
| sl3:props_pts (H25) | **-0.30** | **-0.24** | **-0.49** | +0.23 | +0.07 |
| props 2-fold, chosen w (H25) | xg -0.13, pts -0.07, conservative `_n` variants -0.23 / -0.25 | xg -0.20, pts -0.08 | n/a | n/a | n/a |
| OLS stack pred+props_pts | -0.01; props coefficient -0.18 / -0.02 | +0.01 | n/a | n/a | n/a |

Losers in every regime:
- q90 and p10, at any weight
- p6@>=0.75
- every signal used alone (`@1.0`)
- every `slK:` shortlist rule on H25
- props_xg at any weight

LIVE picks (actual pts):
- base: Bruno 2, Xhaka 6, Bruno 2, Rice 6, Rice 1 (mean 3.4)
- `epp@0.25`: Bruno 2, **Bruno 23**, Bruno 2, Saka 8, Bruno 2 (7.4)
- `props_pts@0.25`: Bruno 2, Haaland 13, Bruno 2, Rogers 8, Bruno 2 (5.4)

## Results: should the point prediction change? (`harness.compare`, pred replaced by the score)

| variant | regime | spearman H25 | spearman H24 | top10 H25 / H24 | cap H25 / H24 | MAE H25 / H24 |
|---|---|---|---|---|---|---|
| 0.75 pred + 0.25 q80 | fast | +0.0019 [+.0013,+.0026] 31/7 | +0.0027 [+.0022,+.0033] 34/4 | +0.02 / -0.04 | +0.13 / +1.32 | +0.027 / +0.039 |
| same | honest fast | +0.0021 32/6 | +0.0018 31/7 | -0.08 / +0.12 | +0.89 / +0.16 | +0.028 / +0.037 |
| same | **FULL** | +0.0007 [+.0002,+.0013] 24/14 | +0.0012 [+.0008,+.0017] 30/8 | -0.09 / +0.04 | +0.08 / -0.37 | **+0.030 / +0.043** (bias +0.11 / +0.13) |
| seed ensemble 0.75/0.25 (reference) | fast | +0.0011 31/7 | +0.0012 33/5 | +0.02 / -0.11 | -0.03 / 0.00 | -0.001 / -0.002 |
| rank_all@0.5 | fast / full | -0.0022 / -0.0039 | -0.0050 / -0.0072 | n/a | n/a | n/a |
| ep@0.25 / epp@0.25 | fast | -0.0007 / -0.0009 | -0.0015 / -0.0027 | n/a | n/a | n/a |

The q80 blend formally clears the Spearman bar in FULL mode, but the effect is tiny:
- About +0.001 per-GW Spearman, and it sits in the bulk: `spearman_rel` is +0.0004 / +0.0001.
- It does nothing for the top (top10, cap and `sim` are all noise).
- It costs +3-4% MAE, a uniform upward bias, and a second 4-booster model set.
- It is mostly an ensemble/robust-loss effect, which the hparams_ensemble agent covers directly (huber, bagging).

**Verdict: do not change the point prediction.** Ranking (lambdarank) and EP blends lower whole-ranking
Spearman, so they are captain-only rules if they are used at all.

## Why the LIVE / live-like numbers disagree with the holdouts

- **Serving gap.** Serve-sim of the fixed H25 model:
  - Spearman -0.0026 (the headline metric barely moves)
  - top10 -0.71, top30 -0.42
  - cap -3.26 (8.13 -> 4.87)
  - sim -0.87

  This is identical to feature_groups `ss_livedead`. Understat drives most of it, and part of that is the
  leak disappearing. Under these conditions every prior helps: EP, ownership, price and props all come out
  at +0.26 to +0.45 pooled `sim`.
- **Props timing.** 2025-26 props are kickoff-2h snapshots for every match (82% after the deadline). Live
  2026-27 snapshots are pre-deadline, 2.5-120 h before kickoff. Premium players (value>=70) who ended up
  with 0 minutes were still quoted 39% of the time in 2025-26, vs 65% in 2026-27: the backfill carries late
  team news. H25 props results are therefore *optimistic*, and they are still negative. See
  `props_timing.json`.

## Recommendation for the live optimizer

- Once the model is served with its training features (or retrained on the servable set), use the plain
  captain term. No change is needed.
- Until then, an optional stop-gap is `epp@0.25`, with a flag to switch it off:
  `captain_points = 0.75 * xP + 0.25 * ep_this`, where `ep_this` comes from the pre-deadline bootstrap
  snapshot.
  - It is never significantly negative: FULL +0.02, honest +0.21, fast +0.16.
  - It helps where the model is starved: live-like +0.38, LIVE +0.16 (4/0 GWs; captain mean 3.4 -> 7.4).
  - Ownership has larger live-like gains but is negative in FULL mode (-0.15, cap -1.41). Price is
    significantly negative (-0.29). Props are negative on every holdout.

Patch sketch. Point prediction untouched; the MILP only gets a separate captain score:
- `optimizer/types.py`: `PlayerCandidate` gets
  `captain_points: float | None = None  # captain-term score (pts); None -> predicted_points`.
- `optimizer/transfer_optimizer.py`:
  - Add `cp = [p.captain_points if p.captain_points is not None else p.predicted_points for p in all_cands]`.
  - In the objective, `pulp.lpSum(xp[i] * c_var[i])` becomes `pulp.lpSum(cp[i] * c_var[i])`.
  - The vice term becomes `VICE_CAPTAIN_WEIGHT * pulp.lpSum(cp[i] * v_var[i])`.
  - The triple-captain extra term becomes `pulp.lpSum(cp[i] * c_var[i])`.
  - XI, bench and hit terms keep `xp`.
- `optimizer/lineup_selector.py`: the same three substitutions (`c`, `v` use `cp`).
- `live/predict.py`:
  ```python
  CAPTAIN_EP_WEIGHT = 0.25
  def captain_points(preds, bootstrap, w=CAPTAIN_EP_WEIGHT):
      """(1-w)*model xP + w*FPL ep_this. At a pre-deadline run ep_this is exactly
      the training feature fpl_xp_lag (harness.fix_xp); tested on H24/H25/LIVE."""
      out = {}
      for el in bootstrap["elements"]:
          p = preds.get(el["id"])
          if p is None:
              continue
          ep = el.get("ep_this")
          out[el["id"]] = p if ep in (None, "") else (1 - w) * p + w * float(ep)
      return out
  ```
- `live/pool.py`: `build_live_candidates(..., captain_points=None)`.
  - Apply the same availability rules as `pts`: status-excluded -> 0.
  - Scale only the model part by `chance`, because `ep_this` already embeds `chance_of_playing`:
    `cp = chance*(1-w)*p + w*ep`.
- `scripts/gameweek.py`:
  - Add `--captain-ep-weight` (default 0.25; 0 disables and gives the exact old behaviour).
  - Compute `captain_points(predictions, bootstrap, w)` when not in `--ep` mode, and pass it to
    `build_live_candidates`.
  - The existing `--captain` / `--vice` overrides stay.
- `optimizer/backtest.py` (optional): the same `captain_points` from the `fpl_xp_lag` column, for replay.

Live servability and point-in-time:
- `ep_this` is read from the bootstrap snapshotted pre-deadline (`collector.snapshot_predeadline()`).
- `fpl_xp_lag` = the previous GW's vaastav xP. It is shifted within (season, code), and live it is exactly
  that snapshot's `ep_this`.
- No same-GW information is used. Props, ownership and price were tested but are not recommended.

## Files (all under EXP/captain_upside/)
- `train_extra.py`: q80/q90/p6/p10 and lambdarank scorers.
  - Flags: `--kinds`, `--suffix rank|extra`, `--usfix`, `--full`.
  - Output directories: `preds/` (fast), `preds_usfix/` (honest), `preds_full/` (full).
- `evaluate.py`: captain metrics (cap/top3/top10/sim), blends, shortlists, props 2-fold, OLS stack, cross-season weight choice, whole-ranking `compare`. Writes `report.json` and `captain_per_gw.parquet`.
- `cheap_preds.py`: model-free signals. `usfix_baseline.py`: honest baseline and `usfix_baseline.json`. `livelike.py`: serve-sim preds.
- `run_eval_{usfix,full,livelike}.py` write to `{usfix,full,livelike}_out/`. `summarize.py` writes `summary.json` and `logs/summary_tables.md`.
- `props_timing.py` writes `props_timing.json`. `export_rows.py` writes `rows/{P}_{fast,usfix,full}.parquet` (baseline schema + `pred_q80blend`, `cap_epp25`, `cap_rank50`, raw signals).
