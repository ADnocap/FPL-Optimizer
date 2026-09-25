---
name: retrain
description: Retrain the FPL point-prediction model of record with the latest data, evaluate it honestly, and promote it for live use. Use mid-season (e.g. after ~GW8, at the January window, or when prediction quality degrades) or when the user asks to retrain/improve the model.
---

# Retrain the point predictor

## Steps

1. **Refresh current-season data** (needs all completed GWs on disk, final
   scores: 09:00 UK the day after the GW's last match):
   ```
   python scripts/gameweek.py --fresh-squad   # cheap way to rebuild 2026-27 files
   ```
   Understat per-match is NOT needed: those features are excluded from the
   model (never servable live, and worthless once the same-day leak was fixed).

2. **Retrain** (eval + live check + production fit):
   ```
   python scripts/train_predictor.py --out models/prod_2026-27 \
       --features-cache <scratch>/features.parquet --rebuild-cache --second-fold
   ```
   - The recipe is `RECIPE` in the script (override with `--recipe '<json>'`):
     minutes-blend predictor, huber + min_child_samples 50, calibration on
     the val split, EXCLUDED_FEATURES (unservable + h2h odds + reconstructed
     team strengths) left out, and the current season's
     completed GWs added to TRAIN automatically (`include_current_season`;
     early stopping stays on the last 8 GWs of 2025-26 — never extend
     PROD_SEASONS with the in-progress season).
   - `--rebuild-cache` is required whenever GWs were played since the cache
     was built or the feature pipeline changed (a version mismatch is refused).
   - EVAL holds out 2025-26 (and 2024-25 with `--second-fold`); compare per-GW
     Spearman, top-10 and captain against the previous training_report.json
     before promoting. The "without xP" split is the honest view of GWs where
     fpl_xp_lag is missing (2025-26 after GW5).

3. **Promote**: automatic and atomic — the script trains into a staging dir,
   keeps the outgoing model at `models/prod_2026-27.prev` (rollback), then
   renames. Compare `training_report.json` against `prod_2026-27.prev/`'s; if
   the new model is worse, roll back by swapping the directories. Any model
   dir loads through `fpl_optimizer.prediction.minutes.load_predictor`
   (dispatches on metadata `kind`).

## Quality bars (history — only compare like-for-like)

- Models trained on vaastav's same-GW `fpl_xp` (every model before 2026-09-25,
  now in `models/archive/`): leak-inflated — never compare against them. Served
  as they run live they scored per-GW Spearman ~0.67-0.69 on holdouts.
- **Model of record v5 (2026-09-25, `models/prod_2026-27`)**, full params,
  121 features (minutes blend + huber + calibration; understat/FBref-only/legacy,
  h2h odds and reconstructed strengths excluded; 2026-27 GW1-5 in TRAIN):
  per-GW Spearman **0.742** (holdout 2025-26), **0.732** (2024-25, `--second-fold`),
  **0.711** live 2026-27 GW1-5 (model trained on complete seasons only);
  MAE 0.949 / 0.988; hauler MAE ~4.9 (the known weakness).
- Rollback `models/prod_2026-27.prev`: same recipe with the 3 strength features
  (124 features), 0.742 / 0.732 / 0.708.

**Promotion gate** (the new training_report.json vs the current model's):
per-GW Spearman on 2025-26 and 2024-25 within -0.003 of the current model or
better, AND the LIVE check (current-season completed GWs) not worse by more than
0.01 (few GWs = noisy). A retrain that only adds new 2026-27 rows should move
these by less than ±0.005. Anything outside that: investigate before promoting
(look at the data-health block of a dry `gameweek.py` run with the new model).

## Rules that bite

- `fpl_xp` (vaastav's same-GW xP) is a LEAK; only `fpl_xp_lag` is a feature.
- Per-group rolling only (shift+groupby per element) — never global rolling.
- Minutes labels (`min_total`, `min_max`, `mcls`) are the row's own GW
  outcome: they are in `_NON_FEATURE_COLS` and must stay there.
- 2025-26+ includes DEFCON scoring — keep 2025-26 and the current season in
  the training set (uniform weights; upweighting the new era lost in tests).
