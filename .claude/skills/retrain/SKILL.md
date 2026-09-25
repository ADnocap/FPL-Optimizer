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
     the val split, unservable features excluded, and the current season's
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

- Old leaky-xP models (prod_2026-27 of 2026-08-22 and earlier): inflated by
  the same-GW fpl_xp leak — do not compare against them.
- Leak-fixed L2 baseline, fast params (lr 0.05): per-GW Spearman 0.723 on
  2025-26, 0.716 on 2024-25; full params 0.726 / 0.719.
- Model-v5 recipe (minutes blend + huber + calibration + train-what-you-serve),
  FAST smoke params (lr 0.05, 600 rounds) on the v3 feature cache: per-GW
  Spearman 0.741 on 2025-26 / 0.732 on 2024-25 vs 0.722 / 0.712 for the
  same-cache L2 point baseline; LIVE 2026-27 GW1-5 0.708 vs 0.681. Full-param
  numbers: see the production training_report.json.
- Anything materially worse than the current training_report.json = don't promote.

## Rules that bite

- `fpl_xp` (vaastav's same-GW xP) is a LEAK; only `fpl_xp_lag` is a feature.
- Per-group rolling only (shift+groupby per element) — never global rolling.
- Minutes labels (`min_total`, `min_max`, `mcls`) are the row's own GW
  outcome: they are in `_NON_FEATURE_COLS` and must stay there.
- 2025-26+ includes DEFCON scoring — keep 2025-26 and the current season in
  the training set (uniform weights; upweighting the new era lost in tests).
