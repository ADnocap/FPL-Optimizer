"""Named feature groups for training recipes.

"Train what you serve": a feature that is always NaN at a live deadline must
not be in the model — the model learns to lean on it from training rows where
it is populated, then sees NaN on every live row (silent train/serve skew;
the 2026-27 GW1-5 run was served 26 such features). Retraining without the
live-dead features beat the model served as live on both holdouts
(feature-groups experiment, 2026-09-25: full mode +0.003 per-GW Spearman).
"""

from __future__ import annotations

# Understat per-match rolling features (features/understat.py) and the two
# derived over/under-performance features built on them. Understat is not
# collected for the live season (the repo collector re-downloads ~420 players
# and never refreshes; not worth building: once the same-day leak was fixed,
# dropping understat entirely was neutral, +0.0007 per-GW Spearman pooled).
UNDERSTAT_ROLLING = [
    "xg_rolling_3", "xg_rolling_5", "xg_rolling_10",
    "xa_rolling_3", "xa_rolling_5", "xa_rolling_10",
    "npxg_rolling_5", "npxg_rolling_10", "shots_rolling_5", "key_passes_rolling_5",
    "xgchain_rolling_5", "xgchain_rolling_10", "xgbuildup_rolling_5",
    "goals_vs_xg_5", "assists_vs_xa_5",
]

# Prior-season features that only FBref provides (FotMob covers the other
# three prior-season FBref features). FBref is Cloudflare-blocked since 2026,
# so these are NaN for every season from 2025-26 on.
PRIOR_FBREF_DEAD = ["prev_sot_per90", "prev_tkl_int_per90", "prev_gls_per90"]

# vaastav detailed stats that exist only for 2016-17..2018-19: NaN in every
# season we evaluate or serve.
LEGACY_DETAIL = [
    "fpl_key_passes_rolling_5", "completed_passes_rolling_5",
    "big_chances_created_rolling_5", "dribbles_rolling_5",
]

# h2h match-odds features (features/odds.py). Servable (football-data.co.uk,
# Odds API fallback) but neutral for the model (+0.0002 per-GW Spearman,
# combiner 2026-09-25) and they are populated in EVERY training season, so a
# missed live refresh (football-data lists a weekend ~2 days ahead; the office
# network blocks the Odds API) would put the model out of distribution.
# Excluded: same accuracy, one fewer live dependency. (Player props stay: they
# are NaN in all pre-2025-26 training rows, so a missing props snapshot is
# in-distribution.)
H2H_ODDS = ["odds_team_win_prob", "odds_team_draw_prob", "odds_team_loss_prob",
            "odds_team_strength"]

# Features that cannot be served at a live 2026-27 deadline.
UNSERVABLE_FEATURES = UNDERSTAT_ROLLING + PRIOR_FBREF_DEAD + LEGACY_DETAIL

# Excluded from training by default (scripts/train_predictor.py
# RECIPE["exclude_features"]).
EXCLUDED_FEATURES = UNSERVABLE_FEATURES + H2H_ODDS
