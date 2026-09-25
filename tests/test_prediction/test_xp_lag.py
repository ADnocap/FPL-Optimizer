"""The xP feature must be point-in-time: only the PREVIOUS GW's xP is used.

vaastav's same-GW ``xP`` is FPL's ep_this recomputed after the GW with that
GW's own points folded into form (a leak, proved 2026-09-24). The live
collector must give the merged_gw ``xP`` column the same post-GW semantics so
that ``fpl_xp_lag`` means the same thing in training and at a deadline.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fpl_optimizer.data.collectors.fpl_live import LiveFPLCollector
from fpl_optimizer.prediction.features.players_raw import (
    FEATURE_COLS,
    compute_players_raw_features,
)


def _merged(rows):
    return pd.DataFrame(rows, columns=["element", "GW", "xP", "total_points", "minutes"])


def test_same_gw_xp_is_not_a_feature():
    assert "fpl_xp" not in FEATURE_COLS
    assert "fpl_xp_lag" in FEATURE_COLS


def test_lag_uses_previous_gw_only(tmp_path):
    merged = _merged([
        (1, 1, 3.0, 2, 90), (1, 2, 9.9, 15, 90), (1, 3, 4.0, 1, 90),
        (2, 1, 1.0, 1, 90), (2, 2, 2.0, 2, 90), (2, 3, 0.0, 0, 0),
    ])
    out = compute_players_raw_features(tmp_path, "2026-27", merged).set_index(["element", "GW"])
    assert np.isnan(out.loc[(1, 1), "fpl_xp_lag"])
    assert out.loc[(1, 2), "fpl_xp_lag"] == 3.0   # GW1's value, not GW2's 9.9
    assert out.loc[(1, 3), "fpl_xp_lag"] == 9.9
    # changing a GW's own xP never changes that GW's feature
    merged.loc[(merged.element == 1) & (merged.GW == 2), "xP"] = -5.0
    out2 = compute_players_raw_features(tmp_path, "2026-27", merged).set_index(["element", "GW"])
    assert out2.loc[(1, 2), "fpl_xp_lag"] == 3.0


def test_missing_scrape_gw_is_nan_not_zero(tmp_path):
    # the scraper missed GW2: every player has xP 0
    merged = _merged([
        (1, 1, 3.0, 2, 90), (1, 2, 0.0, 6, 90), (1, 3, 4.0, 1, 90),
        (2, 1, 1.0, 1, 90), (2, 2, 0.0, 2, 90), (2, 3, 2.0, 0, 90),
    ])
    out = compute_players_raw_features(tmp_path, "2026-27", merged).set_index(["element", "GW"])
    assert np.isnan(out.loc[(1, 3), "fpl_xp_lag"])
    assert np.isnan(out.loc[(2, 3), "fpl_xp_lag"])


def _snapshot(snap_dir, gw, current, ep_this, taken):
    events = [{"id": g, "is_current": g == current, "is_next": g == current + 1}
              for g in range(1, 6)]
    elements = [{"id": 7, "ep_this": ep_this, "ep_next": "9.9"}]
    (snap_dir / f"gw{gw}_bootstrap.json").write_text(
        json.dumps({"events": events, "elements": elements}), encoding="utf-8")
    (snap_dir / f"gw{gw}_meta.json").write_text(
        json.dumps({"gw": gw, "ep_field": "ep_next", "taken_utc": taken}), encoding="utf-8")


def test_live_xp_is_post_gw_ep_this_of_current_event(tmp_path):
    collector = LiveFPLCollector(data_dir=tmp_path, season="2026-27")
    snap = collector.snapshot_dir
    snap.mkdir(parents=True)
    # GW1 pre-deadline: pre-season, no current event -> nothing
    events = [{"id": g, "is_current": False, "is_next": g == 1} for g in range(1, 6)]
    (snap / "gw1_bootstrap.json").write_text(json.dumps(
        {"events": events, "elements": [{"id": 7, "ep_this": None, "ep_next": "5.0"}]}),
        encoding="utf-8")
    # taken before the GW3 deadline: current event is GW2 -> GW2's post-GW value
    _snapshot(snap, 3, current=2, ep_this="4.5", taken="2026-09-04T16:00:00+00:00")
    xp = collector._load_snapshot_xp()
    assert xp == {(7, 2): 4.5}  # never ep_next, never keyed to the upcoming GW


def test_later_snapshot_in_same_window_wins(tmp_path):
    collector = LiveFPLCollector(data_dir=tmp_path, season="2026-27")
    snap = collector.snapshot_dir
    snap.mkdir(parents=True)
    _snapshot(snap, 3, current=2, ep_this="4.5", taken="2026-09-04T16:00:00+00:00")
    # a manual re-snapshot saved under another name, taken earlier in the window
    _snapshot(snap, 30, current=2, ep_this="1.0", taken="2026-08-29T10:00:00+00:00")
    assert collector._load_snapshot_xp()[(7, 2)] == 4.5


def test_live_serving_refuses_models_trained_on_the_leaky_xp():
    """A pre-2026-09-25 model needs fpl_xp; serving it NaN gave nonsense plans."""
    import pytest

    from fpl_optimizer.live.predict import LeakyModelError, check_not_leaky

    class _Stub:
        _feature_names = ["pts_rolling_3", "fpl_xp"]

    with pytest.raises(LeakyModelError):
        check_not_leaky(_Stub(), "models/old")

    class _Fixed:
        _feature_names = ["pts_rolling_3", "fpl_xp_lag"]

    check_not_leaky(_Fixed(), "models/new")  # no error
