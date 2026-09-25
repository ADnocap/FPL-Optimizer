"""Tests for the minutes-history features, the minutes model / blend predictor,
calibration, predictor loading and the training recipe (model-v5)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl_optimizer.prediction.feature_pipeline import FeaturePipeline
from fpl_optimizer.prediction.features.minutes_history import (
    MH_FEATURES,
    MINUTES_LABELS,
    add_depth_and_availability,
    compute_minutes_history_features,
)
from fpl_optimizer.prediction.id_resolver import IDResolver
from fpl_optimizer.prediction.minutes import (
    MinutesBlendPredictor,
    MinutesModel,
    OOF_KEY,
    STACK,
    expanding_minutes_oof,
    load_predictor,
)
from fpl_optimizer.prediction.model import (
    POSITIONS,
    PointPredictor,
    apply_calibration,
    fit_calibration,
)

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _NoResolver:
    """Minutes history only asks the resolver for PREVIOUS seasons' codes."""

    def code_from_element_id(self, season, element_id):
        return None


def _merged_gw(n_players: int = 12, n_gw: int = 10, seed: int = 0) -> pd.DataFrame:
    """Synthetic merged_gw: 3 teams x 4 players, one DGW, a mid-season move."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(1, n_players + 1):
        team = 1 + (e - 1) % 3
        for gw in range(1, n_gw + 1):
            if e == 5 and gw == 6:
                team = 3  # transfer
            n_fix = 2 if (gw == 4 and team == 1) else 1
            for _ in range(n_fix):
                mins = int(rng.choice([0, 0, 25, 70, 90, 90]))
                rows.append({
                    "element": e, "GW": gw, "code": 1000 + e, "team": team,
                    "minutes": mins, "starts": int(mins >= 60),
                    "total_points": 0 if mins == 0 else int(rng.integers(1, 9)),
                })
    return pd.DataFrame(rows)


def _mh_frame(mg: pd.DataFrame) -> pd.DataFrame:
    """compute + depth/availability, driven like FeaturePipeline does."""
    out = compute_minutes_history_features(mg, Path("unused"), "2099-00", _NoResolver())
    pos = {e: POSITIONS[(e - 1) % 4] for e in out["element"].unique()}
    out["position"] = out["element"].map(pos)
    out["value"] = 50.0 + out["element"]
    out["pts_rolling_3"] = 2.0
    out["fpl_xp_lag"] = np.where(out["element"] % 5 == 0, 0.0, 3.0)
    return add_depth_and_availability(out)


def _synthetic_training_frame(seasons=("2021-22", "2022-23", "2023-24"), n_players=48,
                              n_gw=10, seed=1) -> pd.DataFrame:
    """Rows with the columns the minutes blend needs (labels + a few features)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for e in range(1, n_players + 1):
            nailed = rng.uniform()
            for gw in range(1, n_gw + 1):
                u = rng.uniform()
                mcls = 2 if u < nailed * 0.9 else (1 if u < nailed * 0.9 + 0.2 else 0)
                mins = {0: 0, 1: 30, 2: 90}[mcls]
                target = 0 if mcls == 0 else (1 if mcls == 1 else 2) + rng.poisson(1.5 * nailed)
                rows.append({
                    "season": s, "element": e, "GW": gw, "code": 5000 + e,
                    "position": POSITIONS[e % 4], "target": float(target),
                    "min_total": float(mins), "min_max": float(mins), "mcls": mcls,
                    "mins_rolling_3": 90 * nailed + rng.normal(0, 10),
                    "pts_rolling_3": 3 * nailed + rng.normal(0, 0.5),
                    "value": 45 + 40 * nailed,
                    "was_home": float(gw % 2),
                    "mh_lag1": 90 * nailed + rng.normal(0, 20),
                    "mh_p60_r5": nailed + rng.normal(0, 0.1),
                    "noise": rng.normal(),
                })
    return pd.DataFrame(rows)


_TINY = {"n_estimators": 25, "learning_rate": 0.1, "min_child_samples": 5,
         "num_leaves": 7, "verbose": -1, "num_threads": 1, "seed": 3}
_TINY_MM = {"min_child_samples": 10, "num_leaves": 7, "num_threads": 1}


def _split(df):
    last = df["season"].max()
    val = (df["season"] == last) & (df["GW"] > 7)
    return df[~val].copy(), df[val].copy()


@pytest.fixture(scope="module")
def blend_and_data():
    df = _synthetic_training_frame()
    tr, va = _split(df)
    m = MinutesBlendPredictor(params=_TINY, mm_rounds=15, mm_params=_TINY_MM,
                              early_stopping_rounds=5)
    m.train(tr, va)
    m.fit_calibration(va)
    return m, df, tr, va


def _load_train_predictor():
    """scripts/train_predictor.py as a module (not a package)."""
    spec = importlib.util.spec_from_file_location(
        "train_predictor_under_test", REPO / "scripts" / "train_predictor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# minutes-history features
# ---------------------------------------------------------------------------


class TestMinutesHistoryPointInTime:
    @pytest.mark.parametrize("n", [1, 2, 4, 6, 9, 10])
    def test_truncation_future_rows_removed_or_perturbed(self, n: int) -> None:
        """Features of GW <= n do not change when GW > n rows vanish, or when
        every GW >= n minutes/starts value (GW n's own outcome too) changes."""
        mg = _merged_gw()
        full = _mh_frame(mg).set_index(["element", "GW"]).sort_index()

        removed = _mh_frame(mg[mg["GW"] <= n]).set_index(["element", "GW"]).sort_index()
        perturbed_mg = mg.copy()
        later = perturbed_mg["GW"] >= n
        rng = np.random.default_rng(n)
        perturbed_mg.loc[later, "minutes"] = rng.integers(0, 91, later.sum())
        perturbed_mg.loc[later, "starts"] = rng.integers(0, 2, later.sum())
        perturbed = _mh_frame(perturbed_mg).set_index(["element", "GW"]).sort_index()

        keep = full.index.get_level_values("GW") <= n
        for other in (removed, perturbed):
            a = full.loc[keep, MH_FEATURES]
            b = other.loc[a.index, MH_FEATURES]
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12)
        # ...while the GW n labels DID change in the perturbed copy (sanity:
        # the test really perturbed the rows the features must not read)
        gw_n = full.index.get_level_values("GW") == n
        assert not full.loc[gw_n, "min_total"].equals(perturbed.loc[gw_n, "min_total"])

    def test_labels_are_the_rows_own_minutes(self) -> None:
        mg = _merged_gw()
        out = _mh_frame(mg)
        agg = mg.groupby(["element", "GW"])["minutes"].agg(["sum", "max"]).reset_index()
        m = out.merge(agg, on=["element", "GW"])
        assert (m["min_total"] == m["sum"]).all() and (m["min_max"] == m["max"]).all()
        expect = np.select([m["sum"] <= 0, m["sum"] < 60], [0, 1], 2)
        assert (m["mcls"] == expect).all()
        # the first GW has no history
        first = out[out["GW"] == 1]
        assert first["mh_lag1"].isna().all() and (first["mh_rows_prev"] == 0).all()

    def test_depth_chart_ranks_within_team_and_position(self) -> None:
        out = _mh_frame(_merged_gw())
        g = out[(out["GW"] == 5)].dropna(subset=["dp_rank"])
        for _, grp in g.groupby(["position"]):
            assert grp["dp_rank"].min() >= 1
        assert out["av_xp_is0"].isin([0.0, 1.0]).all()
        assert "_mh_team" not in out.columns and "_mh_score" not in out.columns

    def test_pipeline_emits_features_and_labels(self, pred_data_dir: Path) -> None:
        from tests.test_prediction.test_feature_pipeline import _make_merged_gw

        raw = pred_data_dir / "raw" / "2023-24"
        (raw / "gws").mkdir(parents=True)
        rows = []
        for gw in range(1, 5):
            for eid, team, mins in ((14, 1, 90), (24, 2, 0 if gw == 3 else 70)):
                rows.append({"element": eid, "GW": gw, "team": team, "minutes": mins,
                             "total_points": 2 if mins else 0, "opponent_team": 3 - team,
                             "fixture": gw, "was_home": team == 1,
                             "kickoff_time": f"2023-08-{10 + gw * 7} 15:00:00"})
        _make_merged_gw(rows).to_csv(raw / "gws" / "merged_gw.csv", index=False)
        pd.DataFrame({"id": [14, 24], "element_type": [4, 3], "team": [1, 2]}).to_csv(
            raw / "cleaned_players.csv", index=False)
        df = FeaturePipeline(pred_data_dir, IDResolver(pred_data_dir), ["2023-24"]).build()
        for col in MH_FEATURES + MINUTES_LABELS:
            assert col in df.columns, col
        salah = df[df["element"] == 24].set_index("GW")
        assert salah.at[3, "mcls"] == 0 and salah.at[4, "mh_lag1"] == 0
        assert salah.at[4, "mh_zero_streak"] == 1 and salah.at[2, "mh_lag1"] == 70
        assert df["dp_rank"].notna().all()  # positions known -> depth chart


# ---------------------------------------------------------------------------
# minutes model / blend predictor
# ---------------------------------------------------------------------------


class TestMinutesModel:
    def test_labels_never_features(self, blend_and_data) -> None:
        m, _, _, _ = blend_and_data
        for comp in (m.e60, m.esub, m.stack):
            assert not set(MINUTES_LABELS) & set(comp.feature_names)
            assert "target" not in comp.feature_names
        assert not set(MINUTES_LABELS) & set(m.minutes.features)
        assert not set(MINUTES_LABELS) & set(m.feature_names)
        # stacked model: nested minutes probabilities, no minutes-history
        assert set(STACK) <= set(m.stack.feature_names)
        assert not set(MH_FEATURES) & set(m.stack.feature_names)
        assert "mh_lag1" in m.e60.feature_names
        # mm_* are computed inside the predictor, never asked of the pipeline
        assert not set(STACK) & set(m.feature_names)

    def test_point_predictor_ignores_labels(self) -> None:
        df = _synthetic_training_frame(seasons=("2023-24",))
        p = PointPredictor(params=_TINY)
        p.train(df)
        assert not set(MINUTES_LABELS) & set(p.feature_names)

    def test_expanding_oof_is_nested(self) -> None:
        df = _synthetic_training_frame()
        oof = expanding_minutes_oof(df, n_rounds=10, params=_TINY_MM)
        first = df["season"] == "2021-22"
        assert oof.loc[first].isna().all().all()
        assert oof.loc[~first].notna().all().all()
        assert ((oof.loc[~first] >= 0) & (oof.loc[~first] <= 1)).all().all()
        # season S only depends on seasons < S: perturb the last season's labels
        pert = df.copy()
        last = pert["season"] == "2023-24"
        pert.loc[last, "mcls"] = 2 - pert.loc[last, "mcls"]
        oof2 = expanding_minutes_oof(pert, n_rounds=10, params=_TINY_MM)
        pd.testing.assert_frame_equal(oof.loc[~last], oof2.loc[~last])

    def test_precomputed_oof_matches_internal(self, blend_and_data) -> None:
        """A superset OOF (extra later season) reproduces the internal one."""
        m, df, tr, va = blend_and_data
        sup = pd.concat([df, _synthetic_training_frame(seasons=("2024-25",), seed=9)],
                        ignore_index=True)
        oof = pd.concat([sup[OOF_KEY], expanding_minutes_oof(sup, 15, _TINY_MM)], axis=1)
        m2 = MinutesBlendPredictor(params=_TINY, mm_rounds=15, mm_params=_TINY_MM,
                                   early_stopping_rounds=5)
        m2.train(tr, va, minutes_oof=oof)
        m2.fit_calibration(va)
        np.testing.assert_allclose(m2.predict(va), m.predict(va), rtol=1e-10)
        with pytest.raises(ValueError):
            m2.train(tr, va, minutes_oof=oof[oof["season"] != "2022-23"])

    def test_blend_is_sensible(self, blend_and_data) -> None:
        m, df, _, va = blend_and_data
        p = m.predict(va)
        assert np.isfinite(p).all()
        assert np.corrcoef(p, va["target"])[0, 1] > 0.3
        comp = m.predict_components(va)
        assert comp["p_play"].between(0, 1).all() and (comp["p60"] <= comp["p_play"] + 1e-9).all()

    def test_save_load_round_trip_exact(self, blend_and_data, tmp_path: Path) -> None:
        m, df, _, _ = blend_and_data
        m.save(tmp_path / "blend")
        meta = json.loads((tmp_path / "blend" / "metadata.json").read_text())
        assert meta["kind"] == "minutes_blend" and meta["calibration"] == m.calibration
        loaded = load_predictor(tmp_path / "blend")
        assert isinstance(loaded, MinutesBlendPredictor)
        assert loaded.feature_names == m.feature_names
        np.testing.assert_array_equal(loaded.predict(df), m.predict(df))
        np.testing.assert_array_equal(loaded.predict_raw(df), m.predict_raw(df))

    def test_refit_keeps_rounds_and_calibration(self, blend_and_data) -> None:
        m, df, tr, va = blend_and_data
        m2 = MinutesBlendPredictor(params=_TINY, mm_rounds=15, mm_params=_TINY_MM,
                                   early_stopping_rounds=5)
        m2.train(tr, va)
        cal = m2.fit_calibration(va)
        best = m2.best_iterations()
        m2.refit(tr, va)
        assert m2.calibration == cal
        for name in ("e60", "esub", "stack"):
            comp = getattr(m2, name)
            for pos, b in comp._models.items():
                assert b.current_iteration() == best[name][pos]
        assert np.isfinite(m2.predict(va)).all()


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_monotone_quadratic(self) -> None:
        rng = np.random.default_rng(0)
        raw = rng.uniform(0, 4, 5000)
        y = 0.1 + 1.2 * raw + 0.15 * raw ** 2 + rng.normal(0, 1, raw.size)
        cal = fit_calibration(raw, y)
        grid = np.linspace(-10, 20, 3001)
        out = apply_calibration(grid, cal)
        assert np.all(np.diff(out) >= 0)
        inside = apply_calibration(np.linspace(0, 4, 500), cal)
        assert np.all(np.diff(inside) > 0)  # strictly increasing on the data range
        assert cal["coef"][0] > 0  # picked up the curvature

    def test_falls_back_to_linear_when_quadratic_bends(self) -> None:
        raw = np.linspace(0, 5, 400)
        y = 3 * raw - 0.4 * raw ** 2  # concave: a quadratic would turn down
        cal = fit_calibration(raw, y)
        assert cal["coef"][0] == 0.0 and cal["coef"][1] > 0 and cal["vertex"] is None
        assert np.all(np.diff(apply_calibration(np.linspace(-5, 10, 100), cal)) > 0)

    def test_identity_when_absent(self, tmp_path: Path) -> None:
        raw = np.array([-1.0, 0.0, 2.5])
        np.testing.assert_array_equal(apply_calibration(raw, None), raw)
        df = _synthetic_training_frame(seasons=("2023-24",))
        p = PointPredictor(params=_TINY)
        p.train(df)
        assert p.calibration is None
        np.testing.assert_array_equal(p.predict(df), p.predict_raw(df))
        p.fit_calibration(df)
        assert not np.array_equal(p.predict(df), p.predict_raw(df))
        p.save(tmp_path / "m")
        q = PointPredictor.load(tmp_path / "m")
        assert q.calibration == p.calibration
        np.testing.assert_array_equal(q.predict(df), p.predict(df))


# ---------------------------------------------------------------------------
# loading legacy models
# ---------------------------------------------------------------------------


class TestLoadPredictor:
    def test_legacy_point_dir_loads_and_predicts_identically(self, tmp_path: Path) -> None:
        df = _synthetic_training_frame(seasons=("2023-24",))
        p = PointPredictor(params=_TINY)
        p.train(df)
        p.save(tmp_path / "legacy")
        meta_path = tmp_path / "legacy" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        for k in ("kind", "calibration"):  # what models saved before v5 look like
            meta.pop(k)
        meta_path.write_text(json.dumps(meta))
        loaded = load_predictor(tmp_path / "legacy")
        assert type(loaded) is PointPredictor and loaded.calibration is None
        np.testing.assert_array_equal(loaded.predict(df), p.predict(df))

    def test_unknown_kind_raises(self, tmp_path: Path) -> None:
        (tmp_path / "metadata.json").write_text(json.dumps({"kind": "nope"}))
        with pytest.raises(ValueError):
            load_predictor(tmp_path)

    @pytest.mark.parametrize("name", ["prod_2026-27", "prod_2026-27.prev"])
    def test_production_models_are_servable_and_leak_free(self, name: str) -> None:
        """Whatever kind the promoted model and its rollback are, they load via
        load_predictor, never need the leaky same-GW fpl_xp, ship the serving
        reference the data-health check uses, and predict finite values."""
        from fpl_optimizer.live.predict import SERVING_REFERENCE_FILE, check_not_leaky

        d = REPO / "models" / name
        if not (d / "metadata.json").exists():
            pytest.skip(f"{d} not present")
        model = load_predictor(d)
        check_not_leaky(model, d)
        assert "fpl_xp_lag" in model.feature_names
        assert (d / SERVING_REFERENCE_FILE).exists()
        rng = np.random.default_rng(0)
        feats = model.feature_names
        df = pd.DataFrame(rng.normal(1, 1, (40, len(feats))), columns=feats)
        df["position"] = [POSITIONS[i % 4] for i in range(40)]
        got = model.predict(df)
        assert got.shape == (40,) and np.isfinite(got).all()


# ---------------------------------------------------------------------------
# training recipe (scripts/train_predictor.py)
# ---------------------------------------------------------------------------


class TestRecipe:
    def test_exclusion_list_removes_columns(self) -> None:
        from fpl_optimizer.prediction.feature_sets import (
            EXCLUDED_FEATURES, H2H_ODDS, RECONSTRUCTED_STRENGTH, UNSERVABLE_FEATURES,
        )

        tp = _load_train_predictor()
        recipe = tp.resolve_recipe(None)
        assert recipe["exclude_features"] == EXCLUDED_FEATURES
        assert set(EXCLUDED_FEATURES) == (set(UNSERVABLE_FEATURES) | set(H2H_ODDS)
                                          | set(RECONSTRUCTED_STRENGTH))
        assert "xg_rolling_5" in UNSERVABLE_FEATURES and "goals_vs_xg_5" in UNSERVABLE_FEATURES
        # h2h odds: neutral but a fragile live dependency -> excluded by default
        assert all(c.startswith("odds_team") for c in H2H_ODDS)
        df = _synthetic_training_frame(seasons=("2023-24",))
        for c in ["xg_rolling_5", "prev_sot_per90", "dribbles_rolling_5", "odds_team_win_prob"]:
            df[c] = 1.0
        out, dropped = tp.apply_exclusions(df, recipe)
        assert set(dropped) == {"xg_rolling_5", "prev_sot_per90", "dribbles_rolling_5",
                                "odds_team_win_prob"}
        assert not set(dropped) & set(out.columns)
        p = PointPredictor(params=_TINY)
        p.train(out)
        assert not set(EXCLUDED_FEATURES) & set(p.feature_names)

    def test_recipe_override_merges_params(self) -> None:
        tp = _load_train_predictor()
        r = tp.resolve_recipe('{"refit": true, "params": {"learning_rate": 0.05, "alpha": null}}')
        assert r["refit"] is True and r["params"]["learning_rate"] == 0.05
        assert "alpha" not in r["params"] and r["params"]["objective"] == "huber"
        assert tp.RECIPE["params"]["learning_rate"] == 0.01  # default untouched
        with pytest.raises(SystemExit):
            tp.resolve_recipe('{"bogus": 1}')

    def test_default_recipe(self) -> None:
        tp = _load_train_predictor()
        r = tp.RECIPE
        assert r["kind"] == "minutes_blend" and r["calibrate"] and r["include_current_season"]
        assert not r["refit"]
        assert r["params"]["objective"] == "huber" and r["params"]["min_child_samples"] == 50

    def test_current_season_rows_go_to_train_not_val(self) -> None:
        tp = _load_train_predictor()
        cur = tp.CURRENT_SEASON
        rows = []
        for season, n_gw in [(tp.COMPLETE_SEASONS[-2], 38), (tp.COMPLETE_SEASONS[-1], 38),
                             (cur, 5)]:
            for gw in range(1, n_gw + 1):
                rows.append({"season": season, "GW": gw, "target": 1.0})
        df = pd.DataFrame(rows)
        train, val = tp.prod_split(df, include_current=True)
        assert (train["season"] == cur).sum() == 5 and (val["season"] != cur).all()
        assert set(val["season"]) == {tp.COMPLETE_SEASONS[-1]}
        assert sorted(val["GW"].unique()) == list(range(31, 39))
        train_no, val_no = tp.prod_split(df, include_current=False)
        assert (train_no["season"] != cur).all() and val_no.equals(val)
        with pytest.raises(ValueError):
            tp._split_val(df, tp.COMPLETE_SEASONS + [cur])


# ---------------------------------------------------------------------------
# serving health with the blend's feature names
# ---------------------------------------------------------------------------


class TestServingHealth:
    @staticmethod
    def _frame(gws, x_const: bool) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        for gw in gws:
            for e in range(60):
                rows.append({"season": "2025-26", "GW": gw, "mins_rolling_3": 90.0,
                             "gw_phase": gw / 38.0,
                             "x": 1.0 if x_const else rng.normal(),
                             "mh_lag1": np.nan if x_const else 90.0})
        return pd.DataFrame(rows)

    def test_within_gw_constant_features_not_flagged(self) -> None:
        from fpl_optimizer.live.predict import serving_health, serving_reference

        names = ["gw_phase", "x", "mh_lag1"]
        ref = serving_reference(self._frame(range(1, 11), x_const=False), names)
        assert ref["gw_phase"]["const_gw_share"] == 1.0
        assert ref["x"]["const_gw_share"] == 0.0
        live = self._frame([12], x_const=True)
        warns = serving_health(live, names, ref)
        assert any(w.startswith("x: constant live") for w in warns)
        assert any(w.startswith("mh_lag1: 0% populated") for w in warns)
        assert not any(w.startswith("gw_phase") for w in warns)
        # references written before const_gw_share existed keep the old rule
        old = {f: {k: v for k, v in r.items() if k != "const_gw_share"} for f, r in ref.items()}
        assert any(w.startswith("gw_phase") for w in serving_health(live, names, old))

    def test_blend_feature_names_are_pipeline_columns(self, blend_and_data) -> None:
        from fpl_optimizer.live.predict import serving_health

        m, df, _, _ = blend_and_data
        gw = df[(df["season"] == df["season"].max()) & (df["GW"] == 5)]
        assert not any("absent from the pipeline" in w
                       for w in serving_health(gw, m.feature_names, None))


# ---------------------------------------------------------------------------
# horizon predictions with the minutes blend
# ---------------------------------------------------------------------------


def test_horizon_carries_minutes_history_forward(blend_and_data) -> None:
    from fpl_optimizer.prediction.horizon import (
        FixtureContext,
        build_horizon_rows,
        predict_horizon,
    )

    m, df, _, _ = blend_and_data
    fixtures = pd.DataFrame([
        {"id": 1, "event": 1, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4},
        {"id": 2, "event": 2, "team_h": 2, "team_a": 1, "team_h_difficulty": 4, "team_a_difficulty": 2},
        {"id": 3, "event": 3, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4},
    ])
    merged = pd.DataFrame([
        {"element": e, "GW": 1, "fixture": 1, "was_home": e % 2 == 1, "team": 1 + e % 2,
         "opponent_team": 2 - e % 2, "total_points": 2, "minutes": 90, "goals_conceded": 1}
        for e in range(1, 9)
    ])
    ctx = FixtureContext(merged, fixtures)
    last = df[df["season"] == df["season"].max()]
    asof = last[(last["GW"] == 2) & last["element"].between(1, 8)].copy()
    rows = build_horizon_rows(asof, ctx, t=2, horizon=2)
    k0 = rows[rows["k"] == 0].set_index("element")
    k1 = rows[rows["k"] == 1].set_index("element")
    assert len(k1) == 8
    # as-of-t minutes history is carried to the future rows unchanged
    for col in ["mh_lag1", "mh_p60_r5", "mins_rolling_3"]:
        pd.testing.assert_series_equal(k1[col], k0.loc[k1.index, col])
    assert (k1["GW"] == 3).all()
    out = predict_horizon(m, asof, ctx, t=2, horizon=2)
    assert set(out["k"]) == {0, 1} and np.isfinite(out["pred"]).all()
    # the minutes model sees the same player state -> close P(play) at k=0/1
    p0 = m.predict_components(k0.reset_index())["p_play"].to_numpy()
    p1 = m.predict_components(k1.loc[k0.index].reset_index())["p_play"].to_numpy()
    assert np.max(np.abs(p0 - p1)) < 0.35


def test_serving_health_early_season_and_missing_sparse_source():
    """bps_form_delta-like features are constant early in training too (no false
    alarm at GW6); a sparse source that vanished entirely (props) IS flagged."""
    import numpy as np
    import pandas as pd

    from fpl_optimizer.live.predict import serving_health, serving_reference

    rng = np.random.default_rng(0)
    rows = []
    for season in ("2024-25", "2025-26"):
        for gw in range(2, 30):
            for e in range(60):
                rows.append({"season": season, "GW": gw, "mins_rolling_3": 90.0,
                             # constant until GW7 (windows still filling), varies after
                             "delta": 0.0 if gw <= 7 else rng.normal(),
                             # props: quoted for ~1/3 of rows
                             "props_xg": rng.random() if e < 20 else np.nan})
    ref = serving_reference(pd.DataFrame(rows), ["delta", "props_xg"])
    assert ref["delta"]["const_gw_share_early"] == 1.0

    def live(gw, props):
        return pd.DataFrame([{"season": "2026-27", "GW": gw, "mins_rolling_3": 90.0,
                              "delta": 0.0, "props_xg": (0.1 + i / 100 if props else np.nan)}
                             for i in range(60)])

    assert serving_health(live(6, True), ["delta", "props_xg"], ref) == []
    late = serving_health(live(12, True), ["delta", "props_xg"], ref)
    assert any(w.startswith("delta: constant live") for w in late)
    missing = serving_health(live(6, False), ["delta", "props_xg"], ref)
    assert any(w.startswith("props_xg: EMPTY live") for w in missing)
