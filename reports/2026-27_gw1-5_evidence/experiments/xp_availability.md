# xp_availability: how to encode xP and availability point-in-time

Agent `xp_availability`, 2026-09-24/25. The baseline is `fpl_xp_lag` (V2, from `harness.load_features()`),
compared against `EXP/baseline_fixed/{H25,H24,LIVE}_{fast,full}.parquet`.

## Verdict

**No challenger wins. Keep `fpl_xp_lag` (V2) as it is:**
- lag = the previous row's vaastav xP
- GW1 = NaN
- a missing scrape = NaN (no carry-forward)
- live value = the pre-deadline snapshot `gw{n}_bootstrap.json` `ep_this`

Serve-time checks show that V2 holds up under every input it could plausibly get live:
- **FPL's honest `ep_next`:** prediction corr 0.9996, per-GW Spearman +0.0017 (4 of 4 GWs won). Players only move when FPL has just flagged them.
- **The offset-free 2026-27 EP regime:** no change, measured on both full holdouts.
- **A missing snapshot:** +0.0004.

| option | verdict | evidence (per-GW Spearman vs the fixed baseline) |
|---|---|---|
| (a) V1: drop xP | **loser** | H24 fast -0.0043 [-0.0059,-0.0028] 7/31; H24 full -0.0022 [-0.0031,-0.0013] 8/30; on GWs where the lag exists it loses on both holdouts (H25 full -0.0025, 3/7; H24 full -0.0024, 7/27). Flagged-player MAE +0.052 on both holdouts |
| (b) V2: `fpl_xp_lag` | **keep** | reference |
| (c) fixture-adjusted lag (V3a/V3b/V3c) | neutral | 4-seed average: V3a -0.0004 / -0.0005, V3c -0.0007 / +0.0002 (H25 / H24). V3c full: -0.0003 / +0.0001 |
| (d) availability + form decomposition (V4a/b/c) | neutral | all within ±0.0017 (the seed-noise band); the explicit flag improves flagged-player MAE by 0.01-0.03 but doesn't move the ranking |
| (e) missing scrapes: carry forward from GW-2 (V5a) / ffill + age (V5b) | NaN stays | V5a H25 -0.0013 [-0.0021,-0.0003] 9/29; V5b +0.0006 / -0.0003 |

One alternative is still open. V3a is the offset-free lag, which matches the 2026-27 FPL EP formula. It is neutral on both
holdouts but +0.0043 [+0.0008,+0.0081] on LIVE (averaged over 4 seeds; after the pool availability layer
+0.0077 at seed 42). LIVE has only 5 GWs, so this is not a basis for a decision. Re-check it at the ~GW8-12 retrain once
LIVE has more than 10 GWs.

## 1. What vaastav's xP actually is (`formula_check.py`)

FPL form is the mean `total_points` of the player's fixture rows with kickoff in the last 30 days, to 1 decimal. This
reproduces bootstrap `form` for 100% of players in the 2026-27 snapshots (GW3-5). The deadline form recomputed from
merged_gw (`prep.py`) also matches the snapshot `form` 100% for GW1-5.

I rebuilt xP(n) = Σ_f (form(t) + 0.5·(str_team − str_opp)) at several candidate times t. The table gives the share of xP > 0 rows matched exactly, 2020-21 to 2025-26:

| form taken at | with fixture offset | no offset |
|---|---|---|
| deadline n (pre-GW, the honest time) | 0.163 | 0.074 |
| end of GW n + 1 day | **0.501** | 0.276 |
| deadline n+1 − 1h | 0.466 | 0.257 |

The best fit is 0-12 h after GW n's last kickoff (2021-22: 0.58; 2024-25: 0.46). This confirms the leak independently:
xP(n) is computed from form that includes GW n. So the lag at row n is
**(form after GW n-1 + GW n-1's fixture offset) × cop_this(n-1)**.

**The 2026-27 regime is different.** Team `strength` is now null, and the offsets are gone:
- `ep_next − form` has median 0 and IQR 0 in snapshots 3-5.
- `ep_this == form` for 92-94% of players.
- `ep_this == ep_next` for 93-97% of players.

`ep_next` differs from `ep_this` only through `chance_of_playing_next_round` (fresh) versus `_this_round` (frozen at the
previous deadline). Example: element 84 at the GW4 snapshot (calf injury, status i) has ep_this 3.3 and ep_next 0.

**Univariate per-GW Spearman against the target** (rows where the lag exists) shows that the offset is noise for fringe players:

| season | lag | offset removed (V3a) | ep_next-like (V3c) | + GW-n offset (V3b) | form at deadline | pts_rolling_5 |
|---|---|---|---|---|---|---|
| 2021-22 | 0.556 | 0.622 | 0.626 | 0.577 | 0.656 | 0.654 |
| 2024-25 | 0.626 | 0.676 | 0.676 | 0.634 | 0.706 | 0.702 |
| 2025-26 | 0.649 | 0.678 | 0.678 | 0.655 | 0.737 | 0.729 |
| 2026-27 | 0.643 | 0.643 | 0.669 (= ep_next) | 0.617 | 0.667 | 0.671 |

The standalone gains are large, but the model gains nothing from them: the xP "form part" is redundant with
`pts_rolling_*`. The unique value of xP is the availability zero, which is why V1 costs about 0.002-0.004.

## 2. Variants (all point-in-time; definitions in `variants.py`)

**Notation:**
- lag = V2's `fpl_xp_lag`
- `off(n)` = Σ over GW-n fixtures of 0.5·(str_team − str_opp)
- `k(n)` = number of fixtures in GW n
- `c` = cop implied at GW n-1 = lag / (k(n-1)·form_end(n-1) + off(n-1)), rounded to the nearest 0.25. It is 0 when lag = 0 and 1 when undefined.

**Variant definitions:**
- **V1:** drop the xP feature.
- **V3a `xp_nooff`:** lag − off(n-1)·c. Live: `ep_this` (FPL 2026-27 has no offset).
- **V3c `xp_nextlike`:** k(n)/k(n-1)·xp_nooff. Live: `ep_next` (native).
- **V3b `xp_nextrecon`:** V3c + off(n)·c, the task's "remove last GW's offset, add the upcoming one". Live: `ep_next` + own off(n).
- **V4a:** lag plus two availability features:
  - `xp_cop` = the unrounded implied cop (NaN when the denominator is < 0.25)
  - `xp_out` = 1 when lag == 0 even though FPL form ≥ 0.5 (flagged at deadline n-1)
- **V4b:** `form_dl` + `xp_cop` + `xp_out`, no lag (fully decomposed). `form_dl` is FPL form at deadline n; live it is the snapshot `form`.
- **V4c:** lag + `form_dl`.
- **V5a:** a lag whose previous scrape is missing takes the GW-2 value.
- **V5b:** lag forward-filled over any gap, plus `xp_age`.

### Fast mode, all GWs (paired per GW against the fixed fast baseline; `analyze.py`, `make_tables.py`)

| variant | H25 d_spearman [95% CI] won/lost | H25 d_top10 | H25 d_cap | H24 d_spearman [95% CI] won/lost | H24 d_top10 | H24 d_cap |
|---|---|---|---|---|---|---|
| V1 | +0.0009 [-0.0006, +0.0024] 24/14 | -0.07 | -0.29 | **-0.0043 [-0.0059, -0.0028] 7/31** | +0.28 | +0.24 |
| V2 seed 1 (noise) | -0.0007 [-0.0019, +0.0004] 16/22 | +0.14 | -0.95 | -0.0011 [-0.0030, +0.0007] 17/21 | -0.08 | +1.21 |
| V2 seed 2 (noise) | -0.0007 [-0.0021, +0.0008] 16/22 | -0.08 | -0.55 | -0.0021 [-0.0037, -0.0005] 10/28 | -0.11 | +1.45 |
| V2 seed 3 (noise) | -0.0001 15/23 | +0.11 | -1.42 | -0.0011 13/25 | -0.05 | -0.82 |
| V3a | -0.0005 [-0.0016, +0.0006] 19/19 | -0.03 | -0.45 | -0.0010 [-0.0021, +0.0002] 16/22 | +0.14 | +0.97 |
| V3c | -0.0004 [-0.0014, +0.0007] 17/21 | -0.26 | -1.24 | +0.0009 [-0.0005, +0.0024] 20/18 | +0.32 | +1.66 |
| V3b | -0.0017 [-0.0033, -0.0002] 16/22 | -0.09 | +0.03 | +0.0011 [-0.0002, +0.0024] 23/15 | -0.07 | +0.29 |
| V4a | -0.0005 [-0.0018, +0.0006] 15/23 | -0.13 | -1.03 | -0.0016 [-0.0030, -0.0002] 14/24 | +0.16 | +0.95 |
| V4b | -0.0017 [-0.0028, -0.0006] 14/24 | -0.04 | -1.18 | +0.0005 [-0.0009, +0.0020] 20/18 | +0.16 | +0.66 |
| V4c | -0.0008 [-0.0020, +0.0003] 17/21 | +0.14 | -1.50 | -0.0001 [-0.0014, +0.0012] 19/19 | +0.00 | +1.74 |
| V5a | -0.0013 [-0.0021, -0.0003] 9/29 | -0.01 | -0.53 | -0.0003 [-0.0015, +0.0010] 19/19 | +0.21 | -0.66 |
| V5b | +0.0006 [-0.0004, +0.0017] 22/16 | -0.10 | -0.66 | -0.0003 [-0.0018, +0.0013] 15/23 | -0.03 | -0.34 |

**Training-seed noise is as large as every effect here.** Re-running V2 with seeds 1-3 moves the H24 score by -0.0011 to -0.0021,
with up to 10/28 GWs "lost" and a bootstrap CI that excludes 0. The ≥65%-GWs rule and the per-GW bootstrap CI can
therefore be met by seed noise alone.

**Seed-averaged (4 seeds each; `seed_avg.py`), delta against the V2 4-seed average:**

| | H25 | H24 | LIVE (5 GWs) |
|---|---|---|---|
| V3a | -0.0004 [-0.0010, +0.0002] 18/20 | -0.0005 [-0.0011, +0.0002] 16/22 | +0.0043 [+0.0008, +0.0081] 4/1 |
| V3c | -0.0007 [-0.0013, -0.0002] 14/24 | +0.0002 [-0.0005, +0.0009] 21/17 | +0.0043 [-0.0005, +0.0092] 4/1 |
| V1 (1 seed) | +0.0013 | -0.0033 [-0.0047, -0.0018] 8/30 | +0.0011 |

**Split by whether the lag exists in the test GW.** 2025-26 has xP scrapes for only GW1-6, 8, 9, 24, 29 and 38, so on H25
only 10 GWs can see the feature:
- On lag GWs, V1 loses on both holdouts: fast H25 -0.0030 [-0.0055,-0.0004] 2/8, H24 -0.0040 7/27.
- On no-lag GWs, V1 wins on H25 (+0.0023 [+0.0006,+0.0040] 22/6). A model trained without xP handles a missing xP
  better than the lag model's NaN branch, which learnt NaN mostly from 2016-19. This is a training-data gap. It doesn't
  matter live, where every deadline has a snapshot.

### Full mode (PARAMS_FULL, against `baseline_fixed/*_full.parquet`)

| variant | H25 all | H25 lag GWs | H24 all | H24 lag GWs | top10 H25/H24 | cap H25/H24 |
|---|---|---|---|---|---|---|
| V1 | +0.0010 [-0.0001,+0.0019] 26/12 | **-0.0025 [-0.0045,-0.0008] 3/7** | **-0.0022 [-0.0031,-0.0013] 8/30** | -0.0024 7/27 | -0.17 / -0.10 | -0.74 / -1.55 |
| V3c | -0.0003 [-0.0010,+0.0004] 16/22 | -0.0001 | +0.0001 [-0.0006,+0.0007] 20/18 | +0.0002 | -0.17 / -0.18 | -0.13 / -1.63 |

## 3. Serve-time robustness on LIVE (`live_serve.py`, `live_naive.py`, `op_live.py`, `regime_sim.py`)

The V2 model is trained on 2016-17..2025-26 and scored on 2026-27 GW1-5. The xP slot is filled in several ways:

| served value | per-GW Spearman | d vs ep_this | pred corr | rows with \|Δpred\|>0.5 (GW2-5) |
|---|---|---|---|---|
| `ep_this` (default = the lag) | 0.6725 | — | — | — |
| **`ep_next` (honest pre-deadline EP)** | 0.6742 | **+0.0017 [+0.0006,+0.0027] 4/0**, MAE -0.003 (4/0) | 0.9996 | 0.23% |
| `ep_next` also at GW1 (pre-season EP; training GW1 is NaN) | 0.6768 | +0.0043 5/0 | 0.9990 | 0.23% |
| NaN (snapshot missing) | 0.6729 | +0.0004 3/1; hauler_mae +0.064 (0/4) | 0.9977 | 1.4% |
| naive shift of the live merged_gw xP (= `ep_next` of snapshot n-1; only 49% of values identical) | 0.6735 | +0.0011; top10 -0.34 (0/3) | — | — |

**The predictions stay sensible.** All of the top movers under `ep_next` are players FPL had just flagged:
- Jensen (i): 3.18 → 1.64
- Elanga (i): 1.66 → 0.67
- Foden (s, suspended): 1.45 → 0.64

46 player-GWs have `ep_next == 0` while `ep_this ≥ 0.5`, and every one of them scored 0. Under `ep_next` their mean
prediction falls from 1.19 to 0.96.

The operational layer (`live/pool.py`) already scales predictions by `chance_of_playing_next_round` and zeroes statuses
i/s/u/n. After that layer, `ep_this` and `ep_next` give **identical** operational Spearman: 0.7057 vs 0.7057. So using
`ep_next` is a safe fallback and brings no operational gain.

**Operational LIVE results** (after the pool layer, seed 42):

| encoding | per-GW Spearman | d vs baseline |
|---|---|---|
| V2 (`ep_this`) | 0.7057 | — |
| V1 | 0.7130 | +0.0073 |
| V3a | 0.7134 | +0.0077 |
| V3c (`ep_next`) | 0.7108 | +0.0050 |

The V2 seeds alone span 0.0067 raw on LIVE, so these differences are inside the noise.

**2026-27 regime simulated on history** (`regime_sim.py`): the V2 model is served the offset-free lag at test time, as it
will be all season now that FPL has dropped the offsets.

| holdout | V2 served offset-free vs V2 | V3a vs V2 served offset-free |
|---|---|---|
| H24 | +0.0001 [-0.0002,+0.0004] 22/13 | -0.0011 |
| H25 | +0.0000 | -0.0005 |

The offset change FPL made does not affect the model, so there is no parity reason to retrain on V3a.

## 4. Where the encodings differ (`subgroups.py`, pooled MAE change against the fixed baseline)

| group (pre-match state) | H24 n | V1 | V4a | V4b | V5b | V2 seeds 1/2 |
|---|---|---|---|---|---|---|
| flagged (lag 0, form ≥ 0.5) | 2396 | +0.052 | -0.027 | -0.029 | +0.004 | +0.002 / +0.002 |
| lag missing | 2920 | -0.008 | +0.001 | -0.000 | -0.018 | +0.008 / +0.006 |

The same groups on H25:
- **Flagged players (n = 779):** V1 +0.052, V4a -0.010. The baseline over-predicts them: 0.36 predicted vs 0.21 actual on H24.
- **Double GWs (n ≈ 400):** every variant is within the seed spread (±0.04).

## 5. Point-in-time proof and live servability of every constructed input

| input | how it is point-in-time | what serves it live |
|---|---|---|
| `fpl_xp_lag` | xP(n-1): computed after GW n-1 and before deadline n; groupby (season, code) `shift(1)` | snapshot `gw{n}` `ep_this` (FPL's value for GW n-1 at deadline n) |
| `off(n-1)`, `k(n-1)`, `form_end(n-1)` | GW n-1 fixtures and points (kickoff ≤ end of GW n-1 + 5 h); used only on row n | 2026-27: offset is 0 (FPL); `form_end` is the snapshot `form` |
| `off(n)`, `k(n)` | the fixture list and team strengths are known pre-season | fixtures.json + teams.csv (reconstructed strengths) |
| `form_dl(n)` | rows with kickoff ≤ min kickoff(GW n) − 90 min, which is before the deadline | snapshot `gw{n}` `form`; the merged_gw recomputation matches it 100% |
| `xp_cop`, `xp_out` | functions of the lag and the GW n-1 quantities above | `ep_this` / snapshot `form` (= `cop_this`) |
| V5 carry-forward | only xP(n-2) and earlier | never needed live: every deadline has a snapshot |

**Rejected on leakage grounds:** extracting `cop_this(n)` from the same-GW xP(n). The pre-deadline availability part is
clean, but xP(n) = 0 can also come from GW n's own points dragging form to ≤ 0, so it leaks GW-n outcomes.

## 6. Implementation (patch sketch for the recommended encoding; the repo is not touched)

These changes are the same as the xp_fix baseline plus the serving details this study checked.

1. `src/fpl_optimizer/prediction/features/players_raw.py::compute_players_raw_features`
   ```python
   xp = (merged_gw.assign(xP=pd.to_numeric(merged_gw["xP"], errors="coerce"))
         .groupby(["element", "GW"])["xP"].mean().rename("_xp").reset_index())  # DGW rows share one value
   gw_abs = xp["_xp"].fillna(0).abs().groupby(xp["GW"]).transform("sum")
   xp.loc[gw_abs == 0, "_xp"] = np.nan          # missing-scrape GW -> NaN (V5a carry-forward loses; V5b neutral)
   xp = xp.sort_values(["element", "GW"])
   xp["fpl_xp_lag"] = xp.groupby("element")["_xp"].shift(1)   # GW1 -> NaN
   if "xP_prev" in merged_gw.columns:           # live season: snapshot-served value
       live = merged_gw.groupby(["element", "GW"])["xP_prev"].first()
       xp = xp.set_index(["element", "GW"]); xp["fpl_xp_lag"] = pd.to_numeric(live, errors="coerce"); xp = xp.reset_index()
   base = base.merge(xp[["element", "GW", "fpl_xp_lag"]], on=["element", "GW"], how="left")
   ```
   In `FEATURE_COLS`, replace `"fpl_xp"` with `"fpl_xp_lag"`. Then retrain with `scripts/train_predictor.py`.
2. `src/fpl_optimizer/data/collectors/fpl_live.py`: add `_load_snapshot_ep(field)` next to `_load_snapshot_xp`. It
   returns (element, gw) → `ep_this` from `gw{gw}_bootstrap.json`, or NaN when that is null (GW1). Write it as column
   `xP_prev` on every GW-n row: the history rows (`round == gw`) and the synthetic upcoming rows.
   **Do not** derive the live lag by shifting merged_gw `xP`. For GW n-1 that column holds snapshot n-1's `ep_next`,
   which is one GW staler in form; only 49% of the values are identical.
   **Fallback:** if `gw{n}` has no `ep_this`, use `ep_next` of the same snapshot (tested safe: corr 0.9996). Otherwise
   use NaN. Keep GW1 as NaN.
3. **No other feature changes.** `form_dl`, `xp_cop`/`xp_out` and the fixture-adjusted versions are all neutral.
4. **At the ~GW8-12 retrain:** re-run `variants.py fast LIVE V3a` plus seeds (V3a is the only encoding with a LIVE hint).
   Consider backfilling 2025-26 xP (73% of its GWs are missing).

## Files (all under `EXP/xp_availability/`)

**Scripts:**
- `formula_check.py`: what vaastav xP is, and when it was computed.
- `prep.py` → `aux.parquet`: k, off, form_dl, form_end, snapshot ep/cop/status.
- `variants.py`: builds and trains every variant (fast or full; H25, H24, LIVE; seeds via `_sN`).
- `analyze.py`, `make_tables.py`: paired deltas by subset. Outputs: `deltas_{fast,full}.csv`, `analyze_{fast,full}.txt`, `tables_fast.md`.
- `seed_avg.py`: seed-averaged comparison → `seed_avg_fast.csv`.
- `subgroups.py`: MAE by availability group → `subgroups_fast.csv`.
- `live_serve.py`: serve-time robustness on LIVE → `live_serve_fast.json`.
- `live_naive.py`: the naive live implementation pitfall → `live_naive_fast.json`.
- `op_live.py`: LIVE after the pool availability layer → `op_live.csv`.
- `regime_sim.py`: the 2026-27 offset-free regime on history → `regime_sim_fast.json`.

**Per-row predictions** have the baseline schema plus `pred`:
- `{H25,H24,LIVE}_{fast,full}_{variant}[_sN].parquet`
- `LIVE_fast_{enc}_serve-*.parquet`
- `{H24,H25}_fast_serve-V2-nooff.parquet`
