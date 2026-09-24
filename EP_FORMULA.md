> **RETRACTED (2026-09-24). The conclusion of this document is wrong: vaastav's `xP` IS a
> same-gameweek leak.** FPL recomputes a finished GW's `ep_this` from form that already
> includes that GW's points; §4 below tested mid-GW (10 of 13 fixtures played), before
> the recompute. Direct proof on 2026-27 GW5: the pre-deadline `ep_next` snapshot vs
> `ep_this` read after the GW differ for 46% of players (Brobbey 1.8 → 5.5 after 17 pts,
> Palmer 6.5 → 3.8 after 2 pts); Spearman with GW5 points is 0.744 pre-deadline vs 0.855
> post-GW. Historically, players with vaastav xP ≥ 5 returned 5+ points 58-66% of the
> time vs 31% for the honest pre-deadline EP in 2026-27, and same-GW OLS weight on xP is
> 0.29-0.37 in every season 2020-21..2025-26. The model now uses `fpl_xp_lag` (the
> previous GW's xP, point-in-time safe; live = the pre-deadline snapshot's `ep_this`).
> Evidence: `reports/2026-27_gw1-5_review.md`. The text below is kept for the record.

# Is vaastav's `xP` column a post-match leak? No.

This document proves — structurally, empirically, and from source code — that the `xP` column in `vaastav/Fantasy-Premier-League`'s `merged_gw.csv` cannot contain post-match information about the gameweek it labels. It is a pre-match prediction (`ep_this`) captured via the official FPL API during that gameweek's `is_current=True` window. The only known failure mode produces **missing** values, not **leaked** values.

---

## 1. The claim in one paragraph

The FPL bootstrap-static API exposes each player as a single JSON object with one `ep_this` scalar. That scalar is computed at the GW deadline from pre-match inputs (form, team and opponent strength, chance of playing), stored, and never modified. It remains visible and unchanged throughout the entire `is_current=True` window — across the deadline, through every fixture being played, through `finished=True`, up until the next deadline, at which point the slot is overwritten with the next GW's pre-match prediction. Vaastav's scraper reads `ep_this` and the `is_current` flag from the same atomic API response, so the filename label and the values it contains always come from the same moment in time. There is no code path, no API endpoint, and no data flow by which a post-match value for GW N could end up in `xP{N}.csv`.

---

## 2. The two lifecycle boundaries (clear version)

`ep_this` has exactly two state transitions. Both are triggered by deadlines, never by matches finishing.

```
──────────────────────────────────────────────────────────────────────────────
  DEADLINE N                  matches play out                    DEADLINE N+1
       │                             │                                 │
       ▼                             ▼                                 ▼
┌──────┴─────────────────────────────┴─────────────────────────────────┴──────┐
│ ep_this computed,       ep_this STILL SHOWS the GW N deadline       ep_this │
│ frozen, visible         prediction, unchanged, even after all       slot    │
│ (is_current=True,       fixtures finish (is_current=True,           now     │
│  finished=False)        finished=True, data_checked=True)           holds   │
│                                                                     GW N+1  │
│                                                                     value.  │
│                                                                     GW N's  │
│                                                                     value   │
│                                                                     is      │
│                                                                     gone.   │
└─────────────────────────────────────────────────────────────────────────────┘
```

Key points:

- **Transition 1 (deadline N):** `ep_this` is set for GW N. This is a pure pre-match computation.
- **Between deadlines:** `ep_this` is **immutable**. Matches kick off, finish, bonus is awarded, `data_checked` flips — none of this touches `ep_this`.
- **Transition 2 (deadline N+1):** The single `ep_this` slot is overwritten with GW N+1's prediction. GW N's value is no longer queryable.

**"Persists after finished"** and **"no longer scrapable after the GW"** both hold — they refer to different boundaries. The first means ep_this stays the same _during_ the is_current window even once every match has been played. The second means that _after the next deadline_ (a later event) the slot is reused.

---

## 3. Why post-match data cannot reach `xP{N}.csv`

### 3.1 The API has no post-match slot for ep_this

The `bootstrap-static` response gives each player one `ep_this` field. There is no `ep_history`, no per-GW array, no post-match override. For the API to return a post-match value for GW N, FPL would have to _rewrite the live scalar_ after matches ended. They do not, and we prove it below in §4.

### 3.2 The scraper's filename and its values come from the same API call

```python
# global_scraper.py (vaastav/Fantasy-Premier-League)
data = get_data()                                # single API call

xPoints = []
for e in data["elements"]:
    xPoints.append({'id': e['id'], 'xP': e['ep_this']})  # values from this call

gw_num = 0
for event in data["events"]:
    if event["is_current"] == True:
        gw_num = event["id"]                     # label from the same call

if gw_num > 0:
    write(f'xP{gw_num}.csv', xPoints)            # label and values bound together
```

Because the label (`gw_num`) and the values (`ep_this` for every player) come from the same atomic response, it is impossible to produce an `xP{N}.csv` whose values correspond to anything other than what `is_current=True` was pointing at in that snapshot. No mis-labelling is mechanically possible.

### 3.3 The only failure mode is "not scraped at all"

If the scraper is not run inside GW N's `is_current=True` window, `xP{N}.csv` is never written. When the file is missing, the downstream `merge_gw` step leaves the `xP` column as zero/blank for those rows. This is a **gap**, not a **leak**. The three zero-xP gameweeks in vaastav's 2024-25 data (GW22, GW32, GW34) are exactly this: the scraper did not run during those windows, so the file does not exist. None of the rows in those GWs contains post-match data — they contain no data.

### 3.4 Pre-match-only inputs

The formula's inputs are `form` (30-day rolling average of past `total_points`, rounded to 0.5), team and opponent `strength` (season-level integers 1-5), and `chance_of_playing_*` (editorial 0/25/50/75/100 scale set from press conferences). None carry post-match information about the GW being predicted. Post-match quantities (`event_points`, `bonus`, `bps`, live `total_points`, updated `form`) live in separate fields and never feed `ep_this` once it has been frozen.

---

## 4. Empirical proof: live API caught mid-GW33

Captured 2026-04-21, 10/13 GW33 fixtures finished, 3 upcoming. Players whose team has played at least one match have their `form` field updated with the post-match points. If `ep_this` were dynamic, it would reflect the new `form` value. It does not.

For each single-fixture high-scorer, back-solving the formula (`ep_this = round((form + offset) × cop/100, 1)`) recovers the form value _as it was at the deadline_, not the current form:

| Player             | GW33 opp    | offset | form_now | ep_this (API) | form_deadline (back-solved) | Δ form | ep_this if dynamic |
| ------------------ | ----------- | ------ | -------- | ------------- | --------------------------- | ------ | ------------------ |
| Son (Spurs)        | Brighton    | 0      | 6.0      | **3.5**       | 3.5                         | +2.5   | 6.0 ≠ 3.5          |
| Rashford (Man Utd) | Chelsea     | 0      | 5.5      | **3.0**       | 3.0                         | +2.5   | 5.5 ≠ 3.0          |
| Salah (Liverpool)  | Everton     | +0.5   | 5.0      | **3.5**       | 3.0                         | +2.0   | 5.5 ≠ 3.5          |
| Isak (Newcastle)   | Bournemouth | 0      | 4.5      | **2.5**       | 2.5                         | +2.0   | 4.5 ≠ 2.5          |

All four players' teams had played by 04-21, each had event_points ≥ 15, and each had form that has demonstrably moved by +2.0 to +2.5 since the deadline. `ep_this` has moved by 0.0 in every case. The deadline-time form value is what `ep_this` was computed from; the API still returns that original scalar.

A bonus structural case: Man City played Arsenal in GW33 on 04-19 (single fixture at deadline). A second fixture (vs Burnley, 04-22) was added to GW33 after the deadline, making it a DGW for Man City. Haaland's `ep_this = 5.0` still matches a single-fixture computation at deadline-time form 5.5, ignoring both (a) his 22-point haul vs Arsenal and (b) the newly-added second fixture. Frozen in every dimension.

---

## 5. The correlation argument, done properly

[PR #222](https://github.com/vaastav/Fantasy-Premier-League/pull/222) reported three correlations to flag potential lookahead:

- live `ep_this` vs `form` ≈ 0.98 pre-deadline
- scraped `xP` vs `form` ≈ 0.75 in the dataset
- scraped `xP` rolling-3 vs same-GW `total_points` ≈ 0.40

These numbers are real but do not imply lookahead. The dataset is dominated by players who never play — reserves, u21s, injured players, squad filler — for whom both `xP` and `total_points` are (correctly) zero or near-zero. Every such row adds a (0, 0) datapoint that inflates `corr(xP, total_points)` without carrying any predictive content.

Computed on the vaastav 2024-25 `merged_gw.csv` (DGW-deduplicated, 3 missing-xP GWs excluded):

| Filter                     | N      | corr(xP, total_points) |
| -------------------------- | ------ | ---------------------- |
| All players                | 25,059 | **0.72**               |
| Played (minutes > 0)       | 10,565 | **0.59**               |
| Started (minutes ≥ 60)     | 7,144  | **0.55**               |
| Expected to start (xP ≥ 2) | 6,218  | **0.47**               |

A post-match leak would push correlation toward 1.0 for players who played — because xP would "know" the outcome. Instead the correlation **drops by a third** when you filter to players who actually played, exactly as expected for a pre-match estimate: easy "xP=0, scored 0" pairings vanish, and the task collapses to predicting a high-variance stochastic quantity from pre-match features.

Two complementary tests sharpen this:

- **Blanked starters.** Filter to rows with `xP ≥ 3` (confident pre-match prediction), `minutes ≥ 60` (actually started and played most of the match), `total_points ≤ 1` (didn't deliver). Result: **372 rows, mean xP = 4.1, mean total_points = 0.8.** A post-match signal would equal the outcome; xP over-predicts by 3.3 points on average. These are straightforward pre-match misses.
- **Unexpected hauls.** Filter to `xP < 3` and `total_points ≥ 10`. Result: **17 rows, mean xP = 2.1, mean actual = 11.2.** Pre-match underestimates; a leak would have captured these.

MAE between `xP` and same-GW `total_points` for players who played: **1.65 points.** A post-match-contaminated signal would have MAE ≈ 0.

The 0.75 figure for `corr(xP, form)` — low vs 0.98 for live `ep_this` — is also explained without invoking lookahead. `form` in `merged_gw.csv` is a post-scrape snapshot: for a given row, it usually reflects form _after_ that GW's matches, whereas `xP` was computed against deadline-time form. The two quantities are from different moments in the GW lifecycle. Correlating them across thousands of rows introduces a 30-day rolling lag, reducing the live 0.98 toward 0.75. This is an artefact of the merge, not of the source value.

---

## 6. Summary and recommendations

- The `xP` column in `merged_gw.csv` is the pre-match `ep_this` for the labelled gameweek.
- Structurally, the API cannot deliver post-match data to this field; empirically, it does not.
- The caveat flagged in PR #222 is real but operational: scrapes that miss the `is_current=True` window produce **missing** data (empty/zero xP), not **leaked** data.
- For ML users: shifting or dropping `xP` is not required to avoid lookahead. It is required only if you are worried about (a) missing-GW gaps treated as real zeros, or (b) DGW double-counting when the same xP value is stamped on both fixture rows.

Recommended practice:

1. Treat `xP` as a valid pre-match feature for the row's GW.
2. Detect and handle the known missing-GW gaps (3 GWs in 2024-25; check if others exist in other seasons by looking for `sum(xP) == 0` at the GW level).
3. Deduplicate DGW rows before aggregating: use `.first()` or `.max()` on `xP`, never `.sum()`.

---

## Appendix A: Approximate formula

The published FPL rule is approximately:

```
ep_this = round((form + fixture_offset) × chance_of_playing / 100, 1)
fixture_offset = (team_strength - opponent_strength) × 0.5
```

For double gameweeks, `ep_this` is computed per-fixture and summed.

Verified against the live API on 2026-04-06: 426/429 exact matches (99.3%) on players with non-zero `ep_next`. The three mismatches were newly added 0-minute players with a small base expectation injected by FPL's system.

The definition of `form` used in the formula is approximately "mean `total_points` over games with `minutes > 0` in the last 30 calendar days, rounded to the nearest 0.5", but the exact window, rounding, and tiebreak rules are not documented by FPL. The `strength` field is the simple 1-5 integer in `teams.csv`, not `strength_overall_home/away`. The formula is **not** needed to trust the pre-match safety argument above — §3 and §4 stand on API structure and mid-GW observation alone, independent of whether the exact formula is reproducible.

## Appendix B: Lifecycle flags (verbatim from live API, 2026-04-21)

```json
// GW32 — past
{ "id": 32, "is_previous": true,  "is_current": false, "is_next": false,
  "finished": true,  "data_checked": true  }

// GW33 — active, ep_this points here
{ "id": 33, "is_previous": false, "is_current": true,  "is_next": false,
  "finished": false, "data_checked": false }

// GW34 — upcoming, ep_next points here
{ "id": 34, "is_previous": false, "is_current": false, "is_next": true,
  "finished": false, "data_checked": false }
```

At any moment, exactly one event carries `is_current=True` and exactly one carries `is_next=True`. `finished` and `data_checked` are independent flags about match completion; neither causes `ep_this` to change.

## Appendix C: Source references

- Scraper: [`global_scraper.py`](https://github.com/vaastav/Fantasy-Premier-League/blob/master/global_scraper.py), which reads `ep_this` and `is_current` from a single `get_data()` call and writes `xP{gw_num}.csv`.
- PR #222: [docs: add caveat about xP scraping timing and potential lookahead](https://github.com/vaastav/Fantasy-Premier-League/pull/222), merged 2026-04-20.
- 2024-25 missing-xP GWs: 22, 32, 34 (`sum(xP) == 0` despite non-zero minutes played).
