"""Player-prop betting odds features for point prediction.

Computes per-player features from historical player-prop odds snapshots
(The Odds API) stored as ``data/props/{season}/gw{N}_{eventId}.json``.

Anti-lookahead guarantee
------------------------
Each snapshot is taken ~2 hours BEFORE the match kickoff, so all quoted
prices are pre-match public information — no outcome leakage.

Markets used
------------
player_goal_scorer_anytime : "Yes" price per player  → props_xg
player_assists             : "Over 0.5" price        → props_xa
player_shots_on_target     : Over/Under line ladder  → props_sot
player_to_receive_card     : "Yes" price per player  → props_card_prob

Devigging
---------
These are one-sided quotes (no "No"/"Under" prices published), so the
per-market overround cannot be removed pair-wise.  Raw implied
probabilities (1/price) summed per match come to ~6.4 "scorers" vs an
actual ~2.6 goals per match — the margin is heavily concentrated in
longshots (favourite-longshot bias), so a proportional shrink has the
wrong shape.  We instead use a power devig  p = (1/price)^alpha  with a
per-market alpha calibrated once on 2025-26 so that market-implied
per-match totals match the observed season totals:

    alpha_goal   = 1.61  → sum of -ln(1-p) per match ≈ 2.645 actual goals
    alpha_assist = 1.46  → sum of -ln(1-p) per match ≈ 2.479 actual assists
    alpha_card   = 1.52  → sum of p per match ≈ 3.86 actual carded players

Note LightGBM trees are invariant to monotone transforms of a single
feature, so the exact alpha barely affects model fit — it makes the
features live on an interpretable xG/xA/probability scale.

Shots on target is a genuine two-way Over/Under market: the rare quotes
carrying both sides show a stable ~1.064 two-way overround, so the Over
probability is devigged proportionally (p = (1/price) / 1.064) and
converted to an expected-count lambda by Poisson inversion at the
book's lowest quoted line (push-aware for whole-number lines).

Features (per element per GW)
-----------------------------
props_xg        : E[goals] = -ln(1 - p_anytime), devigged consensus.
props_xa        : E[assists] = -ln(1 - p_over0.5_assists).
props_sot       : Expected shots on target (Poisson lambda), consensus.
props_card_prob : Devigged probability of receiving a card.
props_has_line  : 1 if the player has ANY prop quoted this GW, else 0.
                  (The market's implicit minutes/relevance signal —
                  books only price likely starters.)
props_n_books   : Number of distinct bookmakers quoting the player.

Unquoted players in a GW that has props data get has_line=0/n_books=0
and NaN for the four price-derived features (LightGBM handles NaN
natively; 0 would conflate "no line" with "zero probability").
Seasons/GWs without props data get all-NaN columns.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from unidecode import unidecode
except ImportError:  # stdlib fallback: drops combining accents, keeps ASCII
    import unicodedata

    def unidecode(s: str) -> str:
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

from fpl_optimizer.data.collectors.odds import odds_team_to_fpl_name
from fpl_optimizer.prediction.features.odds import _build_team_name_to_id

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "props_xg",
    "props_xa",
    "props_sot",
    "props_card_prob",
    "props_has_line",
    "props_n_books",
]

# Power-devig calibration constants — see module docstring for derivation.
_ALPHA_GOAL = 1.61
_ALPHA_ASSIST = 1.46
_ALPHA_CARD = 1.52
# Two-way overround measured from the SOT quotes that carry both sides.
_SOT_OVERROUND = 1.064
# Cap devigged probabilities away from 1.0 so -ln(1-p) stays finite.
_P_CAP = 0.97

_MARKET_AGS = "player_goal_scorer_anytime"
_MARKET_AST = "player_assists"
_MARKET_SOT = "player_shots_on_target"
_MARKET_CARD = "player_to_receive_card"


def _norm_name(name: str) -> str:
    """Normalise a player name for cross-source matching."""
    s = unidecode(str(name)).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def _poisson_sf(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    term = math.exp(-lam)
    cdf = term
    for i in range(1, k):
        term *= lam / i
        cdf += term
    return max(0.0, 1.0 - cdf)


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def _solve_sot_lambda(line: float, p_over: float) -> float | None:
    """Invert an Over quote at ``line`` into a Poisson expected count.

    Half lines (0.5, 1.5, ...): Over wins iff X >= line + 0.5, so we solve
    P(X >= k) = p_over.  Whole lines (1.0, 2.0, ...) push on X == line and
    the price reflects the win probability conditional on no push:
    P(X >= line+1) / (1 - P(X = line)) = p_over.
    """
    if not (0.001 < p_over < 0.985):
        return None
    is_half = abs(line - math.floor(line) - 0.5) < 1e-9
    k = int(math.floor(line)) + 1  # Over wins iff X >= k, in both cases

    def win_prob(lam: float) -> float:
        p_win = _poisson_sf(k, lam)
        if is_half:
            return p_win
        p_push = _poisson_pmf(int(line), lam)
        denom = 1.0 - p_push
        return p_win / denom if denom > 1e-9 else 1.0

    lo, hi = 1e-4, 15.0
    if win_prob(hi) < p_over:
        return hi
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if win_prob(mid) < p_over:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _load_season_props(data_dir: Path, season: str) -> list[dict]:
    """Load all per-match props snapshots for a season."""
    props_dir = data_dir / "props" / season
    if not props_dir.exists():
        return []

    matches: list[dict] = []
    for path in sorted(props_dir.glob("gw*_*.json")):
        if path.stem.endswith("_events"):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                matches.append(json.load(f))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Props %s: failed to load %s: %s", season, path.name, exc)
    return matches


def _build_name_lookup(
    pool: pd.DataFrame, id_resolver
) -> tuple[dict[str, int], list[tuple[frozenset, int]]]:
    """Build name-matching structures for a two-team candidate pool.

    Returns a dict of normalised-name keys -> element (resolver full
    name, web_name, merged_gw ``name`` column, token-sorted variants for
    "Son Heung-Min" vs "Heung-Min Son", and unique surnames; ambiguous
    keys are dropped), plus a list of (token_set, element) pairs for
    subset matching ("Bruno Guimaraes" vs FPL "Bruno Guimarães
    Rodriguez Moura").
    """
    candidates: dict[str, set[int]] = defaultdict(set)
    surnames: dict[str, set[int]] = defaultdict(set)
    token_sets: list[tuple[frozenset, int]] = []

    for row in pool.itertuples(index=False):
        element = int(row.element)
        names = set()
        code = getattr(row, "code", None)
        if code is not None and not pd.isna(code):
            full = id_resolver.player_full_name(int(code))
            if full:
                names.add(full)
            web = id_resolver.player_name(int(code))
            if web and web != "Unknown":
                names.add(web)
        raw_name = getattr(row, "name", None)
        if raw_name is not None and not pd.isna(raw_name):
            names.add(str(raw_name))

        for name in names:
            norm = _norm_name(name)
            if not norm:
                continue
            candidates[norm].add(element)
            tokens = norm.split()
            if len(tokens) > 1:
                candidates[" ".join(sorted(tokens))].add(element)
                # Any non-first token can be the commonly used surname
                # ("Yéremy Pino Santos" is quoted as "Yeremi Pino")
                for tok in tokens[1:]:
                    surnames[tok].add(element)
                token_sets.append((frozenset(tokens), element))

    lookup = {k: next(iter(v)) for k, v in candidates.items() if len(v) == 1}
    # Surname fallback only when it doesn't clash with a full-name key
    for k, v in surnames.items():
        if len(v) == 1 and k not in lookup:
            lookup[k] = next(iter(v))
    return lookup, token_sets


def _subset_match(
    desc_tokens: frozenset, token_sets: list[tuple[frozenset, int]]
) -> int | None:
    """Match when one name's tokens are a subset of the other's, uniquely."""
    hits = {
        element for tokens, element in token_sets
        if tokens <= desc_tokens or desc_tokens <= tokens
    }
    return next(iter(hits)) if len(hits) == 1 else None


def _extract_quotes(match: dict) -> dict[str, dict[str, dict[str, list]]]:
    """Extract {market: {player_description: {book: [(price, point)]}}}."""
    quotes: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    data = (match.get("odds") or {}).get("data") or {}
    for bk in data.get("bookmakers", []):
        book = bk.get("key", "?")
        for market in bk.get("markets", []):
            mkey = market.get("key")
            if mkey not in (_MARKET_AGS, _MARKET_AST, _MARKET_SOT, _MARKET_CARD):
                continue
            for o in market.get("outcomes", []):
                price = o.get("price")
                desc = o.get("description")
                if not desc or not price or price <= 1.0:
                    continue
                name = o.get("name")
                if mkey in (_MARKET_AGS, _MARKET_CARD) and name != "Yes":
                    continue
                if mkey == _MARKET_AST and (name != "Over" or o.get("point") != 0.5):
                    continue
                if mkey == _MARKET_SOT and name != "Over":
                    continue
                quotes[mkey][desc][book].append((float(price), o.get("point")))
    return quotes


def _consensus_prob(book_quotes: dict[str, list], alpha: float) -> float:
    """Mean raw implied prob across books, then power devig."""
    p_raws = [1.0 / min(q[0] for q in quotes) for quotes in book_quotes.values()]
    p_raw = sum(p_raws) / len(p_raws)
    return min(_P_CAP, p_raw**alpha)


def _consensus_sot(book_quotes: dict[str, list]) -> float | None:
    """Average Poisson lambda across books, each from its lowest line."""
    lambdas = []
    for quotes in book_quotes.values():
        with_line = [(price, pt) for price, pt in quotes if pt is not None]
        if not with_line:
            continue
        price, line = min(with_line, key=lambda q: (q[1], q[0]))
        lam = _solve_sot_lambda(float(line), (1.0 / price) / _SOT_OVERROUND)
        if lam is not None:
            lambdas.append(lam)
    return sum(lambdas) / len(lambdas) if lambdas else None


def compute_props_features(
    merged_gw: pd.DataFrame,
    data_dir: Path,
    season: str,
    id_resolver,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute player-prop odds features per (element, GW).

    Parameters
    ----------
    merged_gw : pd.DataFrame
        Per-player-per-fixture data with ``element``, ``GW``, ``team``
        (numeric ID), ``code``, and (optionally) ``name`` columns.
    data_dir : Path
        Root data directory containing ``props/{season}/``.
    season : str
        Season string, e.g. ``"2025-26"``.
    id_resolver : IDResolver
        For full-name/web-name lookups per code.
    teams_df : pd.DataFrame
        From teams.csv with columns ``id``, ``name``.

    Returns
    -------
    pd.DataFrame
        One row per (element, GW): element, GW, plus :data:`FEATURE_COLS`.
    """
    base = merged_gw[["element", "GW"]].drop_duplicates().copy()
    empty = base.copy()
    for col in FEATURE_COLS:
        empty[col] = float("nan")

    if merged_gw.empty:
        return pd.DataFrame(columns=["element", "GW"] + FEATURE_COLS)

    try:
        matches = _load_season_props(data_dir, season)
        if not matches:
            logger.info("Props %s: no data available, returning NaN features", season)
            return empty

        name_to_id = _build_team_name_to_id(merged_gw, teams_df, data_dir, season)
        if not name_to_id:
            logger.warning("Props %s: no team name->id mapping available", season)
            return empty

        return _compute(merged_gw, matches, name_to_id, id_resolver, base, season)
    except Exception as exc:  # never crash the pipeline over a data defect
        logger.error("Props %s: feature computation failed: %s", season, exc)
        return empty


def _compute(
    merged_gw: pd.DataFrame,
    matches: list[dict],
    name_to_id: dict[str, int],
    id_resolver,
    base: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    def _team_lookup(raw_name: str) -> int | None:
        tid = name_to_id.get(odds_team_to_fpl_name(raw_name))
        if tid is None:
            tid = name_to_id.get(raw_name)
        return tid

    pool_cols = ["element", "GW", "team", "code"]
    if "name" in merged_gw.columns:
        pool_cols.append("name")
    players = merged_gw[pool_cols].drop_duplicates(subset=["element", "GW"])
    players = players[pd.to_numeric(players["team"], errors="coerce").notna()]

    rows: list[dict] = []
    gws_covered: set[int] = set()
    n_quoted = 0
    n_matched = 0

    for match in matches:
        gw = match.get("gw")
        event = match.get("event") or {}
        if gw is None:
            continue
        home_id = _team_lookup(event.get("home_team", ""))
        away_id = _team_lookup(event.get("away_team", ""))
        if home_id is None or away_id is None:
            logger.warning(
                "Props: dropping match, unmapped team(s): %r (id=%s), %r (id=%s)",
                event.get("home_team"), home_id, event.get("away_team"), away_id,
            )
            continue
        gws_covered.add(int(gw))

        quotes = _extract_quotes(match)
        if not quotes:
            continue

        pool = players[
            (players["GW"] == gw)
            & (players["team"].isin([home_id, away_id]))
        ]
        lookup, token_sets = _build_name_lookup(pool, id_resolver)

        # Resolve each quoted player once across all markets
        all_descs = {d for market in quotes.values() for d in market}
        desc_to_element: dict[str, int] = {}
        for desc in all_descs:
            norm = _norm_name(desc)
            if norm in ("no scorer", "no goalscorer"):  # market outcome, not a player
                continue
            n_quoted += 1
            element = lookup.get(norm)
            tokens = norm.split()
            if element is None and len(tokens) > 1:
                element = lookup.get(" ".join(sorted(tokens)))
                if element is None:
                    element = lookup.get(tokens[-1])
            if element is None:
                # Handles both "Bruno Guimaraes" (subset of the FPL full
                # name) and mononyms like "Rodri" (a token of
                # "Rodrigo 'Rodri' Hernandez Cascante")
                element = _subset_match(frozenset(tokens), token_sets)
            if element is not None:
                desc_to_element[desc] = element
                n_matched += 1
            else:
                logger.debug("Props %s GW%s: unmatched player %r", season, gw, desc)

        per_player: dict[int, dict] = defaultdict(
            lambda: {"xg": None, "xa": None, "sot": None, "card": None, "books": set()}
        )
        for mkey, market in quotes.items():
            for desc, book_quotes in market.items():
                element = desc_to_element.get(desc)
                if element is None:
                    continue
                rec = per_player[element]
                rec["books"].update(book_quotes.keys())
                if mkey == _MARKET_AGS:
                    rec["xg"] = -math.log(1 - _consensus_prob(book_quotes, _ALPHA_GOAL))
                elif mkey == _MARKET_AST:
                    rec["xa"] = -math.log(1 - _consensus_prob(book_quotes, _ALPHA_ASSIST))
                elif mkey == _MARKET_CARD:
                    rec["card"] = _consensus_prob(book_quotes, _ALPHA_CARD)
                elif mkey == _MARKET_SOT:
                    rec["sot"] = _consensus_sot(book_quotes)

        for element, rec in per_player.items():
            rows.append({
                "element": element,
                "GW": int(gw),
                "props_xg": rec["xg"],
                "props_xa": rec["xa"],
                "props_sot": rec["sot"],
                "props_card_prob": rec["card"],
                "props_n_books": len(rec["books"]),
            })

    match_rate = 100.0 * n_matched / n_quoted if n_quoted else 0.0
    logger.info(
        "Props %s: matched %d/%d quoted players (%.1f%%) across %d matches",
        season, n_matched, n_quoted, match_rate, len(matches),
    )

    if not rows:
        result = base.copy()
        for col in FEATURE_COLS:
            result[col] = float("nan")
        return result

    props = pd.DataFrame(rows)
    # DGW aggregation: expectations sum across the two matches; card
    # probability combines as 1-(1-p1)(1-p2); books/lines take the max.
    props = props.groupby(["element", "GW"], as_index=False).agg(
        props_xg=("props_xg", lambda s: s.sum(min_count=1)),
        props_xa=("props_xa", lambda s: s.sum(min_count=1)),
        props_sot=("props_sot", lambda s: s.sum(min_count=1)),
        props_card_prob=("props_card_prob", lambda s: 1.0 - (1.0 - s).prod(min_count=1)
                         if s.notna().any() else float("nan")),
        props_n_books=("props_n_books", "max"),
    )
    props["props_has_line"] = 1.0

    result = base.merge(props, on=["element", "GW"], how="left")
    # In GWs where the props snapshot exists, an absent line IS the signal
    covered = result["GW"].isin(gws_covered)
    result.loc[covered, "props_has_line"] = (
        result.loc[covered, "props_has_line"].fillna(0.0)
    )
    result.loc[covered, "props_n_books"] = (
        result.loc[covered, "props_n_books"].fillna(0.0)
    )

    n_xg = result["props_xg"].notna().sum()
    logger.info(
        "Props %s: %d/%d player-GW rows have props_xg (%.1f%%), %d GWs covered",
        season, n_xg, len(result),
        100.0 * n_xg / len(result) if len(result) else 0.0,
        len(gws_covered),
    )
    return result[["element", "GW"] + FEATURE_COLS]
