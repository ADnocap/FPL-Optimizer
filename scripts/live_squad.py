"""Quick live squad recommender from the official FPL API.

Pulls bootstrap-static, builds a candidate pool with FPL's own ep_next as the
point estimate (stopgap until the LightGBM predictor is wired to live data),
and solves the initial-squad MILP.

Usage:
    python scripts/live_squad.py [--budget 1000] [--min-ep 0.0]
"""

from __future__ import annotations

import argparse

import requests

from fpl_optimizer.optimizer.squad_selection import select_squad
from fpl_optimizer.optimizer.types import PlayerCandidate
from fpl_optimizer.utils.constants import Position

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_bootstrap() -> dict:
    resp = requests.get(BOOTSTRAP_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_live_candidates(data: dict, min_ep: float = 0.0) -> list[PlayerCandidate]:
    """Build candidates from live bootstrap data, excluding unavailable players."""
    candidates = []
    for el in data["elements"]:
        # status: a=available, d=doubtful, i=injured, s=suspended, u=unavailable, n=not in squad
        if el["status"] in ("i", "s", "u", "n"):
            continue
        chance = el.get("chance_of_playing_next_round")
        if chance is not None and chance < 75:
            continue
        ep = float(el.get("ep_next") or 0.0)
        if ep < min_ep:
            continue
        candidates.append(
            PlayerCandidate(
                element_id=el["id"],
                position=Position(el["element_type"]),
                price=el["now_cost"],
                team_id=el["team"],
                predicted_points=ep,
            )
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=1000, help="budget in tenths (1000 = 100.0m)")
    parser.add_argument("--min-ep", type=float, default=0.0, help="minimum ep_next to consider")
    args = parser.parse_args()

    data = fetch_bootstrap()
    names = {el["id"]: el["web_name"] for el in data["elements"]}
    team_names = {t["id"]: t["short_name"] for t in data["teams"]}
    teams = {el["id"]: team_names[el["team"]] for el in data["elements"]}
    prices = {el["id"]: el["now_cost"] for el in data["elements"]}
    eps = {el["id"]: float(el.get("ep_next") or 0.0) for el in data["elements"]}
    positions = {el["id"]: Position(el["element_type"]).name for el in data["elements"]}

    candidates = build_live_candidates(data, min_ep=args.min_ep)
    print(f"Candidate pool: {len(candidates)} available players")

    result = select_squad(candidates, budget=args.budget)

    def fmt(eid: int) -> str:
        return (
            f"{names[eid]:<20} {positions[eid]:<4} {teams[eid]:<4} "
            f"{prices[eid] / 10:>5.1f}m  ep={eps[eid]:.1f}"
        )

    print(f"\n=== Recommended squad (cost {result.total_cost / 10:.1f}m, "
          f"objective {result.objective_value:.1f}) ===\n")
    print("Starting XI:")
    for eid in result.lineup_element_ids:
        tag = " (C)" if eid == result.captain_id else " (V)" if eid == result.vice_captain_id else ""
        print(f"  {fmt(eid)}{tag}")
    print("\nBench (in sub priority order):")
    for eid in result.bench_element_ids:
        print(f"  {fmt(eid)}")


if __name__ == "__main__":
    main()
