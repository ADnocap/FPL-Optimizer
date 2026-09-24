"""Data types and helpers for the MILP optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fpl_optimizer.data.loader import SeasonDataLoader
from fpl_optimizer.engine.state import EngineAction, PlayerSlot
from fpl_optimizer.utils.constants import Position


@dataclass(frozen=True)
class PlayerCandidate:
    """A candidate player the optimizer can pick."""

    element_id: int
    position: Position
    price: int  # in tenths (100 = £10.0m)
    team_id: int
    predicted_points: float


@dataclass
class OptimizerResult:
    """Output of any optimizer solve."""

    squad_element_ids: list[int]
    lineup_element_ids: list[int]
    bench_element_ids: list[int]  # ordered by sub priority
    captain_id: int
    vice_captain_id: int
    transfers_in: list[int] = field(default_factory=list)
    transfers_out: list[int] = field(default_factory=list)
    chip: str | None = None
    objective_value: float = 0.0
    total_cost: int = 0
    hit_cost: int = 0


def build_candidate_pool(
    loader: SeasonDataLoader,
    gw: int,
    predicted_points: dict[int, float] | None = None,
) -> list[PlayerCandidate]:
    """Build a candidate pool from loader data for a single GW.

    Parameters
    ----------
    loader : SeasonDataLoader
        Season data loader.
    gw : int
        Gameweek number.
    predicted_points : dict[int, float] | None
        Mapping element_id -> predicted points.  When *None* uses actual
        ``total_points`` from the dataset (hindsight / oracle mode).

    Returns
    -------
    list[PlayerCandidate]
        All available candidates with valid position, price and team.
    """
    gw_data: pd.DataFrame = loader.get_gameweek_data(gw)
    if gw_data.empty:
        return []

    # Deduplicate DGW rows — sum total_points, keep first price
    agg: dict[str, str | object] = {
        "total_points": "sum",
        "value": "first",
    }
    gw_grouped = gw_data.groupby("element", as_index=False).agg(agg)

    candidates: list[PlayerCandidate] = []
    for _, row in gw_grouped.iterrows():
        eid = int(row["element"])
        pos = loader.get_player_position(eid)
        team = loader.get_player_team(eid)
        price = int(row["value"]) if pd.notna(row["value"]) else 0
        if pos is None or team is None or price <= 0:
            continue

        if predicted_points is not None:
            xp = predicted_points.get(eid, 0.0)
        else:
            xp = float(row["total_points"])

        candidates.append(
            PlayerCandidate(
                element_id=eid,
                position=pos,
                price=price,
                team_id=team,
                predicted_points=xp,
            )
        )

    return candidates


def to_engine_action(result: OptimizerResult) -> EngineAction:
    """Convert an OptimizerResult into an EngineAction for the game engine."""
    return EngineAction(
        transfers_out=list(result.transfers_out),
        transfers_in=list(result.transfers_in),
        captain=result.captain_id,
        vice_captain=result.vice_captain_id,
        chip=result.chip,
        lineup=list(result.lineup_element_ids),
        bench=list(result.bench_element_ids),
    )
