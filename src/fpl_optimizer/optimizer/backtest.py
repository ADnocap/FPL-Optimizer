"""Season-long backtest harness for the MILP optimizer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from fpl_optimizer.data.loader import SeasonDataLoader
from fpl_optimizer.engine.engine import FPLGameEngine
from fpl_optimizer.engine.state import (
    ChipState,
    GameState,
    PlayerSlot,
    Squad,
    StepResult,
)
from fpl_optimizer.optimizer.lineup_selector import select_lineup
from fpl_optimizer.optimizer.squad_selection import select_squad
from fpl_optimizer.optimizer.transfer_optimizer import optimize_transfers
from fpl_optimizer.optimizer.types import (
    OptimizerResult,
    PlayerCandidate,
    build_candidate_pool,
    to_engine_action,
)
from fpl_optimizer.utils.constants import GW1_FREE_TRANSFERS, STARTING_BUDGET, Position

logger = logging.getLogger(__name__)


@dataclass
class GWResult:
    """Result for a single gameweek in the backtest."""

    gw: int
    gross_points: int
    hit_cost: int
    net_points: int
    transfers_in: list[int]
    transfers_out: list[int]
    captain_id: int
    chip: str | None = None


@dataclass
class BacktestResult:
    """Full season backtest result."""

    season: str
    total_points: int
    gw_results: list[GWResult] = field(default_factory=list)
    transfers_made: int = 0
    hits_taken: int = 0
    chips_used: list[str] = field(default_factory=list)


def _optimizer_result_to_game_state(
    result: OptimizerResult,
    loader: SeasonDataLoader,
    gw: int,
) -> GameState:
    """Build an initial GameState from a squad-selection result."""
    players: list[PlayerSlot] = []
    for eid in result.squad_element_ids:
        pos = loader.get_player_position(eid)
        price = loader.get_player_price(eid, gw)
        if pos is None:
            raise ValueError(f"No position for element_id={eid}")
        if price <= 0:
            raise ValueError(f"No price for element_id={eid} in GW{gw}")
        players.append(
            PlayerSlot(
                element_id=eid,
                position=pos,
                purchase_price=price,
                selling_price=price,
            )
        )

    # Build lineup/bench as indices into players list
    eid_to_idx = {p.element_id: i for i, p in enumerate(players)}
    lineup = [eid_to_idx[eid] for eid in result.lineup_element_ids]
    bench = [eid_to_idx[eid] for eid in result.bench_element_ids]
    captain_idx = eid_to_idx[result.captain_id]
    vice_captain_idx = eid_to_idx[result.vice_captain_id]

    squad = Squad(
        players=players,
        lineup=lineup,
        bench=bench,
        captain_idx=captain_idx,
        vice_captain_idx=vice_captain_idx,
    )

    return GameState(
        squad=squad,
        bank=STARTING_BUDGET - result.total_cost,
        free_transfers=GW1_FREE_TRANSFERS,  # -> 1 FT for GW2 (FPL rule)
        chips=ChipState(),
        current_gw=gw,
        total_points=0,
    )


def _state_to_candidates(
    state: GameState,
    loader: SeasonDataLoader,
    gw: int,
    predicted_points: dict[int, float] | None,
) -> list[PlayerCandidate]:
    """Build PlayerCandidate list for current squad members."""
    cands: list[PlayerCandidate] = []
    for p in state.squad.players:
        team = loader.get_player_team(p.element_id)
        if predicted_points is not None:
            xp = predicted_points.get(p.element_id, 0.0)
        else:
            data = loader.get_player_gw(p.element_id, gw)
            xp = float(data["total_points"]) if data else 0.0
        cands.append(
            PlayerCandidate(
                element_id=p.element_id,
                position=p.position,
                price=p.selling_price,
                team_id=team if team is not None else 0,
                predicted_points=xp,
            )
        )
    return cands


class SeasonBacktester:
    """Run the MILP optimizer across a full season with the game engine.

    Parameters
    ----------
    loader : SeasonDataLoader
        Data loader for the season.
    model_dir : Path | None
        Directory with trained PointPredictor models. When *None*, uses
        actual points (hindsight/oracle mode).
    """

    def __init__(
        self,
        loader: SeasonDataLoader,
        model_dir: Path | None = None,
    ) -> None:
        self.loader = loader
        self.engine = FPLGameEngine(loader)
        self.model_dir = model_dir
        self._predictions: dict[int, float] | None = None

    def _load_predictions(self, season: str) -> dict[tuple[int, int], float] | None:
        """Load predictions if a model is provided."""
        if self.model_dir is None:
            return None

        from fpl_optimizer.prediction.integration import PredictionIntegrator

        integrator = PredictionIntegrator.from_model(
            self.model_dir,
            self.loader.data_dir,
            season,
        )
        return integrator._predictions

    def run(
        self,
        season: str | None = None,
        max_gw: int | None = None,
    ) -> BacktestResult:
        """Run the full season backtest.

        Parameters
        ----------
        season : str | None
            Season label (for prediction loading). Uses loader.season if None.
        max_gw : int | None
            Stop after this GW (inclusive). Uses all GWs if None.

        Returns
        -------
        BacktestResult
        """
        season = season or self.loader.season
        num_gws = self.loader.get_num_gameweeks()
        if max_gw is not None:
            num_gws = min(num_gws, max_gw)

        # Load predictions if model provided
        all_preds = self._load_predictions(season)

        result = BacktestResult(season=season, total_points=0)

        # --- GW1: initial squad selection ---
        gw1_preds: dict[int, float] | None = None
        if all_preds is not None:
            gw1_preds = {eid: xp for (eid, gw), xp in all_preds.items() if gw == 1}

        candidates = build_candidate_pool(self.loader, 1, gw1_preds)
        if not candidates:
            logger.warning("No candidates for GW1")
            return result

        squad_result = select_squad(candidates)
        state = _optimizer_result_to_game_state(squad_result, self.loader, 1)

        logger.info(
            "GW1 squad selected: cost=%d, bank=%d",
            squad_result.total_cost, state.bank,
        )

        # Step GW1 through the engine
        action = to_engine_action(squad_result)
        action.transfers_in = []
        action.transfers_out = []
        state, step_result = self.engine.step(state, action)

        gw_result = GWResult(
            gw=1,
            gross_points=step_result.gw_points,
            hit_cost=step_result.hit_cost,
            net_points=step_result.net_points,
            transfers_in=[],
            transfers_out=[],
            captain_id=squad_result.captain_id,
        )
        result.gw_results.append(gw_result)
        result.total_points += step_result.net_points

        logger.info("GW1: %d pts (captain=%d)", step_result.net_points, squad_result.captain_id)

        # --- GW2+: transfers + lineup ---
        for gw in range(2, num_gws + 1):
            gw_preds: dict[int, float] | None = None
            if all_preds is not None:
                gw_preds = {eid: xp for (eid, g), xp in all_preds.items() if g == gw}

            candidates = build_candidate_pool(self.loader, gw, gw_preds)
            if not candidates:
                logger.warning("No candidates for GW%d, skipping", gw)
                continue

            transfer_result = optimize_transfers(
                state,
                candidates,
                squad_teams={
                    p.element_id: self.loader.get_player_team(p.element_id)
                    for p in state.squad.players
                    if self.loader.get_player_team(p.element_id) is not None
                },
            )
            action = to_engine_action(transfer_result)

            state, step_result = self.engine.step(state, action)

            num_transfers = len(transfer_result.transfers_out)
            result.transfers_made += num_transfers
            result.hits_taken += step_result.hit_cost
            result.total_points += step_result.net_points

            gw_result = GWResult(
                gw=gw,
                gross_points=step_result.gw_points,
                hit_cost=step_result.hit_cost,
                net_points=step_result.net_points,
                transfers_in=transfer_result.transfers_in,
                transfers_out=transfer_result.transfers_out,
                captain_id=transfer_result.captain_id,
            )
            result.gw_results.append(gw_result)

            logger.info(
                "GW%d: %d pts (transfers=%d, hit=%d, captain=%d)",
                gw, step_result.net_points, num_transfers,
                step_result.hit_cost, transfer_result.captain_id,
            )

        logger.info(
            "Season complete: %d total points, %d transfers, %d hit cost",
            result.total_points, result.transfers_made, result.hits_taken,
        )
        return result
