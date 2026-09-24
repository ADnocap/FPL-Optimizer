"""FPLGameEngine: orchestrates one gameweek step."""

from __future__ import annotations

from fpl_optimizer.data.loader import SeasonDataLoader
from fpl_optimizer.engine.chips import (
    activate_chip,
    clear_active_chip,
    handle_gw19_expiry,
    revert_free_hit,
    validate_chip,
)
from fpl_optimizer.engine.lineup import perform_auto_subs
from fpl_optimizer.engine.scoring import (
    calculate_captain_points,
    calculate_squad_points,
    get_player_points,
)
from fpl_optimizer.engine.state import EngineAction, GameState, StepResult
from fpl_optimizer.engine.transfers import (
    apply_transfers,
    bank_free_transfers,
    update_selling_prices,
)


class FPLGameEngine:
    """Orchestrates one gameweek step of the FPL simulation.

    Stateless — takes a GameState and returns a new GameState.
    Stateless and standalone — the backtest drives it directly.
    """

    def __init__(self, loader: SeasonDataLoader) -> None:
        self.loader = loader
        self._team_map = loader._team_map

    def step(
        self,
        state: GameState,
        action: EngineAction,
        *,
        preseason: bool = False,
    ) -> tuple[GameState, StepResult]:
        """Process one gameweek (or a pre-season squad-building step).

        When preseason=True, only transfers/captain/lineup/selling prices are
        applied. No scoring, auto-subs, FT banking, chip handling, or GW
        advance occurs. Returns a zeroed StepResult.

        Steps (normal mode):
        1. Activate chip (if any)
        2. Apply transfers
        3. Set captain/vice-captain
        4. Set lineup/bench (if specified)
        5. Update selling prices
        6. Perform auto-substitution
        7. Calculate points with captain logic
        8. Calculate total GW points (incl. bench boost, hits)
        9. Update free transfers for next GW
        10. Handle Free Hit revert
        11. Handle GW19 chip expiry
        12. Advance to next GW
        """
        gw = state.current_gw

        # 1. Activate chip (skip during pre-season)
        if action.chip is not None and not preseason:
            state = activate_chip(state, action.chip)
        else:
            state = state.copy()

        is_wildcard = state.active_chip == "wildcard"
        is_free_hit = state.active_chip == "free_hit"
        is_bench_boost = state.active_chip == "bench_boost"
        is_triple_captain = state.active_chip == "triple_captain"

        # 2. Apply transfers
        state, hit_cost = apply_transfers(
            state,
            action.transfers_out,
            action.transfers_in,
            self.loader,
            self._team_map,
        )

        # 3. Set captain/vice-captain
        if action.captain is not None:
            idx = state.squad.find_player_idx(action.captain)
            if idx is not None:
                state.squad.captain_idx = idx
        if action.vice_captain is not None:
            idx = state.squad.find_player_idx(action.vice_captain)
            if idx is not None:
                state.squad.vice_captain_idx = idx

        # 4. Set lineup/bench if specified
        if action.lineup is not None and action.bench is not None:
            new_lineup = []
            new_bench = []
            for eid in action.lineup:
                idx = state.squad.find_player_idx(eid)
                if idx is not None:
                    new_lineup.append(idx)
            for eid in action.bench:
                idx = state.squad.find_player_idx(eid)
                if idx is not None:
                    new_bench.append(idx)
            if len(new_lineup) == 11 and len(new_bench) == 4:
                state.squad.lineup = new_lineup
                state.squad.bench = new_bench

        # 5. Update selling prices
        state.squad = update_selling_prices(state.squad, self.loader, gw)

        # --- Pre-season early return: no scoring, no GW advance ---
        if preseason:
            return state, StepResult(
                gw_points=0, hit_cost=0, net_points=0,
                captain_points=0, bench_points=0,
                auto_subs=[], captain_failover=False,
            )

        # 6. Auto-substitution (before captain calc — captain bonus
        # must reflect the final, post-sub lineup)
        state.squad, auto_subs = perform_auto_subs(state.squad, self.loader, gw)

        # 7. Calculate points with captain logic (post-auto-sub)
        lineup_points, bench_points = calculate_squad_points(
            self.loader, state.squad, gw
        )
        captain_bonus, captain_failover = calculate_captain_points(
            self.loader, state.squad, gw, triple_captain=is_triple_captain
        )

        # 9. Calculate total GW points
        gw_points = lineup_points + captain_bonus
        if is_bench_boost:
            gw_points += bench_points

        net_points = gw_points - hit_cost
        state.total_points += net_points

        # Build result
        result = StepResult(
            gw_points=gw_points,
            hit_cost=hit_cost,
            net_points=net_points,
            captain_points=captain_bonus,
            bench_points=bench_points,
            auto_subs=auto_subs,
            captain_failover=captain_failover,
        )

        # 10. Update free transfers for next GW
        num_transfers = len(action.transfers_out)
        state.free_transfers = bank_free_transfers(
            state.free_transfers, num_transfers, is_wildcard, is_free_hit
        )

        # 11. Handle Free Hit revert
        if is_free_hit:
            state = revert_free_hit(state)

        # 12. Handle GW19 chip expiry
        state = handle_gw19_expiry(state)

        # 13. Clear active chip and advance GW
        state = clear_active_chip(state)
        state.current_gw = gw + 1

        return state, result
