"""Transfer logic: sell prices, free transfer banking, hit costs."""

from __future__ import annotations

import math

from fpl_optimizer.data.loader import SeasonDataLoader
from fpl_optimizer.engine.constraints import check_club_limits, is_valid_squad_composition
from fpl_optimizer.engine.state import GameState, PlayerSlot, Squad
from fpl_optimizer.utils.constants import (
    MAX_FREE_TRANSFERS,
    TRANSFER_HIT_COST,
    Position,
)


def calculate_selling_price(purchase_price: int, current_price: int) -> int:
    """Calculate selling price with 50% appreciation rule.

    If the player has appreciated, you get purchase_price + floor((current - purchase) / 2).
    If the player has depreciated, you sell at current price.
    """
    if current_price <= purchase_price:
        return current_price
    profit = current_price - purchase_price
    return purchase_price + math.floor(profit / 2)


def update_selling_prices(
    squad: Squad, loader: SeasonDataLoader, gw: int
) -> Squad:
    """Update all selling prices based on current market values."""
    squad = squad.copy()
    for player in squad.players:
        current_price = loader.get_player_price(player.element_id, gw)
        if current_price > 0:
            player.selling_price = calculate_selling_price(
                player.purchase_price, current_price
            )
    return squad


def calculate_transfer_cost(num_transfers: int, free_transfers: int) -> int:
    """Calculate the point hit for transfers.

    Returns total hit cost (4 points per extra transfer beyond free allowance).
    """
    extra = max(0, num_transfers - free_transfers)
    return extra * TRANSFER_HIT_COST


def bank_free_transfers(
    current_free: int, transfers_made: int, used_wildcard: bool, used_free_hit: bool
) -> int:
    """Calculate free transfers for next GW.

    2025/26 rules:
    - Base: min(current + 1, MAX_FREE_TRANSFERS)
    - Wildcard and Free Hit do NOT reset banked transfers
    - If you used any free transfers, they're consumed
    """
    if used_wildcard or used_free_hit:
        # WC/FH don't reset banked transfers in 2025/26
        return current_free

    # Free transfers used are consumed
    remaining = max(0, current_free - transfers_made)
    # Bank one additional, up to max
    return min(remaining + 1, MAX_FREE_TRANSFERS)


def apply_transfers(
    state: GameState,
    transfers_out: list[int],
    transfers_in: list[int],
    loader: SeasonDataLoader,
    team_map: dict[int, int],
) -> tuple[GameState, int]:
    """Apply transfers to the game state.

    Args:
        state: Current game state.
        transfers_out: Element IDs to sell.
        transfers_in: Element IDs to buy.
        loader: Data loader for current prices.
        team_map: Element ID -> team ID mapping.

    Returns:
        (new_state, hit_cost): Updated state and point hit incurred.
    """
    if len(transfers_out) != len(transfers_in):
        raise ValueError("transfers_out and transfers_in must have same length")

    if not transfers_out:
        return state, 0

    state = state.copy()
    num_transfers = len(transfers_out)

    # Calculate hit cost
    is_wildcard = state.active_chip == "wildcard"
    is_free_hit = state.active_chip == "free_hit"
    if is_wildcard or is_free_hit:
        hit_cost = 0
    else:
        hit_cost = calculate_transfer_cost(num_transfers, state.free_transfers)

    # Sort transfer pairs so cash-positive swaps execute first.
    # The MILP optimizer guarantees aggregate budget feasibility but the
    # engine processes pairs sequentially, so ordering matters.
    sell_prices = []
    for out_id, in_id in zip(transfers_out, transfers_in):
        out_idx = state.squad.find_player_idx(out_id)
        sp = state.squad.players[out_idx].selling_price if out_idx is not None else 0
        bp = loader.get_player_price(in_id, state.current_gw)
        if bp <= 0:
            bp = loader.get_player_price(in_id, max(1, state.current_gw - 1))
        sell_prices.append(sp - bp)
    pairs = sorted(
        zip(transfers_out, transfers_in, sell_prices),
        key=lambda t: t[2],
        reverse=True,
    )

    # Perform each transfer
    for out_id, in_id, _ in pairs:
        # Find the outgoing player
        out_idx = state.squad.find_player_idx(out_id)
        if out_idx is None:
            raise ValueError(f"Player {out_id} not in squad")

        out_player = state.squad.players[out_idx]

        # Credit the selling price
        state.bank += out_player.selling_price

        # Get incoming player details
        in_price = loader.get_player_price(in_id, state.current_gw)
        if in_price <= 0:
            # Try previous GW price
            in_price = loader.get_player_price(in_id, max(1, state.current_gw - 1))
        in_position = loader.get_player_position(in_id)
        if in_position is None:
            raise ValueError(f"Cannot determine position for player {in_id}")

        # Check budget
        if in_price > state.bank:
            raise ValueError(
                f"Cannot afford player {in_id}: costs {in_price}, bank={state.bank}"
            )

        # Deduct cost
        state.bank -= in_price

        # Create new player slot
        new_player = PlayerSlot(
            element_id=in_id,
            position=in_position,
            purchase_price=in_price,
            selling_price=in_price,  # selling price = purchase price initially
        )

        # Replace in squad
        state.squad.players[out_idx] = new_player

    # Validate the new squad
    if not is_valid_squad_composition(state.squad.players):
        raise ValueError("Transfers resulted in invalid squad composition")
    if not check_club_limits(state.squad.players, team_map):
        raise ValueError("Transfers violated club limit (max 3 per team)")

    return state, hit_cost
