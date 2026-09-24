"""Chip activation, validation, Free Hit revert, and GW19 expiry."""

from __future__ import annotations

from fpl_optimizer.engine.state import GameState
from fpl_optimizer.utils.constants import ALL_CHIPS, FIRST_HALF_END


def validate_chip(state: GameState, chip: str) -> str | None:
    """Validate whether a chip can be activated.

    Returns None if valid, or an error message string.
    """
    if chip not in ALL_CHIPS:
        return f"Unknown chip: {chip}. Valid chips: {ALL_CHIPS}"

    if state.active_chip is not None:
        return f"Cannot use {chip}: already using {state.active_chip} this GW"

    if not state.chips.is_available(chip, state.current_gw):
        return f"Chip {chip} is not available for GW{state.current_gw}"

    return None


def activate_chip(state: GameState, chip: str) -> GameState:
    """Activate a chip for the current GW.

    For Free Hit, stashes the current squad before transfers are applied.
    """
    error = validate_chip(state, chip)
    if error:
        raise ValueError(error)

    state = state.copy()
    state.active_chip = chip
    state.chips.use_chip(chip, state.current_gw)

    # Free Hit: stash current squad AND bank for revert after GW
    if chip == "free_hit":
        state.free_hit_stash = state.squad.copy()
        state.free_hit_bank_stash = state.bank

    return state


def revert_free_hit(state: GameState) -> GameState:
    """Revert squad (and bank) to the pre-Free Hit state after the GW.

    Free Hit transfers are temporary — the squad reverts next GW, and so does
    the money in the bank (the FH sales/purchases are undone).
    Free transfers are NOT reset by Free Hit (2025/26 rule).
    """
    if state.free_hit_stash is None:
        return state

    state = state.copy()
    state.squad = state.free_hit_stash
    state.free_hit_stash = None
    if state.free_hit_bank_stash is not None:
        state.bank = state.free_hit_bank_stash
        state.free_hit_bank_stash = None
    return state


def handle_gw19_expiry(state: GameState) -> GameState:
    """Expire unused first-half chips after GW19.

    Called at the end of GW19 processing.
    """
    if state.current_gw != FIRST_HALF_END:
        return state

    state = state.copy()
    state.chips.expire_first_half()
    return state


def clear_active_chip(state: GameState) -> GameState:
    """Clear the active chip after GW processing."""
    state = state.copy()
    state.active_chip = None
    return state
