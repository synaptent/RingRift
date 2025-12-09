from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.models import GameState, GameStatus, GamePhase, Position
from app.game_engine import GameEngine

"""
Global-actions and ANM helpers for the Python rules engine.

This mirrors src/shared/engine/globalActions.ts and RR-CANON R200–R207
for use in invariants, soaks, and higher-level diagnostics.
"""


@dataclass(frozen=True)
class GlobalLegalActionsSummary:
    """Summary of the R200 global action surface G(state, player).

    Fields:
        has_turn_material:
            Player has turn-material in the sense of RR-CANON-R201
            (at least one controlled stack or a ring in hand).
        has_global_placement_action:
            At least one legal ring placement exists for the player,
            ignoring state.current_phase (RR-CANON-R200, R080–R082).
        has_phase_local_interactive_move:
            At least one phase-local interactive move exists for the
            player in state.current_phase (R200, R204).
        has_forced_elimination_action:
            Host-level forced elimination is available for the player
            under RR-CANON-R072/R100/R205.
    """

    has_turn_material: bool
    has_global_placement_action: bool
    has_phase_local_interactive_move: bool
    has_forced_elimination_action: bool


@dataclass(frozen=True)
class ForcedEliminationOutcome:
    """Result of applying host-level forced elimination for a player."""

    next_state: GameState
    eliminated_player: int
    eliminated_from: Optional[Position]
    eliminated_count: int


def has_turn_material(state: GameState, player: int) -> bool:
    """Return True if player has turn-material in state (RR-CANON-R201)."""

    player_state = next(
        (p for p in state.players if p.player_number == player),
        None,
    )
    if player_state is None:
        return False

    if player_state.rings_in_hand > 0:
        return True

    for stack in state.board.stacks.values():
        if stack.controlling_player == player and stack.stack_height > 0:
            return True

    return False


def has_global_placement_action(state: GameState, player: int) -> bool:
    """Return True if any legal ring placement exists for ``player``."""

    # Delegate to the canonical ring-placement generator, which is
    # phase-agnostic and enforces caps and no-dead-placement.
    moves = GameEngine._get_ring_placement_moves(state, player)
    return bool(moves)


def has_phase_local_interactive_move(
    state: GameState,
    player: int,
) -> bool:
    """Return True if player has an interactive move in the current phase."""

    if state.game_status != GameStatus.ACTIVE:
        return False

    if player != state.current_player:
        return False

    phase = state.current_phase

    if phase == GamePhase.RING_PLACEMENT:
        if has_global_placement_action(state, player):
            return True
        skip_moves = GameEngine._get_skip_placement_moves(state, player)
        return bool(skip_moves)

    if phase in (
        GamePhase.MOVEMENT,
        GamePhase.CAPTURE,
        GamePhase.CHAIN_CAPTURE,
    ):
        if GameEngine._has_valid_movements(state, player):
            return True
        if GameEngine._has_valid_captures(state, player):
            return True
        return False

    if phase == GamePhase.LINE_PROCESSING:
        # During LINE_PROCESSING, the player ALWAYS has a valid move:
        # either an interactive PROCESS_LINE/CHOOSE_LINE_REWARD move,
        # or a NO_LINE_ACTION bookkeeping move synthesized by the host.
        # This prevents false positive ANM (Active No Moves) violations.
        return True

    if phase == GamePhase.TERRITORY_PROCESSING:
        # During TERRITORY_PROCESSING, the player ALWAYS has a valid move:
        # either an interactive PROCESS_TERRITORY_REGION move,
        # or a NO_TERRITORY_ACTION bookkeeping move synthesized by the host.
        # This prevents false positive ANM violations.
        return True

    return False


def has_forced_elimination_action(state: GameState, player: int) -> bool:
    """Return True if host-level forced elimination is available for player."""

    if state.game_status != GameStatus.ACTIVE:
        return False

    moves = GameEngine._get_forced_elimination_moves(state, player)
    return bool(moves)


def global_legal_actions_summary(
    state: GameState,
    player: int,
) -> GlobalLegalActionsSummary:
    """Compute the R200 global legal action summary for the player."""

    material = has_turn_material(state, player)
    placements = has_global_placement_action(state, player)
    phase_moves = has_phase_local_interactive_move(state, player)
    forced = has_forced_elimination_action(state, player)

    return GlobalLegalActionsSummary(
        has_turn_material=material,
        has_global_placement_action=placements,
        has_phase_local_interactive_move=phase_moves,
        has_forced_elimination_action=forced,
    )


def is_anm_state(state: GameState) -> bool:
    """Return True iff ANM(state, currentPlayer) holds (RR-CANON-R202)."""

    if state.game_status != GameStatus.ACTIVE:
        return False

    player = state.current_player
    if not has_turn_material(state, player):
        return False

    summary = global_legal_actions_summary(state, player)

    return (
        not summary.has_global_placement_action
        and not summary.has_phase_local_interactive_move
        and not summary.has_forced_elimination_action
    )


def apply_forced_elimination_for_player(
    state: GameState,
    player: int,
) -> Optional[ForcedEliminationOutcome]:
    """Apply one host-level forced elimination for ``player``, if legal.

    The function mutates ``state`` in-place and returns a
    ForcedEliminationOutcome when an elimination occurs, or None when
    RR-CANON-R072/R100/R205 preconditions are not satisfied.
    """

    if not has_forced_elimination_action(state, player):
        return None

    moves = GameEngine._get_forced_elimination_moves(state, player)
    if not moves:
        return None

    fe_move = moves[0]

    before_total = state.total_rings_eliminated
    GameEngine._apply_forced_elimination(state, fe_move)
    after_total = state.total_rings_eliminated

    eliminated_delta = after_total - before_total
    if eliminated_delta <= 0:
        eliminated_delta = 1

    return ForcedEliminationOutcome(
        next_state=state,
        eliminated_player=player,
        eliminated_from=fe_move.to,
        eliminated_count=eliminated_delta,
    )
