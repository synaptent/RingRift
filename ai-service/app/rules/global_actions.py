from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.models import GameState, GameStatus, GamePhase, MoveType, Position
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
    """Return True if player has an interactive move in the current phase.

    Interactive moves exclude host-synthesised ``no_*_action`` bookkeeping
    moves. This mirrors the TS helper in
    ``src/shared/engine/globalActions.ts``:

    - ``ring_placement``:
        ``place_ring`` or ``skip_placement``.
    - ``movement`` / ``capture`` / ``chain_capture``:
        movement, capture, or ``recovery_slide`` moves.
    - ``line_processing``:
        ``process_line`` or ``choose_line_option`` (legacy: ``choose_line_reward``).
    - ``territory_processing``:
        ``choose_territory_option`` (legacy: ``process_territory_region``),
        ``eliminate_rings_from_stack``, or ``skip_territory_processing``.
    - ``forced_elimination``:
        availability of forced-elimination options for the player.

    For non-active games or non-current players this helper always returns
    False.
    """

    if state.game_status != GameStatus.ACTIVE:
        return False

    if player != state.current_player:
        return False

    phase = state.current_phase

    # Forced elimination is its own phase; treat the existence of any FE option
    # as an interactive choice for this predicate.
    if phase == GamePhase.FORCED_ELIMINATION:
        return has_forced_elimination_action(state, player)

    # LINE_PROCESSING: Call line-processing enumeration directly, matching TS
    # `enumerateProcessLineMoves(state, player, { detectionMode: 'detect_now' })`.
    # Per RR-CANON-R204, this uses fresh line detection to determine if any
    # lines exist for the player. This ensures ANM computation is identical
    # between Python and TS engines.
    if phase == GamePhase.LINE_PROCESSING:
        moves = GameEngine._get_line_processing_moves(state, player)
        return len(moves) > 0

    # TERRITORY_PROCESSING: Call territory enumeration directly, matching TS
    # `enumerateProcessTerritoryRegionMoves` + `enumerateTerritoryEliminationMoves`.
    if phase == GamePhase.TERRITORY_PROCESSING:
        moves = GameEngine._get_territory_processing_moves(state, player)
        return len(moves) > 0

    # For other phases, consult the canonical move generator. It returns only
    # interactive moves; bookkeeping no_* moves are surfaced via
    # get_phase_requirement / synthesize_bookkeeping_move instead.
    moves = GameEngine.get_valid_moves(state, player)
    if not moves:
        return False

    if phase == GamePhase.RING_PLACEMENT:
        interactive_types = {
            MoveType.PLACE_RING,
            MoveType.SKIP_PLACEMENT,
        }
    elif phase == GamePhase.MOVEMENT:
        interactive_types = {
            MoveType.MOVE_STACK,
            MoveType.MOVE_RING,
            MoveType.BUILD_STACK,
            MoveType.OVERTAKING_CAPTURE,
            MoveType.CONTINUE_CAPTURE_SEGMENT,
            MoveType.CHAIN_CAPTURE,
            MoveType.RECOVERY_SLIDE,
        }
    elif phase in (GamePhase.CAPTURE, GamePhase.CHAIN_CAPTURE):
        interactive_types = {
            MoveType.OVERTAKING_CAPTURE,
            MoveType.CONTINUE_CAPTURE_SEGMENT,
        }
    else:
        return False

    return any(move.type in interactive_types for move in moves)


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
