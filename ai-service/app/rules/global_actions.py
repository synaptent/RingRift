from __future__ import annotations

from dataclasses import dataclass

from app.game_engine import GameEngine
from app.models import GamePhase, GameState, GameStatus, MoveType, Position
from app.rules.elimination import EliminationContext, has_eligible_elimination_target


def _did_current_turn_include_recovery_slide(state: GameState, player: int) -> bool:
    """Check if current turn includes a RECOVERY_SLIDE move.

    Used to derive territory elimination context (RR-CANON-R114).
    Mirrors TS didCurrentTurnIncludeRecoverySlide in TerritoryAggregate.ts.
    """
    for move in reversed(state.move_history):
        if move.player != player:
            break
        if move.type == MoveType.RECOVERY_SLIDE:
            return True
    return False


def _derive_territory_elimination_context(state: GameState, player: int) -> EliminationContext:
    """Derive elimination context for territory processing.

    Mirrors TS deriveTerritoryEliminationContext in TerritoryAggregate.ts.

    Args:
        state: Current game state
        player: Player to check

    Returns:
        RECOVERY if turn included recovery slide, TERRITORY otherwise
    """
    if _did_current_turn_include_recovery_slide(state, player):
        return EliminationContext.RECOVERY
    return EliminationContext.TERRITORY

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
    eliminated_from: Position | None
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

    return any(stack.controlling_player == player and stack.stack_height > 0 for stack in state.board.stacks.values())


def has_global_placement_action(state: GameState, player: int) -> bool:
    """Return True if any legal ring placement exists for ``player``.

    This is a boolean predicate used by ANM/global-action checks.

    Important: do **not** enumerate all placement moves here.

    ``GameEngine._get_ring_placement_moves`` is intentionally expensive because
    it fully enumerates placements and runs no-dead-placement validation for
    each candidate. Calling that on every post-move ANM check can turn
    self-play and fitness evaluation into an accidental quadratic/cubic loop.

    ``GameEngine._has_any_valid_placement_fast`` is semantically equivalent for
    existence checks (same caps + no-dead-placement), but returns early.
    """

    return GameEngine._has_any_valid_placement_fast(state, player)


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
        ``process_line`` or ``choose_line_option``.
    - ``territory_processing``:
        ``choose_territory_option``, ``eliminate_rings_from_stack``, or
        ``skip_territory_processing``.
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

    # LINE_PROCESSING: Per RR-CANON-R123, when pending_line_reward_elimination
    # is True, the player must execute an eliminate_rings_from_stack move.
    # This counts as an interactive move for ANM calculation, matching TS
    # hasPhaseLocalInteractiveMove behavior in globalActions.ts:254-267.
    if phase == GamePhase.LINE_PROCESSING:
        if state.pending_line_reward_elimination:
            # Player has controlled stacks -> elimination moves exist
            for stack in state.board.stacks.values():
                if stack.controlling_player == player and stack.stack_height > 0:
                    return True
            # No controlled stacks means no elimination possible (edge case)
            return False

        # Otherwise check for line processing moves using fresh line detection.
        # Per RR-CANON-R204, this uses fresh line detection to determine if any
        # lines exist for the player.
        moves = GameEngine._get_line_processing_moves(state, player)
        return len(moves) > 0

    # TERRITORY_PROCESSING: Per RR-CANON-R145, when the last move by this player
    # was a choose_territory_option (with region data), there's a pending
    # self-elimination that counts as an interactive move. This mirrors TS
    # globalActions.ts which calls BOTH enumerateProcessTerritoryRegionMoves AND
    # enumerateTerritoryEliminationMoves for the ANM check.
    #
    # HEX-PARITY-03 fix: Check for pending territory self-elimination explicitly,
    # similar to the pending_line_reward_elimination check in line_processing.
    if phase == GamePhase.TERRITORY_PROCESSING:
        # Check for pending territory self-elimination (RR-CANON-R145)
        # This matches TS getPendingTerritorySelfEliminationRegion logic
        last_move = state.move_history[-1] if state.move_history else None
        if last_move is not None and last_move.player == player:
            if last_move.type in (MoveType.CHOOSE_TERRITORY_OPTION, MoveType.PROCESS_TERRITORY_REGION):
                # Check if the move has region data (required for pending elimination)
                regions = getattr(last_move, "disconnected_regions", None)
                if regions:
                    # Pending self-elimination: check for eligible elimination targets
                    # outside the processed region using canonical eligibility check.
                    # HEX-PARITY-02 FIX: Derive elimination context like TypeScript does.
                    # If turn included recovery slide, use RECOVERY context (checks for
                    # buried rings). Otherwise use TERRITORY context (checks for control).
                    # This matches TS deriveTerritoryEliminationContext.
                    processed_region_keys = {p.to_key() for p in regions[0].spaces}
                    elimination_context = _derive_territory_elimination_context(state, player)

                    # Build stacks dict for eligibility check
                    stacks_dict = {}
                    for stack in state.board.stacks.values():
                        pos_key = stack.position.to_key()
                        stacks_dict[pos_key] = {
                            "rings": list(stack.rings),
                            "controlling_player": stack.controlling_player,
                        }

                    return has_eligible_elimination_target(
                        stacks_dict,
                        player,
                        elimination_context,
                        exclude_positions=processed_region_keys,
                    )

        # No pending elimination: check for region processing moves
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
) -> ForcedEliminationOutcome | None:
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
