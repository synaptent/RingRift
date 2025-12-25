from __future__ import annotations

"""
Phase state machine for the Python GameEngine.

Architecture Note (2025-12):
----------------------------
This module and ``app.rules.fsm`` have **complementary roles**:

1. **Phase Transitions** (this module): ACTIVE and canonical
   - ``advance_phases()`` is the proven, stable implementation for phase advancement
   - Called by ``GameEngine._update_phase()`` after each move is applied
   - Mirrors TypeScript phaseStateMachine + TurnOrchestrator behaviour

2. **Move Validation** (``app.rules.fsm``): ACTIVE and canonical
   - ``validate_move_for_phase()`` validates moves are appropriate for current phase
   - Used for FSM-level move legality checks

This module centralises phase and turn transitions:

- It is responsible for updating ``current_phase``, ``current_player``,
  and related per-turn bookkeeping after a move has been applied.
- It does **not** mutate board geometry or player inventories; those
  are handled by the move-application layer in ``GameEngine.apply_move``.
- It never fabricates Move objects; bookkeeping moves such as
  ``NO_*_ACTION`` and ``FORCED_ELIMINATION`` are surfaced via
  phase requirements and constructed by hosts.

All logic here operates directly on a mutable GameState instance.
"""

from dataclasses import dataclass

from app.models import GamePhase, GameState, Move, MoveType
from app.rules.legacy.move_type_aliases import convert_legacy_move_type


@dataclass
class PhaseTransitionInput:
    """Input to the phase state machine."""

    game_state: GameState
    last_move: Move
    trace_mode: bool = False


def _is_no_board_state_change_move(move_type: MoveType) -> bool:
    """Return True for moves that do NOT change the board state.

    These moves do NOT prevent forced elimination from triggering. A player who
    only makes these moves during their turn (with no placement, movement, or
    capture) must undergo forced elimination if they control stacks.

    Per RR-CANON-R072/R100: Forced elimination triggers when a player has no
    "real progress" actions. This includes:

    1. **Forced no-ops (NO_*_ACTION)**: Player entered a phase with no options.
       These are bookkeeping moves that record the phase was visited.

    2. **Voluntary skips (SKIP_*)**: Player chose to skip an optional action.
       These don't change the board and thus don't prevent forced elimination.
       - SKIP_PLACEMENT: Player has rings but chose not to place
       - SKIP_TERRITORY_PROCESSING: Player has territory options but chose to skip
       - SKIP_CAPTURE: Player has capture available but chose not to take it
       - SKIP_RECOVERY: Player has recovery slide available but chose to skip

    Note: For LPS (Last Player Standing) purposes, the distinction between
    voluntary skip vs forced no-op matters for determining if a player has
    "real actions available". For forced elimination gating, the criterion is
    simpler: did the player change the board state during their turn?
    """
    return move_type in {
        # Forced no-ops (player had no choice)
        MoveType.NO_PLACEMENT_ACTION,
        MoveType.NO_MOVEMENT_ACTION,
        MoveType.NO_LINE_ACTION,
        MoveType.NO_TERRITORY_ACTION,
        # Voluntary skips (player chose not to act, but didn't change board)
        MoveType.SKIP_PLACEMENT,
        MoveType.SKIP_TERRITORY_PROCESSING,
        MoveType.SKIP_CAPTURE,
        MoveType.SKIP_RECOVERY,
    }


def compute_had_any_board_state_change_this_turn(
    game_state: GameState,
    current_move: Move | None = None,
) -> bool:
    """
    Check if the current player made any board state change this turn.

    Returns True if the player made any move that changed the board (placement,
    movement, capture, territory collapse, etc.). Returns False if the player
    only made no-action or skip moves.

    This is the criterion for forced elimination gating: a player who controls
    stacks but made no board state change this turn must undergo forced
    elimination.

    Per RR-CANON-R072/R100: Forced elimination triggers when P controls stacks
    and had no "real progress" actions during their turn. Voluntary skips
    (SKIP_TERRITORY_PROCESSING, SKIP_CAPTURE, etc.) do NOT prevent forced
    elimination because they don't change the board state.
    """
    current_player = game_state.current_player
    history = game_state.move_history

    # Include the current move being applied when it may not yet be in history.
    if (
        current_move is not None
        and current_move.player == current_player
        and not _is_no_board_state_change_move(current_move.type)
    ):
        return True

    for move in reversed(history):
        if move.player != current_player:
            break
        if not _is_no_board_state_change_move(move.type):
            return True
    return False


# Backward compatibility alias
compute_had_any_action_this_turn = compute_had_any_board_state_change_this_turn
_is_no_action_bookkeeping_move = _is_no_board_state_change_move


def player_has_stacks_on_board(game_state: GameState, player: int) -> bool:
    """
    True if ``player`` currently controls at least one stack on the board.

    A stack is counted when stack_height > 0 and controlling_player == player.
    Mirrors the TS playerHasStacksOnBoard helper used for FE gating.
    """
    for stack in game_state.board.stacks.values():
        if stack.controlling_player == player and stack.stack_height > 0:
            return True
    return False


def _did_process_territory_region(game_state: GameState, move: Move) -> bool:
    """
    True iff a CHOOSE_TERRITORY_OPTION move actually collapsed at least one
    space in the chosen region.

    TS `applyProcessTerritoryRegionDecision` treats non-processable regions as a
    no-op (state unchanged). In that case, the move must NOT be allowed to
    advance/exit TERRITORY_PROCESSING; hosts must still record NO_TERRITORY_ACTION
    to mark the phase as visited (RR-CANON-R075).

    Note: Uses defensive hasattr() checks because Territory class uses 'spaces'
    attribute, not 'positions'. This was the root cause of the hexagonal parity
    bug (fixed in commit 7f43c368).
    """
    board = game_state.board

    # Check if move.to is in collapsed_spaces
    if move.to is not None:
        to_key = move.to.to_key()
        if to_key in board.collapsed_spaces:
            return True

    # Check disconnected_regions - Territory.spaces is a list[Position]
    if move.disconnected_regions:
        for region in move.disconnected_regions:
            if hasattr(region, "spaces"):
                for pos in region.spaces:
                    pos_key = pos.to_key() if hasattr(pos, "to_key") else f"{pos.x},{pos.y}"
                    if pos_key in board.collapsed_spaces:
                        return True

    return False


def _on_line_processing_complete(
    game_state: GameState,
    *,
    trace_mode: bool,
    last_move: Move | None = None,
) -> None:
    """
    Shared helper for exiting LINE_PROCESSING.

    Mirrors TS processPostMovePhases behavior after line-phase completion:

    - RR-CANON-R123: If pending_line_reward_elimination is True, the player must
      execute an eliminate_rings_from_stack move. Stay in LINE_PROCESSING.
    - If at least one canonical territory region exists, the next phase is
      `territory_processing`.
    - If no territory regions exist AND the player had **no actions this turn**
      AND they still control stacks, the next phase is `forced_elimination`
      (per RR-CANON-R070/R204).
    - Otherwise, advance to territory_processing where _on_territory_processing_complete
      will handle the FE/turn-end decision.

    RR-FIX-HEX-PARITY-01-2025-12-21: Previously this function always advanced to
    territory_processing, but TS can skip directly to forced_elimination when
    !hasTerritoryRegions && !hadAnyActionThisTurn && playerHasStacks. This caused
    hex board parity divergence where Python stayed in territory_processing while
    TS transitioned to forced_elimination after no_line_action.

    RR-FIX-HEX-PARITY-03-2025-12-21: Check pending_line_reward_elimination before
    advancing. TS stays in line_processing when this flag is set, requiring an
    explicit eliminate_rings_from_stack move.
    """
    from app.game_engine import GameEngine

    # RR-CANON-R123: If pending line elimination, stay in line_processing
    # This matches TS turnOrchestrator.ts:2766-2785 where eliminationRewardPending
    # causes the engine to stay in line_processing and surface elimination options.
    if game_state.pending_line_reward_elimination:
        # Stay in LINE_PROCESSING - player must execute elimination move
        return

    current_player = game_state.current_player
    had_any_action = compute_had_any_action_this_turn(game_state, current_move=last_move)
    has_stacks = player_has_stacks_on_board(game_state, current_player)

    # Check if any territory regions exist (mirrors TS processPostMovePhases)
    territory_moves = GameEngine._get_territory_processing_moves(game_state, current_player)
    has_territory_regions = len(territory_moves) > 0

    if not has_territory_regions and not had_any_action and has_stacks:
        # Per TS parity: skip territory_processing and go directly to forced_elimination
        # when no territory regions exist and player had no actions but has material.
        # This matches TS onLineProcessingComplete -> 'forced_elimination' branch.
        game_state.current_phase = GamePhase.FORCED_ELIMINATION
        return

    # Otherwise, advance to territory_processing
    GameEngine._advance_to_territory_processing(
        game_state,
        trace_mode=trace_mode,
    )


def _on_territory_processing_complete(
    game_state: GameState,
    *,
    trace_mode: bool,
    last_move: Move | None = None,
) -> None:
    """
    Shared helper for exiting TERRITORY_PROCESSING.

    Mirrors TurnStateMachine.onTerritoryProcessingComplete + orchestrator
    behaviour:

    - If the active player had **no actions at all this turn** and still controls
      stacks, enter FORCED_ELIMINATION.
    - Otherwise, perform a full turn end via GameEngine._end_turn so that the
      next player's FE/ANM gating is applied consistently.
    """
    from app.game_engine import GameEngine

    current_player = game_state.current_player
    had_any_action = compute_had_any_action_this_turn(game_state, current_move=last_move)
    has_stacks = player_has_stacks_on_board(game_state, current_player)

    if not had_any_action and has_stacks:
        game_state.current_phase = GamePhase.FORCED_ELIMINATION
        return

    GameEngine._end_turn(game_state, trace_mode=trace_mode)


def advance_phases(inp: PhaseTransitionInput) -> None:
    """Advance phases and turn rotation for ``inp.game_state``.

    This mirrors the semantics of GameEngine._update_phase plus the helper
    transitions for line/territory/turn-end, but is factored into a
    dedicated module for clarity and easier testing.

    The function mutates ``inp.game_state`` in-place and does not return a
    value. Hosts should call victory and invariant checks separately after
    phase advancement.
    """
    # Local import to avoid circular imports at module load time.
    from app.game_engine import GameEngine

    game_state = inp.game_state
    last_move = inp.last_move
    trace_mode = inp.trace_mode

    current_player = game_state.current_player

    # Normalize the move type to canonical MoveType enum
    raw_move_type = last_move.type.value if hasattr(last_move.type, "value") else str(last_move.type)
    normalized_type_str = convert_legacy_move_type(raw_move_type, warn=False)
    try:
        normalized_type = MoveType(normalized_type_str)
    except ValueError:
        normalized_type = last_move.type if isinstance(last_move.type, MoveType) else None

    if normalized_type == MoveType.FORCED_ELIMINATION:
        # Per RR-CANON-R070 and the shared TS engine, explicit
        # FORCED_ELIMINATION is the seventh and final phase of a turn.
        # After applying the elimination move, the current player has no
        # further interactive actions this turn; we rotate directly to
        # the next active player (skipping players with no material) and
        # start their turn in ring_placement or movement.
        #
        # Victory detection is handled by _check_victory after this
        # phase update, mirroring the TS TurnOrchestrator flow where
        # forced_elimination is followed by a victory check and then
        # turn rotation.
        #
        # RR-PARITY-FIX-2025-12-16: Use _end_turn instead of _rotate_to_next_active_player
        # to match TS semantics. The key difference is that TS computes the next player
        # starting from move.player (via computeNextNonEliminatedPlayer(gameState, move.player, ...)),
        # while _rotate_to_next_active_player starts from game_state.current_player.
        # Using _end_turn ensures:
        # 1. Victory is checked properly if the forced elimination ended the game
        # 2. The rotation uses the correct starting point (current_player at the time
        #    of the move, which should match move.player)
        # 3. Phase transitions to ring_placement for the next player
        GameEngine._end_turn(game_state, trace_mode=trace_mode)

    elif normalized_type == MoveType.PLACE_RING:
        # RR-PARITY-FIX-2025-12-25: Per RR-CANON-R075, all phases must be
        # visited with explicit moves. Always transition to MOVEMENT after
        # placement - if no movement/capture options exist, the player must
        # emit NO_MOVEMENT_ACTION which advances to line_processing.
        # No phase skipping is permitted. This matches TS turnLogic.ts
        # lines 171-178 which unconditionally sets currentPhase: 'movement'.
        game_state.current_phase = GamePhase.MOVEMENT

    elif normalized_type == MoveType.SKIP_PLACEMENT:
        # Skipping placement is only allowed when movement or capture is
        # already available. After a legal skip, always enter movement.
        game_state.current_phase = GamePhase.MOVEMENT

    elif normalized_type == MoveType.NO_PLACEMENT_ACTION:
        # Explicit no-op placement: always advance to MOVEMENT so that the
        # rest of the turn (movement/capture/line/territory/FE) can run.
        game_state.current_phase = GamePhase.MOVEMENT

    elif normalized_type == MoveType.MOVE_STACK:
        # After movement, check for captures from the landing position.
        # Per RR-CANON-R093, post-movement captures are only available
        # from the stack that just moved, at its landing position.
        # Clear any stale chain_capture_state to ensure the first capture
        # is classified as OVERTAKING_CAPTURE.
        game_state.chain_capture_state = None

        # Import here to avoid circular dependency
        from app.rules.capture_chain import enumerate_capture_moves_py

        attacker_pos = last_move.to
        if attacker_pos is None:
            GameEngine._advance_to_line_processing(game_state, trace_mode=trace_mode)
        else:
            capture_moves = enumerate_capture_moves_py(
                game_state,
                current_player,
                attacker_pos,
                kind="initial",
            )

            if capture_moves:
                game_state.current_phase = GamePhase.CAPTURE
            else:
                GameEngine._advance_to_line_processing(game_state, trace_mode=trace_mode)

    elif normalized_type == MoveType.NO_MOVEMENT_ACTION:
        # Movement phase was visited but no legal movement or capture
        # existed anywhere for the player. Per RR-CANON-R075, this is
        # recorded as an explicit NO_MOVEMENT_ACTION and we advance
        # directly to line_processing.
        GameEngine._advance_to_line_processing(game_state, trace_mode=trace_mode)

    elif normalized_type == MoveType.SKIP_CAPTURE:
        # Explicitly decline optional post-movement capture and proceed to
        # line_processing (RR-CANON-R073). Mirrors TS TurnOrchestrator.
        game_state.chain_capture_state = None
        GameEngine._advance_to_line_processing(game_state, trace_mode=trace_mode)

    elif normalized_type == MoveType.RECOVERY_SLIDE:
        # Recovery slide is a movement-phase action (but NOT a "real action" for LPS).
        # After applying it, we proceed to line_processing to record phase traversal
        # (even if no further line decisions remain because collapse was applied).
        game_state.chain_capture_state = None
        GameEngine._advance_to_line_processing(game_state, trace_mode=trace_mode)

    elif normalized_type == MoveType.SKIP_RECOVERY:
        # RR-CANON-R115: recovery-eligible players may explicitly skip recovery
        # to preserve buried rings. This ends MOVEMENT for the player and
        # advances to line_processing (even though no board change occurred).
        game_state.chain_capture_state = None
        GameEngine._advance_to_line_processing(game_state, trace_mode=trace_mode)

    elif normalized_type in (
        MoveType.OVERTAKING_CAPTURE,
        MoveType.CONTINUE_CAPTURE_SEGMENT,
    ):
        # Check for more captures (chain)
        capture_moves = GameEngine._get_capture_moves(
            game_state,
            current_player,
        )
        if capture_moves:
            # After the first capture, subsequent captures are chain captures
            game_state.current_phase = GamePhase.CHAIN_CAPTURE
        else:
            # End of chain
            GameEngine._advance_to_line_processing(game_state, trace_mode=trace_mode)

    elif normalized_type in (
        MoveType.PROCESS_LINE,
        MoveType.CHOOSE_LINE_OPTION,
    ):
        # After processing a line decision, check if there are more interactive
        # line-processing moves for the current player. If not, delegate to the
        # canonical post-line helper which decides between TERRITORY_PROCESSING,
        # FORCED_ELIMINATION, and turn-end based on hadAnyActionThisTurn and the
        # player's remaining material, mirroring the TS
        # TurnStateMachine.onLineProcessingComplete semantics.
        remaining_lines = [
            m
            for m in GameEngine._get_line_processing_moves(
                game_state,
                current_player,
            )
            if m.type != MoveType.NO_LINE_ACTION
        ]
        if remaining_lines:
            # Stay in line_processing; hosts will surface the next PROCESS_LINE or
            # CHOOSE_LINE_OPTION move.
            game_state.current_phase = GamePhase.LINE_PROCESSING
        else:
            _on_line_processing_complete(
                game_state,
                trace_mode=trace_mode,
                last_move=last_move,
            )

    elif normalized_type == MoveType.NO_LINE_ACTION:
        # Forced no-op: player entered line_processing but had no lines to process.
        # Per RR-CANON-R075, this move marks that the phase was visited. After the
        # final line-phase move (including NO_LINE_ACTION), delegate to the shared
        # post-line helper so that we either:
        # - enter TERRITORY_PROCESSING when regions exist;
        # - enter FORCED_ELIMINATION when the player had no actions this turn but
        #   still controls stacks; or
        # - end the turn and rotate to the next player.
        _on_line_processing_complete(
            game_state,
            trace_mode=trace_mode,
            last_move=last_move,
        )
        # If we landed in TERRITORY_PROCESSING with no regions, surface the
        # requirement so hosts record NO_TERRITORY_ACTION (prevents silent
        # territory skip leaving the next placement in the wrong phase).
        remaining_regions = GameEngine._get_territory_processing_moves(
            game_state,
            current_player,
        )
        if game_state.current_phase == GamePhase.TERRITORY_PROCESSING and not remaining_regions:
            game_state.current_phase = GamePhase.TERRITORY_PROCESSING  # explicit
        # If line processing had no interactive decisions and territory also has
        # no regions, make sure hosts see the no-territory requirement so they
        # record NO_TERRITORY_ACTION rather than leaving the recorder mid-turn.
        remaining_regions = GameEngine._get_territory_processing_moves(
            game_state,
            current_player,
        )
        if not remaining_regions and game_state.current_phase == GamePhase.TERRITORY_PROCESSING:
            # Explicitly set the phase; hosts will synthesize NO_TERRITORY_ACTION
            # via get_phase_requirement → synthesize_bookkeeping_move.
            game_state.current_phase = GamePhase.TERRITORY_PROCESSING

    elif normalized_type == MoveType.ELIMINATE_RINGS_FROM_STACK:
        # After an explicit ELIMINATE_RINGS_FROM_STACK decision, re-evaluate whether
        # more territory decisions remain for the **same player**.
        #
        # This follows the same pattern as CHOOSE_TERRITORY_OPTION: check for remaining
        # regions and call _on_territory_processing_complete() when none remain.
        # Per TS processPostMovePhases, after eliminate_rings_from_stack, if no more
        # territory regions exist, advance to the next player's ring_placement.
        remaining_regions = GameEngine._get_territory_processing_moves(
            game_state,
            current_player,
        )

        if remaining_regions:
            # Stay in territory_processing and keep the current player; hosts will
            # surface the next territory decision (choose_territory_option or another
            # eliminate_rings_from_stack).
            game_state.current_phase = GamePhase.TERRITORY_PROCESSING
        else:
            # No further territory decisions remain; delegate to the shared
            # post-territory helper so that we either:
            # - enter FORCED_ELIMINATION when the player had no actions this
            #   entire turn but still controls stacks; or
            # - end the turn and rotate to the next player.
            _on_territory_processing_complete(
                game_state,
                trace_mode=trace_mode,
                last_move=last_move,
            )

    elif normalized_type == MoveType.CHOOSE_TERRITORY_OPTION:
        # TS treat non-processable territory decisions as a no-op, which must NOT
        # end the phase. Keep the current player in TERRITORY_PROCESSING so hosts
        # can emit NO_TERRITORY_ACTION when no processable regions exist.
        if not _did_process_territory_region(game_state, last_move):
            game_state.current_phase = GamePhase.TERRITORY_PROCESSING
            return

        # After processing a disconnected territory region (choose_territory_option),
        # re-evaluate whether more territory decisions remain for the **same player**.
        # This mirrors the TS orchestrator
        # which stays in territory_processing until all regions are resolved.
        remaining_regions = GameEngine._get_territory_processing_moves(
            game_state,
            current_player,
        )

        if remaining_regions:
            # Stay in territory_processing and keep the current player; hosts will
            # surface the next CHOOSE_TERRITORY_OPTION decision.
            game_state.current_phase = GamePhase.TERRITORY_PROCESSING
        else:
            # No further territory decisions remain; delegate to the shared
            # post-territory helper so that we either:
            # - enter FORCED_ELIMINATION when the player had no actions this
            #   entire turn but still controls stacks; or
            # - end the turn and rotate to the next player.
            _on_territory_processing_complete(
                game_state,
                trace_mode=trace_mode,
                last_move=last_move,
            )

    elif normalized_type == MoveType.SKIP_TERRITORY_PROCESSING:
        # Voluntary stop: territory processing is an optional subset. Treat the
        # phase as complete for this player and delegate to the shared
        # post-territory helper so we either enter FORCED_ELIMINATION (ANM +
        # material) or end the turn and rotate to the next player.
        _on_territory_processing_complete(
            game_state,
            trace_mode=trace_mode,
            last_move=last_move,
        )

    elif normalized_type == MoveType.NO_TERRITORY_ACTION:
        # Forced no-op: player entered territory_processing but had no eligible
        # regions. Per RR-CANON-R075, this move marks that the phase was visited.
        # After a NO_TERRITORY_ACTION, treat territory_processing as complete for
        # this player and delegate to the canonical post-territory helper.
        _on_territory_processing_complete(
            game_state,
            trace_mode=trace_mode,
            last_move=last_move,
        )
