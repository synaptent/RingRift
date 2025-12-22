"""Legacy replay phase injection for non-canonical game recordings.

This module provides phase bridging/injection for database recordings that
were created before explicit bookkeeping moves were required. It handles:

1. **Missing bookkeeping moves**: Injects NO_PLACEMENT_ACTION, NO_MOVEMENT_ACTION,
   NO_LINE_ACTION, NO_TERRITORY_ACTION to bridge phase gaps
2. **Phase coercion**: Direct phase mutation for forced_elimination recovery
3. **End-of-turn auto-advance**: Completes turn traversal through empty phases

Violates RR-CANON-R075 (no silent phase fixes) - use only for legacy data replay.

Usage:
    from app.rules.legacy import auto_inject_before_move, auto_inject_no_action_moves

    # Bridge phase gaps before applying recorded move
    state = auto_inject_before_move(state, next_move)

    # Complete turn after move application
    state = auto_inject_no_action_moves(state)

Deprecation Plan:
    - All new recordings should include explicit bookkeeping moves
    - Once canonical database migration is complete, this module can be removed
    - Target removal: Q2 2026
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models import GameState, Move

logger = logging.getLogger(__name__)

# Track usage for deprecation metrics
_injection_calls = 0
_coercion_calls = 0
_warned = False


def auto_inject_before_move(state: "GameState", next_move: "Move") -> "GameState":
    """Auto-inject bookkeeping moves BEFORE applying a recorded move.

    .. deprecated:: 2024.12
        This function violates RR-CANON-R075 (no silent phase fixes).
        Migrate to canonical recordings with explicit bookkeeping moves.
        Target removal: Q2 2026.

    This handles the case where the database recording is missing
    intermediate no-action moves. For example, after NO_LINE_ACTION
    the state is in territory_processing, but the next recorded move
    might be PLACE_RING for a different player. We need to inject
    NO_TERRITORY_ACTION to advance through the territory phase first.

    **WARNING**: This function violates RR-CANON-R075 (no silent phase fixes).
    It should only be used for replaying legacy database recordings that
    were created without explicit bookkeeping moves.

    Args:
        state: Current game state before the next recorded move.
        next_move: The next move from the database that we're about to apply.

    Returns:
        Updated game state with any necessary bookkeeping moves applied.
    """
    warnings.warn(
        "auto_inject_before_move() is deprecated and violates RR-CANON-R075. "
        "Use canonical recordings with explicit bookkeeping moves. "
        "This function will be removed in Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _injection_calls, _coercion_calls, _warned

    # Import here to avoid circular imports
    from app.game_engine import GameEngine
    from app.models import GamePhase, Move as MoveModel, MoveType, Position

    _injection_calls += 1

    # Log deprecation warning once per session
    if not _warned:
        logger.warning(
            "LEGACY_PHASE_INJECTION: Using deprecated phase injection. "
            "This violates RR-CANON-R075. Migrate to canonical recordings with "
            "explicit bookkeeping moves to avoid this code path."
        )
        _warned = True

    # Limit iterations to prevent infinite loops
    max_iterations = 10
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        current_phase = (
            state.current_phase.value
            if hasattr(state.current_phase, "value")
            else str(state.current_phase)
        )

        # Get the next move type for phase coercion checks
        next_type = (
            next_move.type.value
            if hasattr(next_move.type, "value")
            else str(next_move.type)
        )

        # swap_sides is a ring_placement-only meta-move; avoid injecting
        # no-op phase transitions around it and let phase validation handle
        # any out-of-phase records.
        if next_type == "swap_sides":
            break

        # RR-PARITY-FIX: When in ring_placement/movement but the next move
        # is forced_elimination, coerce the phase. This happens when Python's
        # phase machine recorded a forced_elimination mid-turn but the replay
        # engine has already advanced to the next turn's start phase.
        if current_phase in ("ring_placement", "movement") and next_type == "forced_elimination":
            _coercion_calls += 1
            state.current_phase = GamePhase.FORCED_ELIMINATION
            break  # Phase is now correct, exit loop

        # When in ring_placement/movement but the next move is a territory action
        # we need to bridge through earlier phases first.
        territory_moves = (
            "eliminate_rings_from_stack", "choose_territory_option",
            "process_territory_region", "skip_territory_processing", "no_territory_action"
        )
        if current_phase in ("ring_placement", "movement") and next_type in territory_moves:
            # Fall through to specific phase handlers below
            pass

        # When in ring_placement but the next move is NOT a placement move,
        # inject NO_PLACEMENT_ACTION to advance through ring_placement.
        placement_moves = ("place_ring", "no_placement_action", "skip_placement")
        if current_phase == "ring_placement":
            if next_type in placement_moves:
                break
            else:
                no_placement_move = MoveModel(
                    id="legacy-inject-no-placement",
                    type=MoveType.NO_PLACEMENT_ACTION,
                    player=state.current_player,
                    to=Position(x=0, y=0),
                    timestamp=datetime.now(),
                    thinkTime=0,
                    moveNumber=0,
                )
                state = GameEngine.apply_move(state, no_placement_move, trace_mode=True)
                continue

        # When in movement phase but the next move is from a later phase,
        # inject NO_MOVEMENT_ACTION to advance.
        # RR-CANON-FIX (Dec 2025): Check if valid movement moves exist before
        # injecting. This aligns with TypeScript's synthesizeBookkeepingMoves()
        # which respects ANM semantics per RR-CANON-R200.
        movement_moves = (
            "move_stack", "no_movement_action", "recovery_slide",
            "overtaking_capture", "continue_capture_segment", "skip_capture",
            "skip_recovery"
        )
        if current_phase == "movement":
            if next_type not in movement_moves:
                # Check if player has valid movement moves before injecting
                # This prevents incorrect phase advancement when moves exist
                valid_moves = GameEngine.get_valid_moves(state, state.current_player)
                movement_like_types = {
                    MoveType.MOVE_STACK, MoveType.RECOVERY_SLIDE,
                    MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT,
                    MoveType.SKIP_CAPTURE, MoveType.SKIP_RECOVERY,
                }
                has_movement_moves = any(m.type in movement_like_types for m in valid_moves)

                if has_movement_moves:
                    # Player still has movement options - do NOT inject NO_MOVEMENT_ACTION
                    # This matches TS behavior which returns null when moves exist
                    break

                no_movement_move = MoveModel(
                    id="legacy-inject-no-movement",
                    type=MoveType.NO_MOVEMENT_ACTION,
                    player=state.current_player,
                    to=Position(x=0, y=0),
                    timestamp=datetime.now(),
                    thinkTime=0,
                    moveNumber=0,
                )
                state = GameEngine.apply_move(state, no_movement_move, trace_mode=True)
                continue
            else:
                break

        # When in capture phase but next move is from a later phase,
        # inject SKIP_CAPTURE to advance.
        # RR-CANON-FIX (Dec 2025): Check if valid capture moves exist before
        # injecting. This aligns with TypeScript's chain capture semantics.
        capture_moves = ("overtaking_capture", "continue_capture_segment", "skip_capture")
        if current_phase == "capture":
            if next_type not in capture_moves:
                # Check if player has valid capture moves before injecting
                valid_moves = GameEngine.get_valid_moves(state, state.current_player)
                capture_like_types = {
                    MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT,
                }
                has_capture_moves = any(m.type in capture_like_types for m in valid_moves)

                if has_capture_moves:
                    # Player still has capture options - do NOT inject SKIP_CAPTURE
                    break

                skip_capture_move = MoveModel(
                    id="legacy-inject-skip-capture",
                    type=MoveType.SKIP_CAPTURE,
                    player=state.current_player,
                    to=Position(x=0, y=0),
                    timestamp=datetime.now(),
                    thinkTime=0,
                    moveNumber=0,
                )
                state = GameEngine.apply_move(state, skip_capture_move, trace_mode=True)
                continue
            else:
                break

        # Handle territory_processing phase
        if current_phase == "territory_processing":
            if next_type == "forced_elimination":
                _coercion_calls += 1
                state.current_phase = GamePhase.FORCED_ELIMINATION
                break
            territory_actions = (
                "no_territory_action", "process_territory_region",
                "choose_territory_option", "eliminate_rings_from_stack",
                "skip_territory_processing"
            )
            if next_type not in territory_actions:
                no_territory_move = MoveModel(
                    id="legacy-inject-no-territory",
                    type=MoveType.NO_TERRITORY_ACTION,
                    player=state.current_player,
                    to=Position(x=0, y=0),
                    timestamp=datetime.now(),
                    thinkTime=0,
                    moveNumber=0,
                )
                state = GameEngine.apply_move(state, no_territory_move, trace_mode=True)
                continue  # Continue to check if more injections needed
            else:
                break

        # Handle line_processing phase
        elif current_phase == "line_processing":
            line_actions = ("no_line_action", "process_line", "choose_line_option", "choose_line_reward")
            if next_type not in line_actions:
                # RR-PARITY-FIX (Dec 2025): If next move is a territory action,
                # we may need to coerce the phase directly if no_line_action
                # injection would fail (e.g., Python sees lines that TS didn't).
                # This handles hybrid selfplay recordings with different line detection.
                if next_type in territory_moves:
                    # RR-PARITY-FIX (Dec 2025): The recording has a territory action but
                    # Python is in line_processing. This can happen when:
                    # 1. Python sees lines that TS didn't detect (has_line_moves=True)
                    # 2. pending_line_reward_elimination is True (Python detected line reward)
                    # In either case, we need to coerce to territory_processing to match
                    # the recording's expected phase.
                    pending_elim = getattr(state, 'pending_line_reward_elimination', False)
                    valid_moves = GameEngine.get_valid_moves(state, state.current_player)
                    has_line_moves = any(
                        m.type in {MoveType.PROCESS_LINE, MoveType.CHOOSE_LINE_OPTION, MoveType.CHOOSE_LINE_REWARD}
                        for m in valid_moves
                    )
                    if has_line_moves or pending_elim:
                        # Python sees line state that recording skipped - coerce phase
                        _coercion_calls += 1
                        logger.warning(
                            f"PHASE_COERCION: Forcing line_processing → territory_processing "
                            f"(has_line_moves={has_line_moves}, pending_elim={pending_elim})"
                        )
                        state.current_phase = GamePhase.TERRITORY_PROCESSING
                        # Clear pending_line_reward_elimination since we're skipping it
                        if pending_elim:
                            state.pending_line_reward_elimination = False
                        continue

                no_line_move = MoveModel(
                    id="legacy-inject-no-line",
                    type=MoveType.NO_LINE_ACTION,
                    player=state.current_player,
                    to=Position(x=0, y=0),
                    timestamp=datetime.now(),
                    thinkTime=0,
                    moveNumber=0,
                )
                state = GameEngine.apply_move(state, no_line_move, trace_mode=True)
                continue  # Continue to check if more injections needed
            else:
                break
        else:
            # Not in a phase requiring injection
            break

    return state


def auto_inject_no_action_moves(state: "GameState") -> "GameState":
    """Auto-inject NO_LINE_ACTION and NO_TERRITORY_ACTION bookkeeping moves.

    .. deprecated:: 2024.12
        This function violates RR-CANON-R075 (no silent phase fixes).
        Migrate to canonical recordings with explicit bookkeeping moves.
        Target removal: Q2 2026.

    This helper matches TS's replay behavior where the orchestrator
    auto-generates these moves to complete turn traversal through
    phases that have no interactive options.

    **WARNING**: This function violates RR-CANON-R075 (no silent phase fixes).
    Use only for legacy replay compatibility.

    Args:
        state: Current game state after applying a recorded move.

    Returns:
        Updated game state with bookkeeping moves auto-applied.
    """
    warnings.warn(
        "auto_inject_no_action_moves() is deprecated and violates RR-CANON-R075. "
        "Use canonical recordings with explicit bookkeeping moves. "
        "This function will be removed in Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _injection_calls

    from app.game_engine import GameEngine, PhaseRequirementType

    _injection_calls += 1

    # Limit iterations to prevent infinite loops
    max_iterations = 10
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        # Exit if game is not active
        status_value = (
            state.game_status.value
            if hasattr(state.game_status, "value")
            else str(state.game_status)
        )
        if status_value != "active":
            break

        # Check if there's a phase requirement for the current player
        requirement = GameEngine.get_phase_requirement(
            state, state.current_player
        )

        if requirement is None:
            # Interactive moves exist, stop auto-advancing
            break

        # Only auto-inject for line and territory no-action phases
        if requirement.type in (
            PhaseRequirementType.NO_LINE_ACTION_REQUIRED,
            PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED
        ):
            bookkeeping = GameEngine.synthesize_bookkeeping_move(
                requirement, state
            )
            state = GameEngine.apply_move(state, bookkeeping, trace_mode=True)
        else:
            # Other requirements are not auto-injected during replay
            break

    return state


def get_phase_injection_stats() -> dict[str, int]:
    """Get statistics about phase injection usage.

    Returns:
        Dictionary with usage counts for deprecation tracking
    """
    return {
        "injection_calls": _injection_calls,
        "coercion_calls": _coercion_calls,
    }


def reset_phase_injection_stats() -> None:
    """Reset phase injection statistics."""
    global _injection_calls, _coercion_calls, _warned
    _injection_calls = 0
    _coercion_calls = 0
    _warned = False
