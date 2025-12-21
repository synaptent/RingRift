"""Legacy phase auto-advance for selfplay data missing explicit bookkeeping moves.

This module provides phase auto-advancement for game data that was recorded
without explicit bookkeeping moves (NO_LINE_ACTION, NO_TERRITORY_ACTION, etc.).

Violates RR-CANON-R075 (no silent phase fixes) - use only for legacy data replay.

Usage:
    from app.rules.legacy import auto_advance_phase

    # For GPU selfplay or old Gumbel data missing bookkeeping moves
    state = auto_advance_phase(state)

Deprecation Plan:
    - All new selfplay data should include explicit bookkeeping moves
    - Once all legacy data is migrated/regenerated, this module can be removed
    - Target removal: Q2 2026
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

from app.rules.legacy._deprecation import deprecated_legacy

if TYPE_CHECKING:
    from app.models import GameState

logger = logging.getLogger(__name__)

# Track usage for deprecation metrics
_auto_advance_calls = 0
_warned = False


def auto_advance_phase(state: "GameState", max_iterations: int = 100) -> "GameState":
    """Auto-advance through phase-handling moves until at a playable phase.

    .. deprecated:: 2024.12
        This function violates RR-CANON-R075 (no silent phase fixes).
        Use explicit bookkeeping moves in selfplay data instead.
        Target removal: Q2 2026.

    GPU selfplay and some older Gumbel MCTS data skips explicit phase moves
    (line/territory processing). This helper inserts the required bookkeeping
    moves to advance the state to a playable phase.

    **WARNING**: This function violates RR-CANON-R075 (no silent phase fixes).
    It should only be used for replaying legacy game data that was recorded
    without explicit bookkeeping moves.

    Args:
        state: Current game state
        max_iterations: Safety limit to prevent infinite loops

    Returns:
        Game state advanced to a playable phase (placement, movement, capture)

    Note:
        Playable phases are: RING_PLACEMENT, MOVEMENT, CAPTURE, CHAIN_CAPTURE
        Non-playable phases requiring auto-advance: LINE_PROCESSING, TERRITORY_PROCESSING
    """
    warnings.warn(
        "auto_advance_phase() is deprecated and violates RR-CANON-R075. "
        "Generate selfplay data with explicit bookkeeping moves. "
        "This function will be removed in Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _auto_advance_calls, _warned

    # Import here to avoid circular imports
    from app.models import GamePhase, GameStatus, Move, MoveType
    from app.rules import GameEngine

    _auto_advance_calls += 1

    # Log deprecation warning once per session
    if not _warned:
        logger.warning(
            "LEGACY_PHASE_AUTO_ADVANCE: Using deprecated phase auto-advance. "
            "This violates RR-CANON-R075. Generate new selfplay data with "
            "explicit bookkeeping moves to avoid this code path."
        )
        _warned = True

    def _get_status(s: "GameState") -> str:
        return s.game_status.value if hasattr(s.game_status, "value") else str(s.game_status)

    safety = 0
    while _get_status(state) == "active" and safety < max_iterations:
        safety += 1
        phase = state.current_phase
        player = state.current_player

        if phase == GamePhase.LINE_PROCESSING:
            # Auto-process lines
            line_moves = GameEngine._get_line_processing_moves(state, player)
            if line_moves:
                choose_moves = [m for m in line_moves if m.type == MoveType.CHOOSE_LINE_OPTION]
                process_moves = [m for m in line_moves if m.type == MoveType.PROCESS_LINE]
                picked = choose_moves[0] if choose_moves else (process_moves[0] if process_moves else None)
                if picked:
                    state = GameEngine.apply_move(state, picked)
                    continue
            # No lines - insert NO_LINE_ACTION
            no_line = Move(
                id="legacy-auto-no-line",
                type=MoveType.NO_LINE_ACTION,
                player=player,
                timestamp=state.last_move_at,
                think_time=0,
                move_number=0,
            )
            state = GameEngine.apply_move(state, no_line)
            continue

        if phase == GamePhase.TERRITORY_PROCESSING:
            # Auto-process territory
            terr_moves = GameEngine._get_territory_processing_moves(state, player)
            if terr_moves:
                state = GameEngine.apply_move(state, terr_moves[0])
                continue
            # No territory to process
            no_terr = Move(
                id="legacy-auto-no-territory",
                type=MoveType.NO_TERRITORY_ACTION,
                player=player,
                timestamp=state.last_move_at,
                think_time=0,
                move_number=0,
            )
            state = GameEngine.apply_move(state, no_terr)
            continue

        # Check for phase requirements (handles forced elimination, etc.)
        requirement = GameEngine.get_phase_requirement(state, player)
        if requirement is not None:
            synthesized = GameEngine.synthesize_bookkeeping_move(requirement, state)
            if synthesized:
                state = GameEngine.apply_move(state, synthesized)
                continue

        # At a playable phase (placement, movement, capture)
        break

    if safety >= max_iterations:
        logger.error(
            f"LEGACY_PHASE_AUTO_ADVANCE: Hit safety limit ({max_iterations}) "
            f"while auto-advancing. Game may be in invalid state. "
            f"Phase: {state.current_phase}, Player: {state.current_player}"
        )

    return state


def get_auto_advance_stats() -> dict[str, int]:
    """Get statistics about auto-advance usage.

    Returns:
        Dictionary with usage count for deprecation tracking
    """
    return {
        "auto_advance_calls": _auto_advance_calls,
    }


def reset_auto_advance_stats() -> None:
    """Reset auto-advance statistics."""
    global _auto_advance_calls, _warned
    _auto_advance_calls = 0
    _warned = False
