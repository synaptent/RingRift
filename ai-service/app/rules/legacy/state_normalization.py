"""Legacy state normalization for backwards-compatible game replay.

This module handles conversion of legacy game state formats to canonical
representations. This is necessary for replaying games recorded under
previous versions of the rules engine.

Historical State Format Changes:
    - "finished" status was renamed to "completed" (canonical)
    - Phase names were normalized to lowercase
    - Some phase enum values changed during consolidation

Usage:
    from app.rules.legacy.state_normalization import normalize_legacy_state

    canonical_state = normalize_legacy_state(legacy_state_dict)
"""

from __future__ import annotations

import logging
from typing import Any

from app.rules.legacy._deprecation import deprecated_legacy

logger = logging.getLogger(__name__)

# Legacy status values and their canonical equivalents
LEGACY_STATUS_MAPPING: dict[str, str] = {
    "finished": "completed",
    "FINISHED": "completed",
    "Finished": "completed",
    "ended": "completed",
    "ENDED": "completed",
    "done": "completed",
    "DONE": "completed",
    "in_progress": "active",
    "IN_PROGRESS": "active",
    "InProgress": "active",
    "started": "active",
    "STARTED": "active",
    "pending": "active",
    "PENDING": "active",
    "waiting": "active",
    "WAITING": "active",
}

# Legacy phase names and their canonical equivalents
LEGACY_PHASE_MAPPING: dict[str, str] = {
    # Uppercase variants
    "RING_PLACEMENT": "ring_placement",
    "MOVEMENT": "movement",
    "CAPTURE": "capture",
    "LINE_FORMATION": "line_processing",
    "LINE_REWARD": "line_processing",
    "TERRITORY": "territory_processing",
    "RECOVERY": "movement",
    "GAME_OVER": "game_over",
    # Alternative names
    "PLACEMENT": "ring_placement",
    "PLACE_RINGS": "ring_placement",
    "CAPTURING": "capture",
    "LINE": "line_processing",
    "LINES": "line_processing",
    "RECOVER": "movement",
    "ENDED": "game_over",
    "FINISHED": "game_over",
    # CamelCase variants (from old serialization)
    "RingPlacement": "ring_placement",
    "LineFormation": "line_processing",
    "LineReward": "line_processing",
    "GameOver": "game_over",
}


@deprecated_legacy()
def normalize_legacy_status(status: str) -> str:
    """Normalize a legacy game status to canonical form.

    Args:
        status: Game status string (legacy or canonical)

    Returns:
        Canonical status string (lowercase: 'active', 'completed', etc.)

    Examples:
        >>> normalize_legacy_status("finished")
        'completed'
        >>> normalize_legacy_status("active")
        'active'
    """
    if status in LEGACY_STATUS_MAPPING:
        canonical = LEGACY_STATUS_MAPPING[status]
        logger.debug(f"LEGACY_STATUS: Normalized '{status}' -> '{canonical}'")
        return canonical
    return status.lower()


@deprecated_legacy()
def normalize_legacy_phase(phase: str) -> str:
    """Normalize a legacy game phase to canonical form.

    Args:
        phase: Game phase string (legacy or canonical)

    Returns:
        Canonical phase string (lowercase snake_case)

    Examples:
        >>> normalize_legacy_phase("RING_PLACEMENT")
        'ring_placement'
        >>> normalize_legacy_phase("LineFormation")
        'line_formation'
    """
    raw = phase.strip()
    if raw in LEGACY_PHASE_MAPPING:
        canonical = LEGACY_PHASE_MAPPING[raw]
        logger.debug(f"LEGACY_PHASE: Normalized '{phase}' -> '{canonical}'")
        return canonical

    upper = raw.upper()
    if upper in LEGACY_PHASE_MAPPING:
        canonical = LEGACY_PHASE_MAPPING[upper]
        logger.debug(f"LEGACY_PHASE: Normalized '{phase}' -> '{canonical}'")
        return canonical
    return raw.lower()


@deprecated_legacy()
def normalize_legacy_state(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize a legacy game state dictionary to canonical form.

    This function performs in-place normalization of known legacy fields.
    Unknown fields are passed through unchanged.

    Args:
        state_dict: Game state dictionary (will be modified in-place)

    Returns:
        The same dictionary with normalized fields

    Normalized fields:
        - game_status / status: Normalized to canonical status
        - current_phase / phase: Normalized to canonical phase
        - moves[].move_type: Normalized via move_type_aliases

    Examples:
        >>> state = {"game_status": "finished", "current_phase": "MOVEMENT"}
        >>> normalize_legacy_state(state)
        {'game_status': 'completed', 'current_phase': 'movement'}
    """
    # Normalize game status
    for status_key in ("game_status", "status", "gameStatus"):
        if status_key in state_dict:
            state_dict[status_key] = normalize_legacy_status(
                str(state_dict[status_key])
            )

    # Normalize game phase
    for phase_key in ("current_phase", "phase", "currentPhase"):
        if phase_key in state_dict:
            state_dict[phase_key] = normalize_legacy_phase(
                str(state_dict[phase_key])
            )

    # Normalize moves if present
    moves = state_dict.get("moves") or state_dict.get("move_history") or []
    if moves:
        from app.rules.legacy.move_type_aliases import convert_legacy_move_type

        for move in moves:
            if isinstance(move, dict):
                for type_key in ("move_type", "type", "moveType"):
                    if type_key in move:
                        move[type_key] = convert_legacy_move_type(
                            str(move[type_key]), warn=False
                        )

    return state_dict


@deprecated_legacy()
def infer_phase_from_moves(moves: list[dict[str, Any]]) -> str | None:
    """Infer the current game phase from move history.

    This is a fallback for legacy games that may have missing phase data.

    Args:
        moves: List of move dictionaries

    Returns:
        Inferred phase string, or None if cannot be determined
    """
    if not moves:
        return "ring_placement"

    last_move = moves[-1]
    move_type = last_move.get("move_type") or last_move.get("type") or ""
    move_type = move_type.lower()

    # Infer phase from most recent move type
    phase_hints = {
        "place_ring": "ring_placement",
        "move_ring": "movement",
        "move_stack": "movement",
        "build_stack": "movement",
        "overtaking_capture": "capture",
        "continue_capture_segment": "chain_capture",
        "capture": "capture",
        "skip_capture": "capture",
        "process_line": "line_processing",
        "choose_line_option": "line_processing",
        "choose_line_reward": "line_processing",
        "no_line_action": "line_processing",
        "form_line": "line_processing",
        "no_line": "line_processing",
        "choose_line": "line_processing",
        "choose_territory_option": "territory_processing",
        "process_territory_region": "territory_processing",
        "skip_territory_processing": "territory_processing",
        "no_territory_action": "territory_processing",
        "claim_territory": "territory_processing",
        "choose_territory": "territory_processing",
        "recovery_slide": "movement",
        "skip_recovery": "movement",
        "recovery": "movement",
        "forced_elimination": "forced_elimination",
        "swap_sides": "ring_placement",  # Swap happens during placement
    }

    for hint_prefix, phase in phase_hints.items():
        if move_type.startswith(hint_prefix):
            return phase

    return None
