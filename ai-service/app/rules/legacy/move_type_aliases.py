"""Legacy move type aliases for backwards-compatible game replay.

This module handles conversion between legacy move type names and their
canonical equivalents as defined in RULES_CANONICAL_SPEC.md.

Historical Move Type Changes:
    - "CHOOSE_LINE_REWARD" was renamed to "CHOOSE_LINE_OPTION" in Dec 2024
    - "PROCESS_TERRITORY_REGION" was renamed to "CHOOSE_TERRITORY_OPTION"
    - Various phase-specific move types were consolidated

Usage:
    from app.rules.legacy.move_type_aliases import convert_legacy_move_type

    canonical_type = convert_legacy_move_type("CHOOSE_LINE_REWARD")
    # Returns: "choose_line_option"
"""

from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)

# Mapping from legacy move type names to canonical equivalents
# Keys are uppercase legacy names, values are lowercase canonical names
LEGACY_TO_CANONICAL_MOVE_TYPE: Final[dict[str, str]] = {
    # Legacy movement aliases
    "MOVE_RING": "move_stack",
    "BUILD_STACK": "move_stack",
    # Line phase renames (December 2024)
    "CHOOSE_LINE_REWARD": "choose_line_option",
    "LINE_REWARD": "choose_line_option",
    "LINE_CHOICE": "choose_line_option",
    # Territory phase renames
    "PROCESS_TERRITORY_REGION": "choose_territory_option",
    "TERRITORY_REGION": "choose_territory_option",
    "TERRITORY_CHOICE": "choose_territory_option",
    "TERRITORY_CLAIM": "choose_territory_option",
    # Line processing legacy naming
    "LINE_FORMATION": "process_line",
    # Phase-specific action consolidation
    "NO_LINE_REWARD": "no_line_action",
    "SKIP_LINE_REWARD": "no_line_action",
    "NO_TERRITORY_REWARD": "no_territory_action",
    "SKIP_TERRITORY_REWARD": "no_territory_action",
    # Capture phase aliases (legacy capture => overtaking_capture)
    "CAPTURE_RING": "overtaking_capture",
    "PERFORM_CAPTURE": "overtaking_capture",
    "CAPTURE": "overtaking_capture",
    "CHAIN_CAPTURE": "continue_capture_segment",
    # Recovery phase aliases
    "RECOVER_RING": "recovery_slide",
    "STACK_RECOVERY": "recovery_slide",
    # Elimination aliases
    "ELIMINATE_PLAYER": "forced_elimination",
    "FORCED_ELIMINATE": "forced_elimination",
    # Swap rule aliases
    "PIE_RULE": "swap_sides",
    "SWAP_SIDES_ACCEPTED": "swap_sides",
    "SWAP_COLORS": "swap_sides",
}

# Set of known legacy move types for quick lookup
_LEGACY_MOVE_TYPES: Final[frozenset[str]] = frozenset(LEGACY_TO_CANONICAL_MOVE_TYPE.keys())


def is_legacy_move_type(move_type: str) -> bool:
    """Check if a move type string is a legacy name.

    Args:
        move_type: Move type string (case-insensitive)

    Returns:
        True if this is a legacy move type that needs conversion
    """
    return move_type.upper() in _LEGACY_MOVE_TYPES


def convert_legacy_move_type(move_type: str, warn: bool = True) -> str:
    """Convert a legacy move type to its canonical equivalent.

    If the move type is not a legacy alias, it is returned as-is (lowercased).

    Args:
        move_type: Move type string (legacy or canonical)
        warn: If True, log a deprecation warning for legacy types

    Returns:
        Canonical move type string (lowercase)

    Examples:
        >>> convert_legacy_move_type("CHOOSE_LINE_REWARD")
        'choose_line_option'
        >>> convert_legacy_move_type("place_ring")
        'place_ring'
    """
    upper = move_type.upper()

    if upper in LEGACY_TO_CANONICAL_MOVE_TYPE:
        canonical = LEGACY_TO_CANONICAL_MOVE_TYPE[upper]
        if warn:
            logger.warning(
                f"LEGACY_MOVE_TYPE: '{move_type}' is deprecated, "
                f"use canonical '{canonical}' instead. "
                f"This conversion will be removed in a future release."
            )
        return canonical

    # Not a legacy alias, return lowercase canonical form
    return move_type.lower()


def get_all_legacy_aliases() -> dict[str, str]:
    """Get all legacy move type aliases and their canonical equivalents.

    Returns:
        Dictionary mapping legacy names to canonical names
    """
    return dict(LEGACY_TO_CANONICAL_MOVE_TYPE)
