"""Centralized board type utilities.

This module provides a single source of truth for board type handling:

- BoardType enum (re-exported from app.coordination.types)
- Parsing utilities for string -> BoardType conversion
- Config key parsing/building utilities

Usage:
    from app.utils.board_type_utils import (
        BoardType,
        parse_board_type,
        parse_config_key,
        build_config_key,
        ALL_BOARD_TYPES,
    )

    # Parse string to enum
    bt = parse_board_type("hex8")  # Returns BoardType.HEX8
    bt = parse_board_type("Hex-8")  # Also works (case/separator insensitive)

    # Parse config key
    board, players = parse_config_key("hex8_2p")  # Returns (BoardType.HEX8, 2)

    # Build config key
    key = build_config_key(BoardType.HEX8, 2)  # Returns "hex8_2p"

January 2026: Created to consolidate 17+ duplicated board type parsing patterns.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Import canonical BoardType from coordination.types
# This is the single source of truth for BoardType definition
from app.coordination.types import BoardType

__all__ = [
    "BoardType",
    "parse_board_type",
    "parse_config_key",
    "build_config_key",
    "normalize_board_type_string",
    "is_valid_board_type",
    "ALL_BOARD_TYPES",
    "BOARD_TYPE_ALIASES",
    "ParsedConfig",
]


# All valid board types
ALL_BOARD_TYPES: tuple[BoardType, ...] = tuple(BoardType)

# Aliases for board type strings (normalized lowercase, no separators)
BOARD_TYPE_ALIASES: dict[str, BoardType] = {
    # Standard names
    "hex8": BoardType.HEX8,
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hexagonal": BoardType.HEXAGONAL,
    # Short forms
    "sq8": BoardType.SQUARE8,
    "sq19": BoardType.SQUARE19,
    # Descriptive forms
    "hexsmall": BoardType.HEX8,
    "hexlarge": BoardType.HEXAGONAL,
    "fullhex": BoardType.HEXAGONAL,
    "fullhexagonal": BoardType.HEXAGONAL,
    # Legacy aliases
    "hex": BoardType.HEX8,  # Ambiguous but common
}


class ParsedConfig(NamedTuple):
    """Parsed configuration from config key."""

    board_type: BoardType
    num_players: int

    @property
    def config_key(self) -> str:
        """Reconstruct config key."""
        return f"{self.board_type.value}_{self.num_players}p"


def normalize_board_type_string(s: str) -> str:
    """Normalize board type string for matching.

    Converts to lowercase and removes common separators.

    Args:
        s: Raw board type string (e.g., "Hex-8", "SQUARE_19")

    Returns:
        Normalized string (e.g., "hex8", "square19")
    """
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")


def parse_board_type(value: str | BoardType) -> BoardType:
    """Parse string to BoardType enum.

    Handles various input formats:
    - Enum values: BoardType.HEX8 (passed through)
    - Lowercase: "hex8", "square8"
    - Mixed case: "Hex8", "SQUARE19"
    - With separators: "hex-8", "square_19"
    - Aliases: "sq8", "hexsmall"

    Args:
        value: Board type string or enum.

    Returns:
        BoardType enum value.

    Raises:
        ValueError: If board type string is not recognized.

    Examples:
        >>> parse_board_type("hex8")
        <BoardType.HEX8: 'hex8'>
        >>> parse_board_type("Hex-8")
        <BoardType.HEX8: 'hex8'>
        >>> parse_board_type(BoardType.SQUARE8)
        <BoardType.SQUARE8: 'square8'>
    """
    if isinstance(value, BoardType):
        return value

    if not isinstance(value, str):
        raise TypeError(f"Expected str or BoardType, got {type(value).__name__}")

    normalized = normalize_board_type_string(value)

    if normalized in BOARD_TYPE_ALIASES:
        return BOARD_TYPE_ALIASES[normalized]

    # Try matching enum value directly
    for bt in BoardType:
        if bt.value == normalized or bt.name.lower() == normalized:
            return bt

    raise ValueError(
        f"Unknown board type: {value!r}. "
        f"Valid types: {', '.join(bt.value for bt in BoardType)}"
    )


def is_valid_board_type(value: str) -> bool:
    """Check if string is a valid board type.

    Args:
        value: Board type string to check.

    Returns:
        True if valid board type, False otherwise.
    """
    try:
        parse_board_type(value)
        return True
    except (ValueError, TypeError):
        return False


def parse_config_key(config_key: str) -> ParsedConfig:
    """Parse config key into board type and player count.

    Config keys follow the pattern: {board_type}_{num_players}p
    Examples: "hex8_2p", "square19_4p", "hexagonal_3p"

    Args:
        config_key: Configuration key string.

    Returns:
        ParsedConfig with board_type and num_players.

    Raises:
        ValueError: If config key format is invalid.

    Examples:
        >>> parse_config_key("hex8_2p")
        ParsedConfig(board_type=<BoardType.HEX8: 'hex8'>, num_players=2)
        >>> parse_config_key("square19_4p").num_players
        4
    """
    # Try standard format: board_type_Np
    match = re.match(r"^(.+?)_(\d+)p$", config_key, re.IGNORECASE)
    if match:
        board_str, num_players_str = match.groups()
        board_type = parse_board_type(board_str)
        num_players = int(num_players_str)

        if num_players not in (2, 3, 4):
            raise ValueError(
                f"Invalid player count: {num_players}. Must be 2, 3, or 4."
            )

        return ParsedConfig(board_type=board_type, num_players=num_players)

    raise ValueError(
        f"Invalid config key format: {config_key!r}. "
        "Expected format: {{board_type}}_{{num_players}}p (e.g., 'hex8_2p')"
    )


def build_config_key(board_type: str | BoardType, num_players: int) -> str:
    """Build config key from board type and player count.

    Args:
        board_type: Board type string or enum.
        num_players: Number of players (2, 3, or 4).

    Returns:
        Config key string (e.g., "hex8_2p").

    Raises:
        ValueError: If board type is invalid or num_players not in (2, 3, 4).

    Examples:
        >>> build_config_key("hex8", 2)
        'hex8_2p'
        >>> build_config_key(BoardType.SQUARE19, 4)
        'square19_4p'
    """
    bt = parse_board_type(board_type)

    if num_players not in (2, 3, 4):
        raise ValueError(f"Invalid player count: {num_players}. Must be 2, 3, or 4.")

    return f"{bt.value}_{num_players}p"


# Convenience function matching BoardType.from_string() for backward compat
def from_string(s: str) -> BoardType:
    """Parse board type from string (alias for parse_board_type).

    This provides API compatibility with BoardType.from_string().
    """
    return parse_board_type(s)
