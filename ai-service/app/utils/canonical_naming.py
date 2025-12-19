"""Canonical naming utilities for board types and config keys.

This module provides the SINGLE SOURCE OF TRUTH for board type naming
and configuration key formatting throughout the codebase.

CANONICAL BOARD TYPE VALUES:
    - square8    (8x8 board, 64 cells)
    - square19   (19x19 board, 361 cells)
    - hex8       (radius-4 hex, diameter 8, 61 cells) - comparable to square8
    - hexagonal  (radius-12 hex, diameter 24, 469 cells) - larger than square19

NAMING CONVENTION FOR HEX BOARDS:
    The number in hex board names refers to the "diameter" (2 * radius):
    - hex8 = radius 4, diameter 8, 61 cells
    - hex24 (alias for hexagonal) = radius 12, diameter 24, 469 cells

CANONICAL CONFIG KEY FORMAT:
    {board_type}_{num_players}p
    Examples: square8_2p, hexagonal_4p, hex8_3p
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from app.models import BoardType

__all__ = [
    "normalize_board_type",
    "get_board_type_enum",
    "make_config_key",
    "parse_config_key",
    "is_valid_board_type",
    "get_all_config_keys",
    "normalize_database_filename",
    "CANONICAL_CONFIG_KEYS",
]


# Mapping of common aliases to canonical board type values
_BOARD_TYPE_ALIASES: dict[str, str] = {
    # Square 8x8 variants
    "square8": "square8",
    "sq8": "square8",
    "square_8": "square8",
    "square-8": "square8",
    "8x8": "square8",
    # Square 19x19 variants
    "square19": "square19",
    "sq19": "square19",
    "square_19": "square19",
    "square-19": "square19",
    "19x19": "square19",
    # Hex8 (small hex) variants
    "hex8": "hex8",
    "hex_8": "hex8",
    "hex-8": "hex8",
    "smallhex": "hex8",
    "small_hex": "hex8",
    # Hexagonal (large hex) variants - radius 12, diameter 24, 469 cells
    "hexagonal": "hexagonal",
    "hex": "hexagonal",
    "hex24": "hexagonal",  # Diameter convention (2 * radius 12 = 24)
    "hex_24": "hexagonal",
    "largehex": "hexagonal",
    "large_hex": "hexagonal",
    "bighex": "hexagonal",
}


def normalize_board_type(board_type: str | BoardType) -> str:
    """Normalize a board type string to its canonical value.

    Args:
        board_type: Board type string or BoardType enum value.
            Accepts various formats: "square8", "sq8", "square_8", BoardType.SQUARE8

    Returns:
        Canonical board type string: "square8", "square19", "hex8", or "hexagonal"

    Raises:
        ValueError: If board_type cannot be normalized to a known value.

    Examples:
        >>> normalize_board_type("sq8")
        'square8'
        >>> normalize_board_type("hex")
        'hexagonal'
        >>> normalize_board_type(BoardType.HEX8)
        'hex8'
    """
    if isinstance(board_type, BoardType):
        return board_type.value

    # Normalize to lowercase and strip whitespace
    normalized = str(board_type).lower().strip()

    # Check direct alias mapping
    if normalized in _BOARD_TYPE_ALIASES:
        return _BOARD_TYPE_ALIASES[normalized]

    # Check if it's already a valid BoardType value
    try:
        return BoardType(normalized).value
    except ValueError:
        pass

    # Try to match BoardType enum names (e.g., "SQUARE8" -> "square8")
    for bt in BoardType:
        if normalized == bt.name.lower():
            return bt.value

    raise ValueError(
        f"Unknown board type: '{board_type}'. "
        f"Valid values: {', '.join(sorted(_BOARD_TYPE_ALIASES.keys()))}"
    )


def get_board_type_enum(board_type: str | BoardType) -> BoardType:
    """Convert a board type string to its BoardType enum value.

    Args:
        board_type: Board type string or BoardType enum.

    Returns:
        BoardType enum value.

    Raises:
        ValueError: If board_type cannot be converted.
    """
    if isinstance(board_type, BoardType):
        return board_type

    canonical = normalize_board_type(board_type)
    return BoardType(canonical)


def make_config_key(board_type: str | BoardType, num_players: int) -> str:
    """Create a canonical config key from board type and player count.

    Args:
        board_type: Board type string or BoardType enum.
        num_players: Number of players (2-4).

    Returns:
        Canonical config key in format "{board_type}_{num_players}p"

    Raises:
        ValueError: If board_type is invalid or num_players out of range.

    Examples:
        >>> make_config_key("sq8", 2)
        'square8_2p'
        >>> make_config_key(BoardType.HEXAGONAL, 4)
        'hexagonal_4p'
    """
    if not 2 <= num_players <= 4:
        raise ValueError(f"num_players must be 2-4, got {num_players}")

    canonical = normalize_board_type(board_type)
    return f"{canonical}_{num_players}p"


def parse_config_key(config_key: str) -> Tuple[str, int]:
    """Parse a config key into board type and player count.

    Args:
        config_key: Config key string (e.g., "square8_2p", "hexagonal_4p")

    Returns:
        Tuple of (canonical_board_type, num_players)

    Raises:
        ValueError: If config_key format is invalid.

    Examples:
        >>> parse_config_key("square8_2p")
        ('square8', 2)
        >>> parse_config_key("hex8_3p")
        ('hex8', 3)
    """
    # Match patterns like "square8_2p", "hexagonal_4p", etc.
    match = re.match(r"^(.+?)_(\d+)p$", config_key.lower().strip())
    if not match:
        raise ValueError(
            f"Invalid config key format: '{config_key}'. "
            "Expected format: '{board_type}_{num_players}p'"
        )

    board_part = match.group(1)
    num_players = int(match.group(2))

    if not 2 <= num_players <= 4:
        raise ValueError(f"Invalid player count in config key: {num_players}")

    canonical = normalize_board_type(board_part)
    return canonical, num_players


def is_valid_board_type(board_type: str) -> bool:
    """Check if a string is a valid board type (canonical or alias).

    Args:
        board_type: String to check.

    Returns:
        True if valid, False otherwise.
    """
    try:
        normalize_board_type(board_type)
        return True
    except ValueError:
        return False


def get_all_config_keys() -> list[str]:
    """Get all valid canonical config keys.

    Returns:
        List of all config keys: ["hex8_2p", "hex8_3p", ..., "square19_4p"]
    """
    keys = []
    for bt in BoardType:
        for num_players in [2, 3, 4]:
            keys.append(make_config_key(bt, num_players))
    return sorted(keys)


# Canonical config keys for easy import
CANONICAL_CONFIG_KEYS = get_all_config_keys()


def normalize_database_filename(
    board_type: str | BoardType,
    num_players: int,
    prefix: str = "selfplay",
    suffix: str = "",
) -> str:
    """Generate a canonical database filename.

    Args:
        board_type: Board type string or enum.
        num_players: Number of players.
        prefix: Filename prefix (default: "selfplay").
        suffix: Optional suffix before .db extension.

    Returns:
        Canonical filename like "selfplay_square8_2p.db"

    Examples:
        >>> normalize_database_filename("sq8", 2)
        'selfplay_square8_2p.db'
        >>> normalize_database_filename("hex", 4, suffix="_vast123")
        'selfplay_hexagonal_4p_vast123.db'
    """
    config_key = make_config_key(board_type, num_players)
    if suffix:
        return f"{prefix}_{config_key}{suffix}.db"
    return f"{prefix}_{config_key}.db"
