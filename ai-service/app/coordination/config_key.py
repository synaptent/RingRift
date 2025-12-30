#!/usr/bin/env python3
"""Canonical config key parsing and representation.

This module provides the single source of truth for parsing and representing
board configuration keys like 'hex8_2p' or 'square19_4p'.

Usage:
    from app.coordination.config_key import ConfigKey, parse_config_key

    # Parse a config key string
    key = ConfigKey.parse("hex8_2p")
    if key:
        print(f"Board: {key.board_type}, Players: {key.num_players}")

    # Create directly
    key = ConfigKey(board_type="hex8", num_players=2)
    print(str(key))  # "hex8_2p"

    # Use convenience function
    result = parse_config_key("square19_4p")

December 2025: Consolidated from 6 duplicate implementations across the codebase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

# Valid board types in the system
VALID_BOARD_TYPES = frozenset({"hex8", "square8", "square19", "hexagonal"})

# Valid player counts
VALID_PLAYER_COUNTS = frozenset({2, 3, 4})


@dataclass(frozen=True)
class ConfigKey:
    """Canonical config key representation (e.g., 'hex8_2p').

    A configuration key uniquely identifies a board type and player count
    combination used throughout the RingRift AI training system.

    Attributes:
        board_type: Board type string (e.g., 'hex8', 'square8', 'square19', 'hexagonal')
        num_players: Number of players (2, 3, or 4)
        raw: Original config key string as provided (empty if created directly)

    Example:
        >>> key = ConfigKey.parse("hex8_2p")
        >>> key.board_type
        'hex8'
        >>> key.num_players
        2
        >>> key.config_key
        'hex8_2p'
        >>> str(key)
        'hex8_2p'

    Note:
        This is an immutable dataclass (frozen=True). Once created, the
        values cannot be changed.
    """

    board_type: str
    num_players: int
    raw: str = ""

    def __post_init__(self) -> None:
        """Validate the config key values after initialization."""
        if self.num_players not in VALID_PLAYER_COUNTS:
            raise ValueError(
                f"Invalid num_players={self.num_players}. "
                f"Must be one of {sorted(VALID_PLAYER_COUNTS)}"
            )
        # Note: We don't validate board_type to allow for future board types
        # But we log a warning for unknown types
        if self.board_type not in VALID_BOARD_TYPES:
            import logging

            logging.getLogger(__name__).debug(
                f"Unknown board_type '{self.board_type}' - "
                f"known types are {sorted(VALID_BOARD_TYPES)}"
            )

    @classmethod
    def parse(cls, key: str) -> ConfigKey | None:
        """Parse a config key string into a ConfigKey object.

        Supports both 'hex8_2p' and 'hex8_2' formats.

        Args:
            key: Config key string (e.g., 'hex8_2p', 'square19_4')

        Returns:
            ConfigKey if valid, None if parsing fails.

        Examples:
            >>> ConfigKey.parse('hex8_2p')
            ConfigKey(board_type='hex8', num_players=2, raw='hex8_2p')
            >>> ConfigKey.parse('square19_4')
            ConfigKey(board_type='square19', num_players=4, raw='square19_4')
            >>> ConfigKey.parse('invalid')
            None
        """
        if not key or not isinstance(key, str):
            return None

        # Reject whitespace (exact matching required)
        if key != key.strip():
            return None

        # Handle both "board_Np" and "board_N" formats
        # Use rsplit to handle board types with underscores (though we don't have any currently)
        parts = key.rsplit("_", 1)
        if len(parts) != 2:
            return None

        board_type = parts[0]
        player_str = parts[1]

        # Reject empty board type
        if not board_type:
            return None

        # Handle both "2p"/"2P" and "2" formats
        if player_str.lower().endswith("p"):
            player_str = player_str[:-1]

        try:
            num_players = int(player_str)
        except ValueError:
            return None

        # Validate player count range
        if num_players not in VALID_PLAYER_COUNTS:
            return None

        return cls(board_type=board_type, num_players=num_players, raw=key)

    @classmethod
    def from_components(cls, board_type: str, num_players: int) -> ConfigKey:
        """Create a ConfigKey from individual components.

        Args:
            board_type: Board type string
            num_players: Number of players

        Returns:
            ConfigKey instance

        Raises:
            ValueError: If num_players is invalid
        """
        return cls(board_type=board_type, num_players=num_players)

    @property
    def config_key(self) -> str:
        """Return the canonical config key format.

        Always returns the format '{board_type}_{num_players}p'.
        """
        return f"{self.board_type}_{self.num_players}p"

    def __str__(self) -> str:
        """Return the canonical config key string."""
        return self.config_key

    def __hash__(self) -> int:
        """Hash based on canonical form (board_type and num_players only)."""
        return hash((self.board_type, self.num_players))

    def __eq__(self, other: object) -> bool:
        """Compare ConfigKeys by their canonical form."""
        if isinstance(other, ConfigKey):
            return (
                self.board_type == other.board_type
                and self.num_players == other.num_players
            )
        if isinstance(other, str):
            # Allow comparison with string config keys
            parsed = ConfigKey.parse(other)
            if parsed:
                return self == parsed
        return NotImplemented

    def to_tuple(self) -> tuple[str, int]:
        """Convert to tuple for backward compatibility.

        Returns:
            Tuple of (board_type, num_players)

        This method exists for backward compatibility with code that
        expects tuple return values from parse_config_key().
        """
        return (self.board_type, self.num_players)

    def to_dict(self) -> dict[str, str | int]:
        """Convert to dictionary representation.

        Returns:
            Dict with 'board_type', 'num_players', 'config_key' keys
        """
        return {
            "board_type": self.board_type,
            "num_players": self.num_players,
            "config_key": self.config_key,
        }


# Backward compatibility alias
ParsedConfigKey = ConfigKey


class ConfigKeyTuple(NamedTuple):
    """Named tuple version for code expecting specific tuple interface."""

    board_type: str
    num_players: int


def parse_config_key(config_key: str) -> ConfigKey | None:
    """Parse a config key string into a ConfigKey object.

    This is the canonical function for parsing config keys. All other
    implementations should delegate to this function.

    Args:
        config_key: Config key string (e.g., 'hex8_2p', 'square19_4')

    Returns:
        ConfigKey if valid, None if parsing fails.

    Examples:
        >>> parse_config_key('hex8_2p')
        ConfigKey(board_type='hex8', num_players=2, raw='hex8_2p')
        >>> parse_config_key('invalid')
        None
    """
    return ConfigKey.parse(config_key)


def parse_config_key_tuple(config_key: str) -> tuple[str, int] | None:
    """Parse a config key string into a tuple.

    For backward compatibility with code expecting tuple[str, int] returns.

    Args:
        config_key: Config key string

    Returns:
        Tuple (board_type, num_players) if valid, None otherwise.
    """
    result = ConfigKey.parse(config_key)
    return result.to_tuple() if result else None


def parse_config_key_safe(config_key: str) -> tuple[str | None, int | None]:
    """Parse a config key with nullable return values.

    For backward compatibility with code expecting tuple[str | None, int | None].

    Args:
        config_key: Config key string

    Returns:
        Tuple (board_type, num_players) or (None, None) if invalid.
    """
    result = ConfigKey.parse(config_key)
    if result:
        return (result.board_type, result.num_players)
    return (None, None)


def format_config_key(board_type: str, num_players: int) -> str:
    """Format board type and player count into canonical config key.

    Args:
        board_type: Board type string
        num_players: Number of players

    Returns:
        Canonical config key string (e.g., 'hex8_2p')

    Raises:
        ValueError: If num_players is invalid
    """
    return str(ConfigKey.from_components(board_type, num_players))


def is_valid_config_key(config_key: str) -> bool:
    """Check if a string is a valid config key.

    Args:
        config_key: String to validate

    Returns:
        True if valid config key, False otherwise
    """
    return ConfigKey.parse(config_key) is not None


def get_all_config_keys() -> list[str]:
    """Get all valid config key combinations.

    Returns:
        List of all valid config keys (e.g., ['hex8_2p', 'hex8_3p', ...])
    """
    return [
        f"{board}_{players}p"
        for board in sorted(VALID_BOARD_TYPES)
        for players in sorted(VALID_PLAYER_COUNTS)
    ]


__all__ = [
    # Primary class
    "ConfigKey",
    # Backward compat alias
    "ParsedConfigKey",
    # Named tuple for specific interfaces
    "ConfigKeyTuple",
    # Primary function
    "parse_config_key",
    # Backward compat functions
    "parse_config_key_tuple",
    "parse_config_key_safe",
    # Utility functions
    "format_config_key",
    "is_valid_config_key",
    "get_all_config_keys",
    # Constants
    "VALID_BOARD_TYPES",
    "VALID_PLAYER_COUNTS",
]
