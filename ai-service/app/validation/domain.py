"""Domain-specific validators for RingRift.

Provides validators for RingRift-specific data types:
- Config keys (e.g., "square8_2p")
- Board types
- Elo ratings
- Model paths
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.validation.core import ValidationResult

__all__ = [
    "is_valid_board_type",
    "is_valid_config_key",
    "is_valid_elo",
    "is_valid_model_path",
    "is_valid_num_players",
]

# Valid board types
VALID_BOARD_TYPES = {
    "square8", "square19", "square8_forced",
    "hex8", "hex19",
    "square", "hex",  # Generic
}

# Config key pattern: {board_type}_{num_players}p
CONFIG_KEY_PATTERN = re.compile(r"^([a-z0-9_]+)_(\d+)p$")


def is_valid_config_key(value: Any) -> ValidationResult:
    """Validate that value is a valid config key.

    Config keys have format: {board_type}_{num_players}p
    Examples: square8_2p, hex19_4p

    Args:
        value: Value to validate

    Returns:
        ValidationResult
    """
    if not isinstance(value, str):
        return ValidationResult.fail(f"Config key must be string, got {type(value).__name__}")

    match = CONFIG_KEY_PATTERN.match(value)
    if not match:
        return ValidationResult.fail(
            f"Invalid config key format: '{value}'. "
            "Expected format: board_type_Np (e.g., square8_2p)"
        )

    board_type = match.group(1)
    num_players = int(match.group(2))

    # Validate board type
    base_board = board_type.rstrip("_forced").rstrip("0123456789")
    if base_board not in {"square", "hex"}:
        return ValidationResult.fail(f"Unknown board type in config key: {board_type}")

    # Validate player count
    if num_players < 2 or num_players > 8:
        return ValidationResult.fail(f"Invalid player count: {num_players}. Must be 2-8")

    return ValidationResult.ok(value)


def is_valid_board_type(value: Any) -> ValidationResult:
    """Validate that value is a valid board type.

    Args:
        value: Value to validate

    Returns:
        ValidationResult
    """
    if not isinstance(value, str):
        return ValidationResult.fail(f"Board type must be string, got {type(value).__name__}")

    # Normalize and check
    normalized = value.lower()

    # Check for base type
    for valid in VALID_BOARD_TYPES:
        if normalized == valid or normalized.startswith(valid):
            return ValidationResult.ok(value)

    return ValidationResult.fail(
        f"Invalid board type: '{value}'. "
        f"Valid types: {', '.join(sorted(VALID_BOARD_TYPES))}"
    )


def is_valid_elo(value: Any) -> ValidationResult:
    """Validate that value is a valid Elo rating.

    Elo ratings are typically between 0 and 4000.

    Args:
        value: Value to validate

    Returns:
        ValidationResult
    """
    try:
        elo = float(value)
    except (TypeError, ValueError):
        return ValidationResult.fail(f"Elo must be numeric, got {type(value).__name__}")

    if elo < 0:
        return ValidationResult.fail(f"Elo cannot be negative: {elo}")

    if elo > 5000:
        return ValidationResult.fail(f"Elo suspiciously high: {elo}")

    return ValidationResult.ok(value)


def is_valid_model_path(value: Any) -> ValidationResult:
    """Validate that value is a valid model path.

    Checks:
    - Value is a string or Path
    - Has valid extension (.pt, .pth, .onnx, .weights)
    - Optionally exists on filesystem

    Args:
        value: Value to validate

    Returns:
        ValidationResult
    """
    if isinstance(value, str):
        path = Path(value)
    elif isinstance(value, Path):
        path = value
    else:
        return ValidationResult.fail(
            f"Model path must be string or Path, got {type(value).__name__}"
        )

    # Check extension
    valid_extensions = {".pt", ".pth", ".onnx", ".weights", ".bin"}
    if path.suffix.lower() not in valid_extensions:
        return ValidationResult.fail(
            f"Invalid model extension: {path.suffix}. "
            f"Valid: {', '.join(sorted(valid_extensions))}"
        )

    # Check if parent directory exists (if checking filesystem)
    if not path.parent.exists():
        return ValidationResult.fail(f"Parent directory does not exist: {path.parent}")

    return ValidationResult.ok(value)


def is_valid_num_players(value: Any) -> ValidationResult:
    """Validate that value is a valid number of players.

    Args:
        value: Value to validate

    Returns:
        ValidationResult
    """
    try:
        num = int(value)
    except (TypeError, ValueError):
        return ValidationResult.fail(f"Number of players must be integer, got {type(value).__name__}")

    if num < 2:
        return ValidationResult.fail(f"Must have at least 2 players, got {num}")

    if num > 8:
        return ValidationResult.fail(f"Maximum 8 players supported, got {num}")

    return ValidationResult.ok(value)
