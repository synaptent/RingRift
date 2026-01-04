"""Event handler utilities for consistent payload extraction.

This module consolidates common payload extraction patterns used across
event handlers to ensure consistent behavior and reduce code duplication.

Common patterns consolidated:
- Config key extraction (handles "config_key" and "config" aliases)
- Board type extraction with validation
- Model path extraction from various payload locations
- Metadata extraction with defaults

December 2025: Created to reduce ~300 LOC of duplicated extraction logic
across coordination modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeAlias

from app.coordination.event_utils import make_config_key as _make_config_key

logger = logging.getLogger(__name__)

# Type aliases
Payload: TypeAlias = dict[str, Any]
EventData: TypeAlias = dict[str, Any]


def extract_config_key(
    payload: Payload,
    default: str = "",
    allow_empty: bool = True,
) -> str:
    """Extract config key from event payload.

    Handles common aliases: "config_key", "config"

    Args:
        payload: Event payload dictionary.
        default: Default value if not found.
        allow_empty: If False, raises ValueError when key is empty.

    Returns:
        Config key string (e.g., "hex8_2p").

    Raises:
        ValueError: If allow_empty=False and no config key found.

    Example:
        >>> extract_config_key({"config_key": "hex8_2p"})
        'hex8_2p'
        >>> extract_config_key({"config": "square8_4p"})
        'square8_4p'
    """
    config_key = (
        payload.get("config_key")
        or payload.get("config")
        or default
    )

    if not allow_empty and not config_key:
        raise ValueError("No config_key found in payload")

    return config_key


def extract_board_type(
    payload: Payload,
    default: str | None = None,
) -> str | None:
    """Extract board type from event payload.

    Args:
        payload: Event payload dictionary.
        default: Default value if not found.

    Returns:
        Board type string (e.g., "hex8", "square8") or None.

    Example:
        >>> extract_board_type({"board_type": "hex8"})
        'hex8'
    """
    return payload.get("board_type") or default


def extract_num_players(
    payload: Payload,
    default: int = 2,
) -> int:
    """Extract number of players from event payload.

    Args:
        payload: Event payload dictionary.
        default: Default value if not found.

    Returns:
        Number of players (2, 3, or 4).
    """
    value = payload.get("num_players")
    if value is None:
        return default
    return int(value)


def extract_model_path(
    payload: Payload,
    default: str | None = None,
) -> str | None:
    """Extract model path from event payload.

    Handles common aliases: "model_path", "checkpoint_path", "path"

    Args:
        payload: Event payload dictionary.
        default: Default value if not found.

    Returns:
        Model path string or None.

    Example:
        >>> extract_model_path({"model_path": "/models/best.pth"})
        '/models/best.pth'
        >>> extract_model_path({"checkpoint_path": "/models/new.pth"})
        '/models/new.pth'
    """
    return (
        payload.get("model_path")
        or payload.get("checkpoint_path")
        or payload.get("path")
        or default
    )


def extract_config_from_path(model_path: str | Path | None) -> str | None:
    """Extract config key from model path naming convention.

    Assumes canonical naming: canonical_{board}_{n}p.pth

    Args:
        model_path: Path to model checkpoint.

    Returns:
        Config key (e.g., "hex8_2p") or None if not parseable.

    Example:
        >>> extract_config_from_path("models/canonical_hex8_2p.pth")
        'hex8_2p'
        >>> extract_config_from_path("models/ringrift_best_square8_4p.pth")
        'square8_4p'
    """
    if not model_path:
        return None

    stem = Path(model_path).stem

    # Try canonical_* naming
    if "canonical_" in stem:
        config_part = stem.replace("canonical_", "")
        # Remove version suffix if present (e.g., _v5heavy)
        if "_v" in config_part:
            config_part = config_part.rsplit("_v", 1)[0]
        return config_part

    # Try ringrift_best_* naming
    if "ringrift_best_" in stem:
        config_part = stem.replace("ringrift_best_", "")
        return config_part

    # Try to find {board}_{n}p pattern
    for board in ("hex8", "square8", "square19", "hexagonal"):
        for n in (2, 3, 4):
            pattern = f"{board}_{n}p"
            if pattern in stem:
                return pattern

    return None


def extract_metadata(
    payload: Payload,
    include_config: bool = True,
) -> dict[str, Any]:
    """Extract metadata dictionary from payload with defaults.

    Args:
        payload: Event payload dictionary.
        include_config: If True, include config_key in metadata.

    Returns:
        Metadata dictionary with standardized keys.

    Example:
        >>> extract_metadata({"config_key": "hex8_2p", "elo": 1500})
        {'config_key': 'hex8_2p', 'elo': 1500, ...}
    """
    metadata = dict(payload)

    if include_config:
        # Normalize config_key
        config_key = extract_config_key(payload)
        if config_key:
            metadata["config_key"] = config_key

    return metadata


def parse_config_key(config_key: str) -> tuple[str | None, int | None]:
    """Parse config key into board type and num players.

    December 2025: Now delegates to event_utils.parse_config_key() for
    consistent parsing across the codebase.

    Args:
        config_key: Config key string (e.g., "hex8_2p").

    Returns:
        Tuple of (board_type, num_players) or (None, None) if invalid.

    Example:
        >>> parse_config_key("hex8_2p")
        ('hex8', 2)
        >>> parse_config_key("square19_4p")
        ('square19', 4)
    """
    from app.coordination.event_utils import parse_config_key as _parse

    parsed = _parse(config_key)
    if parsed is None:
        return None, None
    return parsed.board_type, parsed.num_players


def build_config_key(board_type: str, num_players: int) -> str:
    """Build config key from board type and num players.

    Args:
        board_type: Board type string (e.g., "hex8").
        num_players: Number of players (2, 3, or 4).

    Returns:
        Config key string (e.g., "hex8_2p").

    Example:
        >>> build_config_key("hex8", 2)
        'hex8_2p'

    Note:
        Delegates to make_config_key() for consistency (January 2026).
    """
    return _make_config_key(board_type, num_players)


def extract_training_completed_data(payload: Payload) -> dict[str, Any]:
    """Extract standardized data from TRAINING_COMPLETED event.

    Args:
        payload: Event payload dictionary.

    Returns:
        Dictionary with standardized keys:
        - config_key: Configuration key
        - model_path: Path to trained model
        - board_type: Board type
        - num_players: Number of players
        - metrics: Training metrics (loss, accuracy, etc.)
    """
    config_key = extract_config_key(payload)
    model_path = extract_model_path(payload)
    board_type = extract_board_type(payload)
    num_players = extract_num_players(payload)

    # If board_type missing but config_key present, parse it
    if not board_type and config_key:
        board_type, parsed_players = parse_config_key(config_key)
        if parsed_players and num_players == 2:  # 2 is default
            num_players = parsed_players

    return {
        "config_key": config_key,
        "model_path": model_path,
        "board_type": board_type,
        "num_players": num_players,
        "metrics": payload.get("metrics", {}),
        "epochs": payload.get("epochs"),
        "final_loss": payload.get("final_loss") or payload.get("loss"),
    }


def extract_evaluation_completed_data(payload: Payload) -> dict[str, Any]:
    """Extract standardized data from EVALUATION_COMPLETED event.

    Args:
        payload: Event payload dictionary.

    Returns:
        Dictionary with standardized keys.
    """
    return {
        "config_key": extract_config_key(payload),
        "model_path": extract_model_path(payload),
        "board_type": extract_board_type(payload),
        "num_players": extract_num_players(payload),
        "elo": payload.get("elo") or payload.get("estimated_elo"),
        "win_rate": payload.get("win_rate"),
        "passed": payload.get("passed", False),
        "games_played": payload.get("games_played", 0),
    }


def extract_sync_completed_data(payload: Payload) -> dict[str, Any]:
    """Extract standardized data from DATA_SYNC_COMPLETED event.

    Args:
        payload: Event payload dictionary.

    Returns:
        Dictionary with standardized keys.
    """
    return {
        "config_key": extract_config_key(payload),
        "sync_type": payload.get("sync_type") or payload.get("data_type"),
        "source_node": payload.get("source_node") or payload.get("source"),
        "target_nodes": payload.get("target_nodes") or payload.get("targets", []),
        "files_synced": payload.get("files_synced", 0),
        "bytes_transferred": payload.get("bytes_transferred", 0),
        "duration_seconds": payload.get("duration_seconds") or payload.get("duration"),
    }


def validate_payload_has_config(
    payload: Payload,
    event_type: str,
    logger_name: str | None = None,
) -> str | None:
    """Validate that payload has a config key, log warning if missing.

    Args:
        payload: Event payload dictionary.
        event_type: Event type for logging.
        logger_name: Optional logger name for logging.

    Returns:
        Config key if found, None otherwise.
    """
    config_key = extract_config_key(payload)
    if not config_key:
        log = logging.getLogger(logger_name) if logger_name else logger
        log.warning(f"{event_type} event missing config_key: {payload}")
        return None
    return config_key


# Backward-compatible aliases
def get_config_key(payload: Payload) -> str:
    """Alias for extract_config_key with default=""."""
    return extract_config_key(payload, default="")


def get_board_type(payload: Payload) -> str | None:
    """Alias for extract_board_type."""
    return extract_board_type(payload)


def get_model_path(payload: Payload) -> str | None:
    """Alias for extract_model_path."""
    return extract_model_path(payload)
