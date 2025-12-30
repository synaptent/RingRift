"""Event extraction utilities for coordination layer.

December 30, 2025: Consolidates duplicate event extraction patterns found across
cascade_training.py, architecture_tracker.py, architecture_feedback_controller.py,
nnue_training_daemon.py, reactive_dispatcher.py, and other coordination modules.

Common pattern: Extract config_key, parse to board_type/num_players, extract elo/model_path.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedConfigKey:
    """Result of parsing a config key like 'hex8_2p'."""

    board_type: str
    num_players: int
    raw: str

    @property
    def config_key(self) -> str:
        """Return the canonical config key format."""
        return f"{self.board_type}_{self.num_players}p"


@dataclass
class EvaluationEventData:
    """Extracted data from EVALUATION_COMPLETED or similar events."""

    config_key: str
    board_type: str
    num_players: int
    model_path: str
    elo: float
    games_played: int
    win_rate: float

    # Optional multi-harness fields
    harness_results: dict[str, Any] | None = None
    best_harness: str | None = None
    composite_participant_ids: list[str] | None = None
    is_multi_harness: bool = False

    @property
    def is_valid(self) -> bool:
        """Check if essential fields are present."""
        return bool(self.config_key and self.board_type and self.num_players > 0)


@dataclass
class TrainingEventData:
    """Extracted data from TRAINING_COMPLETED or similar events."""

    config_key: str
    board_type: str
    num_players: int
    model_path: str
    epochs: int
    final_loss: float
    samples_trained: int

    @property
    def is_valid(self) -> bool:
        """Check if essential fields are present."""
        return bool(self.config_key and self.board_type and self.num_players > 0)


def parse_config_key(config_key: str) -> ParsedConfigKey | None:
    """Parse a config key like 'hex8_2p' into board_type and num_players.

    Args:
        config_key: Config key string (e.g., 'hex8_2p', 'square19_4p')

    Returns:
        ParsedConfigKey if valid, None if parsing fails.

    Examples:
        >>> parse_config_key('hex8_2p')
        ParsedConfigKey(board_type='hex8', num_players=2, raw='hex8_2p')
        >>> parse_config_key('invalid')
        None
    """
    if not config_key:
        return None

    parts = config_key.rsplit("_", 1)
    if len(parts) != 2:
        return None

    board_type = parts[0]
    player_str = parts[1]

    # Handle both "2p" and "2" formats
    if player_str.endswith("p"):
        player_str = player_str[:-1]

    try:
        num_players = int(player_str)
    except ValueError:
        return None

    if num_players < 2 or num_players > 4:
        return None

    return ParsedConfigKey(board_type=board_type, num_players=num_players, raw=config_key)


def extract_config_key(event: dict[str, Any]) -> str:
    """Extract config_key from event, trying multiple field names.

    Args:
        event: Event payload dictionary

    Returns:
        Config key string, or empty string if not found.
    """
    return event.get("config_key") or event.get("config") or ""


def extract_model_path(event: dict[str, Any]) -> str:
    """Extract model path from event, trying multiple field names.

    Args:
        event: Event payload dictionary

    Returns:
        Model path string, or empty string if not found.
    """
    return (
        event.get("model_path")
        or event.get("model_id")
        or event.get("model")
        or event.get("config", "")
    )


def extract_board_type_and_players(event: dict[str, Any]) -> tuple[str, int]:
    """Extract board_type and num_players from event.

    Tries direct fields first, then falls back to parsing config_key.

    Args:
        event: Event payload dictionary

    Returns:
        Tuple of (board_type, num_players). Returns ("", 0) if extraction fails.
    """
    # Try direct fields first
    board_type = event.get("board_type", "")
    num_players = event.get("num_players", 0)

    if board_type and num_players:
        return board_type, num_players

    # Fall back to parsing config_key
    config_key = extract_config_key(event)
    parsed = parse_config_key(config_key)

    if parsed:
        return parsed.board_type, parsed.num_players

    return board_type or "", num_players or 0


def extract_evaluation_data(event: dict[str, Any]) -> EvaluationEventData:
    """Extract all relevant fields from an evaluation event.

    Handles EVALUATION_COMPLETED, ELO_UPDATED, and similar events.

    Args:
        event: Event payload dictionary

    Returns:
        EvaluationEventData with all extracted fields.
    """
    config_key = extract_config_key(event)
    board_type, num_players = extract_board_type_and_players(event)
    model_path = extract_model_path(event)

    return EvaluationEventData(
        config_key=config_key,
        board_type=board_type,
        num_players=num_players,
        model_path=model_path,
        elo=event.get("elo", 1000.0),
        games_played=event.get("games_played", 0) or event.get("games", 0),
        win_rate=event.get("win_rate", 0.0),
        harness_results=event.get("harness_results"),
        best_harness=event.get("best_harness"),
        composite_participant_ids=event.get("composite_participant_ids"),
        is_multi_harness=event.get("is_multi_harness", False),
    )


def extract_training_data(event: dict[str, Any]) -> TrainingEventData:
    """Extract all relevant fields from a training event.

    Handles TRAINING_COMPLETED and similar events.

    Args:
        event: Event payload dictionary

    Returns:
        TrainingEventData with all extracted fields.
    """
    config_key = extract_config_key(event)
    board_type, num_players = extract_board_type_and_players(event)
    model_path = extract_model_path(event)

    return TrainingEventData(
        config_key=config_key,
        board_type=board_type,
        num_players=num_players,
        model_path=model_path,
        epochs=event.get("epochs", 0),
        final_loss=event.get("final_loss", 0.0) or event.get("loss", 0.0),
        samples_trained=event.get("samples_trained", 0) or event.get("samples", 0),
    )


def make_config_key(board_type: str, num_players: int) -> str:
    """Create a canonical config key from components.

    Args:
        board_type: Board type string (e.g., 'hex8', 'square19')
        num_players: Number of players (2, 3, or 4)

    Returns:
        Canonical config key (e.g., 'hex8_2p')
    """
    return f"{board_type}_{num_players}p"
