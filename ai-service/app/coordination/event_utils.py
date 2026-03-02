"""Event extraction utilities for coordination layer.

December 30, 2025: Consolidates duplicate event extraction patterns found across
cascade_training.py, architecture_tracker.py, architecture_feedback_controller.py,
nnue_training_daemon.py, reactive_dispatcher.py, and other coordination modules.

Common pattern: Extract config_key, parse to board_type/num_players, extract elo/model_path.

Note: ParsedConfigKey and parse_config_key are now imported from the canonical
app.coordination.config_key module. Local definitions kept for backward compatibility
but should be considered deprecated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Import canonical implementation
from app.coordination.config_key import (
    ConfigKey,
    parse_config_key as _canonical_parse_config_key,
)


def normalize_event_payload(event: Any) -> dict[str, Any]:
    """Extract payload from an event object with fallback chain.

    December 30, 2025: Consolidated from 225 occurrences across 32 coordination files.
    This is the canonical payload extraction function for the coordination layer.

    The fallback chain handles various event formats:
    1. event.payload - Standard event object with payload attribute
    2. event.metadata - Alternative attribute name used in some contexts
    3. dict event - Event is already a dictionary
    4. Empty dict - Safe fallback when no payload found

    Args:
        event: Event object (may have .payload/.metadata or be a dict)

    Returns:
        Event payload as a dictionary (never None).

    Examples:
        >>> class Event: payload = {"config_key": "hex8_2p"}
        >>> normalize_event_payload(Event())
        {'config_key': 'hex8_2p'}

        >>> normalize_event_payload({"config_key": "square8_4p"})
        {'config_key': 'square8_4p'}

        >>> normalize_event_payload(None)
        {}
    """
    if hasattr(event, "payload"):
        return event.payload
    if hasattr(event, "metadata"):
        return event.metadata
    if isinstance(event, dict):
        return event
    return {}


# Backward compatibility: ParsedConfigKey is now an alias for ConfigKey
# from the canonical config_key module
ParsedConfigKey = ConfigKey


@dataclass
class EvaluationEventData:
    """Extracted data from EVALUATION_COMPLETED or similar events.

    Contains all relevant fields from evaluation events, including optional
    multi-harness gauntlet results. Use `extract_evaluation_data()` to create
    instances from event payloads.

    Example:
        >>> event = {
        ...     "config_key": "hex8_2p",
        ...     "model_path": "models/canonical_hex8_2p.pth",
        ...     "elo": 1450.5,
        ...     "games_played": 100,
        ...     "win_rate": 0.72,
        ... }
        >>> data = extract_evaluation_data(event)
        >>> data.config_key
        'hex8_2p'
        >>> data.elo
        1450.5
        >>> data.is_valid
        True

    Attributes:
        config_key: Config identifier (e.g., 'hex8_2p')
        board_type: Board type extracted from config_key
        num_players: Player count extracted from config_key
        model_path: Path to the evaluated model
        elo: Elo rating after evaluation
        games_played: Number of games in evaluation
        win_rate: Win rate (0.0 to 1.0)
        harness_results: Per-harness results for multi-harness evaluations
        best_harness: Best performing harness type
        composite_participant_ids: Participant IDs for composite evaluations
        is_multi_harness: Whether this was a multi-harness evaluation
    """

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
    """Extracted data from TRAINING_COMPLETED or similar events.

    Contains all relevant fields from training events. Use `extract_training_data()`
    to create instances from event payloads.

    Example:
        >>> event = {
        ...     "config_key": "square8_4p",
        ...     "model_path": "models/canonical_square8_4p.pth",
        ...     "epochs": 50,
        ...     "final_loss": 0.0234,
        ...     "samples_trained": 125000,
        ... }
        >>> data = extract_training_data(event)
        >>> data.config_key
        'square8_4p'
        >>> data.epochs
        50
        >>> data.is_valid
        True

    Attributes:
        config_key: Config identifier (e.g., 'square8_4p')
        board_type: Board type extracted from config_key
        num_players: Player count extracted from config_key
        model_path: Path to the trained model
        epochs: Number of training epochs completed
        final_loss: Final training loss value
        samples_trained: Total samples used in training
    """

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

    Note: This function now delegates to the canonical implementation in
    app.coordination.config_key. Use that module directly for new code.

    Args:
        config_key: Config key string (e.g., 'hex8_2p', 'square19_4p')

    Returns:
        ParsedConfigKey (ConfigKey) if valid, None if parsing fails.

    Examples:
        >>> parse_config_key('hex8_2p')
        ConfigKey(board_type='hex8', num_players=2, raw='hex8_2p')
        >>> parse_config_key('invalid')
        None
    """
    return _canonical_parse_config_key(config_key)


def extract_config_key(event: dict[str, Any] | Any) -> str:
    """Extract config_key from event, trying multiple field names.

    Args:
        event: Event payload dictionary or RouterEvent object.

    Returns:
        Config key string, or empty string if not found.
    """
    payload = normalize_event_payload(event)
    return payload.get("config_key") or payload.get("config") or ""


def extract_model_path(event: dict[str, Any] | Any) -> str:
    """Extract model path from event, trying multiple field names.

    Args:
        event: Event payload dictionary or RouterEvent object.

    Returns:
        Model path string, or empty string if not found.
    """
    payload = normalize_event_payload(event)
    return (
        payload.get("model_path")
        or payload.get("model_id")
        or payload.get("model")
        or payload.get("config", "")
    )


def extract_board_type_and_players(event: dict[str, Any] | Any) -> tuple[str, int]:
    """Extract board_type and num_players from event.

    Tries direct fields first, then falls back to parsing config_key.

    Args:
        event: Event payload dictionary or RouterEvent object.

    Returns:
        Tuple of (board_type, num_players). Returns ("", 0) if extraction fails.
    """
    # Try direct fields first
    payload = normalize_event_payload(event)
    board_type = payload.get("board_type", "")
    num_players = payload.get("num_players", 0)

    if board_type and num_players:
        return board_type, num_players

    # Fall back to parsing config_key
    config_key = extract_config_key(event)
    parsed = parse_config_key(config_key)

    if parsed:
        return parsed.board_type, parsed.num_players

    return board_type or "", num_players or 0


def extract_evaluation_data(event: Any) -> EvaluationEventData:
    """Extract all relevant fields from an evaluation event.

    Handles EVALUATION_COMPLETED, ELO_UPDATED, and similar events.
    Supports both RouterEvent objects and plain dict payloads.

    Args:
        event: Event object (RouterEvent with .payload) or payload dictionary

    Returns:
        EvaluationEventData with all extracted fields.
    """
    # Jan 2026: Normalize payload to handle RouterEvent objects
    payload = normalize_event_payload(event)

    config_key = extract_config_key(payload)
    board_type, num_players = extract_board_type_and_players(payload)
    model_path = extract_model_path(payload)

    return EvaluationEventData(
        config_key=config_key,
        board_type=board_type,
        num_players=num_players,
        model_path=model_path,
        # Mar 2026: Use `or` instead of default kwarg to handle explicit None values.
        # payload.get("elo", 1000.0) returns None when payload has {"elo": None},
        # causing TypeError in downstream handlers that compare elo > threshold.
        elo=payload.get("elo") or 1000.0,
        games_played=payload.get("games_played") or payload.get("games") or 0,
        win_rate=payload.get("win_rate") or 0.0,
        harness_results=payload.get("harness_results"),
        best_harness=payload.get("best_harness"),
        composite_participant_ids=payload.get("composite_participant_ids"),
        is_multi_harness=payload.get("is_multi_harness", False),
    )


def extract_training_data(event: Any) -> TrainingEventData:
    """Extract all relevant fields from a training event.

    Handles TRAINING_COMPLETED and similar events.
    Supports both RouterEvent objects and plain dict payloads.

    Args:
        event: Event object (RouterEvent with .payload) or payload dictionary

    Returns:
        TrainingEventData with all extracted fields.
    """
    # Jan 2026: Normalize payload to handle RouterEvent objects
    payload = normalize_event_payload(event)

    config_key = extract_config_key(payload)
    board_type, num_players = extract_board_type_and_players(payload)
    model_path = extract_model_path(payload)

    return TrainingEventData(
        config_key=config_key,
        board_type=board_type,
        num_players=num_players,
        model_path=model_path,
        epochs=payload.get("epochs", 0),
        final_loss=payload.get("final_loss", 0.0) or payload.get("loss", 0.0),
        samples_trained=payload.get("samples_trained", 0) or payload.get("samples", 0),
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


# =============================================================================
# January 2026 Sprint 12: Additional extraction helpers
# Consolidated from quality_feedback_handler, curriculum_integration,
# model_lifecycle_coordinator, and auto_promotion_daemon
# =============================================================================


@dataclass
class QualityEventData:
    """Extracted data from QUALITY_SCORE_UPDATED or similar quality events.

    Contains all relevant fields from quality assessment events. Use
    `extract_quality_data()` to create instances from event payloads.

    Example:
        >>> event = {
        ...     "config_key": "hex8_2p",
        ...     "quality_score": 0.82,
        ...     "quality_category": "good",
        ...     "training_weight": 0.9,
        ... }
        >>> data = extract_quality_data(event)
        >>> data.quality_score
        0.82
        >>> data.is_trainable
        True

    Attributes:
        config_key: Config identifier (e.g., 'hex8_2p')
        board_type: Board type extracted from config_key
        num_players: Player count extracted from config_key
        quality_score: Quality score (0.0 to 1.0)
        quality_category: Category (excellent/good/adequate/poor/unusable)
        training_weight: Weight multiplier for training samples
        game_id: Optional game ID for per-game quality events
        trend: Quality trend (improving/stable/degrading)
        previous_score: Previous quality score for trend calculation
    """

    config_key: str
    board_type: str
    num_players: int
    quality_score: float
    quality_category: str
    training_weight: float
    game_id: str | None = None
    trend: str | None = None
    previous_score: float | None = None

    @property
    def is_valid(self) -> bool:
        """Check if essential fields are present."""
        return bool(self.config_key and self.quality_score >= 0.0)

    @property
    def is_trainable(self) -> bool:
        """Check if quality is high enough for training."""
        return self.quality_score >= 0.5 and self.quality_category not in ("poor", "unusable")


@dataclass
class CurriculumEventData:
    """Extracted data from CURRICULUM_REBALANCED or curriculum update events.

    Contains all relevant fields from curriculum adjustment events. Use
    `extract_curriculum_data()` to create instances from event payloads.

    Example:
        >>> event = {
        ...     "config_key": "square8_4p",
        ...     "old_weight": 0.8,
        ...     "new_weight": 1.2,
        ...     "source": "elo_velocity_boost",
        ... }
        >>> data = extract_curriculum_data(event)
        >>> data.weight_change
        0.4

    Attributes:
        config_key: Config identifier (e.g., 'square8_4p')
        board_type: Board type extracted from config_key
        num_players: Player count extracted from config_key
        old_weight: Previous curriculum weight
        new_weight: Updated curriculum weight
        source: Source of the weight change
        reason: Human-readable reason for the change
        affected_configs: List of configs affected by rebalance
    """

    config_key: str
    board_type: str
    num_players: int
    old_weight: float
    new_weight: float
    source: str
    reason: str | None = None
    affected_configs: list[str] | None = None

    @property
    def is_valid(self) -> bool:
        """Check if essential fields are present."""
        return bool(self.config_key and self.new_weight >= 0.0)

    @property
    def weight_change(self) -> float:
        """Calculate the weight change (new - old)."""
        return self.new_weight - self.old_weight

    @property
    def is_boost(self) -> bool:
        """Check if this is a weight increase."""
        return self.new_weight > self.old_weight

    @property
    def is_reduction(self) -> bool:
        """Check if this is a weight decrease."""
        return self.new_weight < self.old_weight


@dataclass
class ModelMetadataEventData:
    """Extracted data from MODEL_PROMOTED or model lifecycle events.

    Contains all relevant fields from model promotion and lifecycle events.
    Use `extract_model_metadata()` to create instances from event payloads.

    Example:
        >>> event = {
        ...     "config_key": "hex8_2p",
        ...     "model_path": "models/canonical_hex8_2p.pth",
        ...     "model_version": "v5-heavy",
        ...     "elo": 1580,
        ...     "promoted": True,
        ... }
        >>> data = extract_model_metadata(event)
        >>> data.is_promoted
        True

    Attributes:
        config_key: Config identifier (e.g., 'hex8_2p')
        board_type: Board type extracted from config_key
        num_players: Player count extracted from config_key
        model_path: Path to the model file
        model_version: Architecture version (v2, v4, v5-heavy, etc.)
        elo: Current Elo rating
        previous_elo: Elo before promotion (for delta calculation)
        promoted: Whether model was promoted to canonical
        promotion_reason: Reason for promotion (gauntlet_pass, elo_improvement, etc.)
        checkpoint_path: Path to training checkpoint
    """

    config_key: str
    board_type: str
    num_players: int
    model_path: str
    model_version: str
    elo: float
    previous_elo: float | None = None
    promoted: bool = False
    promotion_reason: str | None = None
    checkpoint_path: str | None = None

    @property
    def is_valid(self) -> bool:
        """Check if essential fields are present."""
        return bool(self.config_key and self.model_path)

    @property
    def is_promoted(self) -> bool:
        """Check if model was promoted."""
        return self.promoted

    @property
    def elo_delta(self) -> float | None:
        """Calculate Elo improvement if previous_elo is available."""
        if self.previous_elo is not None:
            return self.elo - self.previous_elo
        return None


def extract_quality_data(event: dict[str, Any] | Any) -> QualityEventData:
    """Extract all relevant fields from a quality assessment event.

    Handles QUALITY_SCORE_UPDATED, QUALITY_FEEDBACK_ADJUSTED, and similar events.

    Args:
        event: Event payload dictionary or RouterEvent object.

    Returns:
        QualityEventData with all extracted fields.
    """
    config_key = extract_config_key(event)
    board_type, num_players = extract_board_type_and_players(event)
    payload = normalize_event_payload(event)

    return QualityEventData(
        config_key=config_key,
        board_type=board_type,
        num_players=num_players,
        quality_score=payload.get("quality_score", 0.0) or payload.get("score", 0.0),
        quality_category=payload.get("quality_category", "unknown") or payload.get("category", "unknown"),
        training_weight=payload.get("training_weight", 1.0),
        game_id=payload.get("game_id"),
        trend=payload.get("trend"),
        previous_score=payload.get("previous_score") or payload.get("previous_quality"),
    )


def extract_curriculum_data(event: dict[str, Any] | Any) -> CurriculumEventData:
    """Extract all relevant fields from a curriculum update event.

    Handles CURRICULUM_REBALANCED, CURRICULUM_ADVANCED, and similar events.

    Args:
        event: Event payload dictionary or RouterEvent object.

    Returns:
        CurriculumEventData with all extracted fields.
    """
    config_key = extract_config_key(event)
    board_type, num_players = extract_board_type_and_players(event)
    payload = normalize_event_payload(event)

    return CurriculumEventData(
        config_key=config_key,
        board_type=board_type,
        num_players=num_players,
        old_weight=payload.get("old_weight", 1.0) or payload.get("previous_weight", 1.0),
        new_weight=payload.get("new_weight", 1.0) or payload.get("weight", 1.0),
        source=payload.get("source", "unknown"),
        reason=payload.get("reason"),
        affected_configs=payload.get("affected_configs") or payload.get("configs_affected"),
    )


def extract_model_metadata(event: dict[str, Any] | Any) -> ModelMetadataEventData:
    """Extract all relevant fields from a model lifecycle event.

    Handles MODEL_PROMOTED, MODEL_REGISTERED, and similar events.

    Args:
        event: Event payload dictionary or RouterEvent object.

    Returns:
        ModelMetadataEventData with all extracted fields.
    """
    config_key = extract_config_key(event)
    board_type, num_players = extract_board_type_and_players(event)
    model_path = extract_model_path(event)
    payload = normalize_event_payload(event)

    return ModelMetadataEventData(
        config_key=config_key,
        board_type=board_type,
        num_players=num_players,
        model_path=model_path,
        model_version=payload.get("model_version", "") or payload.get("version", "") or payload.get("architecture", ""),
        elo=payload.get("elo", 1000.0),
        previous_elo=payload.get("previous_elo"),
        promoted=payload.get("promoted", False) or payload.get("is_promoted", False),
        promotion_reason=payload.get("promotion_reason") or payload.get("reason"),
        checkpoint_path=payload.get("checkpoint_path") or payload.get("checkpoint"),
    )
