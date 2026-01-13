"""Robust Elo Recording Facade - Single entry point for all Elo match recording.

This module provides THE authoritative entry point for recording match results
and updating Elo ratings. All callers should use this facade instead of directly
calling elo_service.record_match().

Key benefits:
- REQUIRED harness_type - no hidden defaults that pollute data
- Model type detection (NN/NNUE) from participant IDs
- Never-silent failures with proper logging and event emission
- Dead Letter Queue (DLQ) integration for failed recordings
- Harness compatibility validation

Usage:
    from app.training.elo_recording import (
        safe_record_elo,
        record_gauntlet_match,
        EloMatchSpec,
        HarnessType,
        ModelType,
    )

    # Explicit specification
    spec = EloMatchSpec(
        participant_a="model_v1",
        participant_b="heuristic",
        winner="model_v1",
        board_type="hex8",
        num_players=2,
        harness_type=HarnessType.GUMBEL_MCTS,  # REQUIRED - no defaults
    )
    result = safe_record_elo(spec)

    # Convenience function for gauntlet
    result = record_gauntlet_match(
        model_id="model_v1",
        baseline="heuristic",
        model_won=True,
        board_type="hex8",
        num_players=2,
        harness_type=HarnessType.POLICY_ONLY,
    )

Created: January 12, 2026 - Fix Elo harness tracking system (Plan: kind-beaming-gosling)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================
# Enums - Required types for Elo recording
# ============================================


class HarnessType(str, Enum):
    """AI harness types for Elo tracking.

    Each harness represents a different AI evaluation method with different
    strengths and use cases. The harness type MUST be recorded with each
    match to enable meaningful Elo comparisons.

    Jan 2026: Aligned with canonical HarnessType in base_harness.py.
    Removed MCTS (use GUMBEL_MCTS instead), added DESCENT.
    """

    # Neural network harnesses (require policy output)
    POLICY_ONLY = "policy_only"  # Raw policy network output, no search
    GUMBEL_MCTS = "gumbel_mcts"  # Gumbel-based MCTS (quality-focused)
    GPU_GUMBEL = "gpu_gumbel"  # GPU-accelerated Gumbel MCTS
    DESCENT = "descent"  # Gradient descent search

    # NNUE harnesses (require NNUE model)
    MINIMAX = "minimax"  # Alpha-beta minimax (2-player only)
    MAXN = "maxn"  # Max-N algorithm (multiplayer)
    BRS = "brs"  # Best Reply Search

    # Baseline harnesses (no model required)
    HEURISTIC = "heuristic"  # Hand-crafted heuristic evaluation
    RANDOM = "random"  # Random move selection

    @classmethod
    def from_string(cls, value: str) -> HarnessType:
        """Convert string to HarnessType, handling common variations."""
        value_lower = value.lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == value_lower:
                return member
        # Handle common aliases
        aliases = {
            "gumbel": cls.GUMBEL_MCTS,
            "gumbelmcts": cls.GUMBEL_MCTS,
            "policy": cls.POLICY_ONLY,
            "nn": cls.POLICY_ONLY,
            "neural": cls.POLICY_ONLY,
            "alphabeta": cls.MINIMAX,
            "ab": cls.MINIMAX,
        }
        if value_lower in aliases:
            return aliases[value_lower]
        raise ValueError(f"Unknown harness type: {value}")


class ModelType(str, Enum):
    """Model architecture types for Elo tracking.

    Distinguishes between different neural network architectures which
    may have different performance characteristics.

    Jan 2026: Aligned with canonical ModelType in base_harness.py.
    Renamed NN to NEURAL_NET, removed NNUE_MP (use NNUE for all player counts).
    """

    NEURAL_NET = "nn"  # Full neural network (policy + value heads)
    NNUE = "nnue"  # NNUE (all player counts, harness selection per player count)
    NONE = "none"  # No model (baselines like random, heuristic)

    @classmethod
    def from_string(cls, value: str) -> ModelType:
        """Convert string to ModelType."""
        value_lower = value.lower()
        for member in cls:
            if member.value == value_lower:
                return member
        raise ValueError(f"Unknown model type: {value}")


# ============================================
# Data classes
# ============================================


@dataclass
class EloMatchSpec:
    """Specification for recording an Elo match.

    All fields required - no hidden defaults that pollute data.
    This ensures every recorded match has complete tracking information.
    """

    participant_a: str
    participant_b: str
    winner: str | None  # None for draw
    board_type: str
    num_players: int
    harness_type: HarnessType  # REQUIRED - enum, not string

    # Optional fields with explicit defaults
    game_length: int = 0
    duration_sec: float = 0.0
    tournament_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Auto-detected if not provided
    model_type_a: ModelType | None = None
    model_type_b: ModelType | None = None

    def __post_init__(self) -> None:
        """Validate and auto-detect model types."""
        # Convert harness_type string to enum if needed
        if isinstance(self.harness_type, str):
            self.harness_type = HarnessType.from_string(self.harness_type)

        # Auto-detect model types if not provided
        if self.model_type_a is None:
            self.model_type_a = detect_model_type(self.participant_a)
        if self.model_type_b is None:
            self.model_type_b = detect_model_type(self.participant_b)

        # Store model types in metadata for persistence
        if "model_type_a" not in self.metadata:
            self.metadata["model_type_a"] = self.model_type_a.value
        if "model_type_b" not in self.metadata:
            self.metadata["model_type_b"] = self.model_type_b.value

    @property
    def config_key(self) -> str:
        """Return config key like 'hex8_2p'."""
        return f"{self.board_type}_{self.num_players}p"


@dataclass
class EloRecordResult:
    """Result of an Elo recording attempt."""

    success: bool
    match_id: str | None = None
    error: str | None = None
    error_type: str | None = None  # Exception class name
    queued_to_dlq: bool = False
    validation_errors: list[str] = field(default_factory=list)

    # Rating changes (populated on success)
    elo_before_a: float | None = None
    elo_after_a: float | None = None
    elo_before_b: float | None = None
    elo_after_b: float | None = None


# ============================================
# Model type detection
# ============================================


def detect_model_type(participant_id: str) -> ModelType:
    """Detect model type from participant ID or model path.

    Args:
        participant_id: Model identifier, path, or composite ID

    Returns:
        Detected ModelType enum value
    """
    pid_lower = participant_id.lower()

    # Check for baseline markers (none:random, none:heuristic, etc.)
    if pid_lower.startswith("none:"):
        return ModelType.NONE

    # Check for explicit type markers in ID
    if "random" in pid_lower or "baseline" in pid_lower:
        return ModelType.NONE
    if "heuristic" in pid_lower:
        return ModelType.NONE

    # Check for NNUE variants
    if "nnue_mp" in pid_lower or "nnue-mp" in pid_lower:
        return ModelType.NNUE
    if "nnue" in pid_lower:
        return ModelType.NNUE

    # Check if it's a file path
    if "/" in participant_id or "\\" in participant_id:
        return _detect_from_path(participant_id)

    # Default to NN for neural network models
    return ModelType.NEURAL_NET


def _detect_from_path(path: str) -> ModelType:
    """Detect model type from file path or checkpoint."""
    path_lower = path.lower()

    # Check filename patterns
    if "nnue_mp" in path_lower or "nnue-mp" in path_lower:
        return ModelType.NNUE
    if "nnue" in path_lower:
        return ModelType.NNUE

    # Try to load checkpoint and inspect (optional, for accuracy)
    try:
        file_path = Path(path)
        if file_path.exists() and file_path.suffix in (".pth", ".pt"):
            return _detect_from_checkpoint(file_path)
    except Exception:
        pass  # Fall through to default

    return ModelType.NEURAL_NET


def _detect_from_checkpoint(checkpoint_path: Path) -> ModelType:
    """Detect model type by loading checkpoint metadata.

    Only called if path exists and looks like a PyTorch checkpoint.
    """
    try:
        from app.utils.torch_utils import safe_load_checkpoint

        checkpoint = safe_load_checkpoint(str(checkpoint_path))
        if checkpoint is None:
            return ModelType.NEURAL_NET

        # Check metadata
        metadata = checkpoint.get("metadata", {})
        model_type_str = metadata.get("model_type", "").lower()
        if "nnue_mp" in model_type_str:
            return ModelType.NNUE
        if "nnue" in model_type_str:
            return ModelType.NNUE

        # Check architecture info
        arch = metadata.get("architecture", "").lower()
        if "nnue" in arch:
            return ModelType.NNUE

    except Exception:
        pass

    return ModelType.NEURAL_NET


# ============================================
# Validation
# ============================================


def validate_harness_compatibility(
    harness: HarnessType,
    model_type: ModelType,
    num_players: int,
) -> list[str]:
    """Validate that harness is compatible with model type and player count.

    Returns:
        List of validation errors (empty if valid)
    """
    errors: list[str] = []

    # NNUE-only harnesses
    if harness == HarnessType.MINIMAX:
        if num_players != 2:
            errors.append(f"minimax requires 2 players, got {num_players}")
        # Minimax can work with both NN and NNUE

    if harness == HarnessType.BRS:
        if num_players > 4:
            errors.append(f"BRS supports max 4 players, got {num_players}")

    # NN-required harnesses (Jan 2026: replaced MCTS with DESCENT)
    nn_harnesses = {HarnessType.GUMBEL_MCTS, HarnessType.GPU_GUMBEL, HarnessType.DESCENT}
    if harness in nn_harnesses and model_type == ModelType.NONE:
        errors.append(f"{harness.value} requires a neural model, got {model_type.value}")

    # Baseline harnesses shouldn't have NN models (warning, not error)
    if harness in {HarnessType.RANDOM, HarnessType.HEURISTIC} and model_type != ModelType.NONE:
        # This is a warning, not an error - you can run heuristic against NN
        pass

    return errors


# ============================================
# Event emission
# ============================================


def _emit_elo_event(event_type: str, payload: dict[str, Any]) -> None:
    """Emit an Elo-related event via the event router.

    Silently fails if event router not available - this is expected
    in some contexts (tests, standalone scripts).
    """
    try:
        from app.coordination.event_router import DataEventType, emit_event

        # Map string to DataEventType if possible
        try:
            dtype = DataEventType(event_type)
        except ValueError:
            # Event type not in enum - log and skip
            logger.debug(f"[elo_recording] Event type '{event_type}' not in DataEventType enum")
            return

        emit_event(dtype, payload)
    except ImportError:
        pass  # Event router not available
    except Exception as e:
        logger.debug(f"[elo_recording] Failed to emit event: {e}")


def _queue_to_dlq(spec: EloMatchSpec, error: str) -> bool:
    """Queue failed recording to Dead Letter Queue for later retry.

    Returns:
        True if successfully queued, False otherwise
    """
    try:
        from app.coordination.dead_letter_queue import get_dlq, DLQEntry

        dlq = get_dlq()
        if dlq is None:
            return False

        entry = DLQEntry(
            event_type="elo_recording_failed",
            payload={
                "participant_a": spec.participant_a,
                "participant_b": spec.participant_b,
                "winner": spec.winner,
                "board_type": spec.board_type,
                "num_players": spec.num_players,
                "harness_type": spec.harness_type.value,
                "game_length": spec.game_length,
                "error": error,
            },
            source="elo_recording",
        )
        dlq.enqueue(entry)
        return True
    except ImportError:
        return False  # DLQ not available
    except Exception as e:
        logger.debug(f"[elo_recording] Failed to queue to DLQ: {e}")
        return False


# ============================================
# Main entry point
# ============================================


def safe_record_elo(spec: EloMatchSpec, *, use_dlq: bool = True) -> EloRecordResult:
    """THE authoritative entry point for Elo recording.

    This function:
    1. Validates the harness/model compatibility
    2. Records the match via EloService
    3. Emits events for observability
    4. Queues failures to DLQ for retry
    5. NEVER silently fails

    Args:
        spec: Complete match specification (harness_type REQUIRED)
        use_dlq: Whether to queue failures to DLQ (default True)

    Returns:
        EloRecordResult with success status, match_id, and any errors
    """
    result = EloRecordResult(success=False)
    match_id = str(uuid.uuid4())

    # Validate harness compatibility
    validation_errors = validate_harness_compatibility(
        spec.harness_type,
        spec.model_type_a or ModelType.NEURAL_NET,
        spec.num_players,
    )
    if spec.model_type_b and spec.model_type_b != ModelType.NONE:
        validation_errors.extend(
            validate_harness_compatibility(
                spec.harness_type,
                spec.model_type_b,
                spec.num_players,
            )
        )

    if validation_errors:
        result.validation_errors = validation_errors
        result.error = f"Validation failed: {'; '.join(validation_errors)}"
        result.error_type = "ValidationError"
        logger.warning(
            f"[elo_recording] Validation failed for {spec.config_key}: {result.error}"
        )
        _emit_elo_event(
            "elo_validation_failed",
            {
                "config_key": spec.config_key,
                "participant_a": spec.participant_a,
                "participant_b": spec.participant_b,
                "harness_type": spec.harness_type.value,
                "errors": validation_errors,
            },
        )
        # Still attempt recording despite warnings - validation is informational
        # return result  # Uncomment to block on validation errors

    try:
        from app.training.elo_service import get_elo_service

        elo_service = get_elo_service()

        # Get ratings before for reporting
        try:
            rating_a_before = elo_service.get_rating(
                spec.participant_a, spec.board_type, spec.num_players
            )
            result.elo_before_a = rating_a_before.rating
        except Exception:
            result.elo_before_a = None

        try:
            rating_b_before = elo_service.get_rating(
                spec.participant_b, spec.board_type, spec.num_players
            )
            result.elo_before_b = rating_b_before.rating
        except Exception:
            result.elo_before_b = None

        # Record the match
        match_result = elo_service.record_match(
            participant_a=spec.participant_a,
            participant_b=spec.participant_b,
            winner=spec.winner,
            board_type=spec.board_type,
            num_players=spec.num_players,
            game_length=spec.game_length,
            duration_sec=spec.duration_sec,
            tournament_id=spec.tournament_id,
            metadata=spec.metadata,
            harness_type=spec.harness_type.value,  # Pass string value to service
        )

        result.success = True
        result.match_id = match_result.match_id if hasattr(match_result, "match_id") else match_id

        # Get ratings after
        try:
            rating_a_after = elo_service.get_rating(
                spec.participant_a, spec.board_type, spec.num_players
            )
            result.elo_after_a = rating_a_after.rating
        except Exception:
            pass

        try:
            rating_b_after = elo_service.get_rating(
                spec.participant_b, spec.board_type, spec.num_players
            )
            result.elo_after_b = rating_b_after.rating
        except Exception:
            pass

        # Emit success event
        _emit_elo_event(
            "elo_updated",
            {
                "config_key": spec.config_key,
                "participant_a": spec.participant_a,
                "participant_b": spec.participant_b,
                "winner": spec.winner,
                "harness_type": spec.harness_type.value,
                "model_type_a": spec.model_type_a.value if spec.model_type_a else None,
                "model_type_b": spec.model_type_b.value if spec.model_type_b else None,
                "match_id": result.match_id,
            },
        )

        logger.debug(
            f"[elo_recording] Recorded match {result.match_id} for {spec.config_key}: "
            f"{spec.participant_a} vs {spec.participant_b} (harness={spec.harness_type.value})"
        )

    except ImportError as e:
        result.error = f"EloService not available: {e}"
        result.error_type = "ImportError"
        logger.error(f"[elo_recording] {result.error}")

    except Exception as e:
        result.error = str(e)
        result.error_type = type(e).__name__
        logger.error(
            f"[elo_recording] Failed to record Elo for {spec.config_key}: "
            f"{result.error_type}: {result.error}"
        )

        # Queue to DLQ for retry
        if use_dlq:
            result.queued_to_dlq = _queue_to_dlq(spec, result.error)
            if result.queued_to_dlq:
                logger.info(f"[elo_recording] Queued failed recording to DLQ")

        # Emit failure event
        _emit_elo_event(
            "elo_recording_failed",
            {
                "config_key": spec.config_key,
                "participant_a": spec.participant_a,
                "participant_b": spec.participant_b,
                "harness_type": spec.harness_type.value,
                "error": result.error,
                "error_type": result.error_type,
                "queued_to_dlq": result.queued_to_dlq,
            },
        )

    return result


# ============================================
# Convenience functions
# ============================================


def record_gauntlet_match(
    model_id: str,
    baseline: str,
    model_won: bool,
    board_type: str,
    num_players: int,
    harness_type: HarnessType,
    *,
    game_length: int = 0,
    duration_sec: float = 0.0,
    composite_ids: bool = True,
) -> EloRecordResult:
    """Record a gauntlet match result (model vs baseline).

    Args:
        model_id: Model identifier
        baseline: Baseline name (e.g., "heuristic", "random")
        model_won: True if model won, False if baseline won
        board_type: Board type string
        num_players: Number of players
        harness_type: AI harness type (REQUIRED)
        game_length: Number of moves in game
        duration_sec: Game duration in seconds
        composite_ids: Whether to use composite participant IDs

    Returns:
        EloRecordResult
    """
    # Build participant IDs
    if composite_ids:
        # Composite format: {model_id}:{harness}:{config_hash}
        config_hash = f"d{num_players}"
        participant_a = f"{model_id}:{harness_type.value}:{config_hash}"
        participant_b = f"none:{baseline}:{config_hash}"
    else:
        participant_a = model_id
        participant_b = baseline

    winner = participant_a if model_won else participant_b

    spec = EloMatchSpec(
        participant_a=participant_a,
        participant_b=participant_b,
        winner=winner,
        board_type=board_type,
        num_players=num_players,
        harness_type=harness_type,
        game_length=game_length,
        duration_sec=duration_sec,
        tournament_id=f"gauntlet_{board_type}_{num_players}p",
        metadata={"source": "gauntlet"},
    )

    return safe_record_elo(spec)


def record_tournament_match(
    participant_a: str,
    participant_b: str,
    winner: str | None,
    board_type: str,
    num_players: int,
    harness_type: HarnessType,
    tournament_id: str,
    *,
    game_length: int = 0,
    duration_sec: float = 0.0,
    round_number: int | None = None,
) -> EloRecordResult:
    """Record a tournament match result.

    Args:
        participant_a: First participant ID
        participant_b: Second participant ID
        winner: Winner ID (None for draw)
        board_type: Board type string
        num_players: Number of players
        harness_type: AI harness type (REQUIRED)
        tournament_id: Tournament identifier
        game_length: Number of moves in game
        duration_sec: Game duration in seconds
        round_number: Tournament round number

    Returns:
        EloRecordResult
    """
    metadata = {"source": "tournament", "tournament_id": tournament_id}
    if round_number is not None:
        metadata["round_number"] = round_number

    spec = EloMatchSpec(
        participant_a=participant_a,
        participant_b=participant_b,
        winner=winner,
        board_type=board_type,
        num_players=num_players,
        harness_type=harness_type,
        game_length=game_length,
        duration_sec=duration_sec,
        tournament_id=tournament_id,
        metadata=metadata,
    )

    return safe_record_elo(spec)


def record_selfplay_match(
    model_id: str,
    board_type: str,
    num_players: int,
    harness_type: HarnessType,
    winner_player: int | None,
    *,
    game_length: int = 0,
    duration_sec: float = 0.0,
) -> EloRecordResult:
    """Record a selfplay match result (model vs itself).

    For selfplay, both participants are the same model. The winner is
    determined by which player index won (0-indexed).

    Args:
        model_id: Model identifier
        board_type: Board type string
        num_players: Number of players
        harness_type: AI harness type (REQUIRED)
        winner_player: Winning player index (None for draw)
        game_length: Number of moves in game
        duration_sec: Game duration in seconds

    Returns:
        EloRecordResult
    """
    # For selfplay, use player indices in participant IDs
    config_hash = f"d{num_players}"
    participant_a = f"{model_id}:{harness_type.value}:{config_hash}:p0"
    participant_b = f"{model_id}:{harness_type.value}:{config_hash}:p1"

    if winner_player is None:
        winner = None
    elif winner_player == 0:
        winner = participant_a
    else:
        winner = participant_b

    spec = EloMatchSpec(
        participant_a=participant_a,
        participant_b=participant_b,
        winner=winner,
        board_type=board_type,
        num_players=num_players,
        harness_type=harness_type,
        game_length=game_length,
        duration_sec=duration_sec,
        tournament_id=f"selfplay_{board_type}_{num_players}p",
        metadata={"source": "selfplay"},
    )

    return safe_record_elo(spec)


def record_baseline_calibration_match(
    baseline_a: str,
    baseline_b: str,
    winner: str | None,
    board_type: str,
    num_players: int,
    *,
    game_length: int = 0,
) -> EloRecordResult:
    """Record a baseline calibration match (baseline vs baseline).

    Used for calibrating relative baseline strengths (random vs heuristic, etc).

    Args:
        baseline_a: First baseline name
        baseline_b: Second baseline name
        winner: Winner name (None for draw)
        board_type: Board type string
        num_players: Number of players
        game_length: Number of moves in game

    Returns:
        EloRecordResult
    """
    config_hash = f"d{num_players}"
    participant_a = f"none:{baseline_a}:{config_hash}"
    participant_b = f"none:{baseline_b}:{config_hash}"

    winner_id = None
    if winner == baseline_a:
        winner_id = participant_a
    elif winner == baseline_b:
        winner_id = participant_b

    spec = EloMatchSpec(
        participant_a=participant_a,
        participant_b=participant_b,
        winner=winner_id,
        board_type=board_type,
        num_players=num_players,
        harness_type=HarnessType.HEURISTIC,  # Baseline matches use heuristic harness
        game_length=game_length,
        tournament_id=f"baseline_calibration_{board_type}_{num_players}p",
        metadata={"source": "baseline_calibration"},
    )

    return safe_record_elo(spec)


# ============================================
# Bulk operations
# ============================================


def record_batch(specs: list[EloMatchSpec]) -> list[EloRecordResult]:
    """Record multiple matches in batch.

    Args:
        specs: List of match specifications

    Returns:
        List of results in same order as specs
    """
    results = []
    for spec in specs:
        results.append(safe_record_elo(spec))
    return results


# ============================================
# Exports
# ============================================


__all__ = [
    # Enums
    "HarnessType",
    "ModelType",
    # Data classes
    "EloMatchSpec",
    "EloRecordResult",
    # Main function
    "safe_record_elo",
    # Convenience functions
    "record_gauntlet_match",
    "record_tournament_match",
    "record_selfplay_match",
    "record_baseline_calibration_match",
    "record_batch",
    # Utilities
    "detect_model_type",
    "validate_harness_compatibility",
]
