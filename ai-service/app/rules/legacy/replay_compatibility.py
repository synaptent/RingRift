"""Legacy replay compatibility layer for RingRift AI service.

This module provides functions for replaying games that were recorded under
previous versions of the rules engine. It attempts canonical replay first,
falling back to legacy handling when necessary.

Usage:
    from app.rules.legacy import replay_with_legacy_fallback, requires_legacy_replay

    # Check if a game needs legacy handling
    if requires_legacy_replay(game_record, schema_version=5):
        # Use legacy fallback
        result = replay_with_legacy_fallback(game_record)

Design Goals:
    1. Canonical rules are always tried first
    2. Legacy fallback is logged with deprecation warnings
    3. Games using legacy replay are tracked for future migration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.models import GameState

logger = logging.getLogger(__name__)

# Schema versions that definitely require legacy handling
# Games from these versions may have bugs/differences that canonical rules reject
LEGACY_SCHEMA_VERSIONS: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6, 7})

# Schema version where hex geometry changed (radius 10 -> 12)
HEX_GEOMETRY_CHANGE_VERSION: int = 8

# Current canonical schema version
CANONICAL_SCHEMA_VERSION: int = 9


@dataclass
class ReplayResult:
    """Result of a game replay attempt."""

    success: bool
    final_state: GameState | None
    used_legacy: bool
    error_message: str | None = None
    warnings: list[str] | None = None


def requires_legacy_replay(
    game_record: dict[str, Any],
    schema_version: int | None = None,
    board_type: str | None = None,
) -> bool:
    """Determine if a game record requires legacy replay handling.

    Args:
        game_record: Game record dictionary from replay database
        schema_version: Database schema version (if known)
        board_type: Board type string (e.g., 'hexagonal', 'square8')

    Returns:
        True if the game likely needs legacy replay handling

    Detection heuristics:
        1. Old schema versions (< 8) may have move format differences
        2. Hexagonal games from before geometry change need legacy handling
        3. Games with legacy move types need conversion
    """
    # Old schema versions
    if schema_version is not None and schema_version in LEGACY_SCHEMA_VERSIONS:
        logger.debug(
            f"LEGACY_REPLAY: Game from schema v{schema_version} requires legacy handling"
        )
        return True

    # Hexagonal games from before geometry change
    if board_type in ("hexagonal", "hex8"):
        # Check for old geometry markers
        board_data = game_record.get("initial_state", {}).get("board", {})
        if board_data:
            # Old hex geometry had different cell counts
            # radius 10 = 331 cells, radius 12 = 469 cells
            positions = board_data.get("positions") or board_data.get("cells") or {}
            if len(positions) == 331:  # Old hex geometry
                logger.debug("LEGACY_REPLAY: Detected old hex geometry (radius 10)")
                return True

    # Check for legacy move types in move history
    moves = game_record.get("moves", [])
    if moves:
        from app.rules.legacy.move_type_aliases import is_legacy_move_type

        for move in moves[:10]:  # Check first 10 moves
            if isinstance(move, dict):
                move_type = move.get("move_type") or move.get("type") or ""
                if is_legacy_move_type(str(move_type)):
                    logger.debug(
                        f"LEGACY_REPLAY: Detected legacy move type '{move_type}'"
                    )
                    return True

    return False


def replay_with_legacy_fallback(
    game_record: dict[str, Any],
    schema_version: int | None = None,
    strict: bool = False,
) -> ReplayResult:
    """Replay a game with legacy fallback if canonical replay fails.

    This function first attempts to replay using canonical rules. If that fails,
    it normalizes the game record and retries with legacy-compatible settings.

    Args:
        game_record: Game record dictionary from replay database
        schema_version: Database schema version (if known)
        strict: If True, raise exception on failure instead of returning error

    Returns:
        ReplayResult with success status and final state

    Raises:
        ValueError: If strict=True and replay fails
    """
    warnings: list[str] = []

    # First, try canonical replay
    try:
        from app.rules import replay_game_canonical

        final_state = replay_game_canonical(game_record)
        _record_replay_result(used_legacy=False, success=True)
        return ReplayResult(
            success=True,
            final_state=final_state,
            used_legacy=False,
            warnings=warnings if warnings else None,
        )
    except ImportError:
        # replay_game_canonical not available, use fallback
        warnings.append("Canonical replay function not available")
    except Exception as canonical_error:
        warnings.append(f"Canonical replay failed: {canonical_error}")
        logger.info(
            f"LEGACY_FALLBACK: Canonical replay failed, trying legacy handling: "
            f"{type(canonical_error).__name__}: {canonical_error}"
        )

    # Normalize legacy state
    try:
        from app.rules.legacy.state_normalization import normalize_legacy_state

        # Deep copy to avoid modifying original
        import copy

        normalized_record = copy.deepcopy(game_record)

        # Normalize initial state
        if "initial_state" in normalized_record:
            normalize_legacy_state(normalized_record["initial_state"])

        # Normalize moves
        moves = normalized_record.get("moves", [])
        for move in moves:
            if isinstance(move, dict):
                normalize_legacy_state(move)

        warnings.append("Applied legacy state normalization")

    except Exception as norm_error:
        warnings.append(f"State normalization failed: {norm_error}")
        logger.warning(f"LEGACY_FALLBACK: State normalization failed: {norm_error}")
        normalized_record = game_record

    # Try replay with legacy-compatible settings
    try:
        from app._game_engine_legacy import GameEngine

        final_state = _replay_with_legacy_engine(normalized_record)

        logger.warning(
            f"LEGACY_REPLAY_USED: Game replayed using legacy code path. "
            f"This will be deprecated in a future release. "
            f"Schema version: {schema_version}, "
            f"Game ID: {game_record.get('game_id', 'unknown')}"
        )

        # Determine reason for legacy fallback
        reason = "fallback"
        if schema_version is not None and schema_version in LEGACY_SCHEMA_VERSIONS:
            reason = f"schema_v{schema_version}"
        _record_replay_result(used_legacy=True, success=True, reason=reason)

        return ReplayResult(
            success=True,
            final_state=final_state,
            used_legacy=True,
            warnings=warnings,
        )

    except Exception as legacy_error:
        error_msg = f"Legacy replay also failed: {legacy_error}"
        warnings.append(error_msg)
        logger.error(f"LEGACY_FALLBACK_FAILED: {error_msg}")

        _record_replay_result(used_legacy=True, success=False)

        if strict:
            raise ValueError(
                f"Game replay failed with both canonical and legacy handlers: "
                f"{warnings}"
            ) from legacy_error

        return ReplayResult(
            success=False,
            final_state=None,
            used_legacy=True,
            error_message=error_msg,
            warnings=warnings,
        )


def _replay_with_legacy_engine(game_record: dict[str, Any]) -> GameState:
    """Internal helper to replay using the legacy GameEngine.

    This is a minimal implementation for games that cannot be replayed
    with the canonical rules. It uses the legacy GameEngine directly.

    Args:
        game_record: Normalized game record dictionary

    Returns:
        Final game state after applying all moves

    Raises:
        Exception: If replay fails
    """
    from app._game_engine_legacy import GameEngine
    from app.models import BoardType, GameState

    # Extract game configuration
    board_type_str = game_record.get("board_type", "square8")
    num_players = game_record.get("num_players", 2)

    # Convert board type string to enum
    try:
        board_type = BoardType(board_type_str.lower())
    except ValueError:
        board_type = BoardType.SQUARE8

    # Initialize game engine
    engine = GameEngine()
    state = engine.new_game(
        board_type=board_type,
        num_players=num_players,
    )

    # Apply moves
    moves = game_record.get("moves", [])
    for i, move_data in enumerate(moves):
        if isinstance(move_data, dict):
            # Convert move dict to Move object
            from app.models import Move, MoveType

            move_type_str = move_data.get("move_type") or move_data.get("type") or ""
            try:
                from app.rules.legacy.move_type_aliases import convert_legacy_move_type

                move_type_str = convert_legacy_move_type(move_type_str, warn=False)
            except Exception:
                # Fall back to raw move type if conversion fails
                pass
            try:
                move_type = MoveType(move_type_str.lower())
            except ValueError:
                logger.warning(f"Unknown move type '{move_type_str}' at index {i}")
                continue

            move = Move(
                move_type=move_type,
                player=move_data.get("player", state.current_player),
                from_position=move_data.get("from_position") or move_data.get("from"),
                to_position=move_data.get("to_position") or move_data.get("to"),
                ring_index=move_data.get("ring_index"),
                target_position=move_data.get("target_position"),
                line_id=move_data.get("line_id"),
                territory_id=move_data.get("territory_id"),
            )

            state = engine.apply_move(state, move)
        else:
            # Already a Move object
            state = engine.apply_move(state, move_data)

    return state


# =============================================================================
# Legacy Replay Metrics Tracking
# =============================================================================

# Thread-safe counters for replay metrics
import threading
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LegacyReplayMetrics:
    """Thread-safe metrics for legacy replay tracking."""

    legacy_replays: int = 0
    canonical_replays: int = 0
    fallback_replays: int = 0  # Started canonical, fell back to legacy
    failed_replays: int = 0
    legacy_by_reason: dict[str, int] = field(default_factory=dict)
    last_reset: str = field(default_factory=lambda: datetime.now().isoformat())
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_canonical(self) -> None:
        """Record a successful canonical replay."""
        with self._lock:
            self.canonical_replays += 1

    def record_legacy(self, reason: str = "unknown") -> None:
        """Record a legacy replay with reason."""
        with self._lock:
            self.legacy_replays += 1
            self.legacy_by_reason[reason] = self.legacy_by_reason.get(reason, 0) + 1

    def record_fallback(self) -> None:
        """Record a fallback from canonical to legacy."""
        with self._lock:
            self.fallback_replays += 1
            self.legacy_replays += 1

    def record_failure(self) -> None:
        """Record a failed replay attempt."""
        with self._lock:
            self.failed_replays += 1

    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dictionary."""
        with self._lock:
            total = self.canonical_replays + self.legacy_replays
            return {
                "legacy_replays_total": self.legacy_replays,
                "canonical_replays_total": self.canonical_replays,
                "fallback_replays": self.fallback_replays,
                "failed_replays": self.failed_replays,
                "legacy_fallback_rate": (
                    self.legacy_replays / total if total > 0 else 0.0
                ),
                "legacy_by_reason": dict(self.legacy_by_reason),
                "last_reset": self.last_reset,
                "total_replays": total,
            }

    def reset(self) -> None:
        """Reset all counters."""
        with self._lock:
            self.legacy_replays = 0
            self.canonical_replays = 0
            self.fallback_replays = 0
            self.failed_replays = 0
            self.legacy_by_reason.clear()
            self.last_reset = datetime.now().isoformat()


# Global metrics instance
_replay_metrics = LegacyReplayMetrics()


def get_legacy_replay_stats() -> dict[str, Any]:
    """Get statistics about legacy replay usage.

    Returns:
        Dictionary with legacy replay statistics
    """
    return _replay_metrics.to_dict()


def reset_legacy_replay_stats() -> None:
    """Reset legacy replay statistics."""
    _replay_metrics.reset()
    logger.info("LEGACY_REPLAY_STATS: Metrics reset")


def _record_replay_result(
    used_legacy: bool,
    success: bool,
    reason: str | None = None,
) -> None:
    """Internal helper to record replay results.

    Args:
        used_legacy: Whether legacy replay was used
        success: Whether the replay succeeded
        reason: Reason for legacy usage (e.g., "old_schema", "hex_geometry")
    """
    if not success:
        _replay_metrics.record_failure()
    elif used_legacy:
        _replay_metrics.record_legacy(reason or "fallback")
    else:
        _replay_metrics.record_canonical()
