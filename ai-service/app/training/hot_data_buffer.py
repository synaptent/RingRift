"""Hot data buffer for streaming training data.

This module provides an in-memory buffer for recently generated game data,
enabling faster training cycles by avoiding full DB roundtrips.

Key features:
1. LRU eviction when buffer exceeds max size
2. Thread-safe operations for concurrent access
3. Periodic flush to canonical DB (background)
4. Feature extraction compatible with StreamingDataLoader

Event Integration:
- Emits NEW_GAMES_AVAILABLE when batch threshold reached
- Emits TRAINING_THRESHOLD_REACHED when buffer has enough data
- Emits DATA_SYNC_COMPLETED equivalent on flush
- Can subscribe to events to trigger automatic flushes

Usage:
    buffer = HotDataBuffer(max_size=1000)

    # Add games as they complete
    buffer.add_game(game_record)

    # Get training samples
    features, policies, values = buffer.get_training_batch(batch_size=256)

    # Flush to DB periodically
    buffer.flush_to_db(db_path)
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Event system integration (optional - graceful fallback if not available)
try:
    from app.distributed.data_events import (
        DataEventType,
        DataEvent,
        get_event_bus,
    )
    HAS_EVENT_SYSTEM = True
except ImportError:
    HAS_EVENT_SYSTEM = False
    logger.debug("Event system not available - HotDataBuffer running standalone")

# Quality scoring integration (December 2025)
try:
    from app.quality.unified_quality import (
        UnifiedQualityScorer,
        GameQuality,
        get_quality_scorer,
    )
    HAS_QUALITY_SCORER = True
except ImportError:
    HAS_QUALITY_SCORER = False
    UnifiedQualityScorer = None
    GameQuality = None

    def get_quality_scorer():
        return None

# Quality config integration
try:
    from app.config.unified_config import QualityConfig, get_config
    HAS_QUALITY_CONFIG = True
except ImportError:
    HAS_QUALITY_CONFIG = False
    QualityConfig = None

    def get_config():
        return None

# Data validation integration (optional - graceful fallback if not available)
try:
    from app.training.data_validation import (
        DataValidator,
        DataValidatorConfig,
        ValidationResult,
    )
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False
    logger.debug("Data validation not available - HotDataBuffer skipping validation")

# Encoder protocol - any encoder with encode_state method works
from typing import Protocol, runtime_checkable


@runtime_checkable
class StateEncoder(Protocol):
    """Protocol for state encoders compatible with HotDataBuffer."""

    def encode_state(self, state: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Encode a game state to (board_features, global_features)."""
        ...


@dataclass
class GameRecord:
    """Lightweight game record for hot buffer storage."""

    game_id: str
    board_type: str
    num_players: int
    moves: List[Dict[str, Any]]  # List of move records with state/action
    outcome: Dict[str, float]  # Player ID -> final score
    timestamp: float = field(default_factory=time.time)
    source: str = "hot_buffer"
    # Priority fields for experience replay
    avg_elo: float = 1500.0  # Average Elo of players in this game
    priority: float = 1.0  # Base priority (can be updated by TD error)
    from_promoted_model: bool = False  # Was this from a model that got promoted?
    # Manifest quality score (populated from data sync manifest)
    manifest_quality: float = 0.5  # Quality score from unified_manifest [0, 1]

    def to_training_samples(
        self,
        encoder: Optional[StateEncoder] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """Convert game to training samples.

        Args:
            encoder: Optional state encoder for raw game states.
                    If not provided, uses pre-computed features from moves.

        Returns:
            List of (board_features, global_features, policy, value) tuples.
            If encoder is None and no pre-computed features, returns samples
            with board_features only (global_features will be empty).
        """
        samples = []
        for move in self.moves:
            # Try pre-computed features first
            state_features = move.get("state_features")
            global_features = move.get("global_features")
            policy = move.get("policy_target")
            value = move.get("value_target")

            # If we have a raw state and an encoder, encode it
            raw_state = move.get("raw_state")
            if raw_state is not None and encoder is not None:
                try:
                    board_feats, global_feats = encoder.encode_state(raw_state)
                    state_features = board_feats
                    global_features = global_feats
                except Exception as e:
                    logger.warning(f"Failed to encode state in game {self.game_id}: {e}")
                    continue

            if state_features is not None and policy is not None and value is not None:
                board_arr = np.array(state_features, dtype=np.float32)
                global_arr = np.array(global_features, dtype=np.float32) if global_features is not None else np.array([], dtype=np.float32)
                policy_arr = np.array(policy, dtype=np.float32)
                samples.append((
                    board_arr,
                    global_arr,
                    policy_arr,
                    float(value),
                ))
        return samples

    def to_training_samples_legacy(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Legacy method for backward compatibility (board_features, policy, value)."""
        samples = []
        for move in self.moves:
            state = move.get("state_features")
            policy = move.get("policy_target")
            value = move.get("value_target")
            if state is not None and policy is not None and value is not None:
                samples.append((
                    np.array(state, dtype=np.float32),
                    np.array(policy, dtype=np.float32),
                    float(value),
                ))
        return samples


class HotDataBuffer:
    """Thread-safe in-memory buffer for recent game data.

    Implements LRU eviction when buffer exceeds max_size.
    Optionally emits events for coordination with other components.
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 500,
        buffer_name: str = "default",
        enable_events: bool = True,
        training_threshold: int = 100,
        batch_notification_size: int = 50,
        enable_validation: bool = True,
        skip_invalid_games: bool = False,
        encoder: Optional[StateEncoder] = None,
    ):
        """Initialize the hot data buffer.

        Args:
            max_size: Maximum number of games to keep in buffer
            max_memory_mb: Soft memory limit in MB (triggers eviction)
            buffer_name: Identifier for this buffer (used in events)
            enable_events: Whether to emit events
            training_threshold: Games needed before emitting TRAINING_THRESHOLD_REACHED
            batch_notification_size: Emit NEW_GAMES_AVAILABLE every N games
            enable_validation: Whether to validate games before flush
            skip_invalid_games: If True, skip invalid games during flush; if False, log warning but include
            encoder: Optional state encoder for games with raw_state data
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.buffer_name = buffer_name
        self.enable_events = enable_events and HAS_EVENT_SYSTEM
        self.training_threshold = training_threshold
        self.batch_notification_size = batch_notification_size
        self.enable_validation = enable_validation and HAS_VALIDATION
        self.skip_invalid_games = skip_invalid_games
        self._encoder = encoder

        self._buffer: OrderedDict[str, GameRecord] = OrderedDict()
        self._lock = threading.RLock()
        self._sample_cache: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        self._cache_dirty = True
        self._total_samples = 0
        self._flushed_game_ids: set = set()

        # Event tracking
        self._games_since_notification = 0
        self._training_threshold_emitted = False
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Validation
        self._validator: Optional[DataValidator] = None
        if self.enable_validation:
            self._validator = DataValidator(DataValidatorConfig(
                sample_rate=1.0,  # Validate all samples
                max_issues_to_report=10,  # Don't flood logs
            ))
        self._validation_stats = {
            "games_validated": 0,
            "games_with_issues": 0,
            "games_skipped": 0,
        }

        # Manifest quality lookup (populated via set_quality_lookup)
        self._quality_lookup: Dict[str, float] = {}
        self._elo_lookup: Dict[str, float] = {}

    def set_quality_lookup(
        self,
        quality_lookup: Optional[Dict[str, float]] = None,
        elo_lookup: Optional[Dict[str, float]] = None,
    ) -> int:
        """Set manifest quality lookup tables for priority computation.

        Call this with data from DataPipelineController/manifest to enable
        quality-weighted experience replay.

        Args:
            quality_lookup: Dict mapping game_id to manifest quality_score [0, 1]
            elo_lookup: Dict mapping game_id to avg_player_elo

        Returns:
            Number of existing buffer games updated with quality data
        """
        self._quality_lookup = quality_lookup or {}
        self._elo_lookup = elo_lookup or {}

        # Update existing games in buffer
        updated = 0
        with self._lock:
            for game in self._buffer.values():
                if game.game_id in self._quality_lookup:
                    game.manifest_quality = self._quality_lookup[game.game_id]
                    updated += 1
                if game.game_id in self._elo_lookup:
                    game.avg_elo = self._elo_lookup[game.game_id]
                # Recompute priority with new quality data
                game.priority = self.compute_game_priority(game)

        if updated > 0:
            logger.info(f"Updated {updated} buffer games with manifest quality data")

        return updated

    def set_encoder(self, encoder: StateEncoder) -> None:
        """Set or update the state encoder.

        Args:
            encoder: State encoder for games with raw_state data
        """
        self._encoder = encoder
        self._cache_dirty = True  # Invalidate cache to re-encode with new encoder

    def _validate_game(self, game: GameRecord) -> Tuple[bool, Optional[ValidationResult]]:
        """Validate a game record's training samples.

        Args:
            game: GameRecord to validate

        Returns:
            Tuple of (is_valid, validation_result)
        """
        if not self.enable_validation or self._validator is None:
            return True, None

        samples = game.to_training_samples(encoder=self._encoder)
        if not samples:
            # Empty samples - treat as valid (may be a game-level record)
            return True, None

        # Convert samples to arrays for validation
        # Samples are now (board_features, global_features, policy, value)
        try:
            features = np.array([s[0] for s in samples])
            policies = np.array([s[2] for s in samples])  # index 2 is policy
            values = np.array([s[3] for s in samples])    # index 3 is value

            data = {
                'features': features,
                'policy': policies,
                'values': values,
            }

            result = self._validator.validate_arrays(data)
            self._validation_stats["games_validated"] += 1

            if not result.valid:
                self._validation_stats["games_with_issues"] += 1
                logger.warning(
                    f"Validation issues in game {game.game_id}: {result.summary()}"
                )

            return result.valid, result

        except Exception as e:
            logger.warning(f"Validation error for game {game.game_id}: {e}")
            return False, None

    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        return self._validation_stats.copy()

    def _try_emit_event(self, event_type: DataEventType, payload: Dict[str, Any]) -> None:
        """Try to emit an event (fire-and-forget from sync context)."""
        if not self.enable_events:
            return

        try:
            bus = get_event_bus()
            event = DataEvent(
                event_type=event_type,
                payload=payload,
                source=f"hot_buffer:{self.buffer_name}",
            )

            # Try to schedule on an event loop if available
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(bus.publish(event))
            except RuntimeError:
                # No running loop - try to get/create one
                if self._event_loop is None:
                    try:
                        self._event_loop = asyncio.get_event_loop()
                    except RuntimeError:
                        self._event_loop = asyncio.new_event_loop()

                if self._event_loop and not self._event_loop.is_closed():
                    if self._event_loop.is_running():
                        self._event_loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(bus.publish(event))
                        )
                    else:
                        # Can't emit without a running loop - just log
                        logger.debug(f"Event loop not running, skipping event: {event_type.value}")
        except Exception as e:
            logger.debug(f"Failed to emit event {event_type.value}: {e}")

    async def emit_event_async(self, event_type: DataEventType, payload: Dict[str, Any]) -> None:
        """Emit an event asynchronously."""
        if not self.enable_events:
            return

        bus = get_event_bus()
        event = DataEvent(
            event_type=event_type,
            payload=payload,
            source=f"hot_buffer:{self.buffer_name}",
        )
        await bus.publish(event)

    def add_game(self, game: GameRecord) -> None:
        """Add a game to the buffer (thread-safe)."""
        evicted = False
        should_emit_new_games = False
        should_emit_threshold = False

        with self._lock:
            # Remove if already exists (to update LRU position)
            if game.game_id in self._buffer:
                del self._buffer[game.game_id]

            self._buffer[game.game_id] = game
            self._cache_dirty = True

            # Track for event emission
            self._games_since_notification += 1
            if self._games_since_notification >= self.batch_notification_size:
                should_emit_new_games = True
                self._games_since_notification = 0

            # Check training threshold
            if not self._training_threshold_emitted and len(self._buffer) >= self.training_threshold:
                should_emit_threshold = True
                self._training_threshold_emitted = True

            # Evict oldest entries if over capacity
            while len(self._buffer) > self.max_size:
                self._buffer.popitem(last=False)
                evicted = True

        # Emit events outside lock
        if should_emit_new_games:
            self._try_emit_event(DataEventType.NEW_GAMES_AVAILABLE, {
                "buffer_name": self.buffer_name,
                "new_games": self.batch_notification_size,
                "total_games": len(self._buffer),
                "source": "hot_buffer",
            })

        if should_emit_threshold:
            self._try_emit_event(DataEventType.TRAINING_THRESHOLD_REACHED, {
                "buffer_name": self.buffer_name,
                "game_count": len(self._buffer),
                "threshold": self.training_threshold,
                "source": "hot_buffer",
            })
            logger.info(f"Hot buffer '{self.buffer_name}' reached training threshold ({self.training_threshold} games)")

    def add_game_from_dict(self, data: Dict[str, Any]) -> None:
        """Add a game from a dictionary representation."""
        game_id = str(data.get("game_id", ""))

        # Get quality and elo from lookup tables if available
        manifest_quality = float(data.get("manifest_quality", 0.5))
        avg_elo = float(data.get("avg_elo", 1500.0))

        # Override with manifest lookup if available
        if game_id in self._quality_lookup:
            manifest_quality = self._quality_lookup[game_id]
        if game_id in self._elo_lookup:
            avg_elo = self._elo_lookup[game_id]

        game = GameRecord(
            game_id=game_id,
            board_type=str(data.get("board_type", "square8")),
            num_players=int(data.get("num_players", 2)),
            moves=data.get("moves", []),
            outcome=data.get("outcome", {}),
            timestamp=float(data.get("timestamp", time.time())),
            source=str(data.get("source", "hot_buffer")),
            # Priority fields
            avg_elo=avg_elo,
            priority=float(data.get("priority", 1.0)),
            from_promoted_model=bool(data.get("from_promoted_model", False)),
            manifest_quality=manifest_quality,
        )
        # Compute initial priority based on Elo, recency, and quality
        game.priority = self.compute_game_priority(game)
        self.add_game(game)

    def get_game(self, game_id: str) -> Optional[GameRecord]:
        """Get a game by ID (thread-safe)."""
        with self._lock:
            return self._buffer.get(game_id)

    def remove_game(self, game_id: str) -> bool:
        """Remove a game from the buffer (thread-safe)."""
        with self._lock:
            if game_id in self._buffer:
                del self._buffer[game_id]
                self._cache_dirty = True
                return True
            return False

    def get_all_games(self) -> List[GameRecord]:
        """Get all games in the buffer (thread-safe copy)."""
        with self._lock:
            return list(self._buffer.values())

    def get_unflushed_games(self) -> List[GameRecord]:
        """Get games that haven't been flushed to DB yet."""
        with self._lock:
            return [
                g for g in self._buffer.values()
                if g.game_id not in self._flushed_game_ids
            ]

    def __len__(self) -> int:
        """Return number of games in buffer."""
        with self._lock:
            return len(self._buffer)

    def __contains__(self, game_id: str) -> bool:
        """Check if game is in buffer."""
        with self._lock:
            return game_id in self._buffer

    def _rebuild_sample_cache(self) -> None:
        """Rebuild the flattened sample cache from all games."""
        samples = []
        for game in self._buffer.values():
            samples.extend(game.to_training_samples(encoder=self._encoder))
        self._sample_cache = samples
        self._total_samples = len(samples)
        self._cache_dirty = False

    def get_training_batch(
        self,
        batch_size: int = 256,
        shuffle: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get a batch of training samples.

        Args:
            batch_size: Number of samples to return
            shuffle: Whether to randomly sample (vs sequential)

        Returns:
            Tuple of (board_features, global_features, policies, values) as numpy arrays
        """
        with self._lock:
            if self._cache_dirty:
                self._rebuild_sample_cache()

            if not self._sample_cache:
                # Return empty arrays with correct shapes
                return (
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                )

            if shuffle:
                indices = np.random.choice(
                    len(self._sample_cache),
                    size=min(batch_size, len(self._sample_cache)),
                    replace=False,
                )
            else:
                indices = list(range(min(batch_size, len(self._sample_cache))))

            board_features = []
            global_features = []
            policies = []
            values = []

            for idx in indices:
                bf, gf, p, v = self._sample_cache[idx]
                board_features.append(bf)
                global_features.append(gf)
                policies.append(p)
                values.append(v)

            return (
                np.array(board_features, dtype=np.float32),
                np.array(global_features, dtype=np.float32),
                np.array(policies, dtype=np.float32),
                np.array(values, dtype=np.float32),
            )

    def get_sample_iterator(
        self,
        batch_size: int = 256,
        epochs: int = 1,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Iterator over training samples for multiple epochs.

        Yields batches of (board_features, global_features, policies, values).
        """
        with self._lock:
            if self._cache_dirty:
                self._rebuild_sample_cache()

            if not self._sample_cache:
                return

            for _ in range(epochs):
                # Shuffle samples for each epoch
                indices = np.random.permutation(len(self._sample_cache))

                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    board_features = []
                    global_features = []
                    policies = []
                    values = []

                    for idx in batch_indices:
                        bf, gf, p, v = self._sample_cache[idx]
                        board_features.append(bf)
                        global_features.append(gf)
                        policies.append(p)
                        values.append(v)

                    yield (
                        np.array(board_features, dtype=np.float32),
                        np.array(global_features, dtype=np.float32),
                        np.array(policies, dtype=np.float32),
                        np.array(values, dtype=np.float32),
                    )

    @property
    def total_samples(self) -> int:
        """Total number of training samples in buffer."""
        with self._lock:
            if self._cache_dirty:
                self._rebuild_sample_cache()
            return self._total_samples

    @property
    def game_count(self) -> int:
        """Number of games in buffer."""
        return len(self)

    def mark_flushed(self, game_ids: List[str]) -> None:
        """Mark games as flushed to persistent storage."""
        with self._lock:
            self._flushed_game_ids.update(game_ids)

    def clear_flushed(self) -> int:
        """Remove games that have been flushed to persistent storage.

        Returns number of games removed.
        """
        with self._lock:
            to_remove = [
                gid for gid in self._buffer.keys()
                if gid in self._flushed_game_ids
            ]
            for gid in to_remove:
                del self._buffer[gid]
            self._flushed_game_ids.clear()
            self._cache_dirty = True
            return len(to_remove)

    def flush_to_jsonl(self, path: Path) -> int:
        """Flush unflushed games to JSONL file with optional validation.

        Returns number of games written. Emits DATA_SYNC_COMPLETED on success.
        Invalid games are either skipped or included with warnings depending on
        the skip_invalid_games setting.
        """
        games = self.get_unflushed_games()
        if not games:
            return 0

        path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        written = 0
        skipped = 0
        flushed_ids = []

        with open(path, "a", encoding="utf-8") as f:
            for game in games:
                # Validate game before flushing
                is_valid, _ = self._validate_game(game)
                if not is_valid and self.skip_invalid_games:
                    skipped += 1
                    self._validation_stats["games_skipped"] += 1
                    logger.info(f"Skipping invalid game {game.game_id}")
                    flushed_ids.append(game.game_id)  # Still mark as flushed to avoid re-processing
                    continue

                record = {
                    "game_id": game.game_id,
                    "board_type": game.board_type,
                    "num_players": game.num_players,
                    "moves": game.moves,
                    "outcome": game.outcome,
                    "timestamp": game.timestamp,
                    "source": game.source,
                    # Priority fields for experience replay
                    "avg_elo": game.avg_elo,
                    "priority": game.priority,
                    "from_promoted_model": game.from_promoted_model,
                }
                f.write(json.dumps(record) + "\n")
                written += 1
                flushed_ids.append(game.game_id)

        self.mark_flushed(flushed_ids)

        # Emit flush completed event with validation stats
        duration = time.time() - start_time
        self._try_emit_event(DataEventType.DATA_SYNC_COMPLETED, {
            "buffer_name": self.buffer_name,
            "games_flushed": written,
            "games_skipped": skipped,
            "target_path": str(path),
            "duration_seconds": duration,
            "source": "hot_buffer",
            "validation_stats": self._validation_stats.copy(),
        })
        if skipped > 0:
            logger.info(f"Flushed {written} games, skipped {skipped} invalid from '{self.buffer_name}' to {path}")
        else:
            logger.debug(f"Flushed {written} games from '{self.buffer_name}' to {path} in {duration:.2f}s")

        return written

    async def flush_to_jsonl_async(self, path: Path) -> int:
        """Async version of flush_to_jsonl with proper event emission and validation."""
        games = self.get_unflushed_games()
        if not games:
            return 0

        path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        written = 0
        skipped = 0
        flushed_ids = []

        with open(path, "a", encoding="utf-8") as f:
            for game in games:
                # Validate game before flushing
                is_valid, _ = self._validate_game(game)
                if not is_valid and self.skip_invalid_games:
                    skipped += 1
                    self._validation_stats["games_skipped"] += 1
                    flushed_ids.append(game.game_id)
                    continue

                record = {
                    "game_id": game.game_id,
                    "board_type": game.board_type,
                    "num_players": game.num_players,
                    "moves": game.moves,
                    "outcome": game.outcome,
                    "timestamp": game.timestamp,
                    "source": game.source,
                    # Priority fields for experience replay
                    "avg_elo": game.avg_elo,
                    "priority": game.priority,
                    "from_promoted_model": game.from_promoted_model,
                }
                f.write(json.dumps(record) + "\n")
                written += 1
                flushed_ids.append(game.game_id)

        self.mark_flushed(flushed_ids)

        # Emit flush completed event asynchronously with validation stats
        duration = time.time() - start_time
        await self.emit_event_async(DataEventType.DATA_SYNC_COMPLETED, {
            "buffer_name": self.buffer_name,
            "games_flushed": written,
            "games_skipped": skipped,
            "target_path": str(path),
            "duration_seconds": duration,
            "source": "hot_buffer",
            "validation_stats": self._validation_stats.copy(),
        })
        if skipped > 0:
            logger.info(f"Flushed {written} games, skipped {skipped} invalid from '{self.buffer_name}' to {path}")
        else:
            logger.debug(f"Flushed {written} games from '{self.buffer_name}' to {path} in {duration:.2f}s")

        return written

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            if self._cache_dirty:
                self._rebuild_sample_cache()

            # Compute priority statistics
            priorities = [g.priority for g in self._buffer.values()] if self._buffer else [0.0]
            avg_elos = [g.avg_elo for g in self._buffer.values()] if self._buffer else [1500.0]
            promoted_count = sum(1 for g in self._buffer.values() if g.from_promoted_model)

            return {
                "game_count": len(self._buffer),
                "max_size": self.max_size,
                "total_samples": self._total_samples,
                "flushed_count": len(self._flushed_game_ids),
                "unflushed_count": len(self.get_unflushed_games()),
                "utilization": len(self._buffer) / self.max_size if self.max_size > 0 else 0,
                # Priority replay stats
                "avg_priority": float(np.mean(priorities)),
                "max_priority": float(np.max(priorities)),
                "min_priority": float(np.min(priorities)),
                "avg_elo": float(np.mean(avg_elos)),
                "promoted_model_games": promoted_count,
                # Validation stats
                "validation_enabled": self.enable_validation,
                **self._validation_stats,
            }

    # -------------------------------------------------------------------------
    # Priority Experience Replay Methods
    # -------------------------------------------------------------------------

    def compute_game_priority(
        self,
        game: GameRecord,
        base_elo: float = 1500.0,
        elo_scale: float = 400.0,
        recency_half_life_hours: float = 2.0,
        promotion_bonus: float = 1.5,
        quality_weight: float = 0.3,
    ) -> float:
        """Compute priority score for a game.

        Priority = elo_factor × recency_factor × quality_factor × promotion_bonus

        The quality_factor incorporates manifest quality scores from the data sync
        pipeline, prioritizing games that have been identified as high-quality
        based on Elo, game length, and decisiveness.

        Args:
            game: The game record
            base_elo: Reference Elo for normalization
            elo_scale: Elo difference for 2x priority
            recency_half_life_hours: Hours until recency factor halves
            promotion_bonus: Multiplier for games from promoted models
            quality_weight: Weight for manifest quality in final score [0, 1]

        Returns:
            Priority score (higher = more important)
        """
        # Elo factor: games from stronger players are more valuable
        # Formula: 2^((avg_elo - base_elo) / elo_scale)
        elo_factor = 2.0 ** ((game.avg_elo - base_elo) / elo_scale)
        elo_factor = max(0.25, min(4.0, elo_factor))  # Clamp to [0.25, 4.0]

        # Recency factor: recent games are more valuable
        # Exponential decay with half-life
        age_hours = (time.time() - game.timestamp) / 3600.0
        recency_factor = 0.5 ** (age_hours / recency_half_life_hours)
        recency_factor = max(0.1, recency_factor)  # Minimum 10% weight

        # Manifest quality factor: games with high quality scores are prioritized
        # Quality is in [0, 1], so we scale it to [0.5, 1.5] for multiplicative factor
        # If quality is 0.5 (default), this contributes 1.0 (neutral)
        # If quality is 1.0, this contributes up to 1.0 + quality_weight
        # If quality is 0.0, this contributes 1.0 - quality_weight
        quality_factor = 1.0 + quality_weight * (game.manifest_quality - 0.5) * 2

        # Promotion bonus: games that led to model promotion are gold
        promo_factor = promotion_bonus if game.from_promoted_model else 1.0

        # Combine factors
        priority = elo_factor * recency_factor * quality_factor * promo_factor

        return priority

    def update_all_priorities(
        self,
        priority_alpha: float = 0.6,
        **priority_kwargs,
    ) -> None:
        """Recompute priorities for all games in buffer.

        Args:
            priority_alpha: Exponent for priority (0=uniform, 1=full priority)
            **priority_kwargs: Additional args for compute_game_priority
        """
        with self._lock:
            for game in self._buffer.values():
                raw_priority = self.compute_game_priority(game, **priority_kwargs)
                # Apply priority exponent
                game.priority = raw_priority ** priority_alpha

    def update_game_priority(
        self,
        game_id: str,
        td_error: Optional[float] = None,
        from_promoted: Optional[bool] = None,
        avg_elo: Optional[float] = None,
    ) -> bool:
        """Update priority for a specific game.

        Can be called after training to update based on TD error,
        or after promotion to mark games from promoted models.

        Args:
            game_id: ID of game to update
            td_error: Temporal difference error from training (higher = more surprise)
            from_promoted: Mark game as from a promoted model
            avg_elo: Update the average Elo

        Returns:
            True if game was found and updated
        """
        with self._lock:
            game = self._buffer.get(game_id)
            if game is None:
                return False

            if td_error is not None:
                # TD error boost: prioritize surprising samples
                # td_factor = 1 + log(1 + |td_error|)
                td_factor = 1.0 + np.log1p(abs(td_error))
                game.priority *= td_factor

            if from_promoted is not None:
                game.from_promoted_model = from_promoted
                # Recompute priority with promotion bonus
                game.priority = self.compute_game_priority(game)

            if avg_elo is not None:
                game.avg_elo = avg_elo
                # Recompute priority with new Elo
                game.priority = self.compute_game_priority(game)

            return True

    def get_priority_training_batch(
        self,
        batch_size: int = 256,
        importance_beta: float = 0.4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get a batch of training samples with priority-weighted sampling.

        Uses proportional prioritization (Schaul et al., 2015) with
        importance sampling weights to correct for bias.

        Args:
            batch_size: Number of samples to return
            importance_beta: Importance sampling exponent (0=no correction, 1=full)

        Returns:
            Tuple of (states, policies, values, importance_weights) as numpy arrays
        """
        with self._lock:
            if self._cache_dirty:
                self._rebuild_sample_cache()

            if not self._sample_cache:
                return (
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                )

            # Build priority array aligned with sample cache
            # Each sample inherits priority from its parent game
            games_list = list(self._buffer.values())
            sample_priorities = []

            for game in games_list:
                n_samples = len(game.to_training_samples(encoder=self._encoder))
                sample_priorities.extend([game.priority] * n_samples)

            sample_priorities = np.array(sample_priorities, dtype=np.float64)

            # Ensure we have matching sizes
            if len(sample_priorities) != len(self._sample_cache):
                # Rebuild to ensure consistency
                self._rebuild_sample_cache()
                sample_priorities = []
                for game in self._buffer.values():
                    n_samples = len(game.to_training_samples(encoder=self._encoder))
                    sample_priorities.extend([game.priority] * n_samples)
                sample_priorities = np.array(sample_priorities, dtype=np.float64)

            # Compute sampling probabilities
            total_priority = sample_priorities.sum()
            if total_priority <= 0:
                # Fallback to uniform
                probs = np.ones(len(sample_priorities)) / len(sample_priorities)
            else:
                probs = sample_priorities / total_priority

            # Sample indices with priority weighting
            n_samples = min(batch_size, len(self._sample_cache))
            indices = np.random.choice(
                len(self._sample_cache),
                size=n_samples,
                replace=False,
                p=probs,
            )

            # Compute importance sampling weights
            # w_i = (N * P(i))^(-beta) / max_w
            N = len(self._sample_cache)
            weights = (N * probs[indices]) ** (-importance_beta)
            weights = weights / weights.max()  # Normalize to max=1

            # Gather samples
            states = []
            policies = []
            values = []

            for idx in indices:
                s, p, v = self._sample_cache[idx]
                states.append(s)
                policies.append(p)
                values.append(v)

            return (
                np.array(states, dtype=np.float32),
                np.array(policies, dtype=np.float32),
                np.array(values, dtype=np.float32),
                weights.astype(np.float32),
            )

    def get_priority_sample_iterator(
        self,
        batch_size: int = 256,
        epochs: int = 1,
        importance_beta: float = 0.4,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Iterator over priority-weighted training samples.

        Each epoch resamples with current priorities, allowing priorities
        to be updated between epochs.

        Yields batches of (states, policies, values, importance_weights).
        """
        for _ in range(epochs):
            # Get all samples for this epoch
            with self._lock:
                if self._cache_dirty:
                    self._rebuild_sample_cache()

                if not self._sample_cache:
                    return

                n_batches = (len(self._sample_cache) + batch_size - 1) // batch_size

            for _ in range(n_batches):
                yield self.get_priority_training_batch(
                    batch_size=batch_size,
                    importance_beta=importance_beta,
                )

    def mark_games_from_promoted_model(self, model_id: str, game_ids: List[str]) -> int:
        """Mark games as coming from a model that was promoted.

        This boosts their priority for future training.

        Args:
            model_id: ID of the promoted model
            game_ids: IDs of games played by this model

        Returns:
            Number of games marked
        """
        marked = 0
        with self._lock:
            for game_id in game_ids:
                game = self._buffer.get(game_id)
                if game:
                    game.from_promoted_model = True
                    game.priority = self.compute_game_priority(game)
                    marked += 1

        if marked > 0:
            logger.info(f"Marked {marked} games from promoted model {model_id}")

        return marked

    def clear(self) -> None:
        """Clear all games from buffer."""
        with self._lock:
            self._buffer.clear()
            self._sample_cache.clear()
            self._flushed_game_ids.clear()
            self._total_samples = 0
            self._cache_dirty = True

    # -------------------------------------------------------------------------
    # Database Loading with Quality Scores
    # -------------------------------------------------------------------------

    def load_from_db(
        self,
        db_path: Path,
        board_type: str = "square8",
        num_players: int = 2,
        min_quality: float = 0.0,
        limit: int = 1000,
        min_moves: int = 10,
    ) -> int:
        """Load games from SQLite database with quality scores.

        This method loads games directly from the database, including the
        quality_score column computed by game_quality_scorer.py during
        game finalization. Games are added to the buffer with their
        quality scores used for priority computation.

        Args:
            db_path: Path to the SQLite database
            board_type: Board type to filter (e.g., "square8")
            num_players: Number of players to filter
            min_quality: Minimum quality score to include (0.0-1.0)
            limit: Maximum number of games to load
            min_moves: Minimum game length

        Returns:
            Number of games loaded
        """
        import sqlite3

        if not db_path.exists():
            logger.warning(f"Database not found: {db_path}")
            return 0

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Check if quality_score column exists (v9+ schema)
            has_quality_column = True
            try:
                cursor.execute("SELECT quality_score FROM games LIMIT 1")
            except sqlite3.OperationalError:
                has_quality_column = False
                logger.debug(f"Database {db_path} missing quality_score column")

            # Build query with optional quality filtering
            if has_quality_column:
                query = """
                    SELECT game_id, board_type, num_players, winner, total_moves,
                           termination_reason, source, created_at,
                           COALESCE(quality_score, 0.5) as quality_score
                    FROM games
                    WHERE game_status = 'completed'
                      AND winner IS NOT NULL
                      AND board_type = ?
                      AND num_players = ?
                      AND total_moves >= ?
                      AND COALESCE(excluded_from_training, 0) = 0
                      AND COALESCE(quality_score, 0.5) >= ?
                    ORDER BY quality_score DESC, created_at DESC
                    LIMIT ?
                """
                cursor.execute(query, (board_type, num_players, min_moves, min_quality, limit))
            else:
                query = """
                    SELECT game_id, board_type, num_players, winner, total_moves,
                           termination_reason, source, created_at
                    FROM games
                    WHERE game_status = 'completed'
                      AND winner IS NOT NULL
                      AND board_type = ?
                      AND num_players = ?
                      AND total_moves >= ?
                      AND COALESCE(excluded_from_training, 0) = 0
                    ORDER BY created_at DESC
                    LIMIT ?
                """
                cursor.execute(query, (board_type, num_players, min_moves, limit))

            rows = cursor.fetchall()
            loaded = 0

            for row in rows:
                game_id = row['game_id']

                # Skip if already in buffer
                if game_id in self._buffer:
                    continue

                # Get quality score from DB or lookup table
                if has_quality_column:
                    quality_score = float(row['quality_score'])
                elif game_id in self._quality_lookup:
                    quality_score = self._quality_lookup[game_id]
                else:
                    quality_score = 0.5  # Default

                # Get Elo from lookup if available
                avg_elo = self._elo_lookup.get(game_id, 1500.0)

                # Load moves for this game
                moves_query = """
                    SELECT move_number, action_json, state_json
                    FROM moves
                    WHERE game_id = ?
                    ORDER BY move_number
                """
                cursor.execute(moves_query, (game_id,))
                move_rows = cursor.fetchall()

                if not move_rows:
                    continue

                moves = []
                for move_row in move_rows:
                    move_data = {
                        "move_number": move_row['move_number'],
                    }
                    if move_row['action_json']:
                        try:
                            import json
                            move_data["action"] = json.loads(move_row['action_json'])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if move_row['state_json']:
                        try:
                            import json
                            move_data["state"] = json.loads(move_row['state_json'])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    moves.append(move_data)

                # Build outcome dict
                winner = row['winner']
                outcome = {}
                for p in range(1, num_players + 1):
                    if winner == p:
                        outcome[str(p)] = 1.0
                    elif winner is None or winner == 0:
                        outcome[str(p)] = 0.5  # Draw
                    else:
                        outcome[str(p)] = 0.0

                # Create game record
                game = GameRecord(
                    game_id=game_id,
                    board_type=board_type,
                    num_players=num_players,
                    moves=moves,
                    outcome=outcome,
                    timestamp=float(row['created_at']) if row['created_at'] else time.time(),
                    source=row['source'] or "db_load",
                    avg_elo=avg_elo,
                    priority=1.0,  # Will be computed below
                    from_promoted_model=False,
                    manifest_quality=quality_score,
                )

                # Compute priority using quality
                game.priority = self.compute_game_priority(game)
                self.add_game(game)
                loaded += 1

            conn.close()
            logger.info(
                f"Loaded {loaded} games from {db_path.name} "
                f"(min_quality={min_quality:.2f}, has_quality_col={has_quality_column})"
            )
            return loaded

        except sqlite3.Error as e:
            logger.error(f"Failed to load games from {db_path}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error loading games from {db_path}: {e}")
            return 0

    # -------------------------------------------------------------------------
    # Quality Auto-Calibration (December 2025)
    # -------------------------------------------------------------------------

    def get_quality_distribution(self) -> Dict[str, Any]:
        """Get quality score distribution for games in buffer.

        Returns:
            Dict with quality distribution statistics
        """
        with self._lock:
            if not self._buffer:
                return {
                    "count": 0,
                    "avg_quality": 0.0,
                    "min_quality": 0.0,
                    "max_quality": 0.0,
                    "high_quality_count": 0,
                    "low_quality_count": 0,
                    "high_quality_ratio": 0.0,
                    "low_quality_ratio": 0.0,
                }

            qualities = [g.manifest_quality for g in self._buffer.values()]
            avg_quality = sum(qualities) / len(qualities)

            # Use thresholds from config if available
            high_threshold = 0.7
            low_threshold = 0.3
            if HAS_QUALITY_CONFIG:
                config = get_config()
                if config and hasattr(config, 'quality'):
                    high_threshold = getattr(config.quality, 'high_quality_threshold', 0.7)
                    low_threshold = getattr(config.quality, 'min_quality_for_training', 0.3)

            high_quality = [q for q in qualities if q >= high_threshold]
            low_quality = [q for q in qualities if q < low_threshold]

            return {
                "count": len(qualities),
                "avg_quality": avg_quality,
                "min_quality": min(qualities),
                "max_quality": max(qualities),
                "high_quality_count": len(high_quality),
                "low_quality_count": len(low_quality),
                "high_quality_ratio": len(high_quality) / len(qualities),
                "low_quality_ratio": len(low_quality) / len(qualities),
                "high_threshold": high_threshold,
                "low_threshold": low_threshold,
            }

    def calibrate_quality_thresholds(
        self,
        target_high_ratio: float = 0.3,
        target_low_ratio: float = 0.1,
    ) -> Dict[str, float]:
        """Auto-calibrate quality thresholds based on buffer distribution.

        Adjusts thresholds so that approximately:
        - target_high_ratio of games are classified as high-quality
        - target_low_ratio of games are classified as low-quality

        Args:
            target_high_ratio: Target ratio of high-quality games (0-1)
            target_low_ratio: Target ratio of low-quality games (0-1)

        Returns:
            Dict with calibrated thresholds
        """
        with self._lock:
            if len(self._buffer) < 100:
                # Not enough data for calibration
                return {
                    "high_threshold": 0.7,
                    "low_threshold": 0.3,
                    "calibrated": False,
                    "reason": "Insufficient data (< 100 games)",
                }

            qualities = sorted([g.manifest_quality for g in self._buffer.values()])
            n = len(qualities)

            # Compute percentiles for target ratios
            high_idx = int(n * (1.0 - target_high_ratio))
            low_idx = int(n * target_low_ratio)

            high_threshold = qualities[high_idx] if high_idx < n else 0.7
            low_threshold = qualities[low_idx] if low_idx >= 0 else 0.3

            # Sanity check: high must be > low
            if high_threshold <= low_threshold:
                high_threshold = max(low_threshold + 0.1, 0.5)

            calibration = {
                "high_threshold": high_threshold,
                "low_threshold": low_threshold,
                "calibrated": True,
                "sample_size": n,
                "avg_quality": sum(qualities) / n,
            }

            logger.info(
                f"Quality calibration: high={high_threshold:.2f}, "
                f"low={low_threshold:.2f} (n={n})"
            )

            return calibration

    def recompute_quality_with_scorer(
        self,
        elo_lookup: Optional[Dict[str, float]] = None,
    ) -> int:
        """Recompute quality scores using UnifiedQualityScorer.

        This ensures all games in the buffer use the same quality algorithm
        as the sync pipeline.

        Args:
            elo_lookup: Optional dict mapping player_id to Elo rating

        Returns:
            Number of games recomputed
        """
        if not HAS_QUALITY_SCORER:
            logger.debug("UnifiedQualityScorer not available, skipping recomputation")
            return 0

        scorer = get_quality_scorer()
        if scorer is None:
            return 0

        recomputed = 0
        with self._lock:
            for game in self._buffer.values():
                try:
                    # Build game_data dict for scorer
                    game_data = {
                        "game_id": game.game_id,
                        "game_length": len(game.moves),
                        "outcome": game.outcome,
                        "board_type": game.board_type,
                        "num_players": game.num_players,
                    }

                    # Determine if decisive
                    scores = list(game.outcome.values())
                    game_data["is_decisive"] = (1.0 in scores or 0.0 in scores)

                    quality = scorer.compute_game_quality(game_data, elo_lookup)
                    game.manifest_quality = quality.overall_score
                    game.priority = self.compute_game_priority(game)
                    recomputed += 1

                except Exception as e:
                    logger.debug(f"Failed to recompute quality for {game.game_id}: {e}")

        if recomputed > 0:
            logger.info(f"Recomputed quality for {recomputed} buffer games")
            self._cache_dirty = True

        return recomputed

    def auto_calibrate_and_filter(
        self,
        min_quality_percentile: float = 0.1,
        evict_below_percentile: bool = True,
    ) -> Dict[str, Any]:
        """Auto-calibrate quality and optionally evict low-quality games.

        This method:
        1. Computes quality distribution
        2. Calibrates thresholds based on distribution
        3. Optionally evicts games below the low-quality threshold

        Args:
            min_quality_percentile: Percentile below which games are low-quality
            evict_below_percentile: Whether to remove low-quality games

        Returns:
            Dict with calibration results and eviction stats
        """
        # Get distribution and calibrate
        dist = self.get_quality_distribution()
        calibration = self.calibrate_quality_thresholds(
            target_low_ratio=min_quality_percentile,
        )

        result = {
            "distribution": dist,
            "calibration": calibration,
            "evicted": 0,
        }

        if evict_below_percentile and calibration.get("calibrated", False):
            low_threshold = calibration["low_threshold"]
            evicted = 0

            with self._lock:
                games_to_remove = [
                    gid for gid, game in self._buffer.items()
                    if game.manifest_quality < low_threshold
                ]

                for gid in games_to_remove:
                    del self._buffer[gid]
                    evicted += 1

                if evicted > 0:
                    self._cache_dirty = True

            result["evicted"] = evicted
            if evicted > 0:
                logger.info(f"Evicted {evicted} low-quality games (threshold={low_threshold:.2f})")

        return result


def create_hot_buffer(
    *,
    max_size: int = 1000,
    max_memory_mb: int = 500,
    buffer_name: str = "default",
    enable_events: bool = True,
    training_threshold: int = 100,
    batch_notification_size: int = 50,
) -> HotDataBuffer:
    """Factory function to create a hot data buffer.

    Args:
        max_size: Maximum number of games to keep in buffer
        max_memory_mb: Soft memory limit in MB (triggers eviction)
        buffer_name: Identifier for this buffer (used in events)
        enable_events: Whether to emit events to the event bus
        training_threshold: Games needed before emitting TRAINING_THRESHOLD_REACHED
        batch_notification_size: Emit NEW_GAMES_AVAILABLE every N games
    """
    return HotDataBuffer(
        max_size=max_size,
        max_memory_mb=max_memory_mb,
        buffer_name=buffer_name,
        enable_events=enable_events,
        training_threshold=training_threshold,
        batch_notification_size=batch_notification_size,
    )
