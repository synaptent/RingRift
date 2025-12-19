"""
Streaming Training Data Pipeline for RingRift AI.

Replaces batch database reads with streaming data ingestion for
continuous learning. Enables real-time training on newly generated games.

Features:
- Real-time game data ingestion
- Incremental database polling
- Memory-efficient streaming batches
- Data deduplication
- Priority-based sampling
- Integration with PER (Prioritized Experience Replay)

Usage:
    from app.training.streaming_pipeline import StreamingDataPipeline

    pipeline = StreamingDataPipeline(db_path="data/games/selfplay.db")

    # Start streaming
    async for batch in pipeline.stream_batches(batch_size=256):
        train_on_batch(batch)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Set, Tuple

from app.utils.checksum_utils import compute_string_checksum

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming pipeline."""
    # Polling settings
    poll_interval_seconds: float = 5.0
    max_poll_batch: int = 1000

    # Buffer settings
    buffer_size: int = 10000
    min_buffer_fill: float = 0.2  # Start yielding when 20% full

    # Deduplication
    dedupe_enabled: bool = True
    dedupe_window: int = 50000  # Remember last N game_ids

    # Sampling
    priority_sampling: bool = True
    recency_weight: float = 0.3  # Weight for newer samples
    quality_weight: float = 0.4  # Weight for manifest quality score in priority

    # Data augmentation
    augmentation_enabled: bool = True

    # Quality-based sampling (populated from manifest via DataPipelineController)
    # Dict mapping game_id -> quality_score [0, 1]
    quality_lookup: Optional[Dict[str, float]] = None
    elo_lookup: Optional[Dict[str, float]] = None


@dataclass
class GameSample:
    """A single training sample from a game."""
    game_id: str
    move_idx: int
    board_type: str
    num_players: int
    state_hash: str
    timestamp: float
    value_target: float
    policy_target: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    priority: float = 1.0
    quality_score: float = 0.5  # Manifest quality score [0, 1]
    avg_elo: float = 1500.0  # Average player Elo from manifest


class CircularBuffer:
    """Thread-safe circular buffer for streaming samples."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.lock = threading.RLock()

    def append(self, item: Any):
        """Add item to buffer."""
        with self.lock:
            self.buffer.append(item)

    def extend(self, items: List[Any]):
        """Add multiple items to buffer."""
        with self.lock:
            self.buffer.extend(items)

    def sample(self, n: int, weights: Optional[np.ndarray] = None) -> List[Any]:
        """Sample n items from buffer."""
        with self.lock:
            if len(self.buffer) == 0:
                return []

            n = min(n, len(self.buffer))

            if weights is not None and len(weights) == len(self.buffer):
                # Weighted sampling
                indices = np.random.choice(
                    len(self.buffer),
                    size=n,
                    replace=False,
                    p=weights / weights.sum()
                )
            else:
                # Uniform sampling
                indices = np.random.choice(len(self.buffer), size=n, replace=False)

            return [self.buffer[i] for i in indices]

    def get_all(self) -> List[Any]:
        """Get all items in buffer."""
        with self.lock:
            return list(self.buffer)

    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)

    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()


class DatabasePoller:
    """Polls database for new games and samples."""

    def __init__(
        self,
        db_path: Path,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
    ):
        self.db_path = db_path
        self.board_type = board_type
        self.num_players = num_players
        self._last_poll_time: float = 0
        self._last_game_count: int = 0
        self._seen_game_ids: Set[str] = set()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def get_new_games(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get newly completed games since last poll."""
        if not self.db_path.exists():
            return []

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Build query
            query = """
                SELECT game_id, board_type, num_players, winner, move_history,
                       created_at, completed_at
                FROM games
                WHERE status = 'completed'
            """
            params = []

            if self.board_type:
                query += " AND board_type = ?"
                params.append(self.board_type)

            if self.num_players:
                query += " AND num_players = ?"
                params.append(self.num_players)

            query += " ORDER BY completed_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            # Filter to new games only
            new_games = []
            for row in rows:
                game_id = row["game_id"]
                if game_id not in self._seen_game_ids:
                    self._seen_game_ids.add(game_id)
                    new_games.append(dict(row))

            self._last_poll_time = time.time()

            return new_games

        except Exception as e:
            logger.error(f"Error polling database: {e}")
            return []

    def get_game_count(self) -> int:
        """Get total completed game count."""
        if not self.db_path.exists():
            return 0

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = "SELECT COUNT(*) FROM games WHERE status = 'completed'"
            params = []

            if self.board_type:
                query = query.replace("WHERE", "WHERE board_type = ? AND")
                params.append(self.board_type)

            if self.num_players:
                if "board_type" in query:
                    query += " AND num_players = ?"
                else:
                    query = query.replace("WHERE", "WHERE num_players = ? AND")
                params.append(self.num_players)

            cursor.execute(query, params)
            count = cursor.fetchone()[0]
            conn.close()

            return count

        except Exception as e:
            logger.error(f"Error getting game count: {e}")
            return 0

    def reset(self):
        """Reset poller state."""
        self._seen_game_ids.clear()
        self._last_poll_time = 0


def extract_samples_from_game(game: Dict[str, Any]) -> List[GameSample]:
    """Extract training samples from a game record."""
    samples = []

    try:
        move_history = json.loads(game.get("move_history") or "[]")
    except json.JSONDecodeError:
        return samples

    game_id = game["game_id"]
    board_type = game["board_type"]
    num_players = game["num_players"]
    winner = game.get("winner")

    # Parse timestamp
    completed_at = game.get("completed_at", "")
    try:
        if completed_at:
            dt = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            timestamp = dt.timestamp()
        else:
            timestamp = time.time()
    except Exception:
        timestamp = time.time()

    # Create sample for each move
    for move_idx, move in enumerate(move_history):
        player = move_idx % num_players

        # Compute value target
        if winner is None:
            value_target = 0.5  # Draw
        elif winner == player:
            value_target = 1.0  # Win
        else:
            value_target = 0.0  # Loss

        # Create state hash for deduplication
        state_hash = compute_string_checksum(
            f"{game_id}:{move_idx}",
            algorithm="md5",
            truncate=16,
        )

        sample = GameSample(
            game_id=game_id,
            move_idx=move_idx,
            board_type=board_type,
            num_players=num_players,
            state_hash=state_hash,
            timestamp=timestamp,
            value_target=value_target,
            priority=1.0,
        )

        samples.append(sample)

    return samples


class StreamingDataPipeline:
    """Main streaming data pipeline for continuous training.

    Features async DB polling with dual buffers:
    - ThreadPoolExecutor runs DB queries without blocking async event loop
    - Dual buffer prefetching: next query starts before current results processed
    - Near-zero GPU idle time during database operations
    """

    def __init__(
        self,
        db_path: Path,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
        config: Optional[StreamingConfig] = None,
        db_pool_size: int = 2,
    ):
        """
        Initialize streaming pipeline.

        Args:
            db_path: Path to game database
            board_type: Optional filter by board type
            num_players: Optional filter by player count
            config: Pipeline configuration
            db_pool_size: Number of threads for async DB operations
        """
        self.db_path = Path(db_path)
        self.board_type = board_type
        self.num_players = num_players
        self.config = config or StreamingConfig()

        # Initialize components
        self.buffer = CircularBuffer(self.config.buffer_size)
        self.poller = DatabasePoller(db_path, board_type, num_players)

        # Async DB polling with ThreadPoolExecutor
        self._db_executor = ThreadPoolExecutor(max_workers=db_pool_size, thread_name_prefix="db_poll")
        self._prefetch_task: Optional[asyncio.Task] = None
        self._prefetch_buffer: List[Dict[str, Any]] = []

        # Tracking
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        # Use OrderedDict for O(1) FIFO eviction (faster than set + list conversion)
        self._seen_hashes: OrderedDict[str, float] = OrderedDict()
        self._total_samples_ingested: int = 0
        self._total_batches_yielded: int = 0
        self._db_queries_async: int = 0

    async def start(self):
        """Start the streaming pipeline."""
        if self._running:
            return

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info(f"Started streaming pipeline for {self.db_path}")

    async def stop(self):
        """Stop the streaming pipeline."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        # Cancel any pending prefetch
        if self._prefetch_task and not self._prefetch_task.done():
            self._prefetch_task.cancel()
            try:
                await self._prefetch_task
            except asyncio.CancelledError:
                pass
        # Shutdown executor
        self._db_executor.shutdown(wait=False)
        logger.info("Stopped streaming pipeline")

    def set_quality_lookup(
        self,
        quality_lookup: Optional[Dict[str, float]] = None,
        elo_lookup: Optional[Dict[str, float]] = None,
    ) -> None:
        """Set quality lookup tables for priority sampling.

        Call this with data from DataPipelineController to enable
        quality-weighted sampling of training data.

        Args:
            quality_lookup: Dict mapping game_id to quality_score [0, 1]
            elo_lookup: Dict mapping game_id to avg_player_elo
        """
        self.config.quality_lookup = quality_lookup
        self.config.elo_lookup = elo_lookup

        if quality_lookup:
            logger.info(f"Set quality lookup with {len(quality_lookup)} games")

        # Re-score existing buffer samples with new quality data
        if quality_lookup or elo_lookup:
            with self.buffer.lock:
                for sample in self.buffer.buffer:
                    if quality_lookup and sample.game_id in quality_lookup:
                        sample.quality_score = quality_lookup[sample.game_id]
                        sample.priority = sample.quality_score
                    if elo_lookup and sample.game_id in elo_lookup:
                        sample.avg_elo = elo_lookup[sample.game_id]

    async def _async_get_new_games(self, limit: int) -> List[Dict[str, Any]]:
        """Run DB query in thread pool (non-blocking).

        This allows the async event loop to continue while waiting for
        the synchronous sqlite3 query to complete.
        """
        loop = asyncio.get_event_loop()
        games = await loop.run_in_executor(
            self._db_executor,
            self.poller.get_new_games,
            limit
        )
        self._db_queries_async += 1
        return games

    async def _poll_loop(self):
        """Background loop to poll database for new games.

        Uses dual-buffer prefetching:
        1. Start prefetch of next batch before processing current batch
        2. While processing current data, DB query runs in background thread
        3. Near-zero blocking on DB operations
        """
        # Start initial prefetch
        self._prefetch_task = asyncio.create_task(
            self._async_get_new_games(self.config.max_poll_batch)
        )

        while self._running:
            try:
                # Wait for prefetched data
                if self._prefetch_task:
                    new_games = await self._prefetch_task
                else:
                    new_games = await self._async_get_new_games(self.config.max_poll_batch)

                # Immediately start next prefetch (dual buffer)
                self._prefetch_task = asyncio.create_task(
                    self._async_get_new_games(self.config.max_poll_batch)
                )

                # Process current batch while next query runs in background
                if new_games:
                    # Extract samples
                    new_samples = []
                    for game in new_games:
                        samples = extract_samples_from_game(game)
                        new_samples.extend(samples)

                    # Enrich samples with quality data from manifest
                    if self.config.quality_lookup or self.config.elo_lookup:
                        for sample in new_samples:
                            if self.config.quality_lookup and sample.game_id in self.config.quality_lookup:
                                sample.quality_score = self.config.quality_lookup[sample.game_id]
                                # Set priority based on quality for weighted sampling
                                sample.priority = sample.quality_score
                            if self.config.elo_lookup and sample.game_id in self.config.elo_lookup:
                                sample.avg_elo = self.config.elo_lookup[sample.game_id]

                    # Deduplicate using OrderedDict for O(1) FIFO eviction
                    if self.config.dedupe_enabled:
                        unique_samples = []
                        current_time = time.time()
                        for sample in new_samples:
                            if sample.state_hash not in self._seen_hashes:
                                self._seen_hashes[sample.state_hash] = current_time
                                unique_samples.append(sample)

                                # Maintain window size with O(1) eviction
                                while len(self._seen_hashes) > self.config.dedupe_window:
                                    self._seen_hashes.popitem(last=False)

                        new_samples = unique_samples

                    # Add to buffer
                    if new_samples:
                        self.buffer.extend(new_samples)
                        self._total_samples_ingested += len(new_samples)
                        logger.debug(f"Ingested {len(new_samples)} samples, buffer size: {len(self.buffer)}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in poll loop: {e}")

            # Wait before next poll (prefetch continues in background)
            await asyncio.sleep(self.config.poll_interval_seconds)

    def _compute_sample_weights(self, samples: List[GameSample]) -> np.ndarray:
        """Compute sampling weights based on recency, quality, and priority.

        Weight formula:
        - recency_weight: Weight for time-based recency factor
        - quality_weight: Weight for manifest quality score
        - remainder: Weight for base priority

        High-quality games from high-Elo players get sampled more frequently.
        """
        if not samples:
            return np.array([])

        weights = np.ones(len(samples))
        current_time = time.time()

        # Normalize weights to sum to 1.0
        recency_w = self.config.recency_weight
        quality_w = self.config.quality_weight
        priority_w = max(0.0, 1.0 - recency_w - quality_w)

        for i, sample in enumerate(samples):
            # Recency factor (newer samples weighted higher)
            age_hours = (current_time - sample.timestamp) / 3600
            recency_factor = np.exp(-age_hours / 24)  # Half-life of 24 hours

            # Quality factor from manifest (0-1)
            quality_factor = sample.quality_score

            # Priority factor (base priority, could be recency-adjusted externally)
            priority_factor = sample.priority

            # Combined weight
            weights[i] = (
                recency_w * recency_factor
                + quality_w * quality_factor
                + priority_w * priority_factor
            )

        # Ensure minimum weight to prevent starvation
        weights = np.maximum(weights, 0.01)

        return weights

    def get_batch(self, batch_size: int = 256) -> List[GameSample]:
        """Get a batch of samples from the buffer.

        Args:
            batch_size: Number of samples to return

        Returns:
            List of GameSample objects
        """
        all_samples = self.buffer.get_all()

        if len(all_samples) < batch_size:
            return all_samples

        if self.config.priority_sampling:
            weights = self._compute_sample_weights(all_samples)
            return self.buffer.sample(batch_size, weights)
        else:
            return self.buffer.sample(batch_size)

    async def stream_batches(
        self,
        batch_size: int = 256,
        max_batches: Optional[int] = None,
    ) -> AsyncIterator[List[GameSample]]:
        """Stream training batches continuously.

        Args:
            batch_size: Number of samples per batch
            max_batches: Maximum batches to yield (None = infinite)

        Yields:
            Batches of GameSample objects
        """
        # Start pipeline if not running
        if not self._running:
            await self.start()

        # Wait for buffer to fill
        min_fill = int(self.config.buffer_size * self.config.min_buffer_fill)
        while len(self.buffer) < min_fill and self._running:
            logger.info(f"Waiting for buffer to fill: {len(self.buffer)}/{min_fill}")
            await asyncio.sleep(1.0)

        batches_yielded = 0

        while self._running:
            if max_batches and batches_yielded >= max_batches:
                break

            batch = self.get_batch(batch_size)
            if batch:
                yield batch
                batches_yielded += 1
                self._total_batches_yielded += 1

            # Brief pause to allow buffer refill
            await asyncio.sleep(0.01)

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.config.buffer_size,
            "total_samples_ingested": self._total_samples_ingested,
            "total_batches_yielded": self._total_batches_yielded,
            "unique_hashes_tracked": len(self._seen_hashes),
            "running": self._running,
            "db_queries_async": self._db_queries_async,
            "prefetch_active": self._prefetch_task is not None and not self._prefetch_task.done(),
        }

    def update_priorities(self, updates: Dict[str, float]):
        """Update sample priorities (for integration with PER).

        Args:
            updates: Dict mapping state_hash to new priority
        """
        samples = self.buffer.get_all()
        for sample in samples:
            if sample.state_hash in updates:
                sample.priority = updates[sample.state_hash]


class MultiDBStreamingPipeline:
    """Streaming pipeline that aggregates from multiple databases."""

    def __init__(
        self,
        db_paths: List[Path],
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
        config: Optional[StreamingConfig] = None,
    ):
        """
        Initialize multi-database pipeline.

        Args:
            db_paths: List of database paths to stream from
            board_type: Optional filter by board type
            num_players: Optional filter by player count
            config: Pipeline configuration
        """
        self.pipelines = [
            StreamingDataPipeline(db_path, board_type, num_players, config)
            for db_path in db_paths
        ]
        self.config = config or StreamingConfig()

    async def start(self):
        """Start all pipelines."""
        await asyncio.gather(*[p.start() for p in self.pipelines])

    async def stop(self):
        """Stop all pipelines."""
        await asyncio.gather(*[p.stop() for p in self.pipelines])

    async def stream_batches(
        self,
        batch_size: int = 256,
        max_batches: Optional[int] = None,
    ) -> AsyncIterator[List[GameSample]]:
        """Stream batches from all databases.

        Samples are drawn proportionally from each database's buffer.
        """
        await self.start()

        batches_yielded = 0

        while True:
            if max_batches and batches_yielded >= max_batches:
                break

            # Collect samples from all pipelines
            all_samples = []
            for pipeline in self.pipelines:
                samples = pipeline.buffer.get_all()
                all_samples.extend(samples)

            if len(all_samples) >= batch_size:
                # Sample batch
                indices = np.random.choice(len(all_samples), size=batch_size, replace=False)
                batch = [all_samples[i] for i in indices]
                yield batch
                batches_yielded += 1

            await asyncio.sleep(0.01)

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics across all pipelines."""
        stats = {
            "num_databases": len(self.pipelines),
            "total_buffer_size": sum(len(p.buffer) for p in self.pipelines),
            "total_samples_ingested": sum(p._total_samples_ingested for p in self.pipelines),
            "total_batches_yielded": sum(p._total_batches_yielded for p in self.pipelines),
        }
        return stats
