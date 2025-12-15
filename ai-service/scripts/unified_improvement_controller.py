#!/usr/bin/env python3
"""
DEPRECATED: This module is deprecated. Use scripts/unified_ai_loop.py instead.

The unified_ai_loop.py provides all functionality with proper cluster coordination,
event-driven data flow, and feedback loop integration.

To migrate:
    python scripts/unified_ai_loop.py --foreground --verbose

---
Unified AI Improvement Controller - Single entry point for all improvement operations.

This controller integrates all improvement components into a cohesive, pipelined system:
1. Auto-detects cluster mode (local vs distributed)
2. Coordinates selfplay, training, and evaluation in parallel pipeline
3. Maintains unified data manifest for deduplication
4. Tracks model lineage with training source metadata
5. Provides single Elo database across all tournaments
6. Handles automatic failover and recovery

Key Features:
- Pipeline parallelization: Selfplay(i+1) runs while Training(i) executes
- Hot data path: Incremental training on recent games without full reanalysis
- Unified Elo: Single source of truth for all model ratings
- Data manifest: SHA256 fingerprinting prevents duplicate ingestion
- Model lineage: Every checkpoint tracks its training data sources
- Auto-scaling: Dynamically adjusts workers based on resource availability

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │            UnifiedImprovementController                      │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
    │  │ Pipeline │  │  Data    │  │  Model   │  │  Unified │   │
    │  │ Manager  │  │ Manifest │  │ Lineage  │  │   Elo    │   │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
    │       │              │              │              │         │
    │  ┌────▼──────────────▼──────────────▼──────────────▼────┐   │
    │  │                    Stage Coordinator                  │   │
    │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │   │
    │  │  │Selfplay │→→│ Ingest  │→→│  Train  │→→│Evaluate │ │   │
    │  │  │ Stage   │  │  Stage  │  │  Stage  │  │  Stage  │ │   │
    │  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

Usage:
    # Auto-detect mode and run
    python scripts/unified_improvement_controller.py --auto

    # Run in local mode with specific config
    python scripts/unified_improvement_controller.py --mode local --board square8 --players 2

    # Run in cluster mode
    python scripts/unified_improvement_controller.py --mode cluster --config cluster.yaml

    # Pipeline mode (overlapped stages)
    python scripts/unified_improvement_controller.py --pipeline --iterations 100

    # Hot data path (fast iteration)
    python scripts/unified_improvement_controller.py --hot-path --games 500
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Setup path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import BoardType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("UnifiedController")

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Data Manifest System
# =============================================================================

@dataclass
class DataSource:
    """Represents a training data source with deduplication tracking."""
    source_path: str
    fingerprint: str  # SHA256 of first 1MB + file size + mtime
    games_count: int
    board_type: str
    num_players: int
    status: str  # pending, processing, complete, quarantined
    created_at: str
    processed_at: Optional[str] = None
    error_message: Optional[str] = None


class DataManifest:
    """Centralized tracking of all training data sources for deduplication."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or AI_SERVICE_ROOT / "data" / "manifest.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the manifest database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE,
                    games_count INTEGER DEFAULT 0,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    processed_at TEXT,
                    error_message TEXT,
                    UNIQUE(source_path, board_type, num_players)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fingerprint ON data_sources(fingerprint)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON data_sources(status)
            """)
            conn.commit()

    @staticmethod
    def compute_fingerprint(path: Path) -> str:
        """Compute a fingerprint for deduplication."""
        if not path.exists():
            return ""

        stat = path.stat()
        hasher = hashlib.sha256()
        hasher.update(f"{stat.st_size}:{stat.st_mtime}".encode())

        # Hash first 1MB of content
        with open(path, 'rb') as f:
            hasher.update(f.read(1024 * 1024))

        return hasher.hexdigest()[:32]

    def is_duplicate(self, fingerprint: str) -> bool:
        """Check if a data source has already been processed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM data_sources WHERE fingerprint = ? AND status = 'complete'",
                (fingerprint,)
            )
            return cursor.fetchone() is not None

    def register_source(
        self,
        source_path: Path,
        board_type: str,
        num_players: int,
        games_count: int = 0
    ) -> Tuple[bool, str]:
        """Register a new data source. Returns (is_new, fingerprint)."""
        fingerprint = self.compute_fingerprint(source_path)

        if self.is_duplicate(fingerprint):
            logger.info(f"Skipping duplicate source: {source_path} (fingerprint: {fingerprint[:8]})")
            return False, fingerprint

        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO data_sources
                    (source_path, fingerprint, games_count, board_type, num_players, status, created_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?)
                """, (
                    str(source_path),
                    fingerprint,
                    games_count,
                    board_type,
                    num_players,
                    datetime.now(timezone.utc).isoformat()
                ))
                conn.commit()
                logger.info(f"Registered new source: {source_path} ({games_count} games)")
                return True, fingerprint
            except sqlite3.IntegrityError:
                return False, fingerprint

    def mark_processing(self, fingerprint: str):
        """Mark a source as currently being processed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE data_sources SET status = 'processing' WHERE fingerprint = ?",
                (fingerprint,)
            )
            conn.commit()

    def mark_complete(self, fingerprint: str, games_count: int = None):
        """Mark a source as successfully processed."""
        with sqlite3.connect(self.db_path) as conn:
            if games_count is not None:
                conn.execute("""
                    UPDATE data_sources
                    SET status = 'complete', processed_at = ?, games_count = ?
                    WHERE fingerprint = ?
                """, (datetime.now(timezone.utc).isoformat(), games_count, fingerprint))
            else:
                conn.execute("""
                    UPDATE data_sources SET status = 'complete', processed_at = ?
                    WHERE fingerprint = ?
                """, (datetime.now(timezone.utc).isoformat(), fingerprint))
            conn.commit()

    def mark_quarantined(self, fingerprint: str, error: str):
        """Mark a source as quarantined due to errors."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE data_sources
                SET status = 'quarantined', error_message = ?
                WHERE fingerprint = ?
            """, (error, fingerprint))
            conn.commit()

    def get_pending_sources(self, board_type: str = None, num_players: int = None) -> List[DataSource]:
        """Get all pending data sources, optionally filtered."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM data_sources WHERE status = 'pending'"
            params = []

            if board_type:
                query += " AND board_type = ?"
                params.append(board_type)
            if num_players:
                query += " AND num_players = ?"
                params.append(num_players)

            cursor = conn.execute(query, params)
            return [DataSource(**dict(row)) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Get manifest statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    status,
                    COUNT(*) as count,
                    SUM(games_count) as total_games
                FROM data_sources
                GROUP BY status
            """)
            stats = {row[0]: {"count": row[1], "games": row[2] or 0} for row in cursor.fetchall()}
            return stats


# =============================================================================
# Model Lineage System
# =============================================================================

@dataclass
class ModelLineage:
    """Tracks the training lineage of a model."""
    model_id: str
    model_path: str
    board_type: str
    num_players: int
    parent_model_id: Optional[str]
    training_data_fingerprints: List[str]
    training_games_count: int
    training_started_at: str
    training_completed_at: Optional[str]
    epochs: int
    final_loss: Optional[float]
    elo_rating: Optional[float]
    promotion_status: str  # candidate, promoted, rejected
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelLineageTracker:
    """Tracks model lineage and training history."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or AI_SERVICE_ROOT / "data" / "model_lineage.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the lineage database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_path TEXT NOT NULL,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    parent_model_id TEXT,
                    training_data_fingerprints TEXT,
                    training_games_count INTEGER DEFAULT 0,
                    training_started_at TEXT,
                    training_completed_at TEXT,
                    epochs INTEGER DEFAULT 0,
                    final_loss REAL,
                    elo_rating REAL,
                    promotion_status TEXT DEFAULT 'candidate',
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_model_id) REFERENCES models(model_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_board_players
                ON models(board_type, num_players)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_promotion
                ON models(promotion_status)
            """)
            conn.commit()

    def register_model(
        self,
        model_id: str,
        model_path: Path,
        board_type: str,
        num_players: int,
        parent_model_id: Optional[str] = None,
        training_fingerprints: List[str] = None,
        training_games: int = 0,
        epochs: int = 0
    ) -> ModelLineage:
        """Register a new model with its lineage."""
        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO models
                (model_id, model_path, board_type, num_players, parent_model_id,
                 training_data_fingerprints, training_games_count, training_started_at, epochs)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id,
                str(model_path),
                board_type,
                num_players,
                parent_model_id,
                json.dumps(training_fingerprints or []),
                training_games,
                now,
                epochs
            ))
            conn.commit()

        return ModelLineage(
            model_id=model_id,
            model_path=str(model_path),
            board_type=board_type,
            num_players=num_players,
            parent_model_id=parent_model_id,
            training_data_fingerprints=training_fingerprints or [],
            training_games_count=training_games,
            training_started_at=now,
            training_completed_at=None,
            epochs=epochs,
            final_loss=None,
            elo_rating=None,
            promotion_status="candidate"
        )

    def complete_training(self, model_id: str, final_loss: float):
        """Mark model training as complete."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE models
                SET training_completed_at = ?, final_loss = ?
                WHERE model_id = ?
            """, (datetime.now(timezone.utc).isoformat(), final_loss, model_id))
            conn.commit()

    def update_elo(self, model_id: str, elo_rating: float):
        """Update model's Elo rating."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE models SET elo_rating = ? WHERE model_id = ?",
                (elo_rating, model_id)
            )
            conn.commit()

    def promote_model(self, model_id: str):
        """Mark model as promoted (became the new best)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE models SET promotion_status = 'promoted' WHERE model_id = ?",
                (model_id,)
            )
            conn.commit()

    def reject_model(self, model_id: str):
        """Mark model as rejected (failed promotion gate)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE models SET promotion_status = 'rejected' WHERE model_id = ?",
                (model_id,)
            )
            conn.commit()

    def get_current_best(self, board_type: str, num_players: int) -> Optional[ModelLineage]:
        """Get the current best (promoted) model."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM models
                WHERE board_type = ? AND num_players = ? AND promotion_status = 'promoted'
                ORDER BY created_at DESC
                LIMIT 1
            """, (board_type, num_players))
            row = cursor.fetchone()
            if row:
                return self._row_to_lineage(row)
            return None

    def _row_to_lineage(self, row: sqlite3.Row) -> ModelLineage:
        """Convert database row to ModelLineage object."""
        return ModelLineage(
            model_id=row['model_id'],
            model_path=row['model_path'],
            board_type=row['board_type'],
            num_players=row['num_players'],
            parent_model_id=row['parent_model_id'],
            training_data_fingerprints=json.loads(row['training_data_fingerprints'] or '[]'),
            training_games_count=row['training_games_count'],
            training_started_at=row['training_started_at'],
            training_completed_at=row['training_completed_at'],
            epochs=row['epochs'],
            final_loss=row['final_loss'],
            elo_rating=row['elo_rating'],
            promotion_status=row['promotion_status'],
            metadata=json.loads(row['metadata'] or '{}')
        )


# =============================================================================
# Unified Elo System
# =============================================================================

class UnifiedEloSystem:
    """Single source of truth for all model Elo ratings."""

    K_FACTOR = 32.0  # Standard K-factor for updates
    INITIAL_ELO = 1500.0

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or AI_SERVICE_ROOT / "data" / "unified_elo.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the unified Elo database."""
        with sqlite3.connect(self.db_path) as conn:
            # Participants table (models and AI types)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS participants (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    ai_type TEXT NOT NULL,
                    difficulty INTEGER,
                    use_neural_net INTEGER DEFAULT 0,
                    model_id TEXT,
                    created_at REAL
                )
            """)

            # Elo ratings per board/player configuration
            conn.execute("""
                CREATE TABLE IF NOT EXISTS elo_ratings (
                    participant_id TEXT NOT NULL,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    rating REAL DEFAULT 1500.0,
                    games_played INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    draws INTEGER DEFAULT 0,
                    last_update REAL,
                    PRIMARY KEY (participant_id, board_type, num_players),
                    FOREIGN KEY (participant_id) REFERENCES participants(id)
                )
            """)

            # Match history for tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS match_history (
                    id TEXT PRIMARY KEY,
                    participant_ids TEXT NOT NULL,
                    winner_id TEXT,
                    game_length INTEGER,
                    duration_sec REAL,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    timestamp TEXT,
                    worker TEXT,
                    tournament_id TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_elo_board
                ON elo_ratings(board_type, num_players)
            """)
            conn.commit()

    def register_participant(
        self,
        participant_id: str,
        name: str,
        ai_type: str,
        difficulty: int = None,
        use_neural_net: bool = False,
        model_id: str = None
    ):
        """Register a new participant (model or AI type)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO participants
                (id, name, ai_type, difficulty, use_neural_net, model_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                participant_id,
                name,
                ai_type,
                difficulty,
                int(use_neural_net),
                model_id,
                time.time()
            ))
            conn.commit()

    def get_or_create_rating(
        self,
        participant_id: str,
        board_type: str,
        num_players: int
    ) -> float:
        """Get participant's Elo rating, creating initial if needed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT rating FROM elo_ratings
                WHERE participant_id = ? AND board_type = ? AND num_players = ?
            """, (participant_id, board_type, num_players))
            row = cursor.fetchone()

            if row:
                return row[0]

            # Create initial rating
            conn.execute("""
                INSERT INTO elo_ratings
                (participant_id, board_type, num_players, rating, last_update)
                VALUES (?, ?, ?, ?, ?)
            """, (participant_id, board_type, num_players, self.INITIAL_ELO, time.time()))
            conn.commit()
            return self.INITIAL_ELO

    def update_after_match(
        self,
        winner_id: str,
        loser_id: str,
        board_type: str,
        num_players: int,
        is_draw: bool = False
    ) -> Tuple[float, float]:
        """Update Elo ratings after a match. Returns (winner_new_elo, loser_new_elo)."""
        winner_elo = self.get_or_create_rating(winner_id, board_type, num_players)
        loser_elo = self.get_or_create_rating(loser_id, board_type, num_players)

        # Calculate expected scores
        expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
        expected_loser = 1 - expected_winner

        # Actual scores
        if is_draw:
            score_winner = 0.5
            score_loser = 0.5
        else:
            score_winner = 1.0
            score_loser = 0.0

        # New ratings
        new_winner_elo = winner_elo + self.K_FACTOR * (score_winner - expected_winner)
        new_loser_elo = loser_elo + self.K_FACTOR * (score_loser - expected_loser)

        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            # Update winner
            conn.execute("""
                UPDATE elo_ratings
                SET rating = ?, games_played = games_played + 1,
                    wins = wins + ?, draws = draws + ?, last_update = ?
                WHERE participant_id = ? AND board_type = ? AND num_players = ?
            """, (
                new_winner_elo,
                0 if is_draw else 1,
                1 if is_draw else 0,
                now,
                winner_id, board_type, num_players
            ))

            # Update loser
            conn.execute("""
                UPDATE elo_ratings
                SET rating = ?, games_played = games_played + 1,
                    losses = losses + ?, draws = draws + ?, last_update = ?
                WHERE participant_id = ? AND board_type = ? AND num_players = ?
            """, (
                new_loser_elo,
                0 if is_draw else 1,
                1 if is_draw else 0,
                now,
                loser_id, board_type, num_players
            ))
            conn.commit()

        return new_winner_elo, new_loser_elo

    def get_leaderboard(
        self,
        board_type: str = None,
        num_players: int = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get Elo leaderboard, optionally filtered."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = """
                SELECT e.*, p.name, p.ai_type, p.difficulty, p.use_neural_net, p.model_id
                FROM elo_ratings e
                JOIN participants p ON e.participant_id = p.id
                WHERE 1=1
            """
            params = []

            if board_type:
                query += " AND e.board_type = ?"
                params.append(board_type)
            if num_players:
                query += " AND e.num_players = ?"
                params.append(num_players)

            query += " ORDER BY e.rating DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


# =============================================================================
# Pipeline Stage System
# =============================================================================

class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStage:
    """Represents a stage in the improvement pipeline."""
    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None

    @property
    def duration_sec(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class PipelineManager:
    """Manages parallel pipeline execution for overlapped improvement stages."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.stages: Dict[str, Dict[int, PipelineStage]] = {}  # stage_name -> {iteration -> stage}
        self.futures: Dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit_stage(
        self,
        stage_name: str,
        iteration: int,
        func: Callable,
        *args,
        **kwargs
    ) -> Future:
        """Submit a stage for execution."""
        stage = PipelineStage(name=stage_name)

        with self._lock:
            if stage_name not in self.stages:
                self.stages[stage_name] = {}
            self.stages[stage_name][iteration] = stage

        def wrapped():
            stage.status = StageStatus.RUNNING
            stage.started_at = time.time()
            try:
                result = func(*args, **kwargs)
                stage.status = StageStatus.COMPLETE
                stage.result = result
                return result
            except Exception as e:
                stage.status = StageStatus.FAILED
                stage.error = str(e)
                raise
            finally:
                stage.completed_at = time.time()

        future = self.executor.submit(wrapped)
        key = f"{stage_name}_{iteration}"
        self.futures[key] = future
        return future

    def wait_for_stage(self, stage_name: str, iteration: int, timeout: float = None) -> Any:
        """Wait for a specific stage to complete."""
        key = f"{stage_name}_{iteration}"
        if key in self.futures:
            return self.futures[key].result(timeout=timeout)
        return None

    def get_stage_status(self, stage_name: str, iteration: int) -> Optional[PipelineStage]:
        """Get the status of a specific stage."""
        return self.stages.get(stage_name, {}).get(iteration)

    def is_stage_complete(self, stage_name: str, iteration: int) -> bool:
        """Check if a stage has completed."""
        stage = self.get_stage_status(stage_name, iteration)
        return stage and stage.status == StageStatus.COMPLETE

    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        self.executor.shutdown(wait=wait)


# =============================================================================
# Unified Improvement Controller
# =============================================================================

class ControllerMode(Enum):
    LOCAL = "local"
    CLUSTER = "cluster"
    AUTO = "auto"


@dataclass
class ControllerConfig:
    """Configuration for the unified controller."""
    mode: ControllerMode = ControllerMode.AUTO
    board_types: List[str] = field(default_factory=lambda: ["square8", "square19", "hexagonal"])
    player_counts: List[int] = field(default_factory=lambda: [2, 3, 4])

    # Pipeline settings
    enable_pipeline: bool = True
    max_parallel_stages: int = 3

    # Selfplay settings
    games_per_iteration: int = 500
    selfplay_workers: int = 8

    # Training settings
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.001
    use_hot_path: bool = False

    # Evaluation settings
    eval_games: int = 100
    promotion_threshold: float = 0.55
    confidence_level: float = 0.95

    # Auto-scaling
    target_gpu_utilization: float = 0.7
    target_cpu_utilization: float = 0.6

    # Paths
    data_dir: Path = field(default_factory=lambda: AI_SERVICE_ROOT / "data")
    models_dir: Path = field(default_factory=lambda: AI_SERVICE_ROOT / "models")
    logs_dir: Path = field(default_factory=lambda: AI_SERVICE_ROOT / "logs" / "unified_controller")


class UnifiedImprovementController:
    """Main controller that coordinates all improvement components."""

    def __init__(self, config: ControllerConfig = None):
        self.config = config or ControllerConfig()
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.manifest = DataManifest()
        self.lineage = ModelLineageTracker()
        self.elo = UnifiedEloSystem()
        self.pipeline = PipelineManager(max_workers=self.config.max_parallel_stages)

        # State
        self.running = False
        self.current_iteration = 0
        self._shutdown_event = threading.Event()

        logger.info(f"UnifiedImprovementController initialized with config: {asdict(self.config)}")

    def detect_mode(self) -> ControllerMode:
        """Auto-detect whether to run in local or cluster mode."""
        # Check for P2P orchestrator
        p2p_url = os.environ.get("P2P_ORCHESTRATOR_URL")
        if p2p_url:
            try:
                import urllib.request
                req = urllib.request.Request(f"{p2p_url}/health", method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        logger.info(f"Detected P2P orchestrator at {p2p_url}")
                        return ControllerMode.CLUSTER
            except Exception:
                pass

        # Check for multiple GPU nodes
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_count = len(result.stdout.strip().split('\n'))
                if gpu_count > 1:
                    logger.info(f"Detected {gpu_count} GPUs, using cluster mode")
                    return ControllerMode.CLUSTER
        except Exception:
            pass

        logger.info("Using local mode")
        return ControllerMode.LOCAL

    def run_selfplay_stage(
        self,
        board_type: str,
        num_players: int,
        games: int,
        output_dir: Path
    ) -> Tuple[bool, Path]:
        """Run selfplay stage and register data in manifest."""
        output_dir.mkdir(parents=True, exist_ok=True)
        db_path = output_dir / f"selfplay_{board_type}_{num_players}p.db"

        cmd = [
            sys.executable,
            str(AI_SERVICE_ROOT / "scripts" / "run_self_play_soak.py"),
            "--board", board_type,
            "--players", str(num_players),
            "--games", str(games),
            "--output", str(db_path),
            "--workers", str(self.config.selfplay_workers),
        ]

        logger.info(f"Running selfplay: {board_type} {num_players}p, {games} games")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        if result.returncode != 0:
            logger.error(f"Selfplay failed: {result.stderr}")
            return False, db_path

        # Register in manifest
        if db_path.exists():
            is_new, fingerprint = self.manifest.register_source(
                db_path, board_type, num_players, games
            )
            if is_new:
                logger.info(f"Registered selfplay data: {db_path} (fingerprint: {fingerprint[:8]})")

        return True, db_path

    def run_training_stage(
        self,
        board_type: str,
        num_players: int,
        training_db: Path,
        parent_model: Optional[Path] = None
    ) -> Tuple[bool, Optional[Path]]:
        """Run training stage with lineage tracking."""
        model_id = f"{board_type}_{num_players}p_iter{self.current_iteration}_{int(time.time())}"
        output_path = self.config.models_dir / f"{model_id}.pth"

        # Get training data fingerprints
        fingerprint = DataManifest.compute_fingerprint(training_db)

        # Register model lineage
        parent_id = None
        if parent_model:
            parent_id = parent_model.stem

        lineage = self.lineage.register_model(
            model_id=model_id,
            model_path=output_path,
            board_type=board_type,
            num_players=num_players,
            parent_model_id=parent_id,
            training_fingerprints=[fingerprint],
            epochs=self.config.epochs
        )

        # Run training
        cmd = [
            sys.executable,
            str(AI_SERVICE_ROOT / "scripts" / "train_nnue.py"),
            "--db", str(training_db),
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--epochs", str(self.config.epochs),
            "--batch-size", str(self.config.batch_size),
            "--learning-rate", str(self.config.learning_rate),
            "--save-path", str(output_path),
        ]

        if parent_model and parent_model.exists():
            cmd.extend(["--resume-from", str(parent_model)])

        logger.info(f"Training model: {model_id}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)

        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            self.lineage.reject_model(model_id)
            return False, None

        # Parse final loss from output
        final_loss = None
        for line in result.stdout.split('\n'):
            if 'final_loss' in line.lower() or 'loss:' in line.lower():
                try:
                    final_loss = float(line.split()[-1])
                except (ValueError, IndexError):
                    pass

        self.lineage.complete_training(model_id, final_loss or 0.0)
        self.manifest.mark_complete(fingerprint)

        return True, output_path

    def run_evaluation_stage(
        self,
        board_type: str,
        num_players: int,
        candidate_model: Path,
        baseline_model: Path
    ) -> Tuple[bool, float, float]:
        """Run evaluation and update Elo ratings."""
        # Register models in Elo system
        candidate_id = candidate_model.stem
        baseline_id = baseline_model.stem

        self.elo.register_participant(
            candidate_id, candidate_id, "neural_net",
            use_neural_net=True, model_id=candidate_id
        )
        self.elo.register_participant(
            baseline_id, baseline_id, "neural_net",
            use_neural_net=True, model_id=baseline_id
        )

        # Run evaluation games
        cmd = [
            sys.executable,
            str(AI_SERVICE_ROOT / "scripts" / "evaluate_ai_models.py"),
            "--board", board_type,
            "--players", str(num_players),
            "--model-a", str(candidate_model),
            "--model-b", str(baseline_model),
            "--games", str(self.config.eval_games),
            "--output-json", str(self.config.logs_dir / f"eval_{candidate_id}.json"),
        ]

        logger.info(f"Evaluating {candidate_id} vs {baseline_id}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        if result.returncode != 0:
            logger.error(f"Evaluation failed: {result.stderr}")
            return False, 0.0, 0.0

        # Parse results
        eval_json = self.config.logs_dir / f"eval_{candidate_id}.json"
        if eval_json.exists():
            with open(eval_json) as f:
                eval_data = json.load(f)

            wins = eval_data.get("wins", 0)
            losses = eval_data.get("losses", 0)
            draws = eval_data.get("draws", 0)
            total = wins + losses + draws

            if total > 0:
                win_rate = (wins + 0.5 * draws) / total

                # Update Elo for each game
                for _ in range(wins):
                    self.elo.update_after_match(candidate_id, baseline_id, board_type, num_players)
                for _ in range(losses):
                    self.elo.update_after_match(baseline_id, candidate_id, board_type, num_players)
                for _ in range(draws):
                    self.elo.update_after_match(candidate_id, baseline_id, board_type, num_players, is_draw=True)

                # Get updated Elo
                candidate_elo = self.elo.get_or_create_rating(candidate_id, board_type, num_players)
                self.lineage.update_elo(candidate_id, candidate_elo)

                return True, win_rate, candidate_elo

        return False, 0.0, 0.0

    def should_promote(self, win_rate: float, games: int) -> Tuple[bool, float, float]:
        """Determine if model should be promoted using Wilson score interval."""
        from app.training.significance import wilson_score_interval

        lower, upper = wilson_score_interval(
            int(win_rate * games),
            games,
            confidence=self.config.confidence_level
        )

        should = lower > self.config.promotion_threshold
        return should, lower, upper

    def run_iteration(
        self,
        board_type: str,
        num_players: int,
        iteration: int
    ) -> bool:
        """Run a complete improvement iteration."""
        self.current_iteration = iteration
        logger.info(f"Starting iteration {iteration} for {board_type} {num_players}p")

        # Create iteration directory
        iter_dir = self.config.data_dir / "iterations" / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Get current best model
        best_lineage = self.lineage.get_current_best(board_type, num_players)
        best_model = Path(best_lineage.model_path) if best_lineage else None

        # Stage 1: Selfplay
        if self.config.enable_pipeline:
            selfplay_future = self.pipeline.submit_stage(
                "selfplay", iteration,
                self.run_selfplay_stage,
                board_type, num_players,
                self.config.games_per_iteration,
                iter_dir
            )
        else:
            success, db_path = self.run_selfplay_stage(
                board_type, num_players,
                self.config.games_per_iteration,
                iter_dir
            )
            if not success:
                return False

        # Stage 2: Training (waits for selfplay)
        if self.config.enable_pipeline:
            success, db_path = selfplay_future.result()
            if not success:
                return False

        success, new_model = self.run_training_stage(
            board_type, num_players, db_path, best_model
        )
        if not success:
            return False

        # Stage 3: Evaluation
        if best_model and best_model.exists():
            success, win_rate, elo = self.run_evaluation_stage(
                board_type, num_players, new_model, best_model
            )

            if success:
                should_promote, lower, upper = self.should_promote(
                    win_rate, self.config.eval_games
                )

                logger.info(
                    f"Evaluation: win_rate={win_rate:.2%}, "
                    f"CI=[{lower:.2%}, {upper:.2%}], "
                    f"promote={should_promote}"
                )

                if should_promote:
                    self.lineage.promote_model(new_model.stem)
                    logger.info(f"Promoted model: {new_model.stem}")
                    return True
                else:
                    self.lineage.reject_model(new_model.stem)
                    logger.info(f"Rejected model: {new_model.stem}")
        else:
            # No baseline, promote by default
            self.lineage.promote_model(new_model.stem)
            logger.info(f"Initial model promoted: {new_model.stem}")
            return True

        return False

    def run_pipeline(
        self,
        board_type: str,
        num_players: int,
        iterations: int
    ):
        """Run pipelined improvement loop with overlapped stages."""
        logger.info(f"Starting pipeline for {board_type} {num_players}p, {iterations} iterations")

        for i in range(iterations):
            if self._shutdown_event.is_set():
                logger.info("Shutdown requested, stopping pipeline")
                break

            # Start selfplay for next iteration while current trains
            if self.config.enable_pipeline and i < iterations - 1:
                next_iter_dir = self.config.data_dir / "iterations" / f"iter_{i+1}"
                next_iter_dir.mkdir(parents=True, exist_ok=True)

                self.pipeline.submit_stage(
                    "selfplay", i + 1,
                    self.run_selfplay_stage,
                    board_type, num_players,
                    self.config.games_per_iteration,
                    next_iter_dir
                )

            # Run current iteration
            success = self.run_iteration(board_type, num_players, i)
            if success:
                logger.info(f"Iteration {i} succeeded with promotion")
            else:
                logger.info(f"Iteration {i} completed without promotion")

    def run(self):
        """Main entry point for the controller."""
        self.running = True

        # Detect mode if auto
        mode = self.config.mode
        if mode == ControllerMode.AUTO:
            mode = self.detect_mode()

        logger.info(f"Running in {mode.value} mode")

        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Run improvement for each board/player config
            for board_type in self.config.board_types:
                for num_players in self.config.player_counts:
                    if self._shutdown_event.is_set():
                        break

                    self.run_pipeline(board_type, num_players, iterations=10)

        finally:
            self.pipeline.shutdown()
            self.running = False
            logger.info("Controller shutdown complete")

    def get_status(self) -> Dict[str, Any]:
        """Get current controller status."""
        return {
            "running": self.running,
            "current_iteration": self.current_iteration,
            "manifest_stats": self.manifest.get_stats(),
            "elo_leaderboard": self.elo.get_leaderboard(limit=10),
            "pipeline_stages": {
                name: {
                    iter_num: asdict(stage)
                    for iter_num, stage in stages.items()
                }
                for name, stages in self.pipeline.stages.items()
            }
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified AI Improvement Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["local", "cluster", "auto"],
        default="auto",
        help="Execution mode (default: auto-detect)"
    )

    parser.add_argument(
        "--board", "-b",
        type=str,
        nargs="+",
        default=["square8"],
        help="Board types to train"
    )

    parser.add_argument(
        "--players", "-p",
        type=int,
        nargs="+",
        default=[2],
        help="Player counts to train"
    )

    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10,
        help="Number of improvement iterations"
    )

    parser.add_argument(
        "--games", "-g",
        type=int,
        default=500,
        help="Selfplay games per iteration"
    )

    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Enable pipeline parallelization"
    )

    parser.add_argument(
        "--hot-path",
        action="store_true",
        help="Use hot data path (skip reanalysis)"
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status and exit"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create config
    config = ControllerConfig(
        mode=ControllerMode(args.mode),
        board_types=args.board,
        player_counts=args.players,
        games_per_iteration=args.games,
        enable_pipeline=args.pipeline,
        use_hot_path=args.hot_path,
    )

    controller = UnifiedImprovementController(config)

    if args.status:
        status = controller.get_status()
        print(json.dumps(status, indent=2, default=str))
        return

    controller.run()


if __name__ == "__main__":
    main()
