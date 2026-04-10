"""Model Registry Daemon - Cluster-wide model discovery and tracking.

January 2026 - Part of comprehensive multi-harness model evaluation pipeline.

This daemon periodically scans all P2P cluster nodes for models and maintains
a unified registry. It enables:
1. Automatic discovery of all NN/NNUE models across the cluster
2. Tracking which models exist on which nodes
3. Event emission for newly discovered models
4. Persistence to SQLite for crash recovery

Key features:
- Uses ClusterModelEnumerator to query all P2P nodes
- Stores inventory in data/model_registry.db
- Emits MODEL_REGISTRY_UPDATED events on changes
- Tracks unevaluated (model, harness) combinations
- Integrates with EvaluationDaemon for automatic evaluation

Usage:
    from app.coordination.model_registry_daemon import (
        ModelRegistryDaemon,
        get_model_registry_daemon,
    )

    daemon = get_model_registry_daemon()
    await daemon.start()

    # Get all known models
    models = daemon.get_all_models()

    # Get unevaluated combinations
    unevaluated = daemon.get_unevaluated_combinations()
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.contracts import CoordinatorStatus

logger = logging.getLogger(__name__)

__all__ = [
    "ModelRegistryDaemon",
    "ModelRegistryConfig",
    "get_model_registry_daemon",
    "reset_model_registry_daemon",
]


# Try to import ClusterModelEnumerator
try:
    from app.distributed.cluster_model_enumerator import (
        ClusterModelEnumerator,
        ModelInfo,
        EvaluationCombination,
        get_cluster_model_enumerator,
    )
    HAS_ENUMERATOR = True
except ImportError:
    HAS_ENUMERATOR = False
    ClusterModelEnumerator = None
    ModelInfo = None
    EvaluationCombination = None


# Try to import event emission
try:
    from app.coordination.event_emission_helpers import safe_emit_event_async
    HAS_EVENTS = True
except ImportError:
    HAS_EVENTS = False
    safe_emit_event_async = None


@dataclass
class ModelRegistryConfig:
    """Configuration for ModelRegistryDaemon."""

    # Scan interval
    scan_interval_seconds: float = 300.0  # 5 minutes

    # Database path
    db_path: str = "data/model_registry.db"

    # P2P port
    p2p_port: int = 8770

    # Request timeout for node queries
    request_timeout: float = 10.0

    # Cache TTL for enumerator
    cache_ttl_seconds: float = 300.0

    @classmethod
    def from_env(cls) -> ModelRegistryConfig:
        """Create config from environment variables."""
        return cls(
            scan_interval_seconds=float(
                os.environ.get("RINGRIFT_MODEL_REGISTRY_INTERVAL", "300")
            ),
            db_path=os.environ.get(
                "RINGRIFT_MODEL_REGISTRY_DB", "data/model_registry.db"
            ),
            p2p_port=int(os.environ.get("RINGRIFT_P2P_PORT", "8770")),
            request_timeout=float(
                os.environ.get("RINGRIFT_MODEL_REGISTRY_TIMEOUT", "10")
            ),
            cache_ttl_seconds=float(
                os.environ.get("RINGRIFT_MODEL_REGISTRY_CACHE_TTL", "300")
            ),
        )


class ModelRegistryDaemon(HandlerBase):
    """Daemon for cluster-wide model discovery and registry.

    Periodically scans all P2P nodes for models and maintains a unified
    registry in SQLite. Emits events when new models are discovered.
    """

    _event_source = "ModelRegistryDaemon"

    def __init__(self, config: ModelRegistryConfig | None = None):
        """Initialize the daemon.

        Args:
            config: Optional configuration. Uses env vars if not provided.
        """
        self._daemon_config = config or ModelRegistryConfig.from_env()

        super().__init__(
            name="model_registry",
            config=self._daemon_config,
            cycle_interval=self._daemon_config.scan_interval_seconds,
        )

        # Database
        self._db_path = Path(self._daemon_config.db_path)
        self._db_lock = threading.RLock()

        # State tracking
        self._known_models: dict[str, dict[str, Any]] = {}  # hash -> info
        self._last_scan_time: float = 0.0
        self._scan_count: int = 0
        self._models_discovered: int = 0
        self._new_models_this_scan: int = 0

        # Enumerator
        self._enumerator: ClusterModelEnumerator | None = None

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for model registry."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._db_lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()

                # Models table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS models (
                        model_hash TEXT PRIMARY KEY,
                        model_path TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        board_type TEXT NOT NULL,
                        num_players INTEGER NOT NULL,
                        architecture TEXT,
                        size_bytes INTEGER DEFAULT 0,
                        modified_time REAL DEFAULT 0,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        node_ids TEXT  -- JSON array
                    )
                """)

                # Model-node mapping for tracking availability
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_nodes (
                        model_hash TEXT NOT NULL,
                        node_id TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        PRIMARY KEY (model_hash, node_id),
                        FOREIGN KEY (model_hash) REFERENCES models(model_hash)
                    )
                """)

                # Evaluation tracking
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS evaluations (
                        model_hash TEXT NOT NULL,
                        harness TEXT NOT NULL,
                        config_key TEXT NOT NULL,
                        elo_rating REAL,
                        games_played INTEGER DEFAULT 0,
                        last_evaluated TEXT,
                        PRIMARY KEY (model_hash, harness, config_key),
                        FOREIGN KEY (model_hash) REFERENCES models(model_hash)
                    )
                """)

                # Index for fast lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_models_config
                    ON models(board_type, num_players)
                """)

                conn.commit()
                logger.info(f"[ModelRegistry] Database initialized at {self._db_path}")

            finally:
                conn.close()

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Get event subscriptions. Empty for this daemon."""
        return {}

    async def _on_start(self) -> None:
        """Called when daemon starts."""
        # Initialize enumerator
        if HAS_ENUMERATOR:
            self._enumerator = get_cluster_model_enumerator()
            self._enumerator.p2p_port = self._daemon_config.p2p_port
            self._enumerator.request_timeout = self._daemon_config.request_timeout
            self._enumerator.cache_ttl_seconds = self._daemon_config.cache_ttl_seconds

        # Load existing models from database
        await asyncio.to_thread(self._load_known_models)

        logger.info(
            f"[ModelRegistry] Started with {len(self._known_models)} known models"
        )

    async def _run_cycle(self) -> None:
        """Main work loop - scan cluster for models."""
        if not HAS_ENUMERATOR:
            logger.warning("[ModelRegistry] ClusterModelEnumerator not available")
            return

        try:
            self._new_models_this_scan = 0
            self._scan_count += 1

            # Enumerate all models across cluster
            models = await self._enumerator.enumerate_all_models(force_refresh=True)

            logger.info(
                f"[ModelRegistry] Scan {self._scan_count}: found {len(models)} models"
            )

            # Process each model
            for model in models:
                await self._process_model(model)

            self._last_scan_time = time.time()

            # Emit event if new models found
            if self._new_models_this_scan > 0:
                await self._emit_registry_updated()

            logger.debug(
                f"[ModelRegistry] Cycle complete: {self._new_models_this_scan} new, "
                f"{len(self._known_models)} total"
            )

        except Exception as e:
            logger.error(f"[ModelRegistry] Scan failed: {e}")
            self._stats.errors_count += 1
            self._stats.last_error = str(e)

    async def _process_model(self, model: ModelInfo) -> None:
        """Process a discovered model.

        Args:
            model: ModelInfo from ClusterModelEnumerator
        """
        is_new = model.model_hash not in self._known_models

        # Store model info
        model_data = {
            "model_path": model.model_path,
            "model_type": model.model_type,
            "board_type": model.board_type,
            "num_players": model.num_players,
            "architecture": model.architecture,
            "size_bytes": model.size_bytes,
            "modified_time": model.modified_time,
            "node_ids": model.node_ids,
        }
        self._known_models[model.model_hash] = model_data

        # Persist to database
        await asyncio.to_thread(self._upsert_model, model)

        if is_new:
            self._new_models_this_scan += 1
            self._models_discovered += 1
            logger.info(
                f"[ModelRegistry] New model: {model.model_path} "
                f"({model.model_type}, {model.config_key}) on {model.node_ids}"
            )

    def _upsert_model(self, model: ModelInfo) -> None:
        """Insert or update model in database (blocking).

        Args:
            model: ModelInfo to persist
        """
        import json as json_module

        now = datetime.now().isoformat()

        with self._db_lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()

                # Check if exists
                cursor.execute(
                    "SELECT first_seen FROM models WHERE model_hash = ?",
                    (model.model_hash,)
                )
                row = cursor.fetchone()
                first_seen = row[0] if row else now

                # Upsert model
                cursor.execute("""
                    INSERT OR REPLACE INTO models
                    (model_hash, model_path, model_type, board_type, num_players,
                     architecture, size_bytes, modified_time, first_seen, last_seen, node_ids)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model.model_hash,
                    model.model_path,
                    model.model_type,
                    model.board_type,
                    model.num_players,
                    model.architecture,
                    model.size_bytes,
                    model.modified_time,
                    first_seen,
                    now,
                    json_module.dumps(model.node_ids),
                ))

                # Update node mappings
                for node_id in model.node_ids:
                    cursor.execute("""
                        INSERT OR REPLACE INTO model_nodes
                        (model_hash, node_id, last_seen)
                        VALUES (?, ?, ?)
                    """, (model.model_hash, node_id, now))

                # Update evaluation status
                for eval_status in model.evaluations:
                    cursor.execute("""
                        INSERT OR REPLACE INTO evaluations
                        (model_hash, harness, config_key, elo_rating, games_played, last_evaluated)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        model.model_hash,
                        eval_status.harness,
                        eval_status.config_key,
                        eval_status.elo_rating,
                        eval_status.games_played,
                        eval_status.last_evaluated.isoformat() if eval_status.last_evaluated else None,
                    ))

                conn.commit()

            finally:
                conn.close()

    def _load_known_models(self) -> None:
        """Load known models from database (blocking)."""
        import json as json_module

        if not self._db_path.exists():
            return

        with self._db_lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT model_hash, model_path, model_type, board_type,
                           num_players, architecture, size_bytes, modified_time, node_ids
                    FROM models
                """)

                for row in cursor.fetchall():
                    model_hash = row[0]
                    self._known_models[model_hash] = {
                        "model_path": row[1],
                        "model_type": row[2],
                        "board_type": row[3],
                        "num_players": row[4],
                        "architecture": row[5],
                        "size_bytes": row[6],
                        "modified_time": row[7],
                        "node_ids": json_module.loads(row[8]) if row[8] else [],
                    }

            finally:
                conn.close()

    async def _emit_registry_updated(self) -> None:
        """Emit MODEL_REGISTRY_UPDATED event."""
        if not HAS_EVENTS:
            return

        try:
            await safe_emit_event_async(
                "MODEL_REGISTRY_UPDATED",
                {
                    "total_models": len(self._known_models),
                    "new_models": self._new_models_this_scan,
                    "scan_count": self._scan_count,
                    "timestamp": time.time(),
                },
                source="model_registry_daemon",
                context="ModelRegistryDaemon",
            )
        except Exception as e:
            logger.debug(f"[ModelRegistry] Failed to emit event: {e}")

    def get_all_models(self) -> list[dict[str, Any]]:
        """Get all known models.

        Returns:
            List of model info dictionaries
        """
        return [
            {"model_hash": h, **info}
            for h, info in self._known_models.items()
        ]

    def get_models_for_config(
        self, board_type: str, num_players: int
    ) -> list[dict[str, Any]]:
        """Get models for a specific board configuration.

        Args:
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players (2, 3, 4)

        Returns:
            List of model info dictionaries matching the config
        """
        return [
            {"model_hash": h, **info}
            for h, info in self._known_models.items()
            if info["board_type"] == board_type and info["num_players"] == num_players
        ]

    def get_unevaluated_combinations(self) -> list[EvaluationCombination]:
        """Get (model, harness) combinations not yet evaluated.

        Returns:
            List of EvaluationCombination objects
        """
        if not HAS_ENUMERATOR or not self._enumerator:
            return []
        return self._enumerator.get_unevaluated_combinations()

    def get_stale_combinations(
        self, max_age_days: int = 7
    ) -> list[EvaluationCombination]:
        """Get combinations not evaluated recently.

        Args:
            max_age_days: Max days since last evaluation

        Returns:
            List of stale EvaluationCombination objects
        """
        if not HAS_ENUMERATOR or not self._enumerator:
            return []
        return self._enumerator.get_stale_combinations(max_age_days)

    def health_check(self) -> HealthCheckResult:
        """Return health status.

        Returns:
            HealthCheckResult with daemon status
        """
        # Check base status
        if not self._running:
            return HealthCheckResult(
                status=CoordinatorStatus.STOPPED,
                message="ModelRegistryDaemon is stopped",
                details={"running": False},
            )

        # Check scan recency
        time_since_scan = time.time() - self._last_scan_time if self._last_scan_time else 0

        # Determine status
        if self._stats.errors_count > 5:
            status = CoordinatorStatus.DEGRADED
            message = "Multiple scan errors"
        elif time_since_scan > self._cycle_interval * 3:
            status = CoordinatorStatus.DEGRADED
            message = "Scans behind schedule"
        else:
            status = CoordinatorStatus.RUNNING
            message = "ModelRegistryDaemon healthy"

        return HealthCheckResult(
            status=status,
            message=message,
            details={
                "running": True,
                "known_models": len(self._known_models),
                "models_discovered": self._models_discovered,
                "scan_count": self._scan_count,
                "last_scan_time": self._last_scan_time,
                "time_since_scan": time_since_scan,
                "errors_count": self._stats.errors_count,
                "has_enumerator": HAS_ENUMERATOR,
            },
        )


# Singleton accessor
_daemon_instance: ModelRegistryDaemon | None = None
_daemon_lock = threading.Lock()


def get_model_registry_daemon() -> ModelRegistryDaemon:
    """Get singleton ModelRegistryDaemon instance.

    Returns:
        ModelRegistryDaemon singleton
    """
    global _daemon_instance
    if _daemon_instance is None:
        with _daemon_lock:
            if _daemon_instance is None:
                _daemon_instance = ModelRegistryDaemon()
    return _daemon_instance


def reset_model_registry_daemon() -> None:
    """Reset singleton instance (for testing)."""
    global _daemon_instance
    with _daemon_lock:
        if _daemon_instance is not None:
            # Stop if running
            if _daemon_instance._running:
                try:
                    asyncio.get_running_loop()
                    asyncio.create_task(_daemon_instance.stop())
                except RuntimeError:
                    asyncio.run(_daemon_instance.stop())
            _daemon_instance = None
