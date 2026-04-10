"""Training coordination protocol helpers.

This module keeps slot acquisition, status reporting, stale-job cleanup, and
training event emission separate from the daemon/event-shell code in
``training_coordinator.py``.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.core_utils import DistributedLock
from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.event_utils import make_config_key
from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

# January 2026: Cluster manifest integration for data pre-positioning.
try:
    from app.distributed.data_catalog import get_data_registry

    HAS_DATA_CATALOG = True
except ImportError:
    HAS_DATA_CATALOG = False
    get_data_registry = None  # type: ignore[assignment]

try:
    from app.coordination.sync_facade import get_sync_facade

    HAS_SYNC_FACADE = True
except ImportError:
    HAS_SYNC_FACADE = False
    get_sync_facade = None  # type: ignore[assignment]

# NFS path for cluster-wide coordination (Lambda GH200 nodes).
NFS_COORDINATION_PATH = Path(
    os.environ.get("RINGRIFT_NFS_COORDINATION_PATH", "/lambda/nfs/RingRift/coordination")
)

# Training configuration - use centralized defaults (December 2025).
try:
    from app.config.coordination_defaults import HeartbeatDefaults, TrainingDefaults

    MAX_CONCURRENT_TRAINING_SAME_CONFIG = TrainingDefaults.MAX_CONCURRENT_SAME_CONFIG
    MAX_TOTAL_CONCURRENT_TRAINING = TrainingDefaults.MAX_CONCURRENT_TOTAL
    TRAINING_TIMEOUT_HOURS = TrainingDefaults.TIMEOUT_HOURS
    HEARTBEAT_INTERVAL_SECONDS = HeartbeatDefaults.INTERVAL
    STALE_CHECK_INTERVAL_SECONDS = HeartbeatDefaults.STALE_CLEANUP_INTERVAL * 5
except ImportError:
    MAX_CONCURRENT_TRAINING_SAME_CONFIG = 1
    MAX_TOTAL_CONCURRENT_TRAINING = 4
    TRAINING_TIMEOUT_HOURS = 12
    HEARTBEAT_INTERVAL_SECONDS = 60
    STALE_CHECK_INTERVAL_SECONDS = 300

logger = logging.getLogger(__name__)


def _coordinator_compat_attr(name: str, default: Any) -> Any:
    """Read compatibility exports from ``training_coordinator.py``."""
    try:
        from app.coordination import training_coordinator as coordinator_module

        return getattr(coordinator_module, name, default)
    except ImportError:
        return default


def _create_distributed_lock(name: str) -> DistributedLock:
    """Create a lock through the compatibility export when available."""
    lock_cls = _coordinator_compat_attr("DistributedLock", DistributedLock)
    return lock_cls(name)


@dataclass
class TrainingJob:
    """Represents an active or queued training job."""

    job_id: str
    board_type: str
    num_players: int
    node_name: str
    node_ip: str
    pid: int
    started_at: float
    last_heartbeat: float
    status: str = "running"  # running, queued, completed, failed
    model_version: str = ""
    epochs_completed: int = 0
    best_val_loss: float = float("inf")
    current_elo: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def config_key(self) -> str:
        return f"{self.board_type}_{self.num_players}p"

    @property
    def age_hours(self) -> float:
        return (time.time() - self.started_at) / 3600

    @property
    def heartbeat_age_seconds(self) -> float:
        return time.time() - self.last_heartbeat

    @property
    def is_stale(self) -> bool:
        return (
            self.heartbeat_age_seconds > HEARTBEAT_INTERVAL_SECONDS * 3
            or self.age_hours > TRAINING_TIMEOUT_HOURS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "config_key": self.config_key,
            "age_hours": round(self.age_hours, 2),
            "heartbeat_age_seconds": round(self.heartbeat_age_seconds, 1),
            "is_stale": self.is_stale,
        }


class TrainingProtocolMixin:
    """Training slot, progress, status, and event protocol helpers."""

    def get_metrics(self) -> dict[str, Any]:
        """Get coordinator metrics.

        Returns:
            Dictionary of metrics (CoordinatorProtocol compliant)
        """
        active_jobs = self.get_active_jobs()

        return {
            "name": self.name,
            "status": self._status.value,
            "uptime_seconds": self.uptime_seconds,
            "start_time": self._start_time,
            "events_processed": self._events_processed,
            "errors_count": self._errors_count,
            "last_error": self._last_error,
            # Custom metrics
            "active_training_jobs": len(active_jobs),
            "cluster_healthy": self._cluster_healthy,
            "cluster_capacity": self._cluster_capacity,
            "node_name": self._node_name,
            "db_path": str(self._db_path),
            "using_nfs": self._use_nfs and NFS_COORDINATION_PATH.exists(),
        }

    def health_check(self) -> HealthCheckResult:
        """Check coordinator health.

        Returns:
            HealthCheckResult (CoordinatorProtocol compliant)
        """
        if self._status == CoordinatorStatus.ERROR:
            return HealthCheckResult.unhealthy(
                f"Coordinator in error state: {self._last_error}",
                db_path=str(self._db_path),
            )

        if self._status == CoordinatorStatus.STOPPED:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Coordinator is stopped",
            )

        # Check database connectivity
        try:
            conn = self._get_connection()
            conn.execute("SELECT 1")
            db_healthy = True
        except (sqlite3.Error, OSError, AttributeError) as e:
            # Database errors: connection failures, disk I/O, missing connection
            db_healthy = False
            return HealthCheckResult.unhealthy(
                f"Database connection failed: {e}",
                db_path=str(self._db_path),
            )

        # Check cluster health
        if not self._cluster_healthy:
            return HealthCheckResult.degraded(
                "Coordinator running but cluster is unhealthy",
                cluster_capacity=self._cluster_capacity,
                db_healthy=db_healthy,
            )

        active_jobs = self.get_active_jobs()
        return HealthCheckResult(
            healthy=True,
            status=self._status,
            details={
                "active_training_jobs": len(active_jobs),
                "cluster_healthy": self._cluster_healthy,
                "cluster_capacity": self._cluster_capacity,
                "uptime_seconds": self.uptime_seconds,
                "db_path": str(self._db_path),
                "db_healthy": db_healthy,
            },
        )

    # =========================================================================
    # Data Pre-positioning Methods (January 2026 - Phase 4)
    # =========================================================================

    def _select_best_data_source(
        self,
        board_type: str,
        num_players: int,
        target_node: str | None = None,
    ) -> dict[str, Any] | None:
        """Select the best data source for training.

        Queries the cluster manifest to find optimal data sources,
        scoring by locality (prefer target node), quality, and recency.

        Args:
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)
            target_node: Optional target training node for locality scoring

        Returns:
            Dictionary with source info (node_id, path, game_count, score),
            or None if no data available.

        January 2026: Phase 4 - TrainingCoordinator Data Pre-positioning.
        """
        has_data_catalog = _coordinator_compat_attr(
            "HAS_DATA_CATALOG", HAS_DATA_CATALOG
        )
        registry_getter = _coordinator_compat_attr(
            "get_data_registry", get_data_registry
        )
        if not has_data_catalog or registry_getter is None:
            logger.debug("[TrainingCoordinator] Data catalog not available")
            return None

        config_key = make_config_key(board_type, num_players)
        target = target_node or self._node_name

        try:
            registry = registry_getter()
            sources = registry.get_data_sources(board_type, num_players)

            if not sources:
                logger.debug(f"[TrainingCoordinator] No data sources for {config_key}")
                return None

            # Score each source
            scored_sources: list[dict[str, Any]] = []
            for source in sources:
                score = 0.0
                node_id = source.get("node_id", "")
                game_count = source.get("game_count", 0)

                # Locality bonus: prefer data already at target node
                if node_id == target:
                    score += 100.0
                    source["is_local"] = True
                else:
                    source["is_local"] = False

                # Game count bonus: prefer sources with more data
                if game_count >= 10000:
                    score += 50.0
                elif game_count >= 5000:
                    score += 30.0
                elif game_count >= 1000:
                    score += 10.0

                # Recency bonus (if last_updated available)
                last_updated = source.get("last_updated", 0)
                if last_updated:
                    age_hours = (time.time() - last_updated) / 3600
                    if age_hours < 1:
                        score += 20.0
                    elif age_hours < 6:
                        score += 10.0

                source["score"] = score
                scored_sources.append(source)

            if not scored_sources:
                return None

            # Return best source
            best = max(scored_sources, key=lambda s: s.get("score", 0))
            logger.debug(
                f"[TrainingCoordinator] Best source for {config_key}: "
                f"node={best.get('node_id')}, games={best.get('game_count')}, "
                f"local={best.get('is_local')}, score={best.get('score')}"
            )
            return best

        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"[TrainingCoordinator] Error finding data source: {e}")
            return None

    async def _ensure_data_at_node(
        self,
        board_type: str,
        num_players: int,
        target_node: str | None = None,
        sync_timeout: float = 300.0,
    ) -> bool:
        """Ensure training data is available at the target node.

        If the best data source is remote, triggers a priority sync
        to pre-position data before training starts.

        Args:
            board_type: Board type
            num_players: Number of players
            target_node: Target training node (defaults to local node)
            sync_timeout: Maximum time to wait for sync (seconds)

        Returns:
            True if data is available (local or synced), False otherwise.

        January 2026: Phase 4 - TrainingCoordinator Data Pre-positioning.
        """
        import asyncio

        config_key = make_config_key(board_type, num_players)
        target = target_node or self._node_name

        # Find best data source
        best_source = self._select_best_data_source(
            board_type, num_players, target
        )

        if not best_source:
            logger.warning(
                f"[TrainingCoordinator] No data sources found for {config_key}"
            )
            return False

        # If data is already local, we're done
        if best_source.get("is_local", False):
            logger.info(
                f"[TrainingCoordinator] Data for {config_key} already at {target} "
                f"({best_source.get('game_count', 0)} games)"
            )
            return True

        # Need to sync from remote source
        has_sync_facade = _coordinator_compat_attr(
            "HAS_SYNC_FACADE", HAS_SYNC_FACADE
        )
        sync_facade_getter = _coordinator_compat_attr(
            "get_sync_facade", get_sync_facade
        )
        if not has_sync_facade or sync_facade_getter is None:
            logger.warning(
                f"[TrainingCoordinator] Sync facade not available, "
                f"cannot pre-position data for {config_key}"
            )
            return False

        source_node = best_source.get("node_id", "unknown")
        logger.info(
            f"[TrainingCoordinator] Pre-positioning data for {config_key}: "
            f"{source_node} -> {target} ({best_source.get('game_count', 0)} games)"
        )

        try:
            facade = sync_facade_getter()
            response = await facade.trigger_priority_sync(
                reason=f"training_pre_positioning_{config_key}",
                source_node=source_node,
                config_key=config_key,
                data_type="games",
            )

            if response.success:
                logger.info(
                    f"[TrainingCoordinator] Data pre-positioned for {config_key}: "
                    f"synced {response.nodes_synced} nodes"
                )
                self._config_sync_times[config_key] = time.time()
                return True
            else:
                logger.warning(
                    f"[TrainingCoordinator] Pre-positioning sync failed for {config_key}: "
                    f"{response.errors}"
                )
                return False

        except asyncio.TimeoutError:
            logger.warning(
                f"[TrainingCoordinator] Pre-positioning sync timed out for {config_key} "
                f"after {sync_timeout}s"
            )
            return False
        except (RuntimeError, OSError, ConnectionError) as e:
            # Runtime/connection errors during sync
            logger.warning(
                f"[TrainingCoordinator] Pre-positioning sync error for {config_key}: {e}"
            )
            return False

    async def prepare_training_data_async(
        self,
        board_type: str,
        num_players: int,
        target_node: str | None = None,
    ) -> dict[str, Any]:
        """Prepare training data before starting a training job.

        This is the main entry point for data pre-positioning. Call this
        before start_training() to ensure data is available.

        Args:
            board_type: Board type
            num_players: Number of players
            target_node: Target training node

        Returns:
            Dictionary with:
            - ready: bool - whether data is ready for training
            - source: dict - best data source info
            - synced: bool - whether sync was performed
            - games: int - number of games available

        January 2026: Phase 4 - TrainingCoordinator Data Pre-positioning.
        """
        config_key = make_config_key(board_type, num_players)
        target = target_node or self._node_name

        result: dict[str, Any] = {
            "ready": False,
            "source": None,
            "synced": False,
            "games": 0,
            "config_key": config_key,
            "target_node": target,
        }

        # Find best source
        best_source = self._select_best_data_source(board_type, num_players, target)
        if best_source:
            result["source"] = best_source
            result["games"] = best_source.get("game_count", 0)

        # Try to ensure data at node
        data_ready = await self._ensure_data_at_node(
            board_type, num_players, target
        )
        result["ready"] = data_ready

        # Check if we synced
        if config_key in self._config_sync_times:
            sync_age = time.time() - self._config_sync_times[config_key]
            if sync_age < 60:  # Synced within last minute
                result["synced"] = True

        logger.info(
            f"[TrainingCoordinator] Data preparation for {config_key}: "
            f"ready={result['ready']}, games={result['games']}, synced={result['synced']}"
        )

        return result

    def can_start_training(self, board_type: str, num_players: int) -> bool:
        """Check if training can be started for this config.

        Returns:
            True if no active training for this config and slots available
        """
        config_key = make_config_key(board_type, num_players)

        # Check cluster health first (December 2025 - feedback loop)
        if not self._cluster_healthy:
            logger.info("Training blocked: cluster is unhealthy")
            self._emit_slot_unavailable(
                board_type, num_players, reason="cluster_unhealthy"
            )
            return False

        conn = self._get_connection()
        self._cleanup_stale_jobs()

        # Check if this config is already being trained
        cursor = conn.execute(
            '''SELECT job_id, node_name FROM training_jobs
               WHERE board_type = ? AND num_players = ? AND status = 'running' ''',
            (board_type, num_players)
        )
        existing = cursor.fetchone()
        if existing:
            logger.info(
                f"Training for {config_key} already running on {existing['node_name']}"
            )
            self._emit_slot_unavailable(
                board_type, num_players,
                reason="already_running",
                holder_node=existing['node_name'],
                holder_job_id=existing['job_id'],
            )
            return False

        # Check total concurrent training limit
        cursor = conn.execute(
            "SELECT COUNT(*) FROM training_jobs WHERE status = 'running'"
        )
        active_count = cursor.fetchone()[0]
        if active_count >= MAX_TOTAL_CONCURRENT_TRAINING:
            logger.info(
                f"Max concurrent training ({MAX_TOTAL_CONCURRENT_TRAINING}) reached"
            )
            self._emit_slot_unavailable(
                board_type, num_players,
                reason="max_concurrent_reached",
                active_count=active_count,
                max_allowed=MAX_TOTAL_CONCURRENT_TRAINING,
            )
            return False

        return True

    def start_training(
        self,
        board_type: str,
        num_players: int,
        model_version: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Register a new training job.

        Args:
            board_type: Board type (e.g., "square8", "hex8")
            num_players: Number of players
            model_version: Version string for the model
            metadata: Additional metadata

        Returns:
            job_id if registered successfully, None if slot not available
        """
        # First try to acquire distributed lock with retry
        config_key = make_config_key(board_type, num_players)
        lock = _create_distributed_lock(f"training:{config_key}")

        # Retry with increasing timeouts: 30s, 60s, 90s
        lock_timeouts = [30, 60, 90]
        lock_acquired = False

        for attempt, timeout in enumerate(lock_timeouts):
            if lock.acquire(timeout=timeout, blocking=True):
                lock_acquired = True
                break

            logger.warning(
                f"Lock acquisition attempt {attempt + 1}/{len(lock_timeouts)} failed "
                f"for {config_key} (timeout={timeout}s)"
            )

            # Wait before next attempt (except on last try)
            if attempt < len(lock_timeouts) - 1:
                time.sleep(5)

        if not lock_acquired:
            logger.error(f"Could not acquire distributed lock for {config_key} after {len(lock_timeouts)} attempts")

            # January 2, 2026: Track consecutive lock failures for escalation
            self._lock_failure_counts[config_key] = self._lock_failure_counts.get(config_key, 0) + 1
            consecutive_failures = self._lock_failure_counts[config_key]

            # Emit failure event for monitoring
            self._emit_slot_unavailable(
                board_type=board_type,
                num_players=num_players,
                reason="lock_failed",
                attempts=len(lock_timeouts),
                consecutive_failures=consecutive_failures,
            )

            # January 2, 2026: Escalate if threshold exceeded
            if consecutive_failures >= self._lock_failure_escalation_threshold:
                self._escalate_lock_failure(
                    config_key=config_key,
                    board_type=board_type,
                    num_players=num_players,
                    consecutive_failures=consecutive_failures,
                )

            return None

        # January 2, 2026: Reset failure count on successful lock acquisition
        if config_key in self._lock_failure_counts:
            del self._lock_failure_counts[config_key]

        # Emit lock acquired event for monitoring (December 2025 - Phase 14)
        self._emit_training_event(
            "lock_acquired",
            job_id="pending",  # Job ID not yet assigned
            board_type=board_type,
            num_players=num_players,
        )

        try:
            if not self.can_start_training(board_type, num_players):
                lock.release()
                return None

            conn = self._get_connection()
            now = time.time()
            job_id = f"{config_key}_{int(now)}_{os.getpid()}"

            # Jan 6, 2026: P2 - Capture Elo before training starts
            before_elo = self._get_current_elo(board_type, num_players)

            try:
                conn.execute(
                    '''INSERT INTO training_jobs
                       (job_id, board_type, num_players, node_name, node_ip, pid,
                        started_at, last_heartbeat, status, model_version, before_elo, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)''',
                    (
                        job_id, board_type, num_players,
                        self._node_name, self._node_ip, os.getpid(),
                        now, now, model_version, before_elo,
                        json.dumps(metadata or {})
                    )
                )
                conn.commit()
                logger.info(f"Started training job {job_id} on {self._node_name}")

                # Emit TRAINING_STARTED event (December 2025)
                self._emit_training_event(
                    "started",
                    job_id=job_id,
                    board_type=board_type,
                    num_players=num_players,
                    model_version=model_version,
                )

                return job_id

            except sqlite3.IntegrityError:
                # Race condition - another node started training
                logger.warning(f"Race condition: {config_key} training started elsewhere")
                lock.release()
                return None

        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            # Dec 29, 2025: Narrowed from bare Exception
            # Database errors (connection, corruption, etc.)
            logger.error(f"Database error starting training: {e}")
            lock.release()
            return None
        except (AttributeError, TypeError) as e:
            # Programming errors - log critically
            logger.critical(f"Training coordinator bug: {e}")
            lock.release()
            raise
        except (OSError, RuntimeError) as e:
            # System/runtime errors
            logger.error(f"System error starting training: {e}")
            lock.release()
            return None

    def update_progress(
        self,
        job_id: str,
        epochs_completed: int = 0,
        best_val_loss: float = float("inf"),
        current_elo: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update training progress and heartbeat.

        Args:
            job_id: The job ID returned by start_training
            epochs_completed: Number of epochs completed
            best_val_loss: Best validation loss so far
            current_elo: Current Elo rating if evaluated
            metadata: Additional metadata to update

        Returns:
            True if update successful
        """
        conn = self._get_connection()
        now = time.time()

        updates = ["last_heartbeat = ?", "epochs_completed = ?"]
        params: list[Any] = [now, epochs_completed]

        if best_val_loss < float("inf"):
            updates.append("best_val_loss = ?")
            params.append(best_val_loss)

        if current_elo > 0:
            updates.append("current_elo = ?")
            params.append(current_elo)

        if metadata:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        params.append(job_id)
        params.append(os.getpid())

        cursor = conn.execute(
            f'''UPDATE training_jobs
                SET {', '.join(updates)}
                WHERE job_id = ? AND pid = ?''',
            params
        )
        conn.commit()
        return cursor.rowcount > 0

    def complete_training(
        self,
        job_id: str,
        status: str = "completed",
        final_val_loss: float | None = None,
        final_elo: float | None = None,
    ) -> bool:
        """Mark training as complete and archive to history.

        Args:
            job_id: The job ID
            status: Final status (completed, failed)
            final_val_loss: Final validation loss
            final_elo: Final Elo rating

        Returns:
            True if completed successfully
        """
        conn = self._get_connection()
        now = time.time()

        # Get current job info
        cursor = conn.execute(
            "SELECT * FROM training_jobs WHERE job_id = ?", (job_id,)
        )
        job = cursor.fetchone()
        if not job:
            return False

        # Archive to history
        # Jan 6, 2026: P2 - Include before_elo for Elo delta tracking
        conn.execute(
            '''INSERT INTO training_history
               (job_id, board_type, num_players, node_name, started_at,
                completed_at, status, final_val_loss, final_elo,
                epochs_completed, before_elo, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                job_id, job["board_type"], job["num_players"],
                job["node_name"], job["started_at"], now, status,
                final_val_loss or job["best_val_loss"],
                final_elo or job["current_elo"],
                job["epochs_completed"], job["before_elo"],
                job["metadata"]
            )
        )

        # Remove from active jobs
        conn.execute("DELETE FROM training_jobs WHERE job_id = ?", (job_id,))
        conn.commit()

        # Release distributed lock
        config_key = f"{job['board_type']}_{job['num_players']}p"
        lock = _create_distributed_lock(f"training:{config_key}")
        lock.release()

        logger.info(f"Completed training job {job_id} with status {status}")

        # Emit TRAINING_COMPLETE or TRAINING_FAILED event (December 2025)
        # Feb 2026: Include model_path so tournament_daemon can queue evaluation
        event_type = "complete" if status == "completed" else "failed"
        canonical_model = f"models/canonical_{job['board_type']}_{job['num_players']}p.pth"
        self._emit_training_event(
            event_type,
            job_id=job_id,
            board_type=job["board_type"],
            num_players=job["num_players"],
            final_val_loss=final_val_loss or job["best_val_loss"],
            final_elo=final_elo or job["current_elo"],
            epochs_completed=job["epochs_completed"],
            status=status,
            model_path=canonical_model,
            config=config_key,
        )

        return True

    def _emit_training_event(
        self,
        event_type: str,
        job_id: str,
        board_type: str,
        num_players: int,
        **kwargs,
    ) -> None:
        """Emit training-related event via centralized emitters.

        Uses event_emitters.py which handles all routing to stage_events
        and cross-process buses.

        Args:
            event_type: One of "started", "complete", "failed"
            job_id: Training job ID
            board_type: Board type
            num_players: Number of players
            **kwargs: Additional event data
        """
        if event_type == "started":
            # January 2026 - migrated to safe_emit_event
            safe_emit_event(
                "TRAINING_STARTED",
                {
                    "job_id": job_id,
                    "board_type": board_type,
                    "num_players": num_players,
                    "model_version": kwargs.get("model_version", ""),
                    "node_name": self._node_name,
                },
                context="training_coordinator",
            )
            logger.debug(f"Emitted TRAINING_STARTED for job {job_id}")

        elif event_type in ("complete", "failed"):
            # January 2026 - migrated to safe_emit_event
            success = (event_type == "complete")
            event_name = "TRAINING_COMPLETED" if success else "TRAINING_FAILED"
            safe_emit_event(
                event_name,
                {
                    "job_id": job_id,
                    "board_type": board_type,
                    "num_players": num_players,
                    "success": success,
                    "final_loss": kwargs.get("final_val_loss"),
                    "final_elo": kwargs.get("final_elo"),
                    "model_path": kwargs.get("model_path"),
                    "epochs_completed": kwargs.get("epochs_completed", 0),
                    "node_name": self._node_name,
                    "status": kwargs.get("status", "completed" if success else "failed"),
                    "architecture": kwargs.get("architecture"),  # Jan 4, 2026: Multi-architecture support
                },
                context="training_coordinator",
            )
            logger.debug(f"Emitted {event_name} for job {job_id}")

        elif event_type == "lock_acquired":
            # Emit TRAINING_LOCK_ACQUIRED for monitoring
            self._emit_via_router(
                "TRAINING_LOCK_ACQUIRED",
                {
                    "job_id": job_id,
                    "board_type": board_type,
                    "num_players": num_players,
                    "node_name": self._node_name,
                    "config": make_config_key(board_type, num_players),
                    "timestamp": time.time(),
                },
            )
            logger.debug(f"Emitted TRAINING_LOCK_ACQUIRED for {make_config_key(board_type, num_players)}")

    def _emit_slot_unavailable(
        self,
        board_type: str,
        num_players: int,
        reason: str,
        **kwargs,
    ) -> None:
        """Emit TRAINING_SLOT_UNAVAILABLE event.

        Provides visibility into why training couldn't start.

        Args:
            board_type: Board type
            num_players: Number of players
            reason: Why slot is unavailable (cluster_unhealthy, already_running, max_concurrent_reached, lock_failed)
            **kwargs: Additional context (holder_node, holder_job_id, active_count, etc.)
        """
        self._emit_via_router(
            "TRAINING_SLOT_UNAVAILABLE",
            {
                "board_type": board_type,
                "num_players": num_players,
                "config": make_config_key(board_type, num_players),
                "reason": reason,
                "requester_node": self._node_name,
                "timestamp": time.time(),
                **kwargs,
            },
        )
        logger.debug(
            f"Emitted TRAINING_SLOT_UNAVAILABLE for {board_type}_{num_players}p: {reason}"
        )

    def _escalate_lock_failure(
        self,
        config_key: str,
        board_type: str,
        num_players: int,
        consecutive_failures: int,
    ) -> None:
        """Escalate repeated lock acquisition failures.

        January 2, 2026: Added to surface persistent lock issues that may indicate
        infrastructure problems (dead locks, stale lock holders, network partitions).

        This method:
        1. Emits a high-severity TRAINING_LOCK_ESCALATION event
        2. Logs at ERROR level for operator visibility
        3. Increments error counters for health monitoring

        Args:
            config_key: Config key (e.g., "hex8_2p")
            board_type: Board type
            num_players: Number of players
            consecutive_failures: Number of consecutive lock failures
        """
        self._errors_count += 1
        self._last_error = f"Lock escalation for {config_key}: {consecutive_failures} consecutive failures"

        logger.error(
            f"[TrainingCoordinator] ESCALATION: {consecutive_failures} consecutive lock "
            f"failures for {config_key}. Possible causes: dead lock holder, network "
            f"partition, stale NFS lock. Investigation required."
        )

        self._emit_via_router(
            "TRAINING_LOCK_ESCALATION",
            {
                "config_key": config_key,
                "board_type": board_type,
                "num_players": num_players,
                "consecutive_failures": consecutive_failures,
                "escalation_threshold": self._lock_failure_escalation_threshold,
                "node_name": self._node_name,
                "timestamp": time.time(),
                "severity": "high",
                "suggested_actions": [
                    "Check for stale lock files in NFS",
                    "Verify lock holder node is alive",
                    "Check network connectivity",
                    "Consider manual lock cleanup if holder is dead",
                ],
            },
        )

    def _emit_via_router(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit event via unified router.

        Args:
            event_type: Event type string
            payload: Event payload
        """
        try:
            import asyncio

            from app.coordination.event_router import publish, publish_sync
            from app.core.async_context import safe_create_task

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No event loop running - use sync publish
                publish_sync(
                    event_type=event_type,
                    payload=payload,
                    source="TrainingCoordinator",
                )
            else:
                safe_create_task(
                    publish(
                        event_type=event_type,
                        payload=payload,
                        source="TrainingCoordinator",
                    ),
                    name=f"training_coordinator_emit_{event_type.lower()}",
                )
        except Exception as e:
            logger.debug(f"Failed to emit {event_type}: {e}")

    def get_active_jobs(self) -> list[TrainingJob]:
        """Get all active training jobs."""
        conn = self._get_connection()
        self._cleanup_stale_jobs()

        cursor = conn.execute(
            '''SELECT * FROM training_jobs WHERE status = 'running'
               ORDER BY started_at'''
        )

        jobs = []
        for row in cursor.fetchall():
            jobs.append(TrainingJob(
                job_id=row["job_id"],
                board_type=row["board_type"],
                num_players=row["num_players"],
                node_name=row["node_name"],
                node_ip=row["node_ip"],
                pid=row["pid"],
                started_at=row["started_at"],
                last_heartbeat=row["last_heartbeat"],
                status=row["status"],
                model_version=row["model_version"],
                epochs_completed=row["epochs_completed"],
                best_val_loss=row["best_val_loss"],
                current_elo=row["current_elo"],
                metadata=json.loads(row["metadata"] or "{}"),
            ))
        return jobs

    def get_job(self, board_type: str, num_players: int) -> TrainingJob | None:
        """Get the active training job for a config if any."""
        conn = self._get_connection()
        cursor = conn.execute(
            '''SELECT * FROM training_jobs
               WHERE board_type = ? AND num_players = ? AND status = 'running' ''',
            (board_type, num_players)
        )
        row = cursor.fetchone()
        if not row:
            return None

        return TrainingJob(
            job_id=row["job_id"],
            board_type=row["board_type"],
            num_players=row["num_players"],
            node_name=row["node_name"],
            node_ip=row["node_ip"],
            pid=row["pid"],
            started_at=row["started_at"],
            last_heartbeat=row["last_heartbeat"],
            status=row["status"],
            model_version=row["model_version"],
            epochs_completed=row["epochs_completed"],
            best_val_loss=row["best_val_loss"],
            current_elo=row["current_elo"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def get_training_history(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get training history."""
        conn = self._get_connection()

        query = "SELECT * FROM training_history WHERE 1=1"
        params: list[Any] = []

        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)
        if num_players:
            query += " AND num_players = ?"
            params.append(num_players)

        query += " ORDER BY completed_at DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def _get_current_elo(self, board_type: str, num_players: int) -> float:
        """Get the current best Elo rating for a config.

        Jan 6, 2026: P2 - Used to capture before_elo at training start.

        Args:
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players

        Returns:
            Current best Elo for the config, or 0.0 if unavailable
        """
        try:
            # Lazy import to avoid circular dependency
            from app.training.elo_service import get_elo_service

            elo_service = get_elo_service()
            leaderboard = elo_service.get_leaderboard(
                board_type=board_type,
                num_players=num_players,
                limit=1,
            )
            if leaderboard:
                # Return the top Elo rating
                return leaderboard[0].elo
        except ImportError:
            logger.debug("[TrainingCoordinator] EloService not available")
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            logger.debug(f"[TrainingCoordinator] Elo lookup unavailable: {e}")
        except (AttributeError, TypeError, IndexError) as e:
            logger.debug(f"[TrainingCoordinator] Could not get current Elo: {e}")

        return 0.0

    def update_training_final_elo(
        self,
        board_type: str,
        num_players: int,
        final_elo: float,
    ) -> bool:
        """Update the final_elo for the most recent completed training job.

        Called after EVALUATION_COMPLETED to record the Elo rating achieved
        by the model trained in that job. This closes the feedback loop
        between training and evaluation.

        Jan 6, 2026: P1 improvement for model improvement tracking.

        Args:
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players
            final_elo: The Elo rating from evaluation

        Returns:
            True if a training record was updated, False otherwise
        """
        conn = self._get_connection()

        # Find the most recent completed training for this config
        cursor = conn.execute(
            """SELECT history_id, job_id, final_elo
               FROM training_history
               WHERE board_type = ? AND num_players = ?
                 AND status = 'completed'
               ORDER BY completed_at DESC
               LIMIT 1""",
            (board_type, num_players)
        )
        row = cursor.fetchone()

        if not row:
            logger.debug(
                f"[TrainingCoordinator] No completed training found for "
                f"{board_type}_{num_players}p to update final_elo"
            )
            return False

        history_id = row["history_id"]
        old_elo = row["final_elo"]

        # Update the final_elo
        conn.execute(
            "UPDATE training_history SET final_elo = ? WHERE history_id = ?",
            (final_elo, history_id)
        )
        conn.commit()

        logger.info(
            f"[TrainingCoordinator] Updated final_elo for {board_type}_{num_players}p "
            f"training job {row['job_id']}: {old_elo} → {final_elo:.0f}"
        )
        return True

    def _cleanup_stale_jobs(self) -> int:
        """Remove stale training jobs."""
        conn = self._get_connection()
        now = time.time()

        # Find stale jobs
        heartbeat_threshold = now - (HEARTBEAT_INTERVAL_SECONDS * 3)
        age_threshold = now - (TRAINING_TIMEOUT_HOURS * 3600)

        cursor = conn.execute(
            '''SELECT job_id, board_type, num_players, node_name, started_at
               FROM training_jobs
               WHERE status = 'running'
                 AND (last_heartbeat < ? OR started_at < ?)''',
            (heartbeat_threshold, age_threshold)
        )
        stale_jobs = cursor.fetchall()

        for job in stale_jobs:
            logger.warning(
                f"Cleaning up stale training job {job['job_id']} "
                f"from {job['node_name']}"
            )
            # Archive with failed status
            conn.execute(
                '''INSERT INTO training_history
                   (job_id, board_type, num_players, node_name, started_at,
                    completed_at, status, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, 'stale', '{}')''',
                (
                    job["job_id"], job["board_type"], job["num_players"],
                    job["node_name"], job["started_at"], now
                )
            )
            conn.execute(
                "DELETE FROM training_jobs WHERE job_id = ?",
                (job["job_id"],)
            )

            # Release the distributed lock
            config_key = f"{job['board_type']}_{job['num_players']}p"
            lock = _create_distributed_lock(f"training:{config_key}")
            lock.release()

        if stale_jobs:
            conn.commit()
        return len(stale_jobs)

    def get_status(self) -> dict[str, Any]:
        """Get overall training coordination status."""
        conn = self._get_connection()
        self._cleanup_stale_jobs()

        cursor = conn.execute(
            "SELECT COUNT(*) FROM training_jobs WHERE status = 'running'"
        )
        active_count = cursor.fetchone()[0]

        cursor = conn.execute(
            '''SELECT board_type, num_players, node_name, epochs_completed,
                      best_val_loss, (? - started_at) / 3600 as hours_running
               FROM training_jobs WHERE status = 'running'
               ORDER BY started_at''',
            (time.time(),)
        )
        active_jobs = [
            {
                "config": f"{row['board_type']}_{row['num_players']}p",
                "node": row["node_name"],
                "epochs": row["epochs_completed"],
                "best_loss": round(row["best_val_loss"], 4),
                "hours": round(row["hours_running"], 2),
            }
            for row in cursor.fetchall()
        ]

        return {
            "active_jobs": active_count,
            "max_concurrent": MAX_TOTAL_CONCURRENT_TRAINING,
            "slots_available": MAX_TOTAL_CONCURRENT_TRAINING - active_count,
            "coordinator_node": self._node_name,
            "db_path": str(self._db_path),
            "using_nfs": "nfs" in str(self._db_path).lower(),
            "jobs": active_jobs,
        }

    # =========================================================================
    # Async Wrappers for Event Loop Safety (Sprint 17.3 - January 2026)
    # =========================================================================
    #
    # These async methods wrap synchronous TrainingCoordinator operations using
    # asyncio.to_thread() to prevent blocking the event loop. Use these from
    # HandlerBase subclasses and other async code paths.

    async def can_start_training_async(
        self, board_type: str, num_players: int
    ) -> bool:
        """Async wrapper for can_start_training().

        Checks if training can start without blocking the event loop.

        Args:
            board_type: The board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)

        Returns:
            True if training can start, False otherwise

        Sprint 17.3 (Jan 4, 2026): Added for async-safe training coordination.
        """
        import asyncio
        return await asyncio.to_thread(self.can_start_training, board_type, num_players)

    async def start_training_async(
        self,
        board_type: str,
        num_players: int,
        node_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Async wrapper for start_training().

        Starts training without blocking the event loop.

        Args:
            board_type: The board type
            num_players: Number of players
            node_name: Optional node name override
            metadata: Optional training metadata

        Returns:
            Job ID if training started, None otherwise

        Sprint 17.3 (Jan 4, 2026): Added for async-safe training coordination.
        """
        import asyncio
        return await asyncio.to_thread(
            self.start_training, board_type, num_players, node_name, metadata
        )

    async def complete_training_async(
        self,
        job_id: str,
        status: str = "completed",
        final_val_loss: float | None = None,
        final_elo: float | None = None,
    ) -> bool:
        """Async wrapper for complete_training().

        Completes training without blocking the event loop.

        Args:
            job_id: The training job ID
            status: Final status (completed, failed)
            final_val_loss: Final validation loss
            final_elo: Final Elo rating

        Returns:
            True if training was completed, False otherwise

        Sprint 17.3 (Jan 4, 2026): Added for async-safe training coordination.
        """
        import asyncio
        return await asyncio.to_thread(
            self.complete_training, job_id, status, final_val_loss, final_elo
        )

    async def get_status_async(self) -> dict[str, Any]:
        """Async wrapper for get_status().

        Gets training coordination status without blocking the event loop.

        Returns:
            Dictionary with training coordination status

        Sprint 17.3 (Jan 4, 2026): Added for async-safe status queries.
        """
        import asyncio
        return await asyncio.to_thread(self.get_status)

    async def health_check_async(self) -> HealthCheckResult:
        """Async wrapper for health_check().

        Gets health status without blocking the event loop.

        Returns:
            HealthCheckResult with training coordinator health

        Sprint 17.3 (Jan 4, 2026): Added for async-safe health checks.
        """
        import asyncio
        return await asyncio.to_thread(self.health_check)

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
